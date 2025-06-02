import pathlib
import shutil
import threading

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from functools import partial

import matplotlib
import tqdm

matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import numpy
import pandas
import rasterio
import utm

from hotlink.load_hotlink_model import load_hotlink_model
from hotlink import preprocess, support_functions
from skimage.filters import apply_hysteresis_threshold

def _gen_output_dir(
    input_filename: pathlib.Path,
    base_dir: pathlib.Path
) -> pathlib.Path:
    """
    Generate an output directory path based on the date in a filename.

    Constructs a directory path under `base_dir` using the year and month extracted
    from `input_filename.stem`.

    Parameters
    ----------
    input_filename : pathlib.Path
        The input file whose stem contains a date in the format 'YYYYMMDD_HHMM'.
    base_dir : pathlib.Path
        The base directory under which the year/month subdirectories are created.

    Returns
    -------
    pathlib.Path: The constructed output directory (e.g., `base_dir/YYYY/MM`).

    Examples
    --------
    >>> from pathlib import Path
    >>> file = Path("20231106_1230.npy")
    >>> base = Path("/output")
    >>> dir = _gen_output_dir(file, base)
    >>> print(dir)
    /output/2023/11
    """
    # make sure arguments are actually pathlib.Path objects so the rest of the code works.
    stem = pathlib.Path(input_filename).stem
    base_dir = pathlib.Path(base_dir)
    out_dir = base_dir / stem[:4] / stem[4:6]
    return out_dir


lock = threading.Lock()
def _save_fig(img, out, title):
# def save_fig(img, out, title, lat_bounds, lon_bounds):
    # with lock:
        # fig = plt.figure(figsize=(4, 4))

        # # Get the extent in the format [left, right, bottom, top]
        # extent = [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]]

        # # Display the image with coordinates
        # ax = plt.gca()
        # im = ax.imshow(img, extent=extent, origin='lower')

        # # Add scale bar (1km)
        # # Convert distance to decimal degrees (approximately)
        # # At equator, 1 degree ≈ 111 km, but varies with latitude
        # lat_mid = (lat_bounds[0] + lat_bounds[1]) / 2
        # lon_per_km = 1 / (111.32 * np.cos(np.radians(lat_mid)))  # Degrees per km at this latitude

        # # Add scale bar
        # from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        # import matplotlib.font_manager as fm
        # fontprops = fm.FontProperties(size=8)
        # scalebar = AnchoredSizeBar(ax.transData,
                                  # lon_per_km,  # 1 km in decimal degrees
                                  # '1 km',
                                  # 'lower right',
                                  # pad=0.5,
                                  # color='black',
                                  # frameon=False,
                                  # size_vertical=0.005,
                                  # fontproperties=fontprops)
        # ax.add_artist(scalebar)

        # plt.colorbar(im)
        # plt.title(title)
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')

        # fig.savefig(str(out))
        # plt.close(fig)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.colorbar()
    plt.title(title)
    fig.savefig(str(out))
    plt.close(fig)

def load_data_files(files):
    # Load the first file to determine the shape and store its data
    first_data = numpy.load(files[0])

    # pre-allocate the image_dates array. Minor speed up, if any, but why not?
    image_dates = [None] * len(files)
    image_dates[0] = datetime.strptime(files[0].stem, '%Y%m%d_%H%M')

    output_shape = (len(files),) + first_data.shape
    output_array = numpy.empty(output_shape, dtype=first_data.dtype)

    # Assign the first file's data to the output array
    output_array[0] = first_data

    # Load the remaining files into the output array
    for i, file_path in enumerate(tqdm.tqdm(
        files[1:],
        total = len(files) - 1,
        desc = "LOADING DATA",
        unit = "file"
    ), start =1):
        # for i, file_path in enumerate(files[1:], start=1):
        data = numpy.load(file_path)
        date = datetime.strptime(file_path.stem, '%Y%m%d_%H%M')
        output_array[i] = data
        image_dates[i] = date

    return output_array, image_dates

def get_results(
    vent: str | tuple[float, float],
    elevation: int,
    dates: tuple[str, str],
    sensor: str,
    out_dir: str | pathlib.Path | None = None
) -> (pandas.DataFrame, dict):

    """
    Retrieve and process satellite images for a given volcano and date range.

    This function downloads satellite images for the specified volcano or vent
    location from the EarthScope database, processes them using the HotLINK
    machine learning model, and returns a pandas DataFrame containing statistical
    results for each processed image.

    Parameters
    ----------
    vent : str | tuple[float, float]
        The name of the volcano (e.g., "Shishaldin") or the coordinates of
        the vent as a tuple (latitude, longitude).
    elevation : int
        The elevation of the vent in meters above sea level.
    dates : tuple[str, str]
        A tuple specifying the start and end dates for data retrieval in the
        format "YYYY-MM-DD" (e.g., `("2023-01-01", "2023-12-31")`).
    sensor : str
        The satellite sensor to retrieve data from. Must be one of:
        - 'viirs': Visible Infrared Imaging Radiometer Suite
        - 'modis': Moderate Resolution Imaging Spectroradiometer
    out_dir : str | Path, default "Output/{sensor}"
        The directory in which to save output image products. Will be created
        if it does not exist.

    Returns
    -------
    results: pandas.DataFrame
        A DataFrame containing the processed results for each image. Each row
        corresponds to an  input image and includes model output.
    meta: dict
        A Dictionary containing metadata about the run

    Raises
    ------
    ValueError
        If the specified volcano name is not found in the volcano database.

    Notes
    -----
    - The function determines the volcano location based on the name or coordinates.
      If a name is provided, it searches for the volcano in the internal database.
      If coordinates are provided, it finds the nearest volcano to the given location.

    Examples
    --------
    >>> results = get_results(
    ...     vent="Shishaldin",
    ...     elevation=2550,
    ...     dates=("2019-01-01", "2019-12-31"),
    ...     sensor="viirs",
    ...     out_dir="Output Images"
    ... )
    >>> print(results)

    >>> results = get_results(
    ...     vent=(54.7554, -163.9711),
    ...     elevation=2550,
    ...     dates=("2019-01-01", "2019-12-31"),
    ...     sensor="viirs"
    ... )
    >>> print(results)
    """

    meta = {
        'Vent': vent,
        'Elevation': elevation,
        'Data Dates': dates,
        'Sensor': sensor,
        'Run Start': datetime.now(UTC).isoformat(),
    }

    volcs = support_functions.load_volcanoes()

    if isinstance(vent, str):
        volc = volcs[volcs['name']==vent]
        if len(volc) == 0:
            raise ValueError("Specified volcano not found!")
        vent = (volc.iloc[0]['lat'], volc.iloc[0]['lon'])
    else:
        dists = support_functions.haversine_np(vent[1], vent[0], volcs['lon'], volcs['lat'])
        volcs.loc[:, 'dist'] = dists
        volc = volcs[volcs['dist']==volcs['dist'].min()]

    print("Using volcano:", volc.iloc[0]['name'], "location:", vent)

    meta['Volcano Name'] = volc.iloc[0]['name']
    meta['Volcano ID'] = volc.iloc[0]['id']
    meta['Center'] = vent

    if out_dir is None:
        out_dir = pathlib.Path("Output") / sensor

    # Make sure this is a pathlib.Path object, and make sure it exists, creating it if needed.
    output_dir = pathlib.Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents = True, exist_ok = True)

    data_path = pathlib.Path('./data')

    # make sure the data directory exists
    data_path.mkdir(exist_ok = True)

    print("Searching for files to download...")
    download_meta = preprocess.download_preprocess(
        dates,
        vent,
        sensor,
        folder = data_path,
        output=output_dir)

    print("Image files processed. Beginning calculations")

    # Set some constants based on sensor
    if sensor.upper() == 'MODIS':
        RES = 1000
        MIR_WL = support_functions.MODIS_MIR_WL
        TIR_WL = support_functions.MODIS_TIR_WL
        RP_CONSTANT = support_functions.MODIS_MIR_RP_CONSTANT
    else:
        # VIIRS
        RES = 375
        MIR_WL = support_functions.VIIRS_MIR_WL
        TIR_WL = support_functions.VIIRS_TIR_WL
        RP_CONSTANT = support_functions.VIIRS_MIR_RP_CONSTANT

    # Calculate the GeoTransform for output images
    center_x, center_y, utm_zone, utm_lat_band = utm.from_latlon(*vent)
    # resolution = 1000 if sensor.upper() == 'MODIS' else 375
    # size = 24
    # transform = rasterio.transform.from_origin(center_x - (size / 2) * resolution,
                                               # center_y + (size / 2) * resolution,
                                               # resolution, resolution)

    # hemisphere = hemisphere = " +south" if utm_lat_band < 'N' else ""

    # crs = f"+proj=utm +zone={utm_zone}{hemisphere} +datum=WGS84 +units=m +no_defs"
    meta['UTM Zone'] = utm_zone
    meta['UTM Latitude Band'] = utm_lat_band

    data_files = list(data_path.glob('*.npy'))

    if not data_files:
        print("WARNING: No data files to process.")
        # Define the expected columns for an empty DataFrame
        expected_columns = [
            'Data File', 'Number Hotspot Pixels', 'Hotspot Radiative Power (W)',
            'MIR Hotspot Brightness Temperature', 'MIR Background Brightness Temperature',
            'MIR Hotspot Max Brightness Temperature', 'TIR Hotspot Brightness Temperature',
            'TIR Background Brightness Temperature', 'TIR Hotspot Max Brightness Temperature',
            'Day/Night Flag', 'Solar Zenith', 'Solar Azimuth', 'Date', 'Max Probability',
            'Pixels Above 0.5 Probability', 'Sensor', 'Volcano ID', 'Satellite', 'Data URL'
        ]
        # Return an empty DataFrame with the expected structure
        empty_results = pandas.DataFrame(columns=expected_columns)
        # Update meta with the failure reason and end time
        meta['Result Count'] = 0
        meta['Error'] = "No .npy files found in the data directory"
        meta['Run End'] = datetime.now(UTC).isoformat()
        return empty_results, meta

    model = load_hotlink_model()

    img_data, img_dates = load_data_files(data_files)
    # Make sure there are no missing pixels
    mir_data = img_data[:, :, :, 0] # creates a view
    tir_data = img_data[:, :, :, 1]

    # Create masks for NaN values
    nan_mask_tir = numpy.isnan(tir_data)
    nan_mask_mir = numpy.isnan(mir_data)

    # Fill NaN values in the mir/tir arrays with min values for the array
    if nan_mask_mir.any():
        # create arrays with the minimum value for each image
        min_mir_observed = numpy.nanmin(mir_data, axis=(1, 2), keepdims=True)
        min_mir_observed = numpy.broadcast_to(min_mir_observed, mir_data.shape)
        # Fill NaN values with the corresponding minimum values
        mir_data[nan_mask_mir] = min_mir_observed[nan_mask_mir]

    if nan_mask_tir.any():
        min_tir_observed = numpy.nanmin(tir_data, axis=(1, 2), keepdims=True)
        min_tir_observed = numpy.broadcast_to(min_tir_observed, tir_data.shape)
        tir_data[nan_mask_tir] = min_tir_observed[nan_mask_tir]

    mir_analysis = support_functions.crop_center(mir_data, size=24, crop_dimensions=(1, 2))
    tir_analysis = support_functions.crop_center(tir_data, size=24, crop_dimensions=(1, 2))

    mir_bt = support_functions.brightness_temperature(mir_analysis, wl=MIR_WL)
    tir_bt = support_functions.brightness_temperature(tir_analysis, wl=TIR_WL)

    n_data = img_data.copy()
    n_data[:, :, :, 0] = support_functions.normalize_MIR(n_data[:, :, :, 0])
    n_data[:, :, :, 1] = support_functions.normalize_TIR(n_data[:, :, :, 1])

    predict_data = support_functions.crop_center(n_data, crop_dimensions=(1, 2))
    predict_data = predict_data.reshape(n_data.shape[0], 64, 64, 2)

    print("Predicting hotspots...")
    prediction = model.predict(predict_data) #shape=[batch_size, 24, 24, 3], for 3 predicted classes:background, hotspot-adjacent, and hotspot

    # use hysteresis thresholding to generate a binary map of hotspot pixels
    prob_active = prediction[:,:,:,2] #map with probabilities of active class

    #highest probability per image, equated to probability that the image contains a hotspot
    max_prob = numpy.max(prob_active, axis=(1, 2))
    prob_above_05 = numpy.count_nonzero(prob_active>0.5, axis=(1, 2))

    process_progress = tqdm.tqdm(
        total=img_data.shape[0],
        desc="CALCULATING RESULTS"
    )

    def _run_calcs(idx):
        result = {}
        img_file = data_files[idx]
        image_date = img_dates[idx]

        result['Data File'] = img_file.name

        hotspot_mask = apply_hysteresis_threshold(prob_active[idx], low=0.4, high=0.5).astype('bool')
        hotspot_pixels = numpy.count_nonzero(hotspot_mask)
        result['Number Hotspot Pixels'] = hotspot_pixels

        rp = support_functions.radiative_power(
            mir_analysis[idx],
            hotspot_mask,
            cellsize=RES,
            rp_constant=RP_CONSTANT
        ) if hotspot_mask.any() else numpy.nan

        result['Hotspot Radiative Power (W)'] = round(rp, 4)

        # mir hotspot/background brightness temperature analysis
        hotspot_mir_bt = mir_bt[idx][hotspot_mask]
        bg_mir_bt = mir_bt[idx][~hotspot_mask]

        result['MIR Hotspot Brightness Temperature'] = hotspot_mir_bt.mean().round(4)
        result['MIR Background Brightness Temperature'] = bg_mir_bt.mean().round(4)
        result['MIR Hotspot Max Brightness Temperature'] = hotspot_mir_bt.max().round(4) if hotspot_mir_bt.size > 0 else numpy.nan

        # tir hotspot/background brigbhtness temerature analysis
        hotspot_tir_bt = tir_bt[idx][hotspot_mask]
        bg_tir_bt = tir_bt[idx][~hotspot_mask]

        result['TIR Hotspot Brightness Temperature'] = hotspot_tir_bt.mean().round(4)
        result['TIR Background Brightness Temperature'] = bg_tir_bt.mean().round(4)
        result['TIR Hotspot Max Brightness Temperature'] = hotspot_tir_bt.max().round(4) if hotspot_tir_bt.size > 0 else numpy.nan

        day_night = support_functions.get_dn(image_date, vent[1], vent[0], elevation)
        sol_zenith, sol_azimuth = support_functions.get_solar_coords(
            image_date, vent[1], vent[0], elevation
        )

        result['Day/Night Flag'] = day_night
        result['Solar Zenith'] = round(sol_zenith, 1)
        result['Solar Azimuth'] = round(sol_azimuth, 1)

        process_progress.update()
        return result

    # Not sure if this is really needed, as this loop is fast, but might
    # speed things up a bit.
    with ThreadPoolExecutor() as executor:
        results = executor.map(_run_calcs, range(img_data.shape[0]))

    results = pandas.DataFrame(results)
    results.reset_index(drop=True, inplace=True)

    results['Date'] = img_dates
    results['Max Probability'] = max_prob
    results['Pixels Above 0.5 Probability'] = prob_above_05

    # Single values apply to all records
    results['Sensor'] = sensor.upper()
    results['Volcano ID'] = volc.iloc[0]['id']

    # pull in metadata retrieved during the download
    file_meta = results['Data File'].map(lambda x: download_meta.get(x, {}))
    results['Satellite'] = file_meta.map(lambda x: x.get('satelite'))
    results['Data URL'] = file_meta.map(lambda x: x.get('url'))

    SAVE_IMAGES = False # TODO: make this a user passable flag somewhere.

    for idx, (image_date, img_file) in tqdm.tqdm(
        enumerate(zip(img_dates, data_files)),
        total=len(img_dates),
        unit="IMAGES",
        desc="SAVING IMAGES"
    ):
        if SAVE_IMAGES:
            ########## IMAGE SAVE/Data File Archive #################
            # This section deals with saving PNG images and archiving
            # the pre-processed data files. Remove this section if not
            # desired
            #########################################################

            # Save the .png images. Second loop, but this one doesn't lend itself to
            # parallel processing at all.
            file_out_dir = _gen_output_dir(img_file, out_dir)
            file_out_dir.mkdir(parents=True, exist_ok=True)

            # Save MIR images
            mir_image = file_out_dir / f"{img_file.stem}_mir.png"
            results.loc[idx, 'MIR Image'] = str(mir_image)

            _save_fig(
                mir_data[idx],
                mir_image,
                f"Middle Infrared\n{image_date.strftime('%Y-%m-%d %H:%M')}"
            )

            # slice_prob_active = prob_active[idx]

            # Optional: save probability GeoTIFF (currently disabled)
            # NOTE: These files are EXTREAMLY tiny at only 24px x 24px
            # geotiff_file = output_dir / f"{img_file.stem}_probability.tif"
            # result['Probability TIFF'] = str(geotiff_file)

            # with rasterio.open(
                # geotiff_file,
                # 'w',
                # driver = 'GTiff',
                # height = slice_prob_active.shape[0],
                # width = slice_prob_active.shape[1],
                # count = 1,
                # dtype = slice_prob_active.dtype,
                # crs=crs,
                # transform=transform
            # ) as dst:
                # dst.write(slice_prob_active, 1)

            # Move the processed data file to the output directory
            shutil.move(str(img_file), str(file_out_dir / img_file.name))
            ###################### END IMAGE SECTION ###########################

        img_file.unlink(missing_ok=True)

    meta['Result Count'] = len(results)

    if len(results) > 0:
        results = results.sort_values('Date').reset_index(drop = True)

    meta['Run End'] = datetime.now(UTC).isoformat()
    return results, meta
