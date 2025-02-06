import pathlib
import shutil
import threading

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, UTC
from functools import partial

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt
import numpy
import pandas
import rasterio
import utm

from hotlink.load_hotlink_model import load_hotlink_model
from hotlink import preprocess, support_functions
from skimage.filters import apply_hysteresis_threshold

def process_image(
    vent: tuple[float, float],
    elevation: int,
    model: "Model",
    out_dir: pathlib.Path,
    file: pathlib.Path
) -> dict:
    """
    Process a satellite image to detect volcanic hotspots and compute radiative power.

    The function loads a thermal infrared (TIR) and mid-infrared (MIR) image,
    normalizes and processes the data, applies a deep learning model to
    identify hotspots, and calculates various temperature and solar parameters.

    Parameters:
    -----------
    vent : tuple[float, float]
        Latitude and longitude of the volcano vent.
    elevation : float
        Elevation of the volcano in meters.
    model : tensorflow.keras.Model
        Trained model for hotspot classification.
    file : pathlib.Path
        Path to the input image file.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'source file' (str): Original filename.
        - 'date' (datetime.datetime): Timestamp extracted from the filename.
        - 'radiative_power' (float): Estimated radiative power in Watts.
        - 'day_night flag' (str): d if daytime, n if nighttime.
        - 'max_prob' (float): Maximum probability of a hotspot in the image.
        - 'mir_hotspot_bt' (float): Mean MIR brightness temperature of hotspots.
        - 'mir_background_bt' (float): Mean MIR brightness temperature of background.
        - 'mir_hotspot_max_bt' (float): Maximum MIR brightness temperature of hotspots.
        - 'tir_hotspot_bt' (float): Mean TIR brightness temperature of hotspots.
        - 'tir_hotspot_max_bt' (float): Maximum TIR brightness temperature of hotspots.
        - 'tir_background_bt' (float): Mean TIR brightness temperature of background.
        - 'num_hotspot_pixels' (int): Number of detected hotspot pixels.
        - 'solar_zenith' (float): Solar zenith angle at capture time.
        - 'solar_azimuth' (float): Solar azimuth angle at capture time.
        - 'Pixels above 0.5 prob' (int): Count of pixels with a probability > 0.5.

    Notes:
    ------
    - Uses hysteresis thresholding to refine hotspot detection.
    - Deletes the input file after processing.
    """
    filename = file.name
    image_date = datetime.strptime(file.stem, '%Y%m%d_%H%M')

    out_dir = out_dir / str(image_date.year) / str(image_date.month)
    out_dir.mkdir(parents = True, exist_ok = True)

    data = numpy.load(file)

    mir = data[:, :, 0].copy()
    tir = data[:, :, 1].copy()

    # Save MIR and TIR images
    mir_image = out_dir / f"{file.stem}_mir.png"
    tir_image = out_dir / f"{file.stem}_tir.png"
    _save_fig(
        mir,
        mir_image,
        f"Middle Infrared\n{image_date.strftime('%Y-%m-%d %H:%M')}"
    )

    _save_fig(
        tir,
        tir_image,
        f"Thermal Infrared\n{image_date.strftime('%Y-%m-%d %H:%M')}"
    )

    n_mir = support_functions.normalize_MIR(mir) #note, normalize also fills in missing pixels
    n_tir = support_functions.normalize_TIR(tir)
    stacked = numpy.dstack([n_mir, n_tir])

    # Make sure the original mir array doesn't have any missing pixels
    min_mir_observered = numpy.nanmin(mir)
    mir[numpy.isnan(mir)] = min_mir_observered

    # Fill missing pixels for tir
    min_tir_observered = numpy.nanmin(tir)
    tir[numpy.isnan(tir)] = min_tir_observered

    predict_data = support_functions.crop_center(stacked).reshape(1, 64, 64, 2)
    prediction = model.predict(predict_data) #shape=[batch_size, 24, 24, 3], for 3 predicted classes:background, hotspot-adjacent, and hotspot

    # get predicted class for each pixel (highest probability)
    # pred_classes = numpy.array(numpy.argmax(prediction[0,:,:,:], axis=2)) #classes 0,1,2 correspond to bg, hot-adjacent, and hot

    # use hysteresis thresholding to generate a binary map of hotspot pixels
    prob_active = prediction[0,:,:,2] #map with probabilities of active class
    max_prob = numpy.max(prob_active) #highest probability per image, equated to probability that the image contains a hotspot
    prob_above_05 = numpy.count_nonzero(prob_active>0.5)
    hotspot_mask = apply_hysteresis_threshold(prob_active, low=0.4, high=0.5).astype('bool') #hysteresis thresholding active mask
    num_hotspot_pixels = numpy.count_nonzero(hotspot_mask)

    # Save probability matrix to a tiff file (geo transform will be added later)
    geotiff_file = out_dir / f"{file.stem}_probability.tif"
    with rasterio.open(
        geotiff_file,
        'w',
        driver = 'GTiff',
        height = prob_active.shape[0],
        width = prob_active.shape[1],
        count = 1,
        dtype = prob_active.dtype
    ) as dst:
        dst.write(prob_active, 1)

    # generate results
    mir_analysis = support_functions.crop_center(mir, size=24) #crop to output size, for analysis
    tir_analysis = support_functions.crop_center(tir, size = 24) # So we can get the background TIR brightness temperature
    if hotspot_mask.any():
        rp = support_functions.radiative_power(mir_analysis, hotspot_mask) # in Watts
    else:
        rp = numpy.nan

    # mir hotspot/background brightness temperature analysis
    mir_hotspot = mir_analysis[hotspot_mask ]
    mir_background = mir_analysis[~hotspot_mask]
    hotspot_mir_bt = support_functions.brightness_temperature(mir_hotspot, wl=3.74e-6)
    bg_mir_bt = support_functions.brightness_temperature(mir_background, wl=3.74e-6)

    # tir hotspot/background brigbhtness temerature analysis
    tir_hotspot = tir_analysis[hotspot_mask ]
    tir_background = tir_analysis[~hotspot_mask]
    hotspot_tir_bt = support_functions.brightness_temperature(tir_hotspot, wl=3.74e-6)
    bg_tir_bt = support_functions.brightness_temperature(tir_background, wl=3.74e-6)

    try:
        mir_max_hs_bt = hotspot_mir_bt.max().round(4)
        tir_max_hs_bt = hotspot_tir_bt.max().round(4)
    except ValueError:
        # Handle the no detection case
        mir_max_hs_bt = numpy.nan
        tir_max_hs_bt = numpy.nan

    day_night = support_functions.get_dn(image_date, vent[1], vent[0], elevation)
    sol_zenith, sol_azimuth = support_functions.get_solar_coords(image_date, vent[1], vent[0], elevation)

    result = {
        'Date': image_date,
        'Hotspot Radiative Power (W)': round(rp, 4),
        'Day/Night Flag': day_night,
        'Max Probability': round(max_prob, 3),
        'MIR Hotspot Brightness Temperature': hotspot_mir_bt.mean().round(4),
        'MIR Hotspot Max Brightness Temperature': mir_max_hs_bt,
        'MIR Background Brightness Temperature': bg_mir_bt.mean().round(4),
        'TIR Hotspot Brightness Temperature': hotspot_tir_bt.mean().round(4),
        'TIR Hotspot Max Brightness Temperature': tir_max_hs_bt,
        'TIR Background Brightness Temperature': bg_tir_bt.mean().round(4),
        'Number Hotspot Pixels': num_hotspot_pixels,
        'Pixels Above 0.5 Probability': prob_above_05,
        'Solar Zenith': round(sol_zenith, 1),
        'Solar Azimuth': round(sol_azimuth, 1),
        'Data File': filename,
        'MIR Image': str(mir_image),
        'TIR Image': str(tir_image),
        'Probability TIFF': str(geotiff_file),
    }

    # Move the data file into the output directory
    shutil.move(str(file), str(out_dir/file.name))

    return result

lock = threading.Lock()
def _save_fig(img, out, title):
    with lock:
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.colorbar()
        plt.title(title)
        fig.savefig(str(out))
        plt.close(fig)


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

    download_meta = preprocess.download_preprocess(dates, vent, sensor, folder = data_path)
    print("Image files processed. Beginning calculations")

    data_files = list(data_path.glob('*.npy'))

    model = load_hotlink_model()

    # Using the thread pool here provides a very modest (~16% in my testing) speedup.
    # It might not be worth the complexity for smaller image sets, but could help a bit with
    # larger sets. ProcessPoolExecutor is horrible here, due to the need for
    # every individual process to import/set up tensorflow, coupled with
    # inter-process communications.
    process_func = partial(process_image, vent, elevation, model, output_dir)
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_func, data_files)

    results = pandas.DataFrame(results)
    meta['Result Count'] = len(results)

    # Add results that apply to all
    results['Sensor'] = sensor.upper()
    results['Volcano ID'] = volc.iloc[0]['id']

    # pull in metadata retrieved during the download
    file_meta = results['Data File'].map(download_meta)
    results['Satellite'] = file_meta.map(lambda x: x.get('satelite'))
    results['Data URL'] = file_meta.map(lambda x: x.get('url'))

    # Add the geo-transform to the generated TIFF files.
    center_x, center_y, utm_zone, utm_lat_band = utm.from_latlon(*vent)
    resolution = 1000 if sensor == 'MODIS' else 375
    size = 24
    transform = rasterio.transform.from_origin(center_x - (size / 2) * resolution,
                                               center_y + (size / 2) * resolution,
                                               resolution, resolution)
    
    hemisphere = hemisphere = " +south" if utm_lat_band < 'N' else ""
    
    crs = f"+proj=utm +zone={utm_zone}{hemisphere} +datum=WGS84 +units=m +no_defs"
    meta['UTM Zone'] = utm_zone
    meta['UTM Latitude Band'] = utm_lat_band

    def update_geotransform(file_path):
        with rasterio.open(file_path, 'r+') as dst:
            dst.transform = transform
            dst.crs = crs

    for file_path in results['Probability TIFF']:
        update_geotransform(file_path)

    if len(results) > 0:
        results = results.sort_values('Date').reset_index(drop = True)
        
    meta['Run End'] = datetime.now(UTC).isoformat()
    return results, meta
