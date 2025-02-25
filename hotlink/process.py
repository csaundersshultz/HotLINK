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

def _gen_date_and_dir(input_filename, base_dir):
    image_date = datetime.strptime(input_filename.stem, '%Y%m%d_%H%M')
    out_dir = base_dir / str(image_date.year) / str(image_date.month)
    
    return (image_date, out_dir)
    
def process_image(
    vent: tuple[float, float],
    elevation: int,
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
    # Better safe than sorry
    file = pathlib.Path(file)
        
    image_date, out_dir = _gen_date_and_dir(file, out_dir)
    
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
        'Data File': file.name,
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

def load_data_files(files):
    # Load the first file to determine the shape and store its data
    first_data = numpy.load(files[0])
    image_dates = [datetime.strptime(files[0].stem, '%Y%m%d_%H%M')]
    
    output_shape = (len(files),) + first_data.shape
    output_array = numpy.empty(output_shape, dtype=first_data.dtype)
    
    # Assign the first file's data to the output array
    output_array[0] = first_data
    
    # Load the remaining files into the output array
    for i, file_path in enumerate(files[1:], start=1):
        data = numpy.load(file_path)
        date = datetime.strptime(file_path.stem, '%Y%m%d_%H%M')
        output_array[i] = data
        image_dates.append(date)
        
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
    
    # Calculate the GeoTransform for output images
    center_x, center_y, utm_zone, utm_lat_band = utm.from_latlon(*vent)
    resolution = 1000 if sensor.upper() == 'MODIS' else 375
    size = 24
    transform = rasterio.transform.from_origin(center_x - (size / 2) * resolution,
                                               center_y + (size / 2) * resolution,
                                               resolution, resolution)
    
    hemisphere = hemisphere = " +south" if utm_lat_band < 'N' else ""
    
    crs = f"+proj=utm +zone={utm_zone}{hemisphere} +datum=WGS84 +units=m +no_defs"
    meta['UTM Zone'] = utm_zone
    meta['UTM Latitude Band'] = utm_lat_band    

    data_files = list(data_path.glob('*.npy'))

    model = load_hotlink_model()

    img_data, img_dates = load_data_files(data_files)
    # Make sure there are no missing pixels
    mir_data = img_data[:, :, :, 0] # creates a view
    tir_data = img_data[:, :, :, 1]
    
    # Create masks for NaN values
    nan_mask_tir = numpy.isnan(tir_data)
    nan_mask_mir = numpy.isnan(mir_data)        
    
    if nan_mask_mir.any():
        min_mir_observed = numpy.nanmin(mir_data, axis=(1, 2), keepdims=True)
        min_mir_observed = numpy.broadcast_to(min_mir_observed, mir_data.shape)
        mir_data[nan_mask_mir] = min_mir_observed[nan_mask_mir]

    if nan_mask_tir.any():
        # create arrays with the minimum value for each image
        min_tir_observed = numpy.nanmin(tir_data, axis=(1, 2), keepdims=True)
        min_tir_observed = numpy.broadcast_to(min_tir_observed, tir_data.shape)
        # Fill NaN values with the corresponding minimum values
        tir_data[nan_mask_tir] = min_tir_observed[nan_mask_tir]
    
    mir_analysis = support_functions.crop_center(mir_data, size=24, crop_dimensions=(1, 2))
    tir_analysis = support_functions.crop_center(tir_data, size=24, crop_dimensions=(1, 2))
    
    n_data = img_data.copy()
    n_data[:, :, :, 0] = support_functions.normalize_MIR(n_data[:, :, :, 0])
    n_data[:, :, :, 1] = support_functions.normalize_MIR(n_data[:, :, :, 1])
    
    predict_data = support_functions.crop_center(n_data, crop_dimensions=(1, 2))
    predict_data = predict_data.reshape(n_data.shape[0], 64, 64, 2)
    
    print("Predicting hotspots...")
    prediction = model.predict(predict_data) #shape=[batch_size, 24, 24, 3], for 3 predicted classes:background, hotspot-adjacent, and hotspot
    
    # use hysteresis thresholding to generate a binary map of hotspot pixels
    prob_active = prediction[:,:,:,2] #map with probabilities of active class
    
    #highest probability per image, equated to probability that the image contains a hotspot
    max_prob = numpy.max(prob_active, axis=(1, 2)) 
    prob_above_05 = numpy.count_nonzero(prob_active>0.5, axis=(1, 2))
      

    
    result = {
        'Date': img_dates,
        'Hotspot Radiative Power (W)': [],
        'Day/Night Flag': [],
        'Max Probability': max_prob, 
        'MIR Hotspot Brightness Temperature': [],
        'MIR Hotspot Max Brightness Temperature': [],
        'MIR Background Brightness Temperature': [],
        'TIR Hotspot Brightness Temperature': [],
        'TIR Hotspot Max Brightness Temperature': [],
        'TIR Background Brightness Temperature': [],
        'Number Hotspot Pixels': [],
        'Pixels Above 0.5 Probability': prob_above_05,
        'Solar Zenith': [],
        'Solar Azimuth': [],
        'Data File': [],
        'MIR Image': [],
        'TIR Image': [],
        'Probability TIFF': [],
    }
    
    # loop...could we speed things up by using threading?
    for idx in range(img_data.shape[0]):
        img_file = data_files[idx]
        image_date = img_dates[idx]        
        result['Data File'].append(img_file.name)
        
        # Save MIR and TIR images
        mir_image = output_dir / f"{img_file.stem}_mir.png"
        result['MIR Image'].append(str(mir_image))
        tir_image = output_dir / f"{img_file.stem}_tir.png"
        result['TIR Image'].append(str(tir_image))
        _save_fig(
            mir_data[idx],
            mir_image,
            f"Middle Infrared\n{image_date.strftime('%Y-%m-%d %H:%M')}"
        )
    
        _save_fig(
            tir_data[idx],
            tir_image,
            f"Thermal Infrared\n{image_date.strftime('%Y-%m-%d %H:%M')}"
        )
        
        slice_prob_active = prob_active[idx]
        
        # Save probability matrix to a tiff file (geo transform will be added later)
        geotiff_file = output_dir / f"{img_file.stem}_probability.tif"
        result['Probability TIFF'].append(geotiff_file)

        with rasterio.open(
            geotiff_file,
            'w',
            driver = 'GTiff',
            height = slice_prob_active.shape[0],
            width = slice_prob_active.shape[1],
            count = 1,
            dtype = slice_prob_active.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(slice_prob_active, 1)        
        
        hotspot_mask = apply_hysteresis_threshold(slice_prob_active, low=0.4, high=0.5).astype('bool')
        hotspot_pixels = numpy.count_nonzero(hotspot_mask)
        result['Number Hotspot Pixels'].append(hotspot_pixels)        
        
        if hotspot_mask.any():
            rp = support_functions.radiative_power(mir_analysis[idx], hotspot_mask)
        else:
            rp = numpy.nan
            
        result['Hotspot Radiative Power (W)'].append(round(rp, 4))
        
        # mir hotspot/background brightness temperature analysis
        mir_hotspot = mir_analysis[idx][hotspot_mask ]
        mir_background = mir_analysis[idx][~hotspot_mask]
        hotspot_mir_bt = support_functions.brightness_temperature(mir_hotspot, wl=3.74e-6)
        bg_mir_bt = support_functions.brightness_temperature(mir_background, wl=3.74e-6)
        
        result['MIR Hotspot Brightness Temperature'].append(hotspot_mir_bt.mean().round(4))
        result['MIR Background Brightness Temperature'].append(bg_mir_bt.mean().round(4))
        
        # tir hotspot/background brigbhtness temerature analysis
        tir_hotspot = tir_analysis[idx][hotspot_mask ]
        tir_background = tir_analysis[idx][~hotspot_mask]
        hotspot_tir_bt = support_functions.brightness_temperature(tir_hotspot, wl=3.74e-6)
        bg_tir_bt = support_functions.brightness_temperature(tir_background, wl=3.74e-6)
        
        result['TIR Hotspot Brightness Temperature'].append(hotspot_tir_bt.mean().round(4))
        result['TIR Background Brightness Temperature'].append(bg_tir_bt.mean().round(4))
        
        try:
            mir_max_hs_bt = hotspot_mir_bt.max().round(4)
            tir_max_hs_bt = hotspot_tir_bt.max().round(4)
        except ValueError:
            # Handle the no detection case
            mir_max_hs_bt = numpy.nan
            tir_max_hs_bt = numpy.nan        
        
        result['MIR Hotspot Max Brightness Temperature'].append(mir_max_hs_bt)
        result['TIR Hotspot Max Brightness Temperature'].append(tir_max_hs_bt)
        
        day_night = support_functions.get_dn(image_date, vent[1], vent[0], elevation)
        sol_zenith, sol_azimuth = support_functions.get_solar_coords(
            image_date, vent[1], vent[0], elevation
        )
        
        result['Day/Night Flag'].append(day_night)
        result['Solar Zenith'].append(round(sol_zenith, 1))
        result['Solar Azimuth'].append(round(sol_azimuth, 1))

    results = pandas.DataFrame(result)
    meta['Result Count'] = len(results)

    # Add results that apply to all
    results['Sensor'] = sensor.upper()
    results['Volcano ID'] = volc.iloc[0]['id']

    # pull in metadata retrieved during the download
    file_meta = results['Data File'].map(download_meta)
    results['Satellite'] = file_meta.map(lambda x: x.get('satelite'))
    results['Data URL'] = file_meta.map(lambda x: x.get('url'))

    if len(results) > 0:
        results = results.sort_values('Date').reset_index(drop = True)
        
    meta['Run End'] = datetime.now(UTC).isoformat()
    return results, meta
