import os
import pathlib

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

import numpy
import pandas

from hotlink.load_hotlink_model import load_hotlink_model
from hotlink import preprocess, support_functions
from skimage.filters import apply_hysteresis_threshold

def process_image(
    vent: tuple[float, float],
    elevation: int,
    model: "Model",
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
    data = numpy.load(file)
    mir = data[:, :, 0].copy()
    tir = data[:, :, 1].copy()

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

    # generate results
    mir_analysis = support_functions.crop_center(mir, size=24) #crop to output size, for analysis
    tir_analysis = support_functions.crop_center(tir, size = 24) # So we can get the background TIR brightness temperature
    rp = support_functions.radiative_power(mir_analysis, hotspot_mask) # in Watts

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
        mir_max_hs_bt = hotspot_mir_bt.max()
        tir_max_hs_bt = hotspot_tir_bt.max()
    except ValueError:
        # Handle the no detection case
        mir_max_hs_bt = numpy.nan
        tir_max_hs_bt = numpy.nan

    day_night = support_functions.get_dn(image_date, vent[1], vent[0], elevation)
    sol_zenith, sol_azimuth = support_functions.get_solar_coords(image_date, vent[1], vent[0], elevation)

    result = {
        'source file': filename,
        'date': image_date,
        'radiative_power': rp,
        'day_night flag': day_night,
        'max_prob': max_prob,
        'mir_hotspot_bt': hotspot_mir_bt.mean(),
        'mir_background_bt': bg_mir_bt.mean(),
        'mir_hotspot_max_bt': mir_max_hs_bt,
        'tir_hotspot_bt': hotspot_tir_bt.mean(),
        'tir_hotspot_max_bt': tir_max_hs_bt,
        'tir_background_bt': bg_tir_bt.mean(),
        'num_hostspot_pixels': num_hotspot_pixels,
        'solar_zenith': sol_zenith,
        'solar_azimuth': sol_azimuth,
        'Pixels above 0.5 prob': prob_above_05,
    }

    file.unlink()

    return result

def get_results(vent: str | tuple[float, float], elevation: int, dates: tuple[str, str], sensor: str) -> pandas.DataFrame:

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

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the processed results for each image. Each row
        corresponds to an image and includes various statistics such as sensor
        metadata, volcano information, and model output.

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
    ...     sensor="viirs"
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

    print("Using volcano:", volc.iloc[0]['name'])

    data_path = pathlib.Path('./data')

    # make sure the data directory exists
    os.makedirs(data_path, exist_ok = True)

    meta = preprocess.download_preprocess(dates, vent, sensor, folder = data_path)
    print("Image files processed. Beginning calculations")

    data_files = list(data_path.glob('*.npy'))

    model = load_hotlink_model()

    results = []

    # Using the thread pool here provides a very modest (~16% in my testing) speedup.
    # It might not be worth the complexity for smaller image sets, but could help a bit with
    # larger sets. ProcessPoolExecutor is horrible here, due to the need for
    # every individual process to import/set up tensorflow, coupled with
    # inter-process communications.
    process_func = partial(process_image, vent, elevation, model)
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_func, data_files)
    
    results = pandas.DataFrame(results)
    
    # Add results that apply to all
    results['sensor'] = sensor
    results['VolcanoID'] = volc.iloc[0]['id']
    
    # pull in metadata retrieved during the download
    meta_map = results['source file'].map(meta)
    results = results.drop(columns=['source file'])    
    results['satellite'] = meta_map.map(lambda x: x.get('satelite'))
    results['dataURL'] = meta_map.map(lambda x: x.get('url'))
    
    if len(results) > 0:
        results = results.sort_values('date').reset_index(drop = True)
    return results
