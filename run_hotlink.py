import os
import pathlib

from datetime import datetime

import numpy
import pandas
import tensorflow

from hotlink import preprocess
from hotlink.support_functions import crop_center, radiative_power, brightness_temperature, get_dn, normalize_MIR, normalize_TIR, get_solar_coords
from skimage.filters import apply_hysteresis_threshold

def load_model():
    tensorflow_version = ".".join(tensorflow.__version__.split('.')[:2])
    script_directory = pathlib.Path(__file__).parent.absolute()
    model_dir = script_directory / "hotlink_model_new"

    model_path = model_dir / f"hotlink_tf{tensorflow_version}.keras"

    model = tensorflow.keras.models.load_model(model_path)
    return model


def get_results(mir, tir, image_date):
    n_mir = normalize_MIR(mir) #note, normalize also fills in missing pixels
    n_tir = normalize_TIR(tir)
    stacked = numpy.dstack([n_mir, n_tir])

    # Make sure the original mir array doesn't have any missing pixels
    min_mir_observered = numpy.nanmin(mir)
    mir[numpy.isnan(mir)] = min_mir_observered

    # Fill missing pixels for tir
    min_tir_observered = numpy.nanmin(tir)
    tir[numpy.isnan(tir)] = min_tir_observered

    predict_data = crop_center(stacked).reshape(1, 64, 64, 2)
    prediction = model.predict(predict_data) #shape=[batch_size, 24, 24, 3], for 3 predicted classes:background, hotspot-adjacent, and hotspot

    # get predicted class for each pixel (highest probability)
    # pred_classes = numpy.array(numpy.argmax(prediction[0,:,:,:], axis=2)) #classes 0,1,2 correspond to bg, hot-adjacent, and hot

    # use hysteresis thresholding to generate a binary map of hotspot pixels
    prob_active = prediction[0,:,:,2] #map with probabilities of active class
    max_prob = numpy.max(prob_active) #highest probability per image, equated to probability that the image contains a hotspot
    hotspot_mask = apply_hysteresis_threshold(prob_active, low=0.4, high=0.5).astype('bool') #hysteresis thresholding active mask
    num_hotspot_pixels = numpy.count_nonzero(hotspot_mask)

    # generate results
    mir_analysis = crop_center(mir, size=24) #crop to output size, for analysis
    tir_analysis = crop_center(tir, size = 24) # So we can get the background TIR brightness temperature
    rp = radiative_power(mir_analysis, hotspot_mask) # in Watts

    # mir hotspot/background brightness temperature analysis
    mir_hotspot = mir_analysis[hotspot_mask ]
    mir_background = mir_analysis[~hotspot_mask]
    hotspot_mir_bt = brightness_temperature(mir_hotspot, wl=3.74e-6)
    bg_mir_bt = brightness_temperature(mir_background, wl=3.74e-6)

    # tir hotspot/background brigbhtness temerature analysis
    tir_hotspot = tir_analysis[hotspot_mask ]
    tir_background = tir_analysis[~hotspot_mask]
    hotspot_tir_bt = brightness_temperature(tir_hotspot, wl=3.74e-6)
    bg_tir_bt = brightness_temperature(tir_background, wl=3.74e-6)

    day_night = get_dn(image_date, vent[1], vent[0], elevation)
    sol_zenith, sol_azimuth = get_solar_coords(image_date, vent[1], vent[0], elevation)

    result = {
        'date': image_date,
        'radiative_power': rp,
        'day_night flag': day_night,
        'max_prob': max_prob,
        'mir_hotspot_bt': hotspot_mir_bt.mean(),
        'mir_background_bt': bg_mir_bt.mean(),
        'mir_hotspot_max_bt': hotspot_mir_bt.max(),
        'tir_hotspot_bt': hotspot_tir_bt.mean(),
        'tir_hotspot_max_bt': hotspot_tir_bt.max(),
        'tir_background_bt': bg_tir_bt.mean(),
        'num_hostspot_pixels': num_hotspot_pixels,
        'solar_zenith': sol_zenith,
        'solar_azimuth': sol_azimuth,
    }

    return result

if __name__ == "__main__":
    #  Spurr
    # vent = (61.2997,-152.2514) # lat,lon
    # elevation = 3374

    #  Shishaldin
    vent = (54.7554, -163.9711)
    elevation = 2857

    dates = ("2019-07-21 14:00", "2019-07-21 15:00") # Year-month, from to.
    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'viirs'
    data_path = pathlib.Path('./data')

    # make sure the data directory exists
    os.makedirs(data_path, exist_ok = True)

    preprocess.download_preprocess(dates, vent, sensor, folder = data_path)

    data_files = data_path.glob('*.npy')

    model = load_model()

    results = []
    for file in data_files:
        image_date = datetime.strptime(file.stem, '%Y%m%d_%H%M')
        data = numpy.load(file)
        mir = data[:, :, 0].copy()
        tir = data[:, :, 1].copy()

        result = get_results(mir, tir, image_date)

        # Add some global properties
        result['sensor'] = sensor

        results.append(result)
        # file.unlink()

    results = pandas.DataFrame(results)

    # fig,axs = plt.subplots(figsize=(8,4), ncols=2)
    # im0 = axs[0].imshow(mir)
    # plt.show()
    print(results)
