import pathlib

from datetime import datetime

import numpy
import pandas

from hotlink import preprocess, load_hotlink_model
from hotlink.support_functions import crop_center, radiative_power, brightness_temperature, get_dn, normalize_MIR, normalize_TIR
from skimage.filters import apply_hysteresis_threshold

import matplotlib.pyplot as plt
from hotlink import load_example_data

def get_results(mir, tir, image_date):
    n_mir = normalize_MIR(mir) #note, normalize also fills in missing pixels
    n_tir = normalize_TIR(tir)
    stacked = numpy.dstack([n_mir, n_tir])
    
    # Make sure the original mir array doesn't have any missing pixels
    min_mir_observered = numpy.nanmin(mir)                
    mir[numpy.isnan(mir)] = min_mir_observered
    
    # We don't care about tir for now (as we don't use the non-normalized tir array)
    # but I put this code in just in case we need it in the future.
    # min_tir_observered = numpy.nanmin(tir)    
    # tir[numpy.isnan(tir)] = min_tir_observered
    
    predict_data = crop_center(stacked).reshape(1, 64, 64, 2)
    prediction = model.predict(predict_data) #shape=[batch_size, 24, 24, 3], for 3 predicted classes:background, hotspot-adjacent, and hotspot
    
    # get predicted class for each pixel (highest probability)
    # pred_classes = numpy.array(numpy.argmax(prediction[0,:,:,:], axis=2)) #classes 0,1,2 correspond to bg, hot-adjacent, and hot
    
    # use hysteresis thresholding to generate a binary map of hotspot pixels
    prob_active = prediction[0,:,:,2] #map with probabilities of active class
    max_prob = numpy.max(prob_active) #highest probability per image, equated to probability that the image contains a hotspot
    hotspot_mask = apply_hysteresis_threshold(prob_active, low=0.4, high=0.5).astype('int') #hysteresis thresholding active mask
    
    # generate results
    mir_analysis = crop_center(mir, size=24) #crop to output size, for analysis
    rp = radiative_power(mir_analysis, hotspot_mask) # in Watts
    
    # get just hotspot pixels
    hotspot = mir_analysis[ numpy.where(hotspot_mask==1) ]
    hotspot_bt = numpy.mean(brightness_temperature(hotspot, wl=3.74e-6))
    
    day_night = get_dn(image_date, vent[1], vent[0], elevation)
    
    result = {
        'date': image_date,
        'rp-w': rp,
        'dn-flag': day_night,
        'max_prob': max_prob,
        'brightness_temp': hotspot_bt,
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
    sat = 'viirs'
    num_batches = 1
    data_path = './data'
    
    preprocess.download_preprocess(dates, vent, sat, num_batches, data_path)
    
    data_files = pathlib.Path(data_path).glob('*.npy')
    
    model = load_hotlink_model()
    
    results = []
    for file in data_files:
        image_date = datetime.strptime(file.stem, '%Y%m%d_%H%M')
        data = numpy.load(file)
        mir = data[:, :, 0].copy()
        tir = data[:, :, 1].copy()
        
        result = get_results(mir, tir, image_date)
        results.append(result)
        # file.unlink()
        
    # Alternate: load test data.
    mir, tir, image_date = load_example_data(170)
    result = get_results(mir, tir, image_date)
    results.append(result)
    
    results = pandas.DataFrame(results)
    
    # fig,axs = plt.subplots(figsize=(8,4), ncols=2)
    # im0 = axs[0].imshow(mir)
    # plt.show()
    print(results)        
