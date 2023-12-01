#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file for storing functions used in u-net_application.py
Note that some functions here are deprecated and not used in the final version of the script.

@author: Pablo Saunders-Shultz
"""
#IMPORTS
import numpy as np
from math import sqrt
import ephem
from scipy.ndimage import generate_binary_structure
from skimage.morphology import dilation
from skimage.measure import label, regionprops
#warnings.simplefilter('ignore', np.RankWarning)





VIIRS_MIR_RAD_MIN = 0.0015104711801057938
VIIRS_MIR_RAD_MAX = 3.9207917784696003
VIIRS_TIR_RAD_MIN = 0.13924616311648136
VIIRS_TIR_RAD_MAX = 32.78489204255699

def normalize(img, min_rad, max_rad):
    img = np.array(img)
    normalized = (img - min_rad) / (max_rad - min_rad)
    return normalized


def get_dn(datetime, volcano_lat, volcano_lng, volcano_elevation):
    obs = ephem.Observer()
    obs.lat = str(volcano_coords[volcano][0]) #default is veniaminof, REMEMBER TO SWITCH THIS OUT!
    obs.lon = str(volcano_coords[volcano][1]) 
    obs.elevation = elevation      # elevation in meters, set to ~1/2 the peak height
    obs.date = datetime
    
    sun = ephem.Sun(obs)
    sun.compute(obs)
    sun_angle = np.rad2deg(float(sun.alt))
    DNflag = (sun_angle>=-6) #set to -6 for civil twilight --> 0 = pure day
    if DNflag:
        return "D"
    else:
        return "N"

def get_solar_coords(datetime, volcano="veni", elevation=1250):
    obs = ephem.Observer()
    obs.lat = str(volcano_coords[volcano][0]) #default is veniaminof, REMEMBER TO SWITCH THIS OUT!
    obs.lon = str(volcano_coords[volcano][1]) 
    obs.elevation = elevation      # elevation in meters, set to ~1/2 the peak height
    obs.date = datetime
    
    sun = ephem.Sun(obs)
    sun.compute(obs)
    solar_zenith_angle = 90-np.rad2deg(float(sun.alt))
    solar_azimuth_angle = np.rad2deg(float(sun.alt))
    return (solar_zenith_angle, solar_azimuth_angle)

def radiative_power(bI4rad, active_all, cellsize=371, rp_constant=17.34): #FROM HANNAHs script
    #rp_constant is 17.34 for VIIRS and 18.9 for MODIS
    # ID "hotspots" = can be multi-pixel:   
    values = np.array(bI4rad) #raw radiance values
    selem = generate_binary_structure(2, 2) #3x3 matrix filled with TRUE (What is this for?)
    label_img = label(active_all,connectivity=2) #assigns each pixel an integer group number? no change for integer masks
    #regions = regionprops(label_img) #returns various properties of each region in label_img
    # For each region:
    #   - create an image with just that region
    #   - dilate and remove the region to extract background
    #   - Then for each coordinate in the region, calculate RP and sum for the region
    #cellsize= 371#static viirs resolution of 371m. Hannah pulls it from the actual scene #now passed in as a variable
    Apix = cellsize**2;
    counter = 0
    rp = np.zeros([label_img.max(),1])
    bginfo = np.zeros([label_img.max(),2])
    for region in regionprops(label_img):
        subimage = np.zeros(active_all.shape)
        subimage[region.coords[:,0],region.coords[:,1]]=1
        dilated = dilation(subimage, selem)
        bgimage = dilated-subimage
        #fig, axs = plt.subplots(ncols=2)
        #axs[0].imshow(active_all)
        #axs[0].set_title("Active pixels")
        #axs[1].imshow(bgimage)
        #axs[0].set_title("Background pixels")
        #plt.show()
        #plt.close()
        bg = values[bgimage==1].mean()
        bginfo[counter,:] = [bg, values[bgimage==1].shape[0]]
        # Loop through each pixel in region and calculate RP
        rp[counter] = 0
        for row in region.coords:
            L4alert = values[row[0],row[1]]     # Find radiance a ~4 um
            dL4pix = L4alert-bg                 # Find above background radiance
            RPpix = rp_constant*Apix*dL4pix           # Convert to radiative power for 3.8 um (Wooster et al., 2003)
            rp[counter] = rp[counter] + RPpix     # Sum RP for each region
        counter=counter+1
    totalrp = rp.sum()
    return totalrp

#U-NET FUNCTIONS:
I4_bt_min = 208
I4_bt_max = 361.7759
I5_bt_min = 150
I5_bt_max = 423.3373
I4_rad_min = 0.0015104711801057938
I4_rad_max = 3.9207917784696003
I5_rad_min = 0.13924616311648136
I5_rad_max = 32.78489204255699
directory_shit = os.popen('pwd') # print working directory
dirname = directory_shit.read().rstrip() # must be two lines for some reason

def brightness_temp(band, wl=3.74e-6): #actually works, was taken from Hannah D (thank you hannah!)
    wl = wl*1e6 #convert from meters to micrometers
    bt =  c2/(wl*np.log((c1/(band*pi*(wl**5)))+1))
    return bt
    
h = 6.626e-34 #Joules/Hz ?
c = 2.99e+8 #meters/second
k = 1.38e-23

        
        
    

