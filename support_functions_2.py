#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file for storing functions used in pre- and post-processing for hotlink

@author: Pablo Saunders-Shultz
"""
#IMPORTS
import os
import numpy as np
from math import sqrt
import ephem
from scipy.ndimage import generate_binary_structure
from skimage.morphology import dilation
from skimage.measure import label, regionprops


#NORMALIZATION VARIABLES 
#Values obtained from any VIIRS l1b file, by using SCALE and OFFSET factors 
#to convert min and max integer values to radiance values.
#See VIIRS L1b user guide for more information
VIIRS_MIR_MIN_RAD = 0.0015104711801057938
VIIRS_MIR_MAX_RAD = 3.9207917784696003
VIIRS_TIR_MIN_RAD = 0.13924616311648136
VIIRS_TIR_MAX_RAD = 32.78489204255699

def normalize(img, min_rad, max_rad):
    """
    normalize values in img, by min radiance and max radiance values
    """
    img = np.array(img)
    normalized = (img - min_rad) / (max_rad - min_rad+0.00000001)
    return normalized

def normalize_MIR(img):
    return normalize(img, min_rad=VIIRS_MIR_MIN_RAD, max_rad=VIIRS_MIR_MAX_RAD)

def normalize_TIR(img):
    return normalize(img, min_rad=VIIRS_MIR_MIN_RAD, max_rad=VIIRS_MIR_MAX_RAD)


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

def get_solar_coords(datetime, volcano_lat, volcano_lng, volcano_elevation):
    obs = ephem.Observer()
    obs.lat = str(volcano_lat) #default is veniaminof, REMEMBER TO SWITCH THIS OUT!
    obs.lon = str(volcano_lng) 
    obs.elevation = volcano_elevation      # elevation in meters, set to ~1/2 the peak height
    obs.date = datetime
    
    sun = ephem.Sun(obs)
    sun.compute(obs)
    solar_zenith_angle = 90-np.rad2deg(float(sun.alt))
    solar_azimuth_angle = np.rad2deg(float(sun.az)) - 180
    return (solar_zenith_angle, solar_azimuth_angle)

def radiative_power(MIR_rad, active_all, cellsize=371, rp_constant=17.34):
    """
    Modified code from Hannah Dietterich
    #rp_constant is 17.34 for VIIRS and 18.9 for MODIS
    MIR_rad is an np.array of mir spectral radiance values (not normalized!)
    active_all is a binary np.array the same size as MIR_rad, 1s denoting hotspot pixels
    cellsize is the nadir resolution of the sensor in meters, 371 for VIIRS, 1000 for MODIS 
    """
    values = np.array(MIR_rad) #raw radiance values
    selem = generate_binary_structure(2, 2) #3x3 matrix filled with TRUE (What is this for?)
    label_img = label(active_all,connectivity=2) #assigns each pixel an integer group number? no change for integer masks
    #regions = regionprops(label_img) #returns various properties of each region in label_img
    # For each region:
    #   - create an image with just that region
    #   - dilate and remove the region to extract background
    #   - Then for each coordinate in the region, calculate RP and sum for the region
    Apix = cellsize**2 #area of the pixels
    counter = 0
    rp = np.zeros([label_img.max(),1])
    bginfo = np.zeros([label_img.max(),2])
    for region in regionprops(label_img):
        subimage = np.zeros(active_all.shape)
        subimage[region.coords[:,0],region.coords[:,1]]=1
        dilated = dilation(subimage, selem)
        bgimage = dilated-subimage
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

h = 6.626e-34 #Plancks constant, Joules*Seconds
c = 2.99e+8 #Speed of light, meters / second
k = 1.38e-23 # Boltzmann constant, Joules / Kelvin

def inverse_planck(L, wl=3.74e-6):
    """
    Calculates brightness temperature from radiance values,
    L, list or array or single radiance value, in units of W*m-3*steradians-1
    wl, central wavelength of the band, in METERS (not micrometers)
    """
    K2 = (h*c)/(wl*k)
    K1 = (2.0*h*(c**2))/(wl**5)
    BT = K2/(np.log(1+(K1/L)))
    return BT
    
        
        
    

