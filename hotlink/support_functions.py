#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file for storing support functions used in pre- and post-processing for hotlink

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



"""
PRE-Processing Functions
"""

#NORMALIZATION VARIABLES 
#Values obtained from any VIIRS l1b file, by using SCALE and OFFSET factors 
#to convert min and max integer values to radiance values.
#See VIIRS L1b user guide for more information
VIIRS_MIR_MIN_RAD = 0.0015104711801057938
VIIRS_MIR_MAX_RAD = 3.9207917784696003
VIIRS_TIR_MIN_RAD = 0.13924616311648136
VIIRS_TIR_MAX_RAD = 32.78489204255699

def normalize(img, min_rad, max_rad, fill_nan=True):
    """
    normalize values in img, by min radiance and max radiance values
    fills missing values with the lowest observed value
    """
    img = np.array(img) #convert to np array

    #fill missing values
    if fill_nan == True:
        min_observed = np.nanmin(img) # Find the minimum observed value
        img[np.isnan(array)] = min_observed # Replace missing values with the minimum non-missing value

    normalized = (img - min_rad) / (max_rad - min_rad+0.00000001) #add infinitesimal to avoid divide by zero
    return normalized

def normalize_MIR(img, fill_nan=True):
    """
    Normalizes MIR images (array-like), to the viirs sensor min and max values
    """
    return normalize(img, min_rad=VIIRS_MIR_MIN_RAD, max_rad=VIIRS_MIR_MAX_RAD, fill_nan=fill_nan)

def normalize_TIR(img, fill_nan=True):
    """
    Normalizes TIR images (array-like), to the viirs sensor min and max values
    """
    return normalize(img, min_rad=VIIRS_MIR_MIN_RAD, max_rad=VIIRS_MIR_MAX_RAD, fill_nan=fill_nan)

def crop_center(img, size=64):
    """
    Function to crop images to the center.
    Shape can be either [width, height], or [width, height, bands].
    """
    img = np.array(img)

    # Determine center for each dimension
    center_x = img.shape[0] // 2
    center_y = img.shape[1] // 2

    # Calculate the lower and upper bounds for cropping
    lb_x = center_x - (size // 2)
    hb_x = center_x + (size // 2)
    lb_y = center_y - (size // 2)
    hb_y = center_y + (size // 2)

    # Crop the image
    cropped = img[lb_x:hb_x, lb_y:hb_y]

    return cropped


"""
Post-processing functions
radiative_power()
brightness_temperature_from_radiance()
"""

def radiative_power(MIR_rad, active_all, cellsize=371, rp_constant=17.34):
    """
    Modified code from Hannah Dietterich, formula follows Wooster et al., 2003
    returns radiative power in Watts, calculated from MIR bands with background correction
    

    MIR_rad is an np.array of mir spectral radiance values (not normalized!)
    active_all is a binary array the same size as MIR_rad, with 1s denoting hotspot pixels
    cellsize is the nadir resolution of the sensor in meters, 371 for VIIRS, 1000 for MODIS
    rp_constant is 17.34 for VIIRS and 18.9 for MODIS
    """
    values = np.array(MIR_rad) #raw radiance values
    selem = generate_binary_structure(2, 2) #3x3 matrix filled with TRUE
    label_img = label(active_all,connectivity=2) #assigns each pixel an integer group number (no change for binary mask, but kept anyway)
  
    
    Apix = cellsize**2 #area of the pixels
    counter = 0
    rp = np.zeros([label_img.max(),1]) #initialize radiative power 
    bginfo = np.zeros([label_img.max(),2]) #initialize background area

    # For each region:
    #   - create an image with just that region
    #   - dilate and remove the region to extract background
    #   - Then for each coordinate in the region, calculate RP and sum for the region
    for region in regionprops(label_img): #returns various properties of each region in label_img
        subimage = np.zeros(active_all.shape)
        subimage[region.coords[:,0],region.coords[:,1]]=1
        dilated = dilation(subimage, selem)
        bgimage = dilated-subimage
        bg = values[bgimage==1].mean()
        bginfo[counter,:] = [bg, values[bgimage==1].shape[0]]
        # Loop through each pixel in region and calculate RP
        rp[counter] = 0
        for row in region.coords:
            L4alert = values[row[0],row[1]]     # Find MIR
            dL4pix = L4alert-bg                 # Find above background radiance
            RPpix = rp_constant*Apix*dL4pix           # Convert to radiative power for 3.8 um (Wooster et al., 2003)
            rp[counter] = rp[counter] + RPpix     # Sum RP for each region
        counter=counter+1
    totalrp = rp.sum()
    return totalrp


#variables for calculating brightness temperature
h = 6.626e-34 #Plancks constant, Joules*Seconds
c = 2.99e+8 #Speed of light, meters / second
k = 1.38e-23 # Boltzmann constant, Joules / Kelvin

def brightness_temperature_from_radiance(L, wl=3.74e-6):
    """
    Calculates brightness temperature from radiance values,
    L, list or array or single radiance value, in units of W*m-3*steradians-1
    wl, central wavelength of the band, in METERS (not micrometers)
    """
    K2 = (h*c)/(wl*k)
    K1 = (2.0*h*(c**2))/(wl**5)
    BT = K2/(np.log(1+(K1/L)))
    return BT

def generate_detection_figure():
    """
    function to generate annotated figures of hotspot detections
    and optionally save as png?
    """
    pass


"""
Miscellaneous support functions
"""


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
    
        
        
    

