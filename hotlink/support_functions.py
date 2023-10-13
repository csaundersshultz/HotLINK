#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file for storing functions used in u-net_application.py
Note that some functions here are deprecated and not used in the final version of the script.

@author: Pablo Saunders-Shultz
"""
#IMPORTS
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt
np.seterr(invalid='ignore')
import ephem
from scipy.ndimage import generic_filter, convolve
from scipy.optimize import curve_fit
from scipy import stats
import gdal
import os
from skimage.segmentation import find_boundaries
import pandas as pd
#warnings.simplefilter('ignore', np.RankWarning)

#VARIABLES USED INTERNALLY MOSTLY
volcano_coords ={"veni":(56.1979, -159.3931),
                 "bogo":(53.9272, -168.0344),"clev":(52.8220, -169.9450),"okmo":(53.3970, -168.1660),
                 "pavl":(55.4173, -161.8937),"redo":(60.4852, -152.7438), "augu":(59.3626, -153.4350),
                 "shis":(54.7554, -163.9711), 'hawaii':(19.421, -155.287)}
#should these variables be declared inside of the function or outside? 
bI4wv = 3.74 #band I04 wavelength
bI5wv = 11.45 #band I05 wavelength
c1 = 3.74151*10.**8. #planck equation constant 1
c2 = 1.43879*10.**4. #plank equation constant 2
pi = 3.14159

#FUNCTION DEFINITIONS
def calc_nti(bI4, bI5):
    #preserve nans # turns out this isnt necessary, but may still be useful at some later point
    #nans = np.logical_or(np.isnan(bI4), np.isnan(bI5))
    nti = (bI4-bI5)/(bI4+bI5)
    #nti[nans] = np.nan
    return nti
def dist_from_center(coords):
    xcoord, ycoord = coords
    xdist = abs(67-xcoord)
    ydist = abs(67-ycoord)
    return sqrt(xdist**2 + ydist**2)
distance_matrix = [[dist_from_center((x,y)) for x in range(135)] for y in range(135)]
def calc_ntiapp(bI4rad, bI5rad):
    #Calculate brightness Temperature from radiance = this is NEARLY identical
    #bI4bT = c2/(bI4wv*np.log((c1/(bI4rad*pi*(bI4wv**5)))+1))
    bI5bT = c2/(bI5wv*np.log((c1/(bI5rad*pi*(bI5wv**5)))+1))
    bI4app = (c1*(bI4wv**-5))/(pi*(np.exp(c2/(bI4wv*bI5bT))-1))
    ntiapp = (bI4app-bI5rad)/(bI4app+bI5rad)
    #plt.imshow(ntiapp)
    #plt.title('ntiapp')
    #plt.show()
    return ntiapp
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c
def scipy_eti(nti, ntiapp):
    shape=nti.shape
    nti_arr = nti.ravel()
    ntiapp_arr = ntiapp.ravel()
    clean = np.isfinite(nti_arr) & np.isfinite(ntiapp_arr) #ignores pixels where either nti or ntiapp are NaN
    popt, pcov = curve_fit(quadratic_function, ntiapp_arr[clean], nti_arr[clean])
    #perr = np.sqrt(np.diag(pcov))
    ntibk = quadratic_function(ntiapp_arr, popt[0], popt[1], popt[2]).reshape(shape)
    eti = nti-ntibk
    return eti
def numpy_eti(nti, ntiapp):
    nti_arr = nti.ravel()
    ntiapp_arr = ntiapp.ravel()
    clean = np.isfinite(nti_arr) & np.isfinite(ntiapp_arr) #ignores pixels where either nti or ntiapp are NaN
    #numpy polyfit
    polyfitted = np.polyfit(ntiapp_arr[clean], nti_arr[clean], 2, full=True)#returns coefficients of polynomial fit
    p = np.poly1d(polyfitted[0]) #creates a function out of the polynomial values
    ntibk = p(ntiapp)
    eti = nti - ntibk
    return eti
def calc_eti(nti, ntiapp, dnflag):
    nti_arr = nti.ravel()
    ntiapp_arr = ntiapp.ravel()
    clean = np.isfinite(nti_arr) & np.isfinite(ntiapp_arr) #ignores pixels where either nti or ntiapp are NaN
    #numpy polyfit
    polyfitted = np.polyfit(ntiapp_arr[clean], nti_arr[clean], 2, full=True)#returns coefficients of polynomial fit
    p = np.poly1d(polyfitted[0]) #creates a function out of the polynomial values
    ntibk = p(ntiapp)
    eti = nti - ntibk

    """#scipy curve_fit
    popt, pcov = curve_fit(quadratic_function, ntiapp_arr[clean], nti_arr[clean])
    perr = np.sqrt(np.diag(pcov))
    ntibk_2 = quadratic_function(ntiapp_arr, popt[0], popt[1], popt[2]).reshape(135,135)
    plt.scatter(ntiapp_arr, nti_arr, alpha=0.2)
    ln = np.linspace(np.min(ntiapp_arr),np.max(ntiapp_arr), 100)
    ln_y2 = quadratic_function(ln, popt[0], popt[1], popt[2])
    ln_y=p(ln)
    plt.plot(ln, ln_y, 'r-', label='numpy.polyfit')
    plt.plot(ln, ln_y2, 'm-', label='scipy.curve_fit')
    plt.title('ntiapp vs nti - {}'.format(dnflag))
    plt.xlabel("NTIapp")
    plt.ylabel("NTI")
    plt.legend()
    plt.show()    
    eti2 = nti - ntibk_2
    fig, axs = plt.subplots(ncols=3)
    axs[0].imshow(nti)
    axs[1].imshow(eti)
    axs[2].imshow(eti2)
    fig.suptitle("NTI, ETI (np.polyfit), and ETI (scipy.curve_fit)")
    for i in range(3):
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.show()
    #"""
    return eti
def show_image(raster):
    plt.imshow(raster, interpolation='none')
    plt.gca().set_axis_off()
    plt.margins(0,0)
    plt.colorbar()
    return
def interp_nans(array):
    mid = (len(array)-1)/2 #can interpolate over adjustable (ODD) sized surroundings
    if np.isnan(array[mid]):
        return np.nanmean(array)
    else:
        return array[mid]
def get_dn(datetime, volcano="veni", elevation=1250):
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

def fnc(buffer):
    #input is a vector with 0-8 elements, the 4th is the pixel in question
    #calc = pixel - average(surrounding pixels) not including NaN
    #~np.isnan(buffer) returns true for nan values, and is then inverted by ~
    calc = buffer[4]- (np.nansum(buffer[0:4])+np.nansum(buffer[5:]))/(np.count_nonzero(~np.isnan(buffer))-1)
    return calc
def d_filter(img):
    d_img = generic_filter(img, fnc, size=3,mode='nearest')
    #make edges NaN
    d_img[:,0]=float('NaN')
    d_img[0,:]=float('NaN')
    d_img[:,-1]=float('NaN')
    d_img[-1,:]=float('NaN')
    return d_img  
def calc_zscore(img):
    mn = np.nanmean(img.ravel())
    std = np.nanstd(img.ravel())
    zsco = (img-mn)/std
    return zsco
def get_num_hottest(nti, num=5):
    #get the num pixels with highest value of passed matrix
    matrix = nti.copy() # copy so as not to change values in original
    indices_10 = []
    nti_10 = []
    for i in range(num):
        maxx = np.nanmax(matrix) # get maximum
        indxx = np.unravel_index(np.nanargmax(matrix), matrix.shape) #get index of maximum
        matrix[indxx] = -100 # pop maximum
        indices_10.append(indxx)
        nti_10.append(maxx)
    return indices_10, nti_10
def get_num_random(matrix, num=5):
    indices_num = []
    nti_num = []
    for i in range(num):
        rand_xy = tuple(np.random.randint(1,134, size=2))
        indices_num.append(rand_xy)
        nti_num.append(matrix[rand_xy])
    return indices_num, nti_num
def get_num_hottest_in_center(matrix, num=5):
    #create a blank matrix
    new = np.full(matrix.shape, np.float32(-1))
    #copy center of nti matrix to new blank matrix
    center = matrix[62:74,62:74]
    new[62:74,62:74] = center 
    indices_num, nti_num = get_num_hottest(new, num=num)
    return indices_num, nti_num
    
#VERY important MIROVA function definition:
def mirova(dnflag, nti, eti, dnti, deti, dnti_z, deti_z, I4bt):
    #for now only using roi1
    #could definitely be made faster but for now just set all of roi2 to zero after calculating activemask
    if dnflag==0 or dnflag=="N": #night
        K = -0.8
        # Mirova (R1) Baseline:
        #C1=0.003
        #C2=5
        #Values from Hannah's script:
        #C1=0.01
        #C2=15
        # Initial Optimized values:
        #C1=0.0065
        #C2=4.125
        #02/2021 optimized values:#C1=didnt save 8P
        # 3fold Cross validated values:
        #C1=0.00775
        #C2=4.875
        #ROC ideal threshold by gmeans
        #C1=0.01
        #C2=9.1
        #OPTIMIZED to VENI_TRAIN_Data 08/30
        #C1=0.05
        #C2=8.5
        #Optimized to VENI_TRAIN_Data via macro-averaged F1-score 09/10
        #C1 = 0.05
        #C2 = 8.5
        #Optimized to VENI_TRAIN_Data via BINARY-averaged F1-score 09/10
        #C1 = 0.05
        #C2 = 8.5
        #optimized to VENI_TRAIN and CLEV_TRAIN via binary-averaged F1-score 11/30
        C1 = 0.07
        C2 = 5.25
        
        
    elif dnflag==1 or dnflag=="D":#day
        K=-0.6
        #Mirova (R1) Baseline:
        #C1=0.02
        #C2=15
        #Values from Hannah's script:
        #C1=0.04
        #C2=30
        # My optimized values
        #C1=0.0625
        #C2=5.125
        # I never did 3fold cross validation on daytime data since it was already very messy
        #C1=0.11
        #C2=14.8
        #OPTIMIZED to VENI_TRAIN_data 08/30
        #C1=0.15
        #C2=24.625
        #Optimized to VENI_TRAIN_Data via macro-averaged F1-score 09/10
        #C1 = 0.1625
        #C2 = 18
        #Optimized to VENI_TRAIN_Data via BINARY-averaged F1-score 09/10
        #C1 = 0.13
        #C2 = 17.5
        #optimized to VENI_TRAIN and CLEV_TRAIN via binary-averaged F1-score 11/30
        C1 = 0.11
        C2 = 6.25
        
        
    # check for solar glint (???)
    bb=I4bt.ravel()
    meanbI4=np.sum(np.array(bb)>308.15); #calculate how many bI4 pixels >35 C = 308.15 K
    if (meanbI4>300):
        K=-0.5
    #detect pixels
    nti_mask = nti>K #nti thermal index --> OG MODVOLC algorithm
    dnti_mask = np.logical_or( (dnti>C1) , (dnti_z>C2) )
    deti_mask = np.logical_or( (deti>C1) , (deti_z>C2) )
    d_mask = np.logical_and(dnti_mask, deti_mask) #pixels detected by spatial indices
    active_mask = np.logical_or(nti_mask, d_mask)
    #plt.imshow(active_mask)
    #now, remove already detected active pixels, recalc spatial indices, and detect again
    nti2 = nti.copy() #ASK ISRAEL IF THIS IS NECESSARY, would this function otherwise modify the original nti? 
    eti2 = eti.copy()
    nti2[active_mask] = np.nan
    eti2[active_mask] = np.nan
    dnti2 =  d_filter(nti2)
    deti2 =  d_filter(eti2)
    dnti2_zscore = calc_zscore(dnti2)
    deti2_zscore = calc_zscore(deti2)
    dnti_mask2 = np.logical_or( (dnti2>C1) , (dnti2_zscore>C2) )
    deti_mask2 = np.logical_or( (deti2>C1) , (deti2_zscore>C2) )
    d_mask2 = np.logical_and(dnti_mask2, deti_mask2) #pixels detected by spatial indices
    active_mask2 = np.logical_or(active_mask, d_mask2)
    #plt.imshow(active_mask2)
    #return only roi1
    #roi1_mask = np.full(shape=(24,24), fill_value=0)
    #roi1_mask = active_mask2[55:79,55:79]
    #return roi1_mask
    return active_mask2

def mean(arr):
    return np.nanmean(arr.ravel())
def std(arr):
    return np.nanstd(arr.ravel())
def skew(arr):
    return stats.skew(arr.ravel(), nan_policy='omit')
def kurt(arr):
    return stats.kurtosis(arr.ravel(), nan_policy='omit')


dfilt = [[ -0.125, -0.125, -0.125 ],
         [ -0.125,      1, -0.125 ],
         [ -0.125, -0.125, -0.125 ]]


gauss3 = [[-0.25, -0.5, -0.25],
          [-0.5,  3,    -0.5],
          [-0.25, -0.5, -0.25]]
gauss5 = [[-1, -4, -7, -4, -1],
          [-4, -16, -26, -16, -4],
          [-7, -26, 232, -26, -7],
          [-4, -16, -26, -16, -4],
          [-1, -4, -7, -4, -1]]
gauss7= [[-0.000036,	-0.000363,	-0.001446,	-0.002291,	-0.001446,	-0.000363,	-0.000036],
         [-0.000363,	-0.003676,	-0.014662,	-0.023226,	-0.014662,	-0.003676,	-0.000363],
         [-0.001446,	-0.014662,	-0.058488,	-0.092651,	-0.058488,	-0.014662,	-0.001446],
         [-0.002291,	-0.023226,	-0.092651,	0.85324,	-0.092651,	-0.023226,	-0.002291],
         [-0.001446,	-0.014662,	-0.058488,	-0.092651,	-0.058488,	-0.014662,	-0.001446],
         [-0.000363,	-0.003676,	-0.014662,	-0.023226,	-0.014662,	-0.003676,	-0.000363],
         [-0.000036,	-0.000363,	-0.001446,	-0.002291,	-0.001446,	-0.000363,	-0.000036]]
smooth5  =    [[-1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1], 
               [-1, -1, 24, -1, -1],
               [-1, -1, -1, -1, -1],
               [-1, -1, -1, -1, -1]]
smooth7 =    [[-1, -1, -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1],
              [-1, -1, -1, 48, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1, -1, -1], 
              [-1, -1, -1, -1, -1, -1, -1]]
round3 = [[0, -1, 0],
          [-1, 4, -1],
          [0, -1, 0]]
round5 =      [[0, -1, -1, -1, 0],
               [-1, -1, -1, -1, -1],
               [-1, -1, 20, -1, -1],
               [-1, -1, -1, -1, -1],
               [ 0, -1, -1, -1,  0]]
round7 =     [[0, 0, 0,-1, 0, 0, 0],
              [0, 0,-1,-1,-1, 0, 0],
              [0,-1,-1,-1,-1,-1, 0],
              [-1,-1,-1,24,-1,-1,-1],
              [0,-1,-1,-1,-1,-1, 0],
              [0, 0,-1,-1,-1, 0, 0],
              [0, 0, 0,-1, 0, 0, 0]]

myfilt1 =    [[0,  0, -1, -1, -1,  0,  0],
              [0, -1,  0,  0,  0, -1,  0],
             [-1,  0,  0,  2,  0,  0, -1],
             [-1,  0,  2,  8,  2,  0, -1],
             [-1,  0,  0,  2,  0,  0, -1],
              [0, -1,  0,  0,  0, -1,  0],
              [0,  0, -1, -1, -1,  0,  0]]
qfilt = np.array([[-1, -1, -1,  0,  0],
                 [-1, -1,  1,  0,  0],
                 [-1,  1,  2,  1,  0],
                 [0,   0,  1,  0,  0],
                 [0,   0,  0,  0,  0]] )
q2 = np.rot90(qfilt)
q3 = np.rot90(q2)
q4 = np.rot90(q3)

def my_filter(img, plot=False):
    mode = 'nearest'
    m1 = convolve(img, qfilt, mode=mode)
    m2 = convolve(img, q2, mode=mode)
    m3 = convolve(img, q3, mode=mode)
    m4 = convolve(img, q4, mode=mode)
    minny = np.minimum(np.minimum(m1,m2) , np.minimum(m3,m4))
    if plot==True:
        imgs = [img, minny]
        fig, axs = plt.subplots(ncols=2, figsize=(13,5))
        for i in range(2):
            im0 = axs[i].imshow(imgs[i])
            fig.colorbar(im0, ax=axs[i])
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        axs[0].set_title('Original')
        axs[1].set_title('Filtered')
        
    return minny
    
def count_active(active_mask):
    pass
def calc_tadr():
    pass

def crop_center(img, size=24):
    img_center = img.shape[0]//2
    lb= img_center-(size//2)
    hb = img_center+(size//2)
    cropped = img[lb:hb, lb:hb]
    return cropped

def calc_raw_rp(mir_img, mask, mask_active_value=2):
    #print(mask.shape)
    #print(mir_img.shape)
    active_pix = np.where(mask==mask_active_value, mir_img, 0)
    rp = np.sum(active_pix)
    return rp

from scipy.ndimage import generate_binary_structure
from skimage.morphology import dilation
from skimage.measure import label, regionprops

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
def open_img_pair(granule, size=64, normalizeby='radiance'):
    bI4 = gdal.Open('{}/veni45/I04_{}.tif'.format(dirname, granule)).ReadAsArray()
    bI5 = gdal.Open('{}/veni45/I05_{}.tif'.format(dirname, granule)).ReadAsArray()
    lb= 67-(size//2)
    hb = 67+(size//2)
    bI4 = bI4[lb:hb, lb:hb] #crop to pixels in the center
    bI5 = bI5[lb:hb, lb:hb]
    # DATA CLEANING
    # replace all pixels less than 0 with Nan
    I4_minpositive = np.nanmin(np.where(bI4>0, bI4, np.inf)) #lowest valid value
    I5_minpositive = np.nanmin(np.where(bI5>0, bI5, np.inf))
    bI4 = np.nan_to_num(bI4) #replace nans with zero
    bI5 = np.nan_to_num(bI5)
    bI4 = np.where(bI4<=0, I4_minpositive, bI4) #replace zeros and negatives with minpositive value
    bI5 = np.where(bI5<=0, I5_minpositive, bI5)
    stacked = np.dstack([bI4, bI5]) #input to tensorflow, shape: (size, size, 2)
    if normalizeby=='image':
        bI4_n = (bI4 - np.min(bI4))/(np.max(bI4)-np.min(bI4)+0.000001)
        bI5_n = (bI5 - np.min(bI5))/(np.max(bI5)-np.min(bI5)+0.000001)
        stacked = np.dstack([bI4_n, bI5_n])
    if normalizeby=='BT':
        I4bt = c2/(bI4wv*np.log((c1/(bI4*pi*(bI4wv**5)))+1))
        I5bt = c2/(bI5wv*np.log((c1/(bI5*pi*(bI5wv**5)))+1))
        I4bt_n = (I4bt - I4_bt_min) / I4_bt_max 
        I5bt_n = (I5bt - I5_bt_min) / I5_bt_max
        stacked = np.dstack([I4bt_n, I5bt_n])
    if normalizeby=='radiance':
        I4_rad_n = (bI4 - I4_rad_min) / I4_rad_max 
        I5_rad_n = (bI5 - I5_rad_min) / I5_rad_max
        stacked = np.dstack([I4_rad_n, I5_rad_n])
    return stacked

def build_mask(coords, size=24, border='thick'):
    mask = np.zeros((135,135)) #
    if type(coords)==str: #list if it contains active pixels, otherwise empty float (?)
        pixel_coords=np.array(eval(coords)).astype(int)
        for coord in pixel_coords:
            mask[coord[1], coord[0]] = 1 #y to rows, x to cols
            #print(coord)
    lb= 67-(size//2)
    hb = 67+(size//2)
    mask = mask[lb:hb, lb:hb].astype('int8') #background=0, hotspot=1
    if border=='thick':
        boundary = find_boundaries(mask, connectivity=1, mode='outer', background=0)
        boundary2 = find_boundaries(boundary, connectivity=1, mode='outer', background=0)
        bounds_and_mask = np.logical_or(np.logical_or(boundary, boundary2), mask)
        final_mask = bounds_and_mask+mask #background=0, adjacents=1, hotspot=2
    elif border=='thin':
        boundary = find_boundaries(mask, connectivity=2, mode='outer', background=0)
        bounds_and_mask = np.logical_or(boundary, mask) #thin boundary
        final_mask = bounds_and_mask + mask #background=0, adjacents=1, hotspot=2
    elif border=='none':
        final_mask=mask
    return final_mask.astype('float32') #background=0, #adjacents=1, #hotspot=2

#Functions for grouping nearest images together by time
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))
def group_nearest(xy):
    xy['datetime'] = pd.to_datetime(xy.image_time, format="%Y%m%d_%H%M%S")
    xy['group'] = np.nan
    group_num=0
    for idx, row in xy.iterrows():
        if pd.notnull(xy.iloc[idx]['group']): #if group is already defined, continue
            continue     
        row_dt = row['datetime'] #isolate the row
        xy_missing=xy.drop(idx)
        other_dts = xy_missing['datetime'] #list of all datetimes besides row in question
        #print(other_dts)
        nearest_date = nearest(other_dts, row_dt)
        nearest_idx = xy.loc[xy['datetime'] == nearest_date].index
        if pd.isnull(xy.iloc[nearest_idx]['group']).bool(): #if nearest has no group, assign both to new group
            xy.at[idx, 'group'] = group_num
            xy.at[nearest_idx, 'group'] = group_num
            group_num +=1
        else: #otherwise assign to the same group as nearest
            xy.at[idx, 'group'] = xy.iloc[nearest_idx]['group'].item()
    return xy

def get_background_pixels(mask):
    bdr1 = find_boundaries(mask, connectivity=1, mode='outer', background=0)
    bdr2 = find_boundaries(bdr1, connectivity=2, mode='outer', background=0)
    bdr3 = find_boundaries(bdr2, connectivity=2, mode='outer', background=0)
    #pixels withing boundary3, and not mask, b1 or b2 pixels
    bg = np.logical_and(np.logical_and(np.logical_and(bdr3, ~bdr2), ~bdr1), ~mask)  
    return bg

def radiative_power2(hotspot_pix, bg_pix, cellsize=371):
    #my implementation of calculating radiative power
    #defines background as 2 "boundaries" away from the active area
    #sum the hotspot radiation (in W/m^2sr)
    radiance_minus_background = np.sum(hotspot_pix) - (np.mean(bg_pix)*len(hotspot_pix))
    pixel_area = cellsize**2
    radiative_power = 17.34*pixel_area*radiance_minus_background          # Convert to radiative power for 3.8 um (Wooster et al., 2003)
    #so do I need to correct for pixel area and wavelength range?
    return radiative_power

#want to calculate average BT of background, average BT of the hotspot, and BThotspot-BTbg, and BThotspot_max
#all calculated from the I4 (MIR) band
#A "color-index" seems to be down the correct path, of quantifying the difference in observed brightness temperature
#across 2 different bands, which can tell you something perhaps about the true temperature, absorbance,
#and degree of gray-body vs blackbodiniess
#considerations: should I use the hottest pixel? by BT? by I4 radiance? or an average? 
#Maybe here the correct metric would be BT(I4) - Bt(I5) ? This would show, in Kelvin, how much hotter
#the hotspot appears in MIR than in TIR

#Prepare Mirova Function
c1 = 3.74151*10.**8. #planck equation constant 1
c2 = 1.43879*10.**4. #plank equation constant 2
pi = 3.14159
def brightness_temp(band, wl=3.74e-6): #actually works, was taken from Hannah D (thank you hannah!)
    wl = wl*1e6 #convert from meters to micrometers
    bt =  c2/(wl*np.log((c1/(band*pi*(wl**5)))+1))
    return bt
    
h = 6.626e-34 #Joules/Hz ?
c = 2.99e+8 #meters/second
k = 1.38e-23
def inverse_planck(L, wl=3.74e-6):#give wavelength in meters, not microns #L in units of W*m-3*steradians-1
    K2 = (h*c)/(wl*k)
    K1 = (2.0*h*(c**2))/(wl**5)
    BT = K2/(np.log(1+(K1/L)))
    return BT



def hotspot_area_bg_corrected(bI4_hotspot, bI4_bg, cellsize=371):
    #assumes that the temperature of the hotspot is uniform, at the brightest pixel
    #then calculates the area as a mixture model between the background radiance and peak radiance
    L_avg = np.mean(bI4_hotspot) #average radiance of the hotspot
    L_max = np.max(bI4_hotspot) #get radiance at the hottest pixel
    L_bg = np.mean(bI4_bg) #mean radiance of the background
    #solve for percentage of the area that is active in this 2 component mixture model
    percent_active = (L_avg - L_bg)/(L_max - L_bg)
    active_area = percent_active*len(bI4_hotspot)*((cellsize/1000)**2) #in km
    return active_area #in km^2
        
        
        
    

