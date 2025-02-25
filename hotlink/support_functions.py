#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
file for storing support functions used in pre- and post-processing for hotlink

@author: Pablo Saunders-Shultz
"""
# IMPORTS

import json
import urllib

import numpy as np
import ephem
import pandas

from pyproj import Proj
from scipy.ndimage import generate_binary_structure
from skimage.morphology import dilation
from skimage.measure import label, regionprops, find_contours
from skimage.transform import rescale
import matplotlib.pyplot as plt


"""
PRE-Processing Functions
"""

# NORMALIZATION VARIABLES
# Values obtained from any VIIRS l1b file, by using SCALE and OFFSET factors
# to convert min and max integer values to radiance values.
# See VIIRS L1b user guide for more information
VIIRS_MIR_MIN_RAD = 0.0015104711801057938
VIIRS_MIR_MAX_RAD = 3.9207917784696003
VIIRS_TIR_MIN_RAD = 0.13924616311648136
VIIRS_TIR_MAX_RAD = 32.78489204255699


def normalize(img, min_rad, max_rad, fill_nan=True):
    """
    Normalize values in the input image array.

    Parameters:
    - img: Input image array to be normalized.
    - min_rad: Minimum radiance value for normalization.
    - max_rad: Maximum radiance value for normalization.
    - fill_nan: Boolean flag indicating whether to fill missing values with the lowest observed value.

    Returns:
    - normalized: Normalized image array.

    Notes:
    - If fill_nan is True, missing values in the input image will be replaced with the minimum observed value before normalization.
    """

    img = np.array(img)  # Convert to NumPy array

    # Fill missing values if specified
    if fill_nan:
        min_observed = np.nanmin(img)  # Find the minimum observed value
        img[np.isnan(img)] = (
            min_observed  # Replace missing values with the minimum non-missing value
        )

    # Perform normalization
    normalized = (img - min_rad) / (
        max_rad - min_rad + 0.00000001
    )  # Add infinitesimal to avoid divide by zero

    return normalized


def normalize_MIR(img, fill_nan=True):
    """
    Normalizes Mid-Infrared (MIR) images (array-like), to the viirs sensor min and max values

    Parameters:
    - img: Input MIR image array to be normalized.
    - fill_nan: Boolean flag indicating whether to fill missing values with the lowest observed value.

    Returns:
    - normalized: Normalized MIR image array.
    """

    # Utilize the normalize function to perform MIR image normalization
    return normalize(
        img, min_rad=VIIRS_MIR_MIN_RAD, max_rad=VIIRS_MIR_MAX_RAD, fill_nan=fill_nan
    )


def normalize_TIR(img, fill_nan=True):
    """
    Normalizes Thermal-Infrared (TIR) images (array-like), to the viirs sensor min and max values

    Parameters:
    - img: Input MIR image array to be normalized.
    - fill_nan: Boolean flag indicating whether to fill missing values with the lowest observed value.

    Returns:
    - normalized: Normalized MIR image array.
    """
    return normalize(
        img, min_rad=VIIRS_TIR_MIN_RAD, max_rad=VIIRS_TIR_MAX_RAD, fill_nan=fill_nan
    )


def crop_center(img, size=64, crop_dimensions=(0, 1)):
    """
    Crop images to the center.

    Parameters:
    - img: Input image array to be cropped.
    - size: Size of the cropped region. Default is 64.
    - crop_dimensions: Tuple of dimensions to crop. Default is (0, 1).

    Returns:
    - cropped: Cropped image array.

    Notes:
    - The input image array can have shapes [width, height], [width, height, bands], or [batch_size, width, height, channels].
    - The function calculates the center for each specified dimension and then determines the lower and upper bounds for cropping.
    - The resulting cropped image has dimensions [size, size] or [size, size, bands] or [batch_size, size, size, channels].
    """

    img = np.array(img)

    # Initialize slices for cropping
    slices = [slice(None)] * img.ndim

    # Determine center for each specified dimension and create slices
    for dim in crop_dimensions:
        center = img.shape[dim] // 2
        slices[dim] = slice(center - (size // 2), center + (size // 2))

    # Crop the image
    cropped = img[tuple(slices)]

    return cropped


"""
Post-processing functions
radiative_power()
brightness_temperature_from_radiance()
"""
VIIRS_MIR_RP_CONSTANT = 17.34
# TODO
# MODIS_MIR_RP_CONSTANT = 18.something #I do not remember. I'll check my old scripts.


def radiative_power(L_mir, active_map, cellsize=371, rp_constant=17.34):
    """
    @author Hannah Dietterich
    Calculates radiative power in Watts from MIR bands with background correction.

    Parameters:
    - L_mir: np.array of MIR spectral radiance values (not normalized!).
    - active_map: Binary array of the same size as L_mir, with 1s denoting hotspot pixels.
    - cellsize: Nadir resolution of the sensor in meters. Default is 371 for VIIRS.
    - rp_constant: Radiative power constant. Default is 17.34 for VIIRS.

    Returns:
    - totalrp: Total radiative power in Watts.

    Notes:
    - Formula described in Wooster et al., 2003.
    """

    values = np.array(L_mir)  # Raw radiance values
    selem = generate_binary_structure(2, 2)  # 3x3 matrix filled with TRUE
    label_img = label(
        active_map, connectivity=2
    )  # Assigns each pixel an integer group number (no change for binary mask, but kept anyway)

    Apix = cellsize**2  # Area of the pixels
    counter = 0
    rp = np.zeros([label_img.max(), 1])  # Initialize radiative power
    bginfo = np.zeros([label_img.max(), 2])  # Initialize background area

    # For each region:
    #   - create an image with just that region
    #   - dilate and remove the region to extract background
    #   - Then for each coordinate in the region, calculate RP and sum for the region
    for region in regionprops(
        label_img
    ):  # Returns various properties of each region in label_img
        subimage = np.zeros(active_map.shape)
        subimage[region.coords[:, 0], region.coords[:, 1]] = 1
        dilated = dilation(subimage, selem)
        bgimage = dilated - subimage
        bg = values[bgimage == 1].mean()
        bginfo[counter, :] = [bg, values[bgimage == 1].shape[0]]

        # Loop through each pixel in region and calculate RP
        rp[counter] = 0
        for row in region.coords:
            L4alert = values[row[0], row[1]]  # Find MIR
            dL4pix = L4alert - bg  # Find above-background radiance
            RPpix = (
                rp_constant * Apix * dL4pix
            )  # Convert to radiative power for 3.8 um (Wooster et al., 2003)
            rp[counter] = rp[counter] + RPpix  # Sum RP for each region

        counter = counter + 1

    totalrp = rp.sum()
    return totalrp


# Constants for calculating brightness temperature
h = 6.626e-34  # Planck's constant, Joules*Seconds
c = 2.99e8  # Speed of light, meters/second
k = 1.38e-23  # Boltzmann constant, Joules/Kelvin

VIIRS_MIR_WL = 3.74e-6  # wavelength in meters
VIIRS_TIR_WL = 11.45e-6
MODIS_MIR_WL = 3.959e-6
MODIS_TIR_WL = 12.02e-6


def brightness_temperature(L, wl=VIIRS_MIR_WL):
    """
    Calculates brightness temperature from radiance values.

    Parameters:
    - L: Radiance values, list, array, or a single radiance value, in units of W*m^-3*steradians^-1.
    - wl: Central wavelength of the band, in METERS (not micrometers). Default is 3.74e-6.

    Returns:
    - BT: Calculated brightness temperature, in Kelvin

    Notes:
    - The function uses Planck's law to convert radiance values to brightness temperature.
    - The default central wavelength of 3.74e-6 corresponds to VIIRS band I4.
    - central wavelengths for VIIRS I5 = 11.45e-6
    - MODIS b21 = 3.959e-6, b32 = 12.02e-6
    """
    K2 = (h * c) / (wl * k)
    K1 = (2.0 * h * (c**2)) / (wl**5)
    BT = K2 / (
        np.log1p(K1 / L)
    )  # Use np.log1p to avoid issues with K1/L close to zero, equivalent to np.log(1 + (K1/L))

    return BT


def plot_detection(
    radiance_image,
    mask,
    title="",
    save_filename=None,
    figsize=(4, 4),
    dpi=150,
    cmap="viridis",
    outline_color="red",
    outline_thickness=1,
):
    """
    Display a radiance image with highlighted hotspots using a binary mask.

    Parameters:
    - radiance_image (numpy.ndarray): Input radiance image.
    - mask (numpy.ndarray): Binary mask indicating the location of hotspots (same size as radiance_image).
    - title (str): Title for the plot.
    - save_filename (str): Filename to save the plot as a PNG. If None, the plot is not saved.
    - cmap (str): Colormap for displaying the radiance image. Default is 'viridis'.
    - outline_color (str): Color for highlighting hotspots. Default is 'red'.
    - outline_thickness (int): Thickness of the outline for highlighting. Default is 1.

    Notes:
    - The function resizes the input images to force contours to surround the pixels
    - Use this function to visualize radiance images with highlighted hotspots based on a binary mask.
    """
    # Resize the images
    radiance_image_resized = rescale(radiance_image, 10, order=0)
    mask_resized = rescale(mask, 10, order=0)

    # Create a subplot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Display the resized radiance image with specified colormap and interpolation
    im = ax.imshow(radiance_image_resized, cmap=cmap, interpolation="none")
    plt.colorbar(im, shrink=0.8, label="MIR radiance")

    # Find contours in the resized mask
    contours = find_contours(mask_resized, level=0.5)

    # Plot contours on the image
    for contour in contours:
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            linewidth=outline_thickness,
            color=outline_color,
        )

    # Configure plot properties
    ax.axis("off")
    ax.set_title(title)

    # Save the plot as a PNG if a filename is provided
    if save_filename:
        plt.savefig(save_filename, bbox_inches="tight", pad_inches=0.1, dpi=300)

    # Display the plot
    plt.show()


"""
Miscellaneous support functions
"""


def get_dn(datetime, volcano_lat, volcano_lng, volcano_elevation, twilight="CIVIL"):
    """
    Determines whether it is daytime (D) or nighttime (N) at a given location and time.

    Parameters:
    - datetime: Date and time for the observation.
    - volcano_lat: Latitude of the volcano location.
    - volcano_lng: Longitude of the volcano location.
    - volcano_elevation: Elevation of the volcano in meters.
    - twilight: Type of twilight to consider for daytime determination. Options are 'CIVIL', 'NAUTICAL', or 'ASTRONOMICAL'.
                Default is 'CIVIL'.

    Returns:
    - DN_flag: Returns "D" for daytime or "N" for nighttime.

    Notes:
    - default twilight is 'CIVIL', for d/n threshold as the sun is 6º below the horizon
    """
    obs = ephem.Observer()
    obs.lat = str(volcano_lat)  # Observer coordinates set using strings
    obs.lon = str(volcano_lng)
    obs.elevation = volcano_elevation  # Elevation in meters
    obs.date = datetime

    sun = ephem.Sun(obs)
    sun.compute(obs)
    sun_angle = np.rad2deg(float(sun.alt))

    # Set threshold for different twilight types
    if twilight.upper() == "CIVIL":
        threshold = -6
    elif twilight.upper() == "NAUTICAL":
        threshold = -12
    elif twilight.upper() == "ASTRONOMICAL":
        threshold = -18
    else:
        raise ValueError(
            "Invalid twilight type. Choose 'CIVIL', 'NAUTICAL', or 'ASTRONOMICAL'."
        )

    # Set DN_flag to "D" for daytime or "N" for nighttime
    DN_flag = sun_angle >= threshold

    if DN_flag:
        return "D"
    else:
        return "N"


def get_solar_coords(datetime, volcano_lat, volcano_lng, volcano_elevation):
    """
    Computes solar zenith and azimuth angles at a given location and time.

    Parameters:
    - datetime: Date and time for the observation.
    - volcano_lat: Latitude of the volcano location.
    - volcano_lng: Longitude of the volcano location.
    - volcano_elevation: Elevation of the volcano in meters.

    Returns:
    - solar_coords: Tuple containing solar zenith and azimuth angles in degrees.
    """
    obs = ephem.Observer()
    obs.lat = str(volcano_lat)  # Observer coordinates set using strings
    obs.lon = str(volcano_lng)
    obs.elevation = volcano_elevation
    obs.date = datetime

    sun = ephem.Sun(obs)
    sun.compute(obs)

    # Calculate solar zenith and azimuth angles
    solar_zenith_angle = 90 - np.rad2deg(float(sun.alt))
    solar_azimuth_angle = np.rad2deg(float(sun.az)) - 180

    return (solar_zenith_angle, solar_azimuth_angle)


def haversine_np(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Works with both numpy arrays and scalars, or a mix - lon/lat 1
    can be numpy arrays, while lon/lat 2 are scaler values, and it will
    calculate the distance from lon/lat 2 to each point in the lon/lat 1
    arrays.

    Less precise than vincenty, but fine for short distances,
    and works on vector math

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def load_volcanoes() -> pandas.DataFrame:
    """
    Load a list of volcanoes from the USGS Volcano Hazards Program API.

    This function fetches volcano data from `volcanoes.usgs.gov` in GeoJSON format,
    extracts relevant details (longitude, latitude, name, and ID), and returns
    the data as a pandas DataFrame.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - `lon` (float): Longitude of the volcano.
        - `lat` (float): Latitude of the volcano.
        - `name` (str): Name of the volcano.
        - `id` (str): Unique identifier for the volcano.

    Notes:
    ------
    - The function filters out volcanoes that do not have a `volcanoCd` (ID).
    - The API response is expected to be in GeoJSON format.

    Raises:
    -------
    urllib.error.URLError
        If there is an issue connecting to the API.
    json.JSONDecodeError
        If the response cannot be parsed as JSON.
    KeyError
        If the expected keys are missing from the API response.
    """
    url = "https://volcanoes.usgs.gov/vsc/api/volcanoApi/geojson"
    with urllib.request.urlopen(url) as response:
        volcs = json.load(response)

    features = volcs["features"]
    data = [
        {
            "lon": feature["geometry"]["coordinates"][0],
            "lat": feature["geometry"]["coordinates"][1],
            "name": feature["properties"]["volcanoName"],
            "id": feature["properties"]["volcanoCd"],
        }
        for feature in features
        if feature["properties"]["volcanoCd"]
    ]

    df = pandas.DataFrame(data)
    return df


def latlon_to_utm(lat, lon):
    """Convert latitude/longitude to UTM coordinates."""
    proj_utm = Proj(
        proj="utm", zone=int((lon + 180) / 6) + 1, ellps="WGS84", datum="WGS84"
    )
    x, y = proj_utm(lon, lat)
    return x, y, proj_utm.crs
