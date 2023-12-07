#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function for loading example dataset

@author: Pablo Saunders-Shultz
"""


import os
import numpy as np
from PIL.Image import open
from datetime import datetime

# Get directory information
script_directory = os.path.dirname(os.path.realpath(__file__))
example_data_dir = os.path.join(script_directory, "example_data")

# Initialize all 257 file pairs
file_pairs = []
files = os.listdir(example_data_dir)  # Relative path to data

# Filter files that match the pattern "I04_date_time.tif" or "I05_date_time.tif"
i04_files = [file for file in files if file.startswith("I04_")]
i05_files = [file for file in files if file.startswith("I05_")]

# Sort files to ensure corresponding I04 and I05 files are in order
i04_files.sort()
i05_files.sort()

# Iterate through the sorted files and create pairs
for i04_file, i05_file in zip(i04_files, i05_files):
    file_pairs.append([i04_file, i05_file])


def load_example_data(num):
    """
    Open MIR and TIR example files, numbered 0-256.

    Parameters:
    - num: Index of the file pair to load (0-256).

    Returns:
    - mir: NumPy array representing MIR data.
    - tir: NumPy array representing TIR data.
    - date: Datetime object representing the date and time of the files.
    """

    if not (0 <= num <= 256):
        raise ValueError("Input 'num' must be in the range 0-256.")

    files = file_pairs[num]
    mir = np.array(open(os.path.join(example_data_dir, files[0])))
    tir = np.array(open(os.path.join(example_data_dir, files[1])))
    date = datetime.strptime(files[0].lstrip('I04_').rstrip('_shis.tif'), "%Y%m%d_%H%M%S")
    
    return mir, tir, date


def load_entire_dataset():
    """
    Load the entire dataset of 256 image pairs.

    Returns:
    - dataset: NumPy array representing the dataset with shape (256, 64, 64, 2).
    - dates: List of datetime objects representing the date and time of each file pair.
    """

    dataset = []
    dates = []

    for num in range(256):
        mir, tir, date = load_example_data(num)
        dataset.append(np.stack([mir, tir], axis=-1))
        dates.append(date)

    return np.array(dataset, dtype=np.float32), dates





