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


#get directory information
script_directory = os.path.dirname(os.path.realpath(__file__))
example_data_dir = os.path.join(script_directory, "example_data")

#Initialize all 257 file pairs
file_pairs = []
files = os.listdir(example_data_dir) #relative path to data

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
	Open MIR and TIR example files, numbered 0-256

	"""
	files = file_pairs[num]
	mir = np.array( open( os.path.join(example_data_dir, files[0]) ) )
	tir = np.array( open( os.path.join(example_data_dir, files[1]) ) )
	date = datetime.strptime( files[0].lstrip('I04_').rstrip('_shis.tif'), "%Y%m%d_%H%M%S" )
	return mir, tir, date