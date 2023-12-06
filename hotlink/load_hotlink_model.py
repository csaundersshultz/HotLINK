#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function to load the hotlink model

@author: Pablo Saunders-Shultz
"""


from tensorflow.keras.models import load_model
import os

def load_hotlink_model():
	"""
	Function loads the hotlink model as a tensorflow.model object
	some model functions are: predict, fit (to fine-tune), etc.
	input to the model has shape [X, 64, 64, 2], where X is the number of input images,
	and 2 corresponds to normalized MIR and TIR images
	"""
	#get the path to the model directory
	script_directory = os.path.dirname(os.path.realpath(__file__))

	#load the model
	hotlink_model = load_model( os.path.join(script_directory, "hotlink_model") )

	return hotlink_model