#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function to load the hotlink model

@author: Pablo Saunders-Shultz
"""


from tensorflow.keras.models import load_model
import os

def load_hotlink_model(**kwargs):
	 """
    Function loads the hotlink model as a tensorflow.model object
    some model functions are: predict, fit (to fine-tune), etc.
    input to the model has shape [X, 64, 64, 2], where X is the number of input images,
    and 2 corresponds to normalized MIR and TIR images

    Parameters:
    - **kwargs: Additional keyword arguments to pass to the load_model function.

    Returns:
    - hotlink_model: Loaded TensorFlow model.
    """

	#get the path to the model directory
	script_directory = os.path.dirname(os.path.realpath(__file__))
	model_directory = os.path.join(script_directory, "hotlink_model")

	#load the model
	hotlink_model = load_model( model_directory, **kwargs)

	return hotlink_model