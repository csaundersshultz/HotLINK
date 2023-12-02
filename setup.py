from setuptools import setup, find_packages

setup(
    name='HotLINK',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    	# DEPENDENCIES
    	'numpy',
    	'pandas',
    	'ephem',
    	'scipy',
    	'scikit-image',
    ],
)