from setuptools import setup, find_packages

setup(
    name='your-package-name',
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