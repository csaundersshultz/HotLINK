from setuptools import setup, find_packages

setup(
    name='hotlink',
    version='0.1',
    packages=find_packages(),
    install_requires=[
    	# DEPENDENCIES
    	'os',
    	'numpy',
    	'pandas',
    	'ephem',
    	'scipy',
    	'scikit-image',
    ],
)