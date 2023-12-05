from setuptools import setup, find_packages

setup(
    name='hotlink',
    version='0.1',
    url='https://github.com/csaundersshultz/HotLINK',
    author='Pablo Saunders-Shultz',
    author_email='csaundersshultz@gmail.com',
    license='MIT',
    description='An example of a python package from pre-existing code',
    packages=find_packages(),
    install_requires=[
    	# DEPENDENCIES
    	'numpy',
    	'pandas',
    	'ephem',
    	'scipy',
    	'scikit-image',
    	'Pillow'
    ],
    zip_safe=False,

)