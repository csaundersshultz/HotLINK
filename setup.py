from setuptools import setup, find_packages

setup(
    name="hotlink",
    version="1.3",
    url="https://github.com/csaundersshultz/HotLINK",
    author="Pablo Saunders-Shultz",
    author_email="csaundersshultz@gmail.com",
    license="MIT",
    description="An example of a python package from pre-existing code",
    packages=find_packages(),
    install_requires=[
        # DEPENDENCIES
        "numpy",
        "pandas",
        "ephem",
        "scipy",
        "scikit-image",
        "Pillow",
        "tensorflow==2.15",
        "matplotlib",
    ],
    include_package_data=True,
    zip_safe=False,
)
