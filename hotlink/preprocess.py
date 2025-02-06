import functools
import math
import pathlib
import time
import traceback
import warnings

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import earthaccess
import numpy as np
import utm

from datetime import datetime
from pyresample import geometry
from satpy import Scene
from tqdm import tqdm

def area_definition(area_id, lat_lon,sat='modis'):
    utm_x, utm_y, utm_zone, utm_lat_band = utm.from_latlon(lat_lon[0], lat_lon[1])
    #print(utm_x, utm_y, utm_zone)
    projection = {'proj':'utm', 'zone':utm_zone, 'ellps':'WGS84', 'datum': 'WGS84', 'units':'m'}
    centerpt = (utm_x, utm_y)
    #resolution =1000 for modis
    if sat=='modis':
        resolution= 1000 #static resolution of 371 meters, Hannah pulls it from the scene resolution...
    else:
        resolution=371
    area_def = geometry.AreaDefinition.from_area_of_interest(area_id, projection, [70,70], centerpt, resolution)
    return area_def

def make_query(dates,bounding_box,sat='modis'):
    if sat=='modis':
        names=['MOD021KM','MYD021KM']
        names2 = []
    elif sat=='viirs':
        names = ['VJ102IMG','VJ202IMG','VNP02IMG']
        names2 = ['VJ103IMG','VJ203IMG','VNP03IMG']
    elif sat=='viirsj2':
        names=['VJ202IMG']
        names2 = ['VJ203IMG']
        sat='viirs'
    elif sat=='viirsj1':
        names = ['VJ102IMG']
        names2 = ['VJ103IMG']
        sat='viirs'
    elif sat=='viirsn':
        names=['VNP02IMG']
        names2 = ['VNP03IMG']
        sat='viirs'

    results=[]
    results2 = []

    for name in names:
        results += earthaccess.search_data(
                    short_name=name,
                    bounding_box=bounding_box,
                    temporal=dates
                )

    for name in names2:
        results2 += earthaccess.search_data(
                    short_name=name,
                    bounding_box=bounding_box,
                    temporal=dates
                )

    return results, results2

class AuthenticationError(Exception):
    pass


def download_batch(results, batch, num, dest):
    threads = min(8, num) # Don't try for more than 8 threads (the default).
    to_download = results[num*batch:num*batch+num]

    auth = earthaccess.login(persist=True)
    if not auth.authenticated:
        print("Authentication to earthaccess failed. Unable to download.")
        raise AuthenticationError("Unable to authenticate")

    try:
        earthaccess.download(to_download, dest, threads=threads)
    except Exception as e:
        print('Downloading Error')
        print(e)
        print('Trying again with 1 thread')
        earthaccess.download(to_download, dest, threads=1)

def download_preprocess(dates,vent,sat='modis',batchsize=200, folder='./data'):
    lat,lon=vent
    bounding_box=(float(lon)-0.05,float(lat)-0.05,float(lon)+0.05,float(lat)+0.05)

    # TODO: Ensure that results and results2 actually match up 1:1 for VIIRS
    results, results2 = make_query(dates,bounding_box,sat)
    if results2:
        # "flatten" the two results lists, keeping paired files together so they download
        # in the same batch
        results = [
            item
            for pair in zip(results, results2)
            for item in pair
        ]

    meta = {
        pathlib.Path(x.data_links()[0]).name:
        {
            'satelite': x['umm']['Platforms'][0]['ShortName'],
            'url': x.data_links()[0],
            'DayNightFlag': x['umm']['DataGranule']['DayNightFlag'],
        }
        for x in results
    }

    if sat in ['viirsj1','viirsj2','viirsn']:
        sat='viirs'

    # We always want batchsize to be even, so VIIRS files will be paired correctly.
    batchsize = batchsize + 1 if batchsize % 2 != 0 else batchsize
    num_results = len(results)
    batches = math.ceil(num_results / batchsize)
    num = min(num_results, batchsize) # number of images per batch

    print(f"Found {num_results} files. Downloading in {batches} batches of {num}")

    area=area_definition('name',vent,sat)
    dest = pathlib.Path(folder)

    if sat == 'viirs':
        reader = 'viirs_l1b'
        datasets = ['I04','I05']
    else:
        reader = 'modis_l1b'
        datasets = ['21', '32']

    # Because we can. Not really, but keeps the submit call later cleaner.
    process_func = functools.partial(load_and_resample, datasets, reader, area)

    with ProcessPoolExecutor() as executor:
        # Download/process in batches of no more than batchsize files to save disk space
        # Each file takes around 200MB of space, so 200 files ~=40GB disk space. Processed
        # files are much smaller.
        for k in range(batches):
            futures = []
            args = {}

            print(f'Downloading files (batch {k + 1}/{batches})')            
            download_batch(results, k, num, folder)

            # VIIRS files are paired
            if sat=='viirs':
                ars2=sorted(dest.glob('VNP02*'))
                ars2+=sorted(dest.glob('VJ102*'))
                ars2+=sorted(dest.glob('VJ202*'))

                ars3=sorted(dest.glob('VNP03*'))
                ars3+=sorted(dest.glob('VJ103*'))
                ars3+=sorted(dest.glob('VJ203*'))

                input_files = zip(ars2, ars3)
            else:
                # We run zip here to keep the file list in a consistant format with VIIRS.
                # Each element will be a single-element tuple.
                input_files=zip(dest.glob('M[OY]D0*'))

            input_files = tuple(input_files)

            t1 = time.time()
            print("Downloading batch", k + 1, "complete. Beginning resampling.")

            for files in tqdm(input_files, total = len(input_files), desc = "SUBMITTING TASKS", unit = "file"):
                name_parts = files[0].stem.split('.')
                img_date = datetime.strptime(".".join(name_parts[1:3]), 'A%Y%j.%H%M')
                out_file =  dest / img_date.strftime('%Y%m%d_%H%M.npy')

                future = executor.submit(process_func, files, out_file)
                futures.append(future)
                args[future] = (files, out_file.name)

            # Verify completion of all resampling operations.
            # We could do things like retry here if we wanted.
            for future in tqdm(as_completed(futures), total = len(futures), desc ="PRE-PROCESSING IMAGES", unit = "file"):
                files, out_filename = args[future]
                try:
                    future.result()
                except Exception as e:
                    # if we wanted to retry, we could get the original URL for this file
                    # by calling meta[files[0].name]
                    print(f"Unable to process file(s) {files} Exception occured:\n{e}")
                    traceback.print_exc()
                    continue

                # Adjust the metadata filename to key off the output file rather than the input.
                out_meta = meta.pop(files[0].name)

                if len(files) > 1: # the viirs paired file, if viirs
                    out_meta['url'] = [
                        out_meta['url'],
                        meta.pop(files[1].name)['url']
                    ]

                meta[out_filename] = out_meta
            print("Resampling of batch", k + 1, "complete in", time.time() - t1, "seconds")

    return meta

def load_and_resample(
    datasets: Sequence[str],
    reader: str,
    area: geometry.AreaDefinition,
    in_files: Sequence,
    out_file: str
) -> None:
    """
    Load datasets, resample to a specified area, and save the combined data to a file.

    Parameters
    ----------
    datasets : Sequence[str]
        List of dataset names to load and process.
    reader : str
        Reader type used by SatPy to load the datasets.
    area : geometry.AreaDefinition
        The area to which the data should be resampled.
    in_files : Sequence
        List of input file paths containing the datasets.
    out_file : str
        Path to the output file where the resampled data will be saved.

    Returns
    -------
    None
        Output is saved to a Numpy file.

    Notes
    -----
    - source files are deleted after processing.

    Warnings
    --------
    - Warnings related to inefficient chunking operations are suppressed.
    """
    # Loading the scene results in warnings about an ineficient chunking operations
    # Since this is SatPy, and we can't do anything about it, just ignore the warnings.
    warnings.simplefilter("ignore", UserWarning)

    scn=Scene(reader=reader,filenames=[str(f.absolute()) for f in in_files])
    scn.load(datasets,calibration='radiance')
    cropscn = scn.resample(destination=area, datasets=datasets)
    mir = cropscn[datasets[0]].to_numpy()
    tir = cropscn[datasets[1]].to_numpy()

    # Fill missing values
    mir[np.isnan(mir)] = np.nanmin(mir)
    tir[np.isnan(tir)] = np.nanmin(tir)

    data = np.dstack((mir, tir))
    np.save(out_file, data)

    for file in in_files:
        file.unlink()
