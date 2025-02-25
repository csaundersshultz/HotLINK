import functools
import math
import pathlib
import shutil
import time
import traceback
import warnings

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import earthaccess
import numpy as np
import pandas
import utm

from datetime import datetime
from pyresample import geometry
from satpy import Scene
from tqdm import tqdm

from .process import _gen_date_and_dir

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


# Define the pairing map
viirs_pair_map = {"VNP02IMG": "VNP03IMG", "VJ102IMG": "VJ103IMG", "VJ202IMG": "VJ203IMG"}

# Function to get the match key, with substitution for list1
def get_match_key(granule, substitute_prefix=False):
    url = granule.data_links()[0]
    filename = url.split("/")[-1]  # Extract filename from URL
    parts = filename.split(".")
    # If substituting (for list1), map the prefix
    if substitute_prefix and parts[0] in viirs_pair_map:
        parts[0] = viirs_pair_map[parts[0]]
    # Keep everything up to the version, drop processing time and extension
    match_key = ".".join(parts[:4])  # e.g., VJ103IMG.A2018005.1148.021
    return match_key

_process_func: functools.partial = None
def download_preprocess(
    dates,
    vent,
    sat='modis',
    batchsize=200,
    folder='./data',
    output=pathlib.Path('./Output')
):
    global _process_func
    from . import wingdbstub

    dest = pathlib.Path(folder)
    lat,lon=vent
    bounding_box=(float(lon)-0.05,float(lat)-0.05,float(lon)+0.05,float(lat)+0.05)

    results, results2 = make_query(dates,bounding_box,sat)
    num_hits = len(results) + len(results2)
    if results2:
        # Match the VIIRS products, checking the file names since the two result
        # lists may not match 1:1
        results1_data = [{"granule": g, "match_key": get_match_key(g, substitute_prefix=True)} for g in results]
        results2_data = [{"granule": g, "match_key": get_match_key(g, substitute_prefix=False)} for g in results2]
        df1 = pandas.DataFrame(results1_data)
        df2 = pandas.DataFrame(results2_data)
        merged = pandas.merge(df1, df2, how="inner", on="match_key", suffixes=("_1", "_2"))
        results = merged[["granule_1", "granule_2"]].to_numpy().tolist()
    else:
        # Just to keep the data structures identical
        results = list(zip(results))

    to_download = []
    meta = {}
    for items in results:
        item_names =  [pathlib.Path(x.data_links()[0]) for x in items]
        item_links = [x.data_links()[0] for x in items]
        meta_value = {
            'satelite': items[0]['umm']['Platforms'][0]['ShortName'],
            'url': item_links, # for all items
            'DayNightFlag': items[0]['umm']['DataGranule']['DayNightFlag'],
        }

        processed_name: pathlib.Path = _gen_output_name(dest, item_names)
        meta[processed_name.name] = meta_value # Left over from the for loop above.

        # See if we have downloaded and pre-processed,
        # but not fully processed these files
        if processed_name.exists():
            # We already have this file, no need to download it again.
            continue

        # See if we have downloaded AND PROCESSED this file.
        _, out_dir = _gen_date_and_dir(processed_name, output)
        out_file = out_dir / processed_name.name
        if out_file.exists():
            #  We have this in the final output directory, move it into
            # the "to be processed" directory for re-processing.
            shutil.move(str(out_file), str(processed_name))
            continue

        to_download.extend(items)

    results = to_download

    if sat in ['viirsj1','viirsj2','viirsn']:
        sat='viirs'

    # We always want batchsize to be even, so VIIRS files will be paired correctly.
    batchsize = batchsize + 1 if batchsize % 2 != 0 else batchsize
    num_results = len(results)
    batches = math.ceil(num_results / batchsize)
    num = min(num_results, batchsize) # number of images per batch

    print(f"Found {num_hits} files. Need to download {num_results}. Downloading in {batches} batches of {num}")

    area=area_definition('name',vent,sat)

    if sat == 'viirs':
        reader = 'viirs_l1b'
        datasets = ['I04','I05']
    else:
        reader = 'modis_l1b'
        datasets = ['21', '32']

    _process_func = functools.partial(load_and_resample, datasets, reader, area)

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
                out_file =  _gen_output_name(dest, files)

                future = executor.submit(_process_func, files, out_file)
                futures.append(future)
                args[future] = (files, out_file.name)

            # Verify completion of all resampling operations.
            for future in tqdm(as_completed(futures), total = len(futures), desc ="PRE-PROCESSING IMAGES", unit = "file"):
                files, out_filename = args[future]

                try:
                    future.result()
                except Exception as e:
                    # if we wanted to retry, we could get the original URL for this file
                    # by calling meta[files[0].name]
                    if out_filename not in meta:
                        print(f"Unable to process file {files[0].name}. No download URL found when attempting to retry. Skipping.")
                        print("ERROR:", e)

                        for file in files:
                            file.unlink(missing_ok = True)
                        continue

                    _retry_file(files, dest, meta[out_filename])

                    print(f"Unable to process file(s) {files} Exception occured:\n{e}")
                    traceback.print_exc()
                    continue

            print("Resampling of batch", k + 1, "complete in", time.time() - t1, "seconds")

    return meta

def _gen_output_name(dest, files):
    name_parts = files[0].stem.split('.')
    img_date = datetime.strptime(".".join(name_parts[1:3]), 'A%Y%j.%H%M')
    out_file =  dest / img_date.strftime('%Y%m%d_%H%M.npy')
    return out_file

def _retry_file(files, dest, download_meta):
    print("Retrying download/process of file(s):", files)
    out_file = _gen_output_name(dest, files)
    download_files = download_meta['url']
    if not isinstance(download_files, (list, tuple)):
        download_files = (download_files, )

    # Download one at a time
    for k in range(len(download_files)):
        download_batch(download_files, k, 1, dest)

    try:
        _process_func(files, out_file)
    except Exception as e:
        print("Unable to re-process. Exception occured:", e)

    # Clean up after the attempt
    for file in files:
        file.unlink(missing_ok = True) # Just make sure it is gone.


def load_and_resample(
    datasets: Sequence[str],
    reader: str,
    area: geometry.AreaDefinition,
    in_files: Sequence,
    out_file: str
) -> None:
    """
    Load datasets, resample to a specified area, and save the combined data to a file.
    Resampled data file will be in UTM.

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
