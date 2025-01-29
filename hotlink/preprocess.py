import math
import pathlib

import earthaccess
import numpy as np
import os
import subprocess
import utm

from datetime import datetime
from pyresample import geometry
from satpy import Scene

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
    elif sat=='viirs':
        names=['VJ102IMG','VJ202IMG','VNP02IMG']
        names1=['VJ103IMG','VJ203IMG','VNP03IMG']
    elif sat=='viirsj2':
        names=['VJ202IMG']
        names1=['VJ203IMG']
        sat='viirs'
    elif sat=='viirsj1':
        names=['VJ102IMG']
        names1=['VJ103IMG']
        sat='viirs'
    elif sat=='viirsn':
        names=['VNP02IMG']
        names1=['VNP03IMG']
        sat='viirs'

    results=[]
    results1=[]
    if sat=='modis':
        for name in names:
            results += earthaccess.search_data(
                        short_name=name,
                        bounding_box=bounding_box,
                        temporal=dates
                    )
    elif sat=='viirs':

        for i in range(len(names)):
            try:
                results += earthaccess.search_data(
                            short_name=names[i],
                            bounding_box=bounding_box,
                            temporal=dates
                        )
            except IndexError as e:
                print('error',e)
                pass
            try:
                results1 += earthaccess.search_data(
                            short_name=names1[i],
                            bounding_box=bounding_box,
                            temporal=dates
                        )
            except IndexError as e:
                print('error',e)
                pass
    return results, results1

class AuthenticationError(Exception):
    pass

def download_preprocess(dates,vent,sat='modis',batchsize=100,folder='./data'):
    lat,lon=vent
    bounding_box=(float(lon)-0.05,float(lat)-0.05,float(lon)+0.05,float(lat)+0.05)

    results,results1=make_query(dates,bounding_box,sat)

    if sat in ['viirsj1','viirsj2','viirsn']:
        sat='viirs'

    area=area_definition('name',vent,sat)

    cwd=os.getcwd()

    num_results = len(results)
    batches = math.ceil(num_results / batchsize)
    num = min(num_results + 1, batchsize) # number of images per batch
    threads = min(5, num) # Don't try for more than 5 threads.

    dest = pathlib.Path(folder)

    for k in range(batches):
        auth = earthaccess.login(persist=True)
        if not auth.authenticated:
            print("Authentication to earthaccess failed. Unable to download.")
            raise AuthenticationError("Unable to authenticate")

        try:
            print(f'Downloading MIR/Combined files (batch {k}/{batches})')
            earthaccess.download(results[num*k:num*k+num], str(dest), threads=threads)
        except Exception as e:
            print('Downloading Error')
            print(e)
            print('Trying again with 1 thread')
            earthaccess.download(results[num*k:num*k+num], str(dest), threads=1)

        if sat=='viirs':
            try:
                print(f'Downloading VIIRS TIR files (batch {k}/{batches})')
                earthaccess.download(results1[num*k:num*k+num], str(dest), threads=num)
            except Exception as e:
                print('Download error on VIIRS TIR')
                print(e)
                print('Trying again with 1 thread')
                earthaccess.download(results1[num*k:num*k+num], str(dest), threads=1)


        if sat=='viirs':
            ars2=sorted(dest.glob('VNP02*'))
            ars2+=sorted(dest.glob('VJ102*'))
            ars2+=sorted(dest.glob('VJ202*'))

            ars3=sorted(dest.glob('VNP03*'))
            ars3+=sorted(dest.glob('VJ103*'))
            ars3+=sorted(dest.glob('VJ203*'))
        else:
            ars2=dest.glob('MOD0*')
            ars2+=dest.glob('MYD0*')


        for i in range(len(ars2)):
            os.chdir(cwd)

            files=[ars2[i]]
            reader = 'modis_l1b'
            if sat=='viirs':
                reader = 'viirs_l1b'
                files.append(ars3[i])

            name_parts = files[0].stem.split('.')
            img_date = datetime.strptime(".".join(name_parts[1:3]), 'A%Y%j.%H%M')
            try:
                scn=Scene(reader=reader,filenames=[str(f.absolute()) for f in files])

                if sat=='viirs':
                    scn.load(['I04','I05'],calibration='radiance')
                    try:
                        cropscn = scn.resample(destination=area, datasets=['I04','I05'])
                    except ValueError:
                        file, file1 = files
                        print('Retrying',str(file),str(file1))
                        file.unlink()
                        file1.unlink()
                        filename=file.name
                        link=results[0].data_links()[0].split(filename.split('.')[0])[0]
                        link+=filename.split('.')[0]+'/'+filename
                        subprocess.call('wget -P '+str(dest)+' '+link,shell=True)
                        filename1=file1.name
                        link1=results1[0].data_links()[0].split(filename1.split('.')[0])[0]
                        link1+=filename1.split('.')[0]+'/'+filename1
                        subprocess.call('wget -P '+str(dest)+' '+link1,shell=True)

                        scn=Scene(reader='viirs_l1b',filenames=[str(file.absolute()),str(file1.absolute())])
                        scn.load(['I04','I05'],calibration='radiance')
                        cropscn = scn.resample(destination=area, datasets=['I04','I05'])
                    mir = cropscn['I04'].to_numpy()
                    tir = cropscn['I05'].to_numpy()

                else:
                    scn.load(['21','32'],calibration='radiance')
                    try:
                        cropscn = scn.resample(destination=area, datasets=['21','32'])
                    except ValueError:
                        file = files[0]
                        file.unlink()
                        filename=file.name

                        link=results[0].data_links()[0].split(filename.split('.')[0])[0]
                        link+=filename.split('.')[0]+'/'+filename
                        subprocess.call('wget -P '+str(dest)+' '+link,shell=True)
                        scn=Scene(reader='modis_l1b',filenames=[str(file.absolute())])
                        scn.load(['21','32'],calibration='radiance')
                        cropscn = scn.resample(destination=area, datasets=['21','32'])
                    mir = cropscn['21'].to_numpy()
                    tir = cropscn['32'].to_numpy()

                # Fill missing values
                min_mir_observered = np.nanmin(mir)
                min_tir_observered = np.nanmin(tir)
                mir[np.isnan(mir)] = min_mir_observered
                tir[np.isnan(tir)] = min_tir_observered

                data=np.ones((70,70,2))*np.nan
                data[:,:,0]=mir
                data[:,:,1]=tir

                np.save(dest / img_date.strftime('%Y%m%d_%H%M.npy'),data)
            except Exception as e:
                print(f"Unable to process {files[0]}. Skipping")
                print(e)
            finally:
                for file in files:
                    file.unlink()
