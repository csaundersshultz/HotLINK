import earthaccess
import subprocess
import utm
import glob
import numpy as np
import os
from datetime import datetime, timedelta
from hotlink.support_functions import normalize_MIR, normalize_TIR
from pyresample import geometry
from satpy import Scene
import time

def area_definition(area_id, lat_lon):
    utm_x, utm_y, utm_zone, utm_lat_band = utm.from_latlon(lat_lon[0], lat_lon[1])
    #print(utm_x, utm_y, utm_zone)
    projection = {'proj':'utm', 'zone':utm_zone, 'ellps':'WGS84', 'datum': 'WGS84', 'units':'m'}
    centerpt = (utm_x, utm_y)
    #resolution =1000 for modis
    resolution= 1000 #static resolution of 371 meters, Hannah pulls it from the scene resolution...
    area_def = geometry.AreaDefinition.from_area_of_interest(area_id, projection, [64,64], centerpt, resolution)
    return area_def

def make_query(dates,bounding_box,sat='modis'):
    if sat=='modis':
        names=['MOD021KM','MYD021KM']
    elif sat=='viirs':
        names=['VJ202IMG','VNP02IMG']
        names1=['VJ203IMG','VNP03IMG']

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
            results += earthaccess.search_data(
                        short_name=names[i],
                        bounding_box=bounding_box,
                        temporal=dates
                    )
            results1 += earthaccess.search_data(
                        short_name=names1[i],
                        bounding_box=bounding_box,
                        temporal=dates
                    )
    return results, results1

def download_preprocess(dates,vent,sat='modis',num=5):
    lon,lat=vent
    bounding_box=(lon-0.05,lat-0.05,lon+0.05,lat+0.05)
    results,results1=make_query(dates,bounding_box,sat)
    cwd=os.getcwd()
    for k in range(int(len(results)/num)):
        earthaccess.login()
        earthaccess.download(results[num*k:num*k+num], "./data")
        if sat=='viirs':
            earthaccess.download(results1[num*k:num*k+num], "./data")
        if sat=='modis':
            ars2=glob.glob('./data/*')
        else:
            ars2=glob.glob('./data/VNP02*')
            ars2+=glob.glob('./data/VJ202*')
            ars3=glob.glob('./data/VNP03*')
            ars3+=glob.glob('./data/VJ203*')
        for i in range(len(ars2)):
            os.chdir('./data')
            try:
                os.chdir(cwd)
                file=ars2[i]
                if sat=='viirs':
                    file1=ars3[i]
                fecha=os.path.basename(file).split('.')[1]+'.'+os.path.basename(file).split('.')[2]
                fecha=datetime.strptime(fecha,'A%Y%j.%H%M')
                if sat=='viirs':
                    scn=Scene(reader='viirs_l1b',filenames=['./data/'+os.path.basename(file),'./data/'+os.path.basename(file1)])
                else:
                    scn=Scene(reader='modis_l1b',filenames=['./data/'+os.path.basename(file)])
                if sat=='modis':
                    scn.load(['21','32'],calibration='radiance')
                else:
                    scn.load(['I04','I05'],calibration='radiance')
                area=area_definition('name',vent)
                if sat=='modis':
                    cropscn = scn.resample(destination=area, datasets=['21','32'])
                else:
                    cropscn = scn.resample(destination=area, datasets=['I04','I05'])
                data=np.ones((64,64,2))*np.nan
                if sat=='modis':
                    data[:,:,0]=cropscn['21'].values
                    data[:,:,1]=cropscn['32'].values
                else:
                    data[:,:,0]=cropscn['I04'].values
                    data[:,:,1]=cropscn['I05'].values
                n_mir = normalize_MIR(data[:,:,0]) #note, normalize also fills in missing pixels
                n_tir = normalize_TIR(data[:,:,1])
                stacked = np.dstack([n_mir, n_tir])
                np.save('./data/'+fecha.strftime('%Y%m%d_%H%M.npy'),stacked)
                subprocess.call('rm -rf ./data/'+os.path.basename(file),shell=True)
                if sat=='viirs':
                    subprocess.call('rm -rf ./data/'+os.path.basename(file1),shell=True)
            except:
                subprocess.call('rm -rf ./data/'+os.path.basename(file),shell=True)
                if sat=='viirs':
                    subprocess.call('rm -rf ./data/'+os.path.basename(file1),shell=True)
                continue
                