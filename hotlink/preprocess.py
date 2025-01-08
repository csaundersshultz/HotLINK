import earthaccess
import glob
import multiprocessing
import numpy as np
import os
import subprocess
import time
import utm

from datetime import datetime, timedelta
from functools import partial
from hotlink.support_functions import normalize_MIR, normalize_TIR
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

def download_preprocess(dates,vent,sat='modis',num=5,folder='./data'):
    lon,lat=vent
    bounding_box=(float(lon)-0.05,float(lat)-0.05,float(lon)+0.05,float(lat)+0.05)
    print(bounding_box)
    results,results1=make_query(dates,bounding_box,sat)
    if sat in ['viirsj1','viirsj2','viirsn']:
        sat='viirs'
    cwd=os.getcwd()
    for k in range(int(len(results)/num)):
        earthaccess.login()
        try:
            print('Descargando2')
            earthaccess.download(results[num*k:num*k+num], folder, threads=num)
        except Exception as e:
            print('Fallo descargando2!!!!')
            print(e)
            print('Intentando nuevamente')
            earthaccess.download(results[num*k:num*k+num], folder, threads=1)
        if sat=='viirs':
            try:
                print('Descargando3')
                earthaccess.download(results1[num*k:num*k+num], folder, threads=num)
            except Exception as e:
                print('Fallo descargando3!!!!')
                print(e)
                print('Intentando nuevamente')
                earthaccess.download(results1[num*k:num*k+num], folder, threads=1)
        if sat=='modis':
            ars2=glob.glob(folder+'/MOD0*')
            ars2+=glob.glob(folder+'/MYD0*')
        else:
            ars2=sorted(glob.glob(folder+'/VNP02*'))
            ars2+=sorted(glob.glob(folder+'/VJ102*'))
            ars2+=sorted(glob.glob(folder+'/VJ202*'))
            ars3=sorted(glob.glob(folder+'/VNP03*'))
            ars3+=sorted(glob.glob(folder+'/VJ103*'))
            ars3+=sorted(glob.glob(folder+'/VJ203*'))
        for i in range(len(ars2)):
            os.chdir(folder)
            os.chdir(cwd)
            file=ars2[i]
            if sat=='viirs':
                file1=ars3[i]
                #file1=glob.glob(folder+'/VNP03*'+file.split('.')[1]+'*')
            fecha=os.path.basename(file).split('.')[1]+'.'+os.path.basename(file).split('.')[2]
            fecha=datetime.strptime(fecha,'A%Y%j.%H%M')
            if sat=='viirs':
                scn=Scene(reader='viirs_l1b',filenames=[folder+'/'+os.path.basename(file),folder+'/'+os.path.basename(file1)])
            else:
                scn=Scene(reader='modis_l1b',filenames=[folder+'/'+os.path.basename(file)])
            if sat=='modis':
                scn.load(['21','32'],calibration='radiance')
            else:
                scn.load(['I04','I05'],calibration='radiance')
            vent=(lat,lon)
            area=area_definition('name',vent,sat)
            if sat=='modis':
                try:
                    cropscn = scn.resample(destination=area, datasets=['21','32'])
                except ValueError:
                    subprocess.call('rm -rf '+folder+'/'+os.path.basename(file),shell=True)
                    filename=file.split('/')[-1]
                    link=results[0].data_links()[0].split(filename.split('.')[0])[0]
                    link+=filename.split('.')[0]+'/'+filename
                    subprocess.call('wget -P '+folder+' '+link,shell=True)
                    scn=Scene(reader='modis_l1b',filenames=[folder+'/'+os.path.basename(file)])
                    scn.load(['21','32'],calibration='radiance')
                    cropscn = scn.resample(destination=area, datasets=['21','32'])
            else:
                try:
                    cropscn = scn.resample(destination=area, datasets=['I04','I05'])
                except ValueError:
                    print('REINTENTANDO',file,file1)
                    subprocess.call('rm -rf '+folder+'/'+os.path.basename(file),shell=True)
                    subprocess.call('rm -rf '+folder+'/'+os.path.basename(file1),shell=True)
                    filename=file.split('/')[-1]
                    link=results[0].data_links()[0].split(filename.split('.')[0])[0]
                    link+=filename.split('.')[0]+'/'+filename
                    subprocess.call('wget -P '+folder+' '+link,shell=True)
                    filename1=file1.split('/')[-1]
                    link1=results1[0].data_links()[0].split(filename1.split('.')[0])[0]
                    link1+=filename1.split('.')[0]+'/'+filename1
                    subprocess.call('wget -P '+folder+' '+link1,shell=True)
                    scn=Scene(reader='viirs_l1b',filenames=[folder+'/'+os.path.basename(file),folder+'/'+os.path.basename(file1)])
                    scn.load(['I04','I05'],calibration='radiance')
                    cropscn = scn.resample(destination=area, datasets=['I04','I05'])
            data=np.ones((70,70,2))*np.nan
            if sat=='modis':
                data[:,:,0]=cropscn['21'].values
                data[:,:,1]=cropscn['32'].values
            else:
                data[:,:,0]=cropscn['I04'].values
                data[:,:,1]=cropscn['I05'].values
            np.save(folder+'/'+fecha.strftime('%Y%m%d_%H%M.npy'),data)
            subprocess.call('rm -rf '+folder+'/'+os.path.basename(file),shell=True)
            if sat=='viirs':
                subprocess.call('rm -rf '+folder+'/'+os.path.basename(file1),shell=True)
                
