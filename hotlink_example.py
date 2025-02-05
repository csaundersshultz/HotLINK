import time

import rasterio

from pyproj import Proj

if __name__ == "__main__":
    import hotlink

    vent = "Shishaldin"
    elevation = 2857

    dates = ("2019-07-30 22:00", "2019-07-31") # YYYY-MM-DD HH:MM, from to.

    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'viirs'

    t1 = time.time()
    results = hotlink.get_results(vent, elevation, dates, sensor, out_dir = f'Output/{sensor}')
    print(results)
    results.to_csv(f'Output/{sensor}/HotLINK Results.csv', index = False)
    print(f"Calculated results in {time.time() - t1}")
