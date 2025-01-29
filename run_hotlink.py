import time

import HotLINK_runner

if __name__ == "__main__":
    #  Spurr
    # vent = (61.2997,-152.2514) # lat,lon
    # elevation = 3374

    #  Shishaldin
    # vent = (54.7554, -163.9711)
    vent = "Shishaldin"
    elevation = 2857

    dates = ("2019-07-01", "2019-07-21") # YYYY-MM-DD HH:MM, from to.

    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'viirs'

    t1 = time.time()
    results = HotLINK_runner.get_results(vent, elevation, dates, sensor)
    print(results)
    results.to_csv('HotLINK Results.csv', index = False)
    print(f"Calculated results in {time.time() - t1}")
