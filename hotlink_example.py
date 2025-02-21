import json
import time

if __name__ == "__main__":
    import hotlink

    vent = "Shishaldin" # Can also be defined as a latitude, longitude pair for more precise positioning.
    elevation = 2857 # Used to calculate solar zenith/azimuth angles

    dates = ("2019-07-29", "2019-07-31 00:00") # YYYY-MM-DD HH:MM, from to.

    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'viirs'

    t1 = time.time()

    results, meta = hotlink.get_results(
        vent,
        elevation,
        dates,
        sensor,
        out_dir = f'Output/{sensor}'
    )
    
    print(results)
    
    results.to_csv(f'Output/{sensor}/Results.csv', index = False, float_format='%.4f')

    with open(f'Output/{sensor}/metadata.json', 'w') as f:
        json.dump(meta, f, indent = 4)

    print(f"Calculated results in {time.time() - t1}")
