import json
import time

if __name__ == "__main__":
    import hotlink

    vent = "Spurr" # Can also be defined as a latitude, longitude pair for more precise positioning.
    elevation = 3370 # Used to calculate solar zenith/azimuth angles


    dates = ("2013-01-01", "2025-03-01") # YYYY-MM-DD HH:MM, from to.

    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'viirs'

    t1 = time.time()

    results, meta = hotlink.get_results(
        vent,
        elevation,
        dates,
        sensor,
        out_dir = f'/Volumes/Transfer/Output/{sensor}'
    )

    print(results)

    results.to_csv(f'Output/{sensor}/Results.csv', index = False, float_format='%.4f')

    with open(f'Output/{sensor}/metadata.json', 'w') as f:
        json.dump(meta, f, indent = 4)

    print(f"Calculated results in {time.time() - t1}")
