import json
import time

from datetime import UTC, datetime

if __name__ == "__main__":
    import hotlink

    vent = "Shishaldin"
    elevation = 2857

    dates = ("2019-07-30", "2019-07-31 00:00") # YYYY-MM-DD HH:MM, from to.

    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'viirs'

    t1 = time.time()
    results = hotlink.get_results(vent, elevation, dates, sensor, out_dir = f'Output/{sensor}')
    print(results)
    results.to_csv(f'Output/{sensor}/Results.csv', index = False, float_format='%.4f')

    meta = {
        'Vent': vent,
        'Elevation': elevation,
        'Data Dates': dates,
        'Sensor': sensor,
        'Num Results': len(results),
        'Run Date': datetime.now(UTC).isoformat(),
    }

    with open(f'Output/{sensor}/metadata.json', 'w') as f:
        json.dump(meta, f, indent = 4)

    print(f"Calculated results in {time.time() - t1}")
