import argparse
import time

def parse_vent(value):
    """Parses vent argument: returns as string if a name, or tuple if coordinates."""
    if "," in value:  # Assuming coordinates are provided as "lat,lon"
        try:
            lat, lon = map(float, value.split(","))
            return (lat, lon)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid coordinates format. Use 'lat,lon' (e.g., '54.7554,-163.9711').")
    return value  # Otherwise, treat as a name

if __name__ == "__main__":
    import hotlink

    # parser = argparse.ArgumentParser(description="Run HotLINK_runner.get_results from the command line.")
    # parser.add_argument("vent", type=parse_vent, help="Vent name (e.g., 'Shishaldin') or coordinates (e.g., '54.7554,-163.9711').")
    # parser.add_argument("elevation", type=float, help="Elevation value")
    # parser.add_argument("dates", type=str, help="Date range (e.g., '2023-01-01,2023-12-31')")
    # parser.add_argument("sensor", type=str, help="Sensor name")

    # args = parser.parse_args()
    
    # # Convert dates if needed (assuming it's a comma-separated range)
    # dates = args.dates.split(",")
        
    #  Spurr
    # vent = (61.2997,-152.2514) # lat,lon
    # elevation = 3374

    #  Shishaldin
    # vent = (54.7554, -163.9711)
    vent = "Shishaldin"
    elevation = 2857

    dates = ("2019-07-30", "2019-07-31") # YYYY-MM-DD HH:MM, from to.

    # Options: modis,viirs,viirsj2,viirsj1,viirsn
    sensor = 'modis'

    t1 = time.time()
    results = hotlink.get_results(vent, elevation, dates, sensor)
    print(results)
    results.to_csv('HotLINK Results.csv', index = False)
    print(f"Calculated results in {time.time() - t1}")
