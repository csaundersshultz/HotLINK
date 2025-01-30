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

def main():
    import hotlink
    
    parser = argparse.ArgumentParser(description="Run HotLINK_runner.get_results from the command line.")
    parser.add_argument("vent", type=parse_vent, help="Vent name (e.g., 'Shishaldin') or coordinates (e.g., '54.7554,-163.9711').")
    parser.add_argument("elevation", type=float, help="Elevation value")
    parser.add_argument("dates", type=str, help="Date range (e.g., '2023-01-01,2023-12-31')")
    parser.add_argument("sensor", type=str, help="Sensor name")

    args = parser.parse_args()
    dates = args.dates.split(",")
    
    t1 = time.time()
    results = hotlink.get_results(args.vent, args.elevation, dates, args.sensor)
    results.to_csv('HotLINK Results.csv', index = False)

    print(f"Calculated results in {time.time() - t1}")    