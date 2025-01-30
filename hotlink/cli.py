import argparse
import time
import hotlink

def parse_vent(value):
    """Parses vent argument: returns as string if a name, or tuple if coordinates."""
    if "," in value:  # Assuming coordinates are provided as "lat,lon"
        try:
            lat, lon = map(float, value.split(","))
            return (lat, lon)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid coordinates format. Use 'lat,lon' (e.g., '54.7554,-163.9711').")
    return value  # Otherwise, treat as a name

def parse_dates(value):
    """Parses date value, returns a tuple of dates"""
    dates = value.split(',')
    if len(dates) != 2:
        raise argparse.ArgumentTypeError("Invalid date format. Use \"start_date,stop_date\"")
    return tuple(dates)
    

def main():
    parser = argparse.ArgumentParser(description="Download VIIRS or MODIS images and run HotLINK analysis")
    parser.add_argument("vent", type=parse_vent, help="Vent name (e.g., 'Shishaldin') or coordinates (e.g., '54.7554,-163.9711').")
    parser.add_argument("elevation", type=float, help="Elevation, in meters")
    parser.add_argument("dates", type=parse_dates, help="Date range (e.g., '2023-01-01,2023-12-31')")
    parser.add_argument("sensor", type=str, choices=['modis', 'viirs'], help="Sensor name (viirs or modis)")

    args = parser.parse_args()
    
    t1 = time.time()
    results = hotlink.get_results(args.vent, args.elevation, args.dates, args.sensor.lower())
    results.to_csv('HotLINK Results.csv', index = False)

    print(f"Calculated results in {time.time() - t1}")
    
if __name__ == "__main__":
    main()