import argparse
import json
import pathlib
import time

from datetime import datetime, UTC

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
    # Parse the arguments to run the model
    
    parser = argparse.ArgumentParser(
        description="""Download VIIRS or MODIS images and run HotLINK analysis.
Output will be saved to a HotLINK Results.csv file, and associated images
will be saved to the output directory, in a year/month sub-folder""",
        formatter_class=argparse.RawTextHelpFormatter  # For better help formatting
    )
    parser.add_argument("vent", type=parse_vent, help="Vent name (e.g., 'Shishaldin') or coordinates (e.g., '54.7554,-163.9711').")
    parser.add_argument("elevation", type=float, help="Elevation, in meters")
    parser.add_argument("dates", type=parse_dates, help="Date range (e.g., '2023-01-01,2023-12-31')")
    parser.add_argument("sensor", type=str, choices=['modis', 'viirs'], help="Sensor name (viirs or modis)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Output",  # Set the default value to "Output"
        help="Directory to save output files (default: Output)"  # Update the help text
    )    

    try:
        args = parser.parse_args()
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}", file=sys.stderr)  # Print errors to stderr
        parser.print_help(sys.stderr) # Print help to stderr
        sys.exit(2)  # Exit with a non-zero code indicating an error
    except Exception as e: # Catch other exceptions during processing
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)        
    
    ##################### RUN ##################
    t1 = time.time()
    out_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    results, meta = hotlink.get_results(
        args.vent,
        args.elevation,
        args.dates,
        args.sensor.lower(),
        out_dir
    )
    
    ################## Handle Results ############
    # Save results to a CSV file
    csv_file = out_dir / 'HotLINK Results.csv'
    results.to_csv(csv_file, index = False, float_format='%.4f')
    
    # And metadata to a json file
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent = 4)

    print(f"Calculated results in {time.time() - t1}")
    
if __name__ == "__main__":
    main()