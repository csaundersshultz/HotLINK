import json
import urllib

import numpy
import pandas


def haversine_np(lon1, lat1, lon2, lat2) -> numpy.ndarray:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Works with both numpy arrays and scalars, or a mix - lon/lat 1
    can be numpy arrays, while lon/lat 2 are scaler values, and it will
    calculate the distance from lon/lat 2 to each point in the lon/lat 1
    arrays.

    Less precise than vincenty, but fine for short distances,
    and works on vector math

    """
    lon1, lat1, lon2, lat2 = map(numpy.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        numpy.sin(dlat / 2.0) ** 2
        + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(dlon / 2.0) ** 2
    )

    c = 2 * numpy.arcsin(numpy.sqrt(a))
    km = 6367 * c
    return km

def load_volcanoes():
    url = 'https://volcanoes.usgs.gov/vsc/api/volcanoApi/geojson'
    with urllib.request.urlopen(url) as response:
        volcs = json.load(response)

    features = volcs['features']
    data = [
        {
            "lon": feature['geometry']['coordinates'][0],
            "lat": feature['geometry']['coordinates'][1],
            "name": feature['properties']['volcanoName'],
            "id": feature['properties']['volcanoCd'],
        }
        for feature in features
        if feature['properties']['volcanoCd']
    ]

    df = pandas.DataFrame(data)
    return df
