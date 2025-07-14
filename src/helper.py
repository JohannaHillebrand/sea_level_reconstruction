def index_to_lat_lon(x, y, lat_min, lon_min, resolution) -> (float, float):
    """
    Convert an index to a latitude and longitude
    :param x:
    :param y:
    :param lat_min:
    :param lon_min:
    :param resolution:
    :return:
    """
    lat = lat_min + x * resolution
    lon = lon_min + y * resolution
    return lat, lon
