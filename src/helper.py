import os
import pickle
import time

import xarray
from haversine import haversine
from loguru import logger

from src import plotting
from src.settings.settings import GlobalSettings
from src.tide_gauge_station import TideGaugeStation


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


def assign_tide_gauge_stations_to_cluster(cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                                          tide_gauge_data: dict[int, TideGaugeStation], clustering_path: str):
    """
    Assign tide gauge stations to clusters
    :param clustering_path:
    :param cluster_id_to_lat_lon_pairs:
    :param tide_gauge_data:
    :return:
    """
    if not os.path.exists(f"{clustering_path}/cluster_id_to_tide_gauge.pkl") or not os.path.exists(
            f"{clustering_path}/tide_gauge_to_lat_lon.pkl"):
        cluster_id_to_tide_gauge = {}
        tide_gauge_to_lat_lon = {}
        # parallelizing this does not improve performance
        time1 = time.time()

        for tide_gauge_id, tide_gauge_station in tide_gauge_data.items():
            closest_cluster_id, closest_lat_lon = find_closest_cluster_for_tide_gauge(tide_gauge_station,
                                                                                      cluster_id_to_lat_lon_pairs)
            try:
                cluster_id_to_tide_gauge[closest_cluster_id].append(tide_gauge_id)
            except KeyError:
                cluster_id_to_tide_gauge[closest_cluster_id] = [tide_gauge_id]
            tide_gauge_to_lat_lon[tide_gauge_id] = closest_lat_lon
        logger.info(f"time taken for tide gauge assignment: {time.time() - time1}")

        pickle.dump(cluster_id_to_tide_gauge, open(f"{clustering_path}/cluster_id_to_tide_gauge.pkl", "wb"))
        pickle.dump(tide_gauge_to_lat_lon, open(f"{clustering_path}/tide_gauge_to_lat_lon.pkl", "wb"))
    else:
        cluster_id_to_tide_gauge = pickle.load(open(f"{clustering_path}/cluster_id_to_tide_gauge.pkl", "rb"))
        tide_gauge_to_lat_lon = pickle.load(open(f"{clustering_path}/tide_gauge_to_lat_lon.pkl", "rb"))
    logger.info(
        f"number of tide gauge stations assigned to clusters: "
        f"{sum(len(tide_gauge_ids) for cluster_id, tide_gauge_ids in cluster_id_to_tide_gauge.items())}")
    return cluster_id_to_tide_gauge, tide_gauge_to_lat_lon


def find_closest_cluster_for_tide_gauge(tide_gauge_station, cluster_id_to_lat_lon_pairs):
    """
    Find the closest cluster for a tide gauge
    :param tide_gauge_station:
    :param cluster_id_to_lat_lon_pairs:
    :return:
    """
    # find cluster that has the closest lat/lon to the tide gauge
    closest_cluster_id = 0
    closest_distance = float('inf')
    closest_lat_lon = (None, None)
    for cluster_id, lat_lon_pairs in cluster_id_to_lat_lon_pairs.items():
        for lat, lon in lat_lon_pairs:
            distance = haversine((tide_gauge_station.latitude, tide_gauge_station.longitude),
                                 (lat, lon))
            if distance < closest_distance:
                closest_distance = distance
                closest_cluster_id = cluster_id
                closest_lat_lon = (lat, lon)
    if closest_distance > 200:
        closest_cluster_id = -99999
    return closest_cluster_id, closest_lat_lon


def filter_tide_gauge_stations(tide_gauge_data: dict[int, TideGaugeStation], timeframe: tuple[int, int],
                               cluster_id_to_tide_gauge: dict[int, list[int]]):
    """
    Select suitable tide gauge stations for reconstruction
    The tide gauges that are chosen should have at least one point in time in comon with the satellite data
    :param cluster_id_to_tide_gauge:
    :param timeframe:
    :param tide_gauge_data:
    :return:
    """
    # check if the tide gauge has data for the whole timeframe
    logger.info(f"Number of tide gauges before checking: {len(tide_gauge_data)}")
    invalid_tide_gauge_ids = []
    for tide_gauge_id, tide_gauge_station in tide_gauge_data.items():
        if not is_tide_gauge_in_timeframe(tide_gauge_station, timeframe):
            invalid_tide_gauge_ids.append(tide_gauge_id)

    # These are tide gauges that are not in any cluster
    invalid_tide_gauge_ids.extend(list(cluster_id_to_tide_gauge[-99999]))

    invalid_tide_gauge_ids = set(invalid_tide_gauge_ids)
    for invalid_tide_gauge_id in invalid_tide_gauge_ids:
        tide_gauge_data.pop(invalid_tide_gauge_id)
        for cluster_id in cluster_id_to_tide_gauge.keys():
            if invalid_tide_gauge_id in cluster_id_to_tide_gauge[cluster_id]:
                cluster_id_to_tide_gauge[cluster_id].remove(invalid_tide_gauge_id)
    cluster_id_to_tide_gauge.pop(-99999)

    logger.info(f"Number of tide gauges after checking: {len(tide_gauge_data)}")
    return tide_gauge_data


def lat_lon_to_grid_point_id(lat, lon, min_lat, min_lon, resolution):
    """
    Convert latitude and longitude to grid point id
    :param lat:
    :param lon:
    :param min_lat:
    :param min_lon:
    :param resolution:
    :return:
    """
    id_x = int((lat - min_lat) / resolution)
    id_y = int((lon - min_lon) / resolution)
    return id_x, id_y


def correct_reference_datum_for_all_tide_gauges(tide_gauge_data, sea_level_data, tide_gauge_to_lat_lon, data_path: str):
    """
    Correct the reference datum of the tide gauges by fitting it to the satellite sea level data
    :param tide_gauge_data:
    :param sea_level_data:
    :param tide_gauge_to_lat_lon:
    :param data_path:
    :return:
    """
    if not os.path.exists(f"{data_path}/tide_gauge_data_corrected_reference_datum.pkl"):
        for tide_gauge_id, tide_gauge_station in tide_gauge_data.items():
            lat, lon = tide_gauge_to_lat_lon[tide_gauge_id]
            time_series_of_closest_grid_point = sea_level_data.sel(latitude=lat, longitude=lon, method="nearest")["sla"]
            tide_gauge_station.correct_reference_datum(time_series_of_closest_grid_point.values,
                                                       time_series_of_closest_grid_point.time.values)
            # add the closest lat/lon and grid point to the tide gauge station
            tide_gauge_station.closest_lat_lon = (lat, lon)
            tide_gauge_station.closest_grid_point = lat_lon_to_grid_point_id(lat, lon,
                                                                             sea_level_data["latitude"].min(),
                                                                             sea_level_data["longitude"].min(),
                                                                             sea_level_data["latitude"].values[1] -
                                                                             sea_level_data["latitude"].values[0])
        # save to file
        with open(f"{data_path}/tide_gauge_data_corrected_reference_datum.pkl", "wb") as f:
            pickle.dump(tide_gauge_data, f)
    else:
        with open(f"{data_path}/tide_gauge_data_corrected_reference_datum.pkl", "rb") as f:
            tide_gauge_data = pickle.load(f)
    return tide_gauge_data


def prepare_tide_gauges(cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                        global_settings: GlobalSettings, sea_level_data: xarray.Dataset,
                        tide_gauge_data: dict[int, TideGaugeStation]) -> tuple[:dict[int, TideGaugeStation],
                                                                         dict[int, list[int]],
                                                                         dict[int, tuple[float, float]]]:
    """
    Prepare the tide gauge data for the global mean sea level calculation or the reconstruction.
    :param cluster_id_to_lat_lon_pairs:
    :param global_settings:
    :param sea_level_data:
    :param tide_gauge_data:
    :return:
    """
    cluster_id_to_tide_gauge, tide_gauge_to_lat_lon = assign_tide_gauge_stations_to_cluster(cluster_id_to_lat_lon_pairs,
                                                                                            tide_gauge_data,
                                                                                            global_settings.clustering_data_path)
    plotting.plot_clustering_with_tide_gauges(cluster_id_to_lat_lon_pairs, global_settings.output_path,
                                              "clusters_with_tide_gauges",
                                              sea_level_data["latitude"].values[1] - sea_level_data["latitude"].values[
                                                  0], cluster_id_to_tide_gauge, tide_gauge_data)
    tide_gauge_data_filtered = filter_tide_gauge_stations(tide_gauge_data, global_settings.timeframe,
                                                          cluster_id_to_tide_gauge)
    plotting.plot_clustering_with_tide_gauges(cluster_id_to_lat_lon_pairs, global_settings.output_path,
                                              f"clusters_with_tide_gauges_filtered",
                                              sea_level_data["latitude"].values[1] -
                                              sea_level_data["latitude"].values[
                                                  0], cluster_id_to_tide_gauge, tide_gauge_data_filtered)
    logger.info("Correcting reference datum for tide gauges")
    # correct the reference datum of the tide gauge data
    tide_gauge_data_corrected = correct_reference_datum_for_all_tide_gauges(tide_gauge_data_filtered, sea_level_data,
                                                                            tide_gauge_to_lat_lon,
                                                                            global_settings.clustering_data_path)
    return tide_gauge_data_corrected, cluster_id_to_tide_gauge, tide_gauge_to_lat_lon


def is_tide_gauge_in_timeframe(tide_gauge_station: TideGaugeStation, timeframe: tuple[int, int]) -> bool:
    """
    Check if the tide gauge has data for the whole timeframe
    :param tide_gauge_station:
    :param timeframe:
    :return:
    """
    valid = False
    current_timeseries = tide_gauge_station.timeseries
    # check if the tide gauge has one valid point within the timeframe
    set_years = set(date.year for date in list(current_timeseries.keys()))
    if not set_years:
        return False
    for i in range(timeframe[0], timeframe[1] + 1):
        if i in set_years:
            valid = True
    return valid
