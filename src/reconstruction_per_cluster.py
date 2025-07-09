import os
import pickle

import xarray

from src.tide_gauge_station import TideGaugeStation


def select_tide_gauge_stations(cluster_id: int, grid_point_ids: list[tuple[int, int]],
                               tide_gauge_data: dict[int, TideGaugeStation]):
    """
    Select suitable tide gauge stations for reconstruction
    :param cluster_id:
    :param grid_point_ids:
    :param tide_gauge_data:
    :return:
    """
    pass


def assign_tide_gauge_stations_to_cluster(cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                                          tide_gauge_data: dict[int, TideGaugeStation]):
    """
    Assign tide gauge stations to clusters
    :param cluster_id_to_lat_lon_pairs:
    :param tide_gauge_data:
    :return:
    """
    if not os.path.exists("../data/clustering/cluster_id_to_tide_gauge.pkl"):
        cluster_id_to_tide_gauge = {}
        # TODO: parallelize this
        for tide_gauge_id, tide_gauge_station in tide_gauge_data.items():
            # find cluster that has the closest lat/lon to the tide gauge
            closest_cluster_id = 0
            closest_distance = float('inf')
            closest_lat_lon_pair = (None, None)
            for cluster_id, lat_lon_pairs in cluster_id_to_lat_lon_pairs.items():
                for lat, lon in lat_lon_pairs:
                    distance = abs(lat - tide_gauge_station.latitude) + abs(lon - tide_gauge_station.longitude)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_cluster_id = cluster_id
                        closest_lat_lon_pair = (lat, lon)
            try:
                cluster_id_to_tide_gauge[closest_cluster_id].append(tide_gauge_id)
            except KeyError:
                cluster_id_to_tide_gauge[closest_cluster_id] = [tide_gauge_id]

        pickle.dump(cluster_id_to_tide_gauge, open("../data/clustering/cluster_id_to_tide_gauge.pkl", "wb"))
    else:
        cluster_id_to_tide_gauge = pickle.load(open("../data/clustering/cluster_id_to_tide_gauge.pkl", "rb"))
    return cluster_id_to_tide_gauge


def start_reconstruction(sea_level_data: xarray.Dataset,
                         cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                         cluster_id_to_grid_point_id: dict[int, tuple[int, int]],
                         tide_gauge_data: dict[int, TideGaugeStation]):
    """
    Start the reconstruction
    :param cluster_id_to_grid_point_id:
    :param cluster_id_to_lat_lon_pairs:
    :param sea_level_data:
    :param tide_gauge_data:
    :return:
    """
    sea_level_data_array = sea_level_data['sla'].values
    cluster_id_to_tide_gauge = assign_tide_gauge_stations_to_cluster(cluster_id_to_lat_lon_pairs, tide_gauge_data)
    # TODO: plot each cluster with its associated tide gauges 
    # for each cluster, perform reconstruction
    for cluster_id, grid_point_ids in cluster_id_to_grid_point_id.items():
        # first identify suitable tide gauge stations
        tide_gauge_stations = select_tide_gauge_stations(cluster_id, grid_point_ids, tide_gauge_data)
        # perform PCA on the sea level data
        # fit the tide gauge data to the eof
        pass
    return None
