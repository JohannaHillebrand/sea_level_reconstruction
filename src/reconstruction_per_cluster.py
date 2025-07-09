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
    cluster_id_to_tide_gauge = {}
    pass


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
    # for each cluster, perform reconstruction
    for cluster_id, grid_point_ids in cluster_id_to_grid_point_id.items():
        # first identify suitable tide gauge stations
        tide_gauge_stations = select_tide_gauge_stations(cluster_id, grid_point_ids, tide_gauge_data)
        # perform PCA on the sea level data
        # fit the tide gauge data to the eof
        pass
    return None
