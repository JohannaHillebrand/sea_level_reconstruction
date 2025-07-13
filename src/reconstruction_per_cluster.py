import os
import pickle
import random
import time

import numpy as np
import xarray
from haversine import haversine
from loguru import logger
from sklearn.decomposition import PCA

from src import plotting
from src.settings import GlobalSettings
from src.tide_gauge_station import TideGaugeStation


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
        f"number of tide gauge stations assigned to clusters: {sum(len(tide_gauge_ids) for cluster_id, tide_gauge_ids in cluster_id_to_tide_gauge.items())}")
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


def reconstruct_cluster(cluster_id: int, lat_lon_pairs: list[tuple[float, float]], sea_level_data: xarray.Dataset,
                        tide_gauge_stations: dict[int, TideGaugeStation],
                        cluster_id_to_tide_gauge: dict[int, list[int]], timeframe: tuple[int, int],
                        grid_points: list[tuple[float, float]], output_path: str,
                        tide_gauge_to_lat_lon: dict[int, tuple[float, float]], number_of_principal_components: int):
    """
    Reconstruct a cluster
    :param number_of_principal_components:
    :param tide_gauge_to_lat_lon:
    :param output_path:
    :param grid_points:
    :param cluster_id:
    :param lat_lon_pairs:
    :param sea_level_data:
    :param tide_gauge_stations:
    :param cluster_id_to_tide_gauge:
    :param timeframe:
    :return:
    """
    # create data array for pca
    data_array_for_pca = np.zeros((len(sea_level_data["time"].values), len(lat_lon_pairs)))
    # TODO: weighting of sla data
    sla_data_array = sea_level_data['sla'].values  # shape time, lat, lon
    print(sla_data_array.shape)
    print(data_array_for_pca.shape)
    index_to_grid_point_id = {}
    counter = 0
    nan_counter = 0
    for counter, (idx, idy) in enumerate(grid_points):
        data_array_for_pca[:, counter] = sla_data_array[:, idx, idy]
        index_to_grid_point_id[counter] = (idx, idy)
        if np.isnan(data_array_for_pca[:, counter]).any():
            nan_counter += 1
    print(f"nan_counter: {nan_counter}")
    print(f"not nans: {len(data_array_for_pca) - nan_counter}")

    pca = PCA(n_components=number_of_principal_components)
    pcs = pca.fit_transform(data_array_for_pca)
    eofs = pca.components_
    print(f"eofs shape: {eofs.shape}")
    print(f"pcs shape: {pcs.shape}")
    explained_variance_ratio = pca.explained_variance_ratio_
    logger.info(f"cluster_id: {cluster_id}")
    logger.info(f"explained_variance_ratio: {sum(explained_variance_ratio)}")
    # create eof dataset and save to file
    eof_data_array = np.full((number_of_principal_components, len(sea_level_data["latitude"].values),
                              len(sea_level_data["longitude"].values)), np.nan)
    print(f"eof_data_array shape: {eof_data_array.shape}")
    for i in range(number_of_principal_components):
        for j in index_to_grid_point_id.keys():
            eof_data_array[i, index_to_grid_point_id[j][0], index_to_grid_point_id[j][1]] = eofs[i, j]

    eof_dataset = xarray.Dataset(
        data_vars={
            "eof": (("components", "latitude", "longitude"), eof_data_array)
        },
        coords={
            "latitude": sea_level_data["latitude"].values,
            "longitude": sea_level_data["longitude"].values,
            "components": range(number_of_principal_components)})

    # logger.info("plotting eofs")
    # for i in range(number_of_principal_components):
    #     data = eof_dataset.eof.isel(components=i)
    #     fig = plt.figure(figsize=(15, 10))
    #     ax = plt.axes(projection=ccrs.PlateCarree())
    #     data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap="jet", add_colorbar=True)
    #     ax.coastlines()
    #     ax.gridlines(draw_labels=True)
    #     plt.savefig(f"{output_path}/cluster_{cluster_id}_eof_{i}.pdf", dpi=500)
    #     plt.close(fig)

    # save to file
    with open(f"{output_path}/cluster_{cluster_id}_explained_variance_ratio.txt", "wb") as f:
        f.write("Explained variance ratio: ".encode("utf-8"))
        f.write(str(sum(explained_variance_ratio)).encode("utf-8"))

    # fit tide gauges in a least squares sense to the grid points in the eofs that they are closest to
    # for each point in time, take all availabe tide gauges and use the EOFs to approximate the principal components
    # with linear least squares regression
    if cluster_id not in cluster_id_to_tide_gauge:
        return
    current_tide_gauge_ids = cluster_id_to_tide_gauge[cluster_id]
    all_dates = set()
    for tide_gauge_id in current_tide_gauge_ids:
        for date in tide_gauge_stations[tide_gauge_id].timeseries.keys():
            all_dates.add(date)
    all_dates_sorted = sorted(list(all_dates))
    for date in all_dates_sorted:
        valid_tide_gauges_for_current_date = []
        testing_tide_gauges_for_current_date = []
        training_tide_gauges_for_current_date = []
        # check if the tide gauges have valid data for this date
        for tide_gauge_id in current_tide_gauge_ids:
            current_tide_gauge_station = tide_gauge_stations[tide_gauge_id]
            if date in current_tide_gauge_station.timeseries.keys():
                if current_tide_gauge_station.timeseries[date] != -99999:
                    valid_tide_gauges_for_current_date.append(tide_gauge_id)
        # of the tide gauges that have valid data, select 90% of the data for training and 10% for testing
        number_of_tide_gauges_for_training = int(len(valid_tide_gauges_for_current_date) * 0.9)
        training_tide_gauges_for_current_date = random.sample(valid_tide_gauges_for_current_date,
                                                              number_of_tide_gauges_for_training)
        testing_tide_gauges_for_current_date = [tide_gauge for tide_gauge in valid_tide_gauges_for_current_date if
                                                tide_gauge not in training_tide_gauges_for_current_date]
        if len(training_tide_gauges_for_current_date) <= number_of_principal_components:
            continue
        print(f"date: {date}")
        print(f"training : {len(training_tide_gauges_for_current_date)}")
        print(f"testing: {len(testing_tide_gauges_for_current_date)}")

        # get the principal components for the training data
        reduced_eofs = np.array(np.zeros((len(training_tide_gauges_for_current_date), number_of_principal_components)))
        for i, tide_gauge_id in enumerate(current_tide_gauge_ids):
            lat, lon = tide_gauge_to_lat_lon[tide_gauge_id]
    pass


def correct_reference_datum(tide_gauge_data, sea_level_data, tide_gauge_to_lat_lon, output_path):
    """
    Correct the reference datum of the tide gauges by fitting it to the satellite sea level data
    :param tide_gauge_data:
    :param sea_level_data:
    :param tide_gauge_to_lat_lon:
    :param output_path:
    :return:
    """
    if not os.path.exists(f"{output_path}/tide_gauge_data_corrected_reference_datum.pkl"):
        for tide_gauge_id, tide_gauge_station in tide_gauge_data.items():
            lat, lon = tide_gauge_to_lat_lon[tide_gauge_id]
            time_series_of_closest_grid_point = sea_level_data.sel(latitude=lat, longitude=lon, method="nearest")["sla"]
            tide_gauge_station.correct_reference_datum(time_series_of_closest_grid_point.values,
                                                       time_series_of_closest_grid_point.time.values)
        # save to file
        with open(f"{output_path}/tide_gauge_data_corrected_reference_datum.pkl", "wb") as f:
            pickle.dump(tide_gauge_data, f)
    else:
        with open(f"{output_path}/tide_gauge_data_corrected_reference_datum.pkl", "rb") as f:
            tide_gauge_data = pickle.load(f)
    return tide_gauge_data


def start_reconstruction(sea_level_data: xarray.Dataset,
                         cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                         cluster_id_to_grid_point_id: dict[int, list[tuple[float, float]]],
                         tide_gauge_data: dict[int, TideGaugeStation], timeframe: tuple[int, int],
                         global_settings: GlobalSettings):
    """
    Start the reconstruction
    :param global_settings:
    :param timeframe:
    :param cluster_id_to_grid_point_id:
    :param cluster_id_to_lat_lon_pairs:
    :param sea_level_data:
    :param tide_gauge_data:
    :return:
    """
    logger.info("Assigning tide gauge stations to clusters")
    cluster_id_to_tide_gauge, tide_gauge_to_lat_lon = assign_tide_gauge_stations_to_cluster(cluster_id_to_lat_lon_pairs,
                                                                                            tide_gauge_data,
                                                                                            global_settings.clustering_data_path)
    plotting.plot_clustering_with_tide_gauges(cluster_id_to_lat_lon_pairs, global_settings.output_path,
                                              "clusters_with_tide_gauges",
                                              sea_level_data["latitude"].values[1] - sea_level_data["latitude"].values[
                                                  0], cluster_id_to_tide_gauge, tide_gauge_data)

    tide_gauge_data_filtered = filter_tide_gauge_stations(tide_gauge_data, timeframe, cluster_id_to_tide_gauge)
    plotting.plot_clustering_with_tide_gauges(cluster_id_to_lat_lon_pairs, global_settings.output_path,
                                              f"clusters_with_tide_gauges_filtered",
                                              sea_level_data["latitude"].values[1] -
                                              sea_level_data["latitude"].values[
                                                  0], cluster_id_to_tide_gauge, tide_gauge_data_filtered)

    # correct the reference datum of the tide gauge data
    tide_gauge_data_corrected = correct_reference_datum(tide_gauge_data_filtered, sea_level_data, tide_gauge_to_lat_lon,
                                                        global_settings.output_path)
    # for each cluster, perform reconstruction
    logger.info("Starting reconstruction")
    for cluster_id, lat_lon_pairs in cluster_id_to_lat_lon_pairs.items():
        # select suitable tide gauge stations -> later
        # perform PCA on the sea level data
        reconstruct_cluster(cluster_id, lat_lon_pairs, sea_level_data, tide_gauge_data_corrected,
                            cluster_id_to_tide_gauge, timeframe, cluster_id_to_grid_point_id[cluster_id],
                            global_settings.output_path, tide_gauge_to_lat_lon,
                            global_settings.number_of_principal_components)
        # fit the tide gauge data to the eof
        pass
    return None
