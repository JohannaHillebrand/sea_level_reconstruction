import cProfile
import pstats
import random
import tempfile
from datetime import datetime

import numpy as np
import xarray
from joblib import delayed, Parallel
from loguru import logger
from loky import set_loky_pickler
from sklearn.decomposition import PCA

from src import plotting, helper
from src.helper import prepare_tide_gauges
from src.settings.settings import GlobalSettings
from src.tide_gauge_station import TideGaugeStation


def reconstruct_cluster(cluster_id: int, lat_lon_pairs: list[tuple[float, float]], sea_level_data_array: np.ndarray,
                        tide_gauge_stations_for_cluster: dict[int, TideGaugeStation],
                        grid_points: list[tuple[int, int]],
                        output_path: str,
                        tide_gauge_to_lat_lon: dict[int, tuple[float, float]],
                        settings: GlobalSettings, min_lat, min_lon, resolution):
    """
    Reconstruct a cluster
    :param resolution:
    :param min_lon:
    :param min_lat:
    :param sea_level_data_array:
    :param tide_gauge_stations_for_cluster:
    :param settings:
    :param tide_gauge_to_lat_lon:
    :param output_path:
    :param grid_points:
    :param cluster_id:
    :param lat_lon_pairs:
    :return:
    """
    # create data array for pca
    data_array_for_pca, index_to_grid_point_id = create_data_array_for_pca(cluster_id, grid_points, lat_lon_pairs,
                                                                           sea_level_data_array, min_lat, min_lon,
                                                                           resolution)

    eof_data_array, eofs, explained_variance_ratio, pcs = perform_pca_on_cluster(data_array_for_pca,
                                                                                 index_to_grid_point_id,
                                                                                 settings.number_of_principal_components,
                                                                                 sea_level_data_array)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as temp_file:
        np.save(temp_file.name, eof_data_array)
        eof_data_array_path = temp_file.name

    # save to file
    with open(f"{output_path}/cluster_{cluster_id}_explained_variance_ratio.txt", "wb") as f:
        f.write("Explained variance ratio: ".encode("utf-8"))
        f.write(str(sum(explained_variance_ratio)).encode("utf-8"))

    # fit tide gauges in a least squares sense to the grid points in the eofs that they are closest to
    # for each point in time, take all availabe tide gauges and use the EOFs to approximate the principal components
    # with linear least squares regression

    all_dates = set()
    for tide_gauge_id in tide_gauge_stations_for_cluster.keys():
        for date in tide_gauge_stations_for_cluster[tide_gauge_id].timeseries.keys():
            if date.year < settings.cut_off_year_beginning:
                continue
            all_dates.add(date)
    all_dates_sorted = sorted(list(all_dates))
    # print(all_dates_sorted)
    # exit()
    # reconstruct the sea level data for each date using the tide gauges
    (mean_reconstruction_error_for_date, min_reconstruction_error_for_date, max_reconstruction_error_for_date,
     number_of_testing_tide_gauges_for_date, number_of_training_tide_gauges_for_date) = (
        reconstruct_sla_for_date(all_dates_sorted, tide_gauge_stations_for_cluster, tide_gauge_to_lat_lon, settings,
                                 eof_data_array_path, min_lat, min_lon, resolution))

    # plot reconstruction error for cluster over time
    plotting.plot_reconstruction_error_over_time(mean_reconstruction_error_for_date, max_reconstruction_error_for_date,
                                                 min_reconstruction_error_for_date, cluster_id, settings.output_path,
                                                 number_of_testing_tide_gauges_for_date,
                                                 number_of_training_tide_gauges_for_date, sum(explained_variance_ratio))
    return mean_reconstruction_error_for_date, min_reconstruction_error_for_date, max_reconstruction_error_for_date


def perform_pca_on_cluster(data_array_for_pca: np.array,
                           index_to_grid_point_id: dict[int, tuple[float, float]], number_of_principal_components: int,
                           sea_level_data_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on the data array for the given cluster
    :param sea_level_data_array:
    :param data_array_for_pca:
    :param index_to_grid_point_id:
    :param number_of_principal_components:
    :return:
    """
    time, indices_x, indices_y = sea_level_data_array.shape
    pca = PCA(n_components=number_of_principal_components)
    pcs = pca.fit_transform(data_array_for_pca)
    eofs = pca.components_
    explained_variance_ratio = pca.explained_variance_ratio_
    # logger.info(f"cluster_id: {cluster_id}")
    # logger.info(f"explained_variance_ratio: {sum(explained_variance_ratio)}")
    # create eof dataset and save to file
    eof_data_array = np.full((number_of_principal_components, indices_x, indices_y), np.nan)

    for i in range(number_of_principal_components):
        for j in index_to_grid_point_id.keys():
            eof_data_array[i, index_to_grid_point_id[j][0], index_to_grid_point_id[j][1]] = eofs[i, j]

    return eof_data_array, eofs, explained_variance_ratio, pcs


def create_data_array_for_pca(cluster_id: int, grid_points: list[tuple[int, int]],
                              lat_lon_pairs: list[tuple[float, float]],
                              sla_data_array: np.ndarray, min_lat, min_lon, resolution) -> tuple[
    np.ndarray, dict[int, tuple[int, int]]]:
    """
    Create a data array for PCA from the sea level data
    The data array has the shape (time, number of grid points)
    :param resolution:
    :param min_lon:
    :param min_lat:
    :param sla_data_array:
    :param cluster_id:
    :param grid_points:
    :param lat_lon_pairs:
    :return:
    """
    time, indices_x, indices_y = sla_data_array.shape
    data_array_for_pca = np.zeros((time, len(lat_lon_pairs)))  # shape time, grid points
    # weight each grid point by the cosine of the latitude
    index_to_grid_point_id = {}
    nan_counter = 0
    for counter, (idx, idy) in enumerate(grid_points):
        lat, lon = helper.index_to_lat_lon(idx, idy, min_lat, min_lon, resolution)
        weight = np.cos(np.deg2rad(lat))  # weight by the cosine of the latitude
        time_series = sla_data_array[:, idx, idy] * weight
        if np.isnan(time_series).any():
            nan_counter += 1
            continue
        data_array_for_pca[:, counter] = time_series
        index_to_grid_point_id[counter] = (idx, idy)
    
    if np.isnan(data_array_for_pca).any():
        logger.warning(f"NaN in data array for cluster {cluster_id}: {counter}")
        logger.warning("Removing column with NaN values from data array for PCA, but this should be handled before")
        exit(1)

    return data_array_for_pca, index_to_grid_point_id


def reconstruct_sla_for_date(all_dates_sorted: list[datetime],
                             current_tide_gauge_stations: dict[int, TideGaugeStation],
                             tide_gauge_to_lat_lon: dict[int, tuple[float, float]],
                             settings: GlobalSettings, eof_data_array_path: str, min_lat: float, min_lon: float,
                             resolution: float):
    """
    Reconstruct the sea level anomaly for each date using the tide gauges
    for each date, perform reconstruction a certain number of times with different, randomly selected tide gauges
    the number of iterations as well as the split are defined in the settings
    :param current_tide_gauge_stations:
    :param eof_data_array:
    :param settings:
    :param all_dates_sorted:
    :param number_of_principal_components:
    :param tide_gauge_to_lat_lon:
    :return:
    """

    tasks = [[date, current_tide_gauge_stations, settings, tide_gauge_to_lat_lon, eof_data_array_path, min_lat, min_lon,
              resolution] for date in
             all_dates_sorted]
    set_loky_pickler('pickle')
    results = Parallel(n_jobs=-2, verbose=1)(delayed(process_date)(*task) for task in tasks)

    min_reconstruction_error_for_date = {}
    mean_reconstruction_error_for_date = {}
    max_reconstruction_error_for_date = {}
    number_of_training_tide_gauges_for_date = {}
    number_of_testing_tide_gauges_for_date = {}
    counter = 0
    for (date, mean_, max_, min_, no_testing_tide_gauges, no_training_tide_gauges) in results:
        # print(f"{date}, {mean_}, {max_}, {min_}, {no_testing_tide_gauges}, {no_training_tide_gauges}")
        if date is None:
            counter = +1
            continue
        mean_reconstruction_error_for_date[date] = mean_
        min_reconstruction_error_for_date[date] = min_
        max_reconstruction_error_for_date[date] = max_
        number_of_testing_tide_gauges_for_date[date] = no_testing_tide_gauges
        number_of_training_tide_gauges_for_date[date] = no_training_tide_gauges
    if counter > 0:
        logger.warning(f"Reconstruction did not work for {counter} dates")
    return (mean_reconstruction_error_for_date, min_reconstruction_error_for_date, max_reconstruction_error_for_date,
            number_of_testing_tide_gauges_for_date, number_of_training_tide_gauges_for_date)


def process_date(date: datetime, current_tide_gauge_stations: dict[int, TideGaugeStation], settings: GlobalSettings,
                 tide_gauge_to_lat_lon: dict[int, tuple[float, float]], eof_data_array_path: str, lat_min, lon_min,
                 resolution):
    """
    Process a single date for reconstruction.
    :return:
    """
    eof_data_array = np.load(eof_data_array_path, mmap_mode='r')
    valid_tide_gauges_for_current_date = []
    # check if the tide gauges have valid data for this date
    for tide_gauge_id, current_tide_gauge_station in current_tide_gauge_stations.items():
        if date in current_tide_gauge_station.timeseries.keys():
            if current_tide_gauge_station.timeseries[date] != -99999:
                valid_tide_gauges_for_current_date.append(tide_gauge_id)
    # of the tide gauges that have valid data, select 90% of the data for training and 10% for testing
    number_of_tide_gauges_for_training = int(len(valid_tide_gauges_for_current_date) * 0.9)
    if ((
            len(valid_tide_gauges_for_current_date) - number_of_tide_gauges_for_training) <=
            settings.baseline_number_of_tide_gauges_for_testing):
        number_of_tide_gauges_for_training = len(
            valid_tide_gauges_for_current_date) - settings.baseline_number_of_tide_gauges_for_testing
    if number_of_tide_gauges_for_training < settings.number_of_principal_components:
        return date, None, None, None, None, None
    min_reconstruction_error_for_date = float('inf')
    max_reconstruction_error_for_date = 0
    mean_reconstruction_error_for_date = 0
    number_of_training_tide_gauges_for_date = number_of_tide_gauges_for_training
    number_of_testing_tide_gauges_for_date = len(
        valid_tide_gauges_for_current_date) - number_of_tide_gauges_for_training
    for _ in range(settings.reconstruction_iterations):
        training_tide_gauges_for_current_date = random.sample(valid_tide_gauges_for_current_date,
                                                              number_of_tide_gauges_for_training)
        testing_tide_gauges_for_current_date = [tide_gauge for tide_gauge in valid_tide_gauges_for_current_date if
                                                tide_gauge not in training_tide_gauges_for_current_date]
        tide_gauge_value_for_reconstruction = np.empty(
            (len(training_tide_gauges_for_current_date), 1))  # or just .reshape(-1, 1) later

        for i, station_id in enumerate(training_tide_gauges_for_current_date):
            current_tide_gauge_station = current_tide_gauge_stations[station_id]
            value = current_tide_gauge_station.timeseries_corrected_reference_datum[date]
            tide_gauge_value_for_reconstruction[i] = value

        # get the principal components for the training data
        # reduced_eofs = np.array(
        #     np.zeros((len(tide_gauge_value_for_reconstruction), settings.number_of_principal_components)))
        reduced_eofs = np.zeros((len(tide_gauge_value_for_reconstruction), settings.number_of_principal_components))

        for i, tide_gauge_id in enumerate(training_tide_gauges_for_current_date):
            current_tide_gauge_station = current_tide_gauge_stations[tide_gauge_id]
            lat, lon = tide_gauge_to_lat_lon[tide_gauge_id]
            if current_tide_gauge_station.closest_grid_point is None:
                current_tide_gauge_station.closest_grid_point = helper.lat_lon_to_grid_point_id(lat, lon, lat_min,
                                                                                                lon_min, resolution)
            idx, idy = current_tide_gauge_stations[tide_gauge_id].closest_grid_point
            try:
                reduced_eofs[i] = eof_data_array[:settings.number_of_principal_components, idx, idy]
                # reduced_eofs[i, :] = eof_dataset.eof.loc[
                #                          dict(latitude=lat, longitude=lon)
                #                      ][:number_of_principal_components].values
            except KeyError:
                logger.warning(
                    f"{lat}, {lon} not in eof dataset for date {date}, skipping tide gauge {tide_gauge_id}")
        # assume error for the tide gauge data
        sigma_squared = 1.0
        error_covariance_matrix = sigma_squared * np.eye(len(tide_gauge_value_for_reconstruction))
        inverse_error_covariance_matrix = np.linalg.inv(error_covariance_matrix)

        # weigth tide gauges with inverse error covariance matrix
        weighted_tide_gauges_for_current_date = np.dot(inverse_error_covariance_matrix,
                                                       tide_gauge_value_for_reconstruction)

        # least squares regression to estimate the PCs
        coefficients, residuals, rank, singular_values = np.linalg.lstsq(reduced_eofs,
                                                                         weighted_tide_gauges_for_current_date,
                                                                         rcond=None)
        estimated_pc_for_date = coefficients
        # reconstruct sea level for current date
        # _, lat_size, lon_size = eof_data_array.shape
        # reconstructed_data_array = np.zeros((lat_size, lon_size))  # shape (latitude, longitude)
        # for i in range(settings.number_of_principal_components):
        #     current_eof = np.array(eof_data_array[i, :, :])  # shape (latitude, longitude)
        #     current_alpha = coefficients[i]
        #     current_h_r = current_eof * current_alpha
        #     reconstructed_data_array = np.sum([reconstructed_data_array, current_h_r], axis=0)

        reconstructed_data_array = np.tensordot(coefficients.flatten(),
                                                eof_data_array[:settings.number_of_principal_components], axes=(0, 0))

        # for each testing tide gauge, take the value for the current date and check how close the value is to the
        # reconstructed sla at the closest lat-lon-point
        reconstruction_error = 0
        for testing_station_id in testing_tide_gauges_for_current_date:
            testing_station = current_tide_gauge_stations[testing_station_id]
            if testing_station.closest_grid_point is None:
                lat, lon = tide_gauge_to_lat_lon[testing_station_id]
                testing_station.closest_grid_point = helper.lat_lon_to_grid_point_id(lat, lon, lat_min, lon_min,
                                                                                     resolution)
            idx, idy = testing_station.closest_grid_point
            value_reconstructed = reconstructed_data_array[idx, idy]
            current_testing_station = current_tide_gauge_stations[testing_station_id]
            value_testing_station = current_testing_station.timeseries_corrected_reference_datum[date]
            difference = abs(value_testing_station - value_reconstructed)
            reconstruction_error += difference
        reconstruction_error /= len(testing_tide_gauges_for_current_date)
        if reconstruction_error < min_reconstruction_error_for_date:
            min_reconstruction_error_for_date = reconstruction_error
        if reconstruction_error > max_reconstruction_error_for_date:
            max_reconstruction_error_for_date = reconstruction_error
        mean_reconstruction_error_for_date += reconstruction_error
    # calculate mean reconstruction error for the date
    mean_reconstruction_error_for_date /= settings.reconstruction_iterations

    return (date, mean_reconstruction_error_for_date, max_reconstruction_error_for_date,
            min_reconstruction_error_for_date,
            number_of_testing_tide_gauges_for_date, number_of_training_tide_gauges_for_date)


def reconstruct_cluster_wrapper(args):
    return reconstruct_cluster(*args)


def start_reconstruction(sea_level_data: xarray.Dataset,
                         cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                         cluster_id_to_grid_point_id: dict[int, list[tuple[float, float]]],
                         tide_gauge_data: dict[int, TideGaugeStation], global_settings: GlobalSettings):
    """
    Start the reconstruction process for all clusters.
    :param sea_level_data:
    :param cluster_id_to_lat_lon_pairs:
    :param cluster_id_to_grid_point_id:
    :param tide_gauge_data:
    :param global_settings:
    :return:
    """
    profiler = cProfile.Profile()
    profiler.enable()

    logger.info("Assigning tide gauge stations to clusters")
    tide_gauge_data_corrected, cluster_id_to_tide_gauge, tide_gauge_to_lat_lon = prepare_tide_gauges(
        cluster_id_to_lat_lon_pairs,
        global_settings, sea_level_data,
        tide_gauge_data)

    logger.info("Starting reconstruction")

    # Build task list
    sea_level_data_array = sea_level_data.sla.values
    min_lat = sea_level_data.latitude.min().item()
    min_lon = sea_level_data.longitude.min().item()
    resolution = sea_level_data.latitude.values[1] - sea_level_data.latitude.values[0]
    print(type(resolution))
    tasks = []
    for cluster_id in cluster_id_to_tide_gauge:
        if len(cluster_id_to_tide_gauge[
                   cluster_id]) > global_settings.lower_bound_for_number_of_tide_gauges_per_cluster:
            tasks.append((
                cluster_id,
                cluster_id_to_lat_lon_pairs[cluster_id],
                sea_level_data_array,
                {tide_gauge_id: tide_gauge_data_corrected[tide_gauge_id] for tide_gauge_id in
                 cluster_id_to_tide_gauge[cluster_id]},
                cluster_id_to_grid_point_id[cluster_id],
                global_settings.output_path,
                tide_gauge_to_lat_lon,
                global_settings,
                min_lat, min_lon, resolution
            ))
        else:
            logger.info(
                f"Skipping cluster {cluster_id}, only {len(cluster_id_to_tide_gauge[cluster_id])} tide gauges")
        print(f"number of tide gauges for {cluster_id}: {len(cluster_id_to_tide_gauge[cluster_id])}")

    results = []
    for task in tasks:
        result = reconstruct_cluster(*task)
        results.append(result)
    # # Run in parallel
    # num_workers = min(multiprocessing.cpu_count(), 4)  # Tune this based on memory
    # results = []
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [
    #         executor.submit(reconstruct_cluster_wrapper, task)
    #         for task in tasks
    #     ]
    #
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Reconstructing clusters"):
    #         result = future.result()
    #         results.append(result)

    # Aggregate reconstruction errors
    mean_reconstruction_error_for_date_all_clusters = {}
    min_reconstruction_error_for_date_all_clusters = {}
    max_reconstruction_error_for_date_all_clusters = {}
    reconstructed_cluster_counter_for_date = {}

    for result in results:
        if result is None:
            continue
        mean_reconstruction_error_for_date, min_reconstruction_error_for_date, max_reconstruction_error_for_date = (
            result)
        for date in mean_reconstruction_error_for_date:
            if date not in mean_reconstruction_error_for_date_all_clusters:
                mean_reconstruction_error_for_date_all_clusters[date] = 0
                min_reconstruction_error_for_date_all_clusters[date] = float('inf')
                max_reconstruction_error_for_date_all_clusters[date] = 0
                reconstructed_cluster_counter_for_date[date] = 0
            mean_reconstruction_error_for_date_all_clusters[date] += mean_reconstruction_error_for_date[date]
            reconstructed_cluster_counter_for_date[date] += 1
            if min_reconstruction_error_for_date[date] < min_reconstruction_error_for_date_all_clusters[date]:
                min_reconstruction_error_for_date_all_clusters[date] = min_reconstruction_error_for_date[date]
            if max_reconstruction_error_for_date[date] > max_reconstruction_error_for_date_all_clusters[date]:
                max_reconstruction_error_for_date_all_clusters[date] = max_reconstruction_error_for_date[date]

    for date in mean_reconstruction_error_for_date_all_clusters:
        mean_reconstruction_error_for_date_all_clusters[date] /= reconstructed_cluster_counter_for_date[date]

    profiler.disable()
    # Print results sorted by time
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(30)  # print top 30 lines
    return mean_reconstruction_error_for_date_all_clusters, min_reconstruction_error_for_date_all_clusters, \
        max_reconstruction_error_for_date_all_clusters
