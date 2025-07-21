import os
import pickle

import numpy as np
import xarray
from loguru import logger
from xarray import Dataset

from src import plotting
from src.settings.settings import GlobalSettings
from src.tide_gauge_station import read_and_create_stations, TideGaugeStation


def compute_nan_mask(dataset: xarray.Dataset, var: str) -> xarray.DataArray:
    """
    Compute a nan mask for the dataset
    :param dataset:
    :param var:
    :return:
    """
    nan_mask = dataset[var].isnull().any(dim="time")
    return nan_mask


def change_clustering_resolution(sea_level_data: xarray.Dataset, clustering_data: xarray.Dataset, clustering_path: str,
                                 output_path: str):
    """
    Change the clustering resolution to match the sea level data
    :param output_path:
    :param clustering_path:
    :param sea_level_data:
    :param clustering_data:
    :return:
    """
    if not os.path.exists(f"{clustering_path}/clustering_15_025_resolution.nc") or not os.path.exists(
            f"{clustering_path}/clustering_masked.nc"):
        logger.info("Regridding clustering data to match sea level data resolution")
        # interpolate the sea level data to the clustering resolution
        interpolated_sea_level_data = sea_level_data.interp(latitude=clustering_data['latitude'],
                                                            longitude=clustering_data['longitude'])
        # change every point in the clustering to nan, where the sea level data is nan in any time step
        nan_mask_da = compute_nan_mask(interpolated_sea_level_data, "sla")
        clustering_data = clustering_data.where(~nan_mask_da, np.nan)
        # save masked clustering
        clustering_data.to_netcdf(f"{clustering_path}/clustering_masked.nc")
        # regrid the clustering data to match the sea level data
        ref_grid = sea_level_data.isel(time=0)
        fine_resolution_clusters = clustering_data.interp_like(ref_grid, method="nearest")
        # save the fine resolution clusters
        fine_resolution_clusters.to_netcdf(f"{clustering_path}/clustering_15_025_resolution.nc")
        # put nan values where there are nan values in the sea level data
        # masking happens twice as the interpolation might leak into the nan areas
        nan_mask_da = compute_nan_mask(sea_level_data, "sla")
        fine_resolution_clusters = fine_resolution_clusters.where(~nan_mask_da, np.nan)
    else:
        fine_resolution_clusters = xarray.open_dataset(f"{clustering_path}/clustering_15_025_resolution.nc")
        clustering_data = xarray.open_dataset(f"{clustering_path}/clustering_masked.nc")

    # plot clustering before and after regridding
    plotting.plot_xarray_dataset_on_map(clustering_data, output_path, "clustering_before_regridding")
    plotting.plot_xarray_dataset_on_map(fine_resolution_clusters, output_path, "clustering_after_regridding")
    return fine_resolution_clusters


def read_data(global_settings: GlobalSettings) -> tuple[
    Dataset, dict[int, list[tuple[float, float]]], dict[int, list[tuple[float, float]]], dict[int, TideGaugeStation]]:
    """
    Read data from file
    :param global_settings:
    :return:
    """
    logger.info(f"Reading sea level data")
    sea_level_data = xarray.open_dataset(global_settings.sea_level_data_path)
    clusters = {}
    clustering_data = xarray.open_dataset(
        f"{global_settings.clustering_data_path}/clustering_{global_settings.number_of_clusters}.nc")
    # change clustering resolution to match the sea level data
    fine_resolution_clustering_data = change_clustering_resolution(sea_level_data, clustering_data,
                                                                   global_settings.clustering_data_path,
                                                                   global_settings.output_path)
    cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id = extract_clusters_from_xarray_dataset(
        fine_resolution_clustering_data, global_settings.clustering_data_path)
    # save the clustering data via pickle
    logger.info(f"Number of clusters: {len(cluster_id_to_lat_lon_pairs.keys())}")
    logger.info(f"Reading tide gauge data")
    tide_gauge_data = read_and_create_stations(global_settings.tide_gauge_data_folder,
                                               global_settings.cut_off_year_beginning)

    return sea_level_data, cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id, tide_gauge_data


def extract_clusters_from_xarray_dataset(clustering_data: xarray.Dataset, clustering_path):
    """
    Extract the clusters
    :param clustering_path:
    :param clustering_data:
    :return:
    """
    if not (os.path.exists(f"{clustering_path}/cluster_id_to_lat_lon_pairs.pkl") and os.path.exists(
            f"{clustering_path}/cluster_id_to_grid_point_id.pkl")):
        cluster_data = clustering_data.to_array().values[0]
        unique_clusters = np.unique(cluster_data)
        unique_clusters = unique_clusters[~np.isnan(unique_clusters)]
        extended_lons, extended_lats = np.meshgrid(clustering_data['longitude'].values,
                                                   clustering_data['latitude'].values)
        cluster_id_to_lat_lon_pairs = {}
        cluster_id_to_grid_point_id = {}
        for cluster_id in unique_clusters:
            # find lat/lon pairs that belong to the cluster
            current_cluster_mask = cluster_data == cluster_id
            lat_lon_pairs = list(zip(extended_lats[current_cluster_mask], extended_lons[current_cluster_mask]))
            cluster_id_to_lat_lon_pairs[cluster_id] = lat_lon_pairs
            # find the grid point ids that belong to the cluster
            grid_point_ids = []
            for lat, lon in lat_lon_pairs:
                x, y = lat_lon_to_index(lat, lon, clustering_data['latitude'].min(), clustering_data['longitude'].min(),
                                        clustering_data['latitude'].values[1] - clustering_data['latitude'].values[0])
                grid_point_id = (x, y)
                grid_point_ids.append(grid_point_id)
            cluster_id_to_grid_point_id[cluster_id] = grid_point_ids
        print(cluster_id_to_lat_lon_pairs.keys())
        # save the cluster data via pickle
        pickle.dump(cluster_id_to_lat_lon_pairs, open(f"{clustering_path}/cluster_id_to_lat_lon_pairs.pkl", "wb"))
        pickle.dump(cluster_id_to_grid_point_id, open(f"{clustering_path}/cluster_id_to_grid_point_id.pkl", "wb"))
    else:
        cluster_id_to_lat_lon_pairs = pickle.load(open(f"{clustering_path}/cluster_id_to_lat_lon_pairs.pkl", "rb"))
        cluster_id_to_grid_point_id = pickle.load(open(f"{clustering_path}/cluster_id_to_grid_point_id.pkl", "rb"))
    return cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id


def lat_lon_to_index(lat, lon, lat_min, lon_min, resolution) -> (int, int):
    """
    Convert a latitude and longitude to an index
    :param lat:
    :param lon:
    :param lat_min:
    :param lon_min:
    :param resolution:
    :return:
    """
    x = int((lat - lat_min) / resolution)
    y = int((lon - lon_min) / resolution)
    return x, y
