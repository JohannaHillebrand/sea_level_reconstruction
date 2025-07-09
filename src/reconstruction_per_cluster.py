import numpy as np
import xarray

from src.tide_gauge_station import TideGaugeStation


def extract_clusters_from_xarray_dataset(clustering_data: xarray.Dataset):
    """
    Extract the clusters
    :param clustering_data:
    :return:
    """
    clusters = {}
    cluster_data = clustering_data["__xarray_dataarray_variable__"].values
    unique_clusters = np.unique(cluster_data)
    unique_clusters = unique_clusters[~np.isnan(unique_clusters)]
    extended_lons, extended_lats = np.meshgrid(clustering_data['longitude'].values, clustering_data['latitude'].values)
    print(extended_lats.shape)
    print(extended_lons.shape)

    cluster_id_to_lat_lon_pairs = {}
    cluster_id_to_grid_point_id = {}
    for cluster_id in unique_clusters:
        if np.isnan(cluster_id):
            continue
        # find lat/lon pairs that belong to the cluster
        current_cluster_mask = cluster_data == cluster_id
        lat_lon_pairs = list(zip(extended_lats[current_cluster_mask], extended_lons[current_cluster_mask]))
        cluster_id_to_lat_lon_pairs[cluster_id] = lat_lon_pairs
        # find the grid point ids that belong to the cluster
        grid_point_ids = np.where(current_cluster_mask)[0]
        print(grid_point_ids)
        print(lat_lon_pairs)
        exit()
    return clusters


def start_reconstruction(sea_level_data: xarray.Dataset, clustering_data: xarray.Dataset,
                         tide_gauge_data: dict[int, TideGaugeStation]):
    """
    Start the reconstruction
    :param sea_level_data:
    :param clustering_data:
    :param tide_gauge_data:
    :return:
    """
    clusters = extract_clusters_from_xarray_dataset(clustering_data)
    return None
