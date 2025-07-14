#  for each cluster select tide gauges
#  perform pca on the satellite sea level data
# TODO: fit tide gauge data to eof
# TODO: reconstruct sea level data throughout history
# TODO: plot sea level data
# TODO: compare reconstructed sea level data with observed sea level data
# TODO: compare reconstructed sea level data with regional sea level data that was not used for the reconstruction
# TODO: check if there are steps in between the different clusters
# TODO: do reconstruction on unclustered data
# TODO: compare
import os

from src import reconstruction_per_cluster
from src.preprocessing import read_data
from src.settings.settings import GlobalSettings


def main():
    global_settings = GlobalSettings()
    if not os.path.exists(global_settings.output_path):
        os.makedirs(global_settings.output_path)
    sea_level_data, cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id, tide_gauge_data = read_data(
        global_settings)

    reconstruction_per_cluster.start_reconstruction(sea_level_data, cluster_id_to_lat_lon_pairs,
                                                    cluster_id_to_grid_point_id, tide_gauge_data,
                                                    global_settings.timeframe, global_settings)

    # calculate global mean sea level, by calculating the mean of the tide gauges in a given cluster, then weight the
    # resulting time series by the area of the cluster, check if the result is close to the global mean sea level as
    # calculated from the satellite data


if __name__ == '__main__':
    main()
