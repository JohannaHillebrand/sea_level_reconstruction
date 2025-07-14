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

from src import reconstruction_per_cluster, compute_global_mean_sea_level, calculate_eofs_for_entire_dataset
from src.preprocessing import read_data
from src.settings.settings import GlobalSettings


def main():
    global_settings = GlobalSettings()
    if not os.path.exists(global_settings.output_path):
        os.makedirs(global_settings.output_path)
    sea_level_data, cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id, tide_gauge_data = read_data(
        global_settings)

    if global_settings.plot_eofs_for_entire_globe:
        current_outdir = "../output/components_entire_globe"
        if not os.path.exists(current_outdir):
            os.makedirs(current_outdir)
        calculate_eofs_for_entire_dataset.start(sea_level_data, current_outdir)

    if global_settings.reconstruction:
        reconstruction_per_cluster.start_reconstruction(sea_level_data, cluster_id_to_lat_lon_pairs,
                                                        cluster_id_to_grid_point_id, tide_gauge_data,
                                                        global_settings.timeframe, global_settings)

    # calculate global mean sea level, by calculating the mean of the tide gauges in a given cluster, then weight the
    # resulting time series by the area of the cluster, check if the result is close to the global mean sea level as
    # calculated from the satellite data
    if global_settings.calc_gmsl:
        compute_global_mean_sea_level.start_calculating_gmsl(sea_level_data, cluster_id_to_lat_lon_pairs,
                                                             cluster_id_to_grid_point_id, tide_gauge_data,
                                                             global_settings)


if __name__ == '__main__':
    main()
