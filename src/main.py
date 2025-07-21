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
import pickle

from loguru import logger

from src import reconstruction_per_cluster, compute_global_mean_sea_level, calculate_eofs_for_entire_dataset, plotting
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
        if not os.path.exists("../output/comparison/mean_sc.pkl") or not os.path.exists(
                "../output/comparison/min_sc.pkl") or not os.path.exists("../output/comparison/max_sc.pkl"):
            logger.info(f"{global_settings.output_path}")
            mean_sc, min_sc, max_sc = (
                reconstruction_per_cluster.start_reconstruction(sea_level_data, cluster_id_to_lat_lon_pairs,
                                                                cluster_id_to_grid_point_id, tide_gauge_data,
                                                                global_settings))

            # save to file
            if not os.path.exists("../output/comparison"):
                os.mkdir("../output/comparison")
            with open(f"../output/comparison/mean_sc.pkl", "wb") as outfile:
                pickle.dump(mean_sc, outfile)
            with open(f"../output/comparison/min_sc.pkl", "wb") as outfile:
                pickle.dump(min_sc, outfile)
            with open(f"../output/comparison/max_sc.pkl", "wb") as outfile:
                pickle.dump(max_sc, outfile)
        else:
            with open(f"../output/comparison/mean_sc.pkl", "rb") as infile:
                mean_sc = pickle.load(infile)
            with open(f"../output/comparison/min_sc.pkl", "rb") as infile:
                min_sc = pickle.load(infile)
            with open(f"../output/comparison/max_sc.pkl", "rb") as infile:
                max_sc = pickle.load(infile)

        if not os.path.exists("../output/comparison/mean_w.pkl") or not os.path.exists(
                "../output/comparison/min_w.pkl") or not os.path.exists("../output/comparison/max_w.pkl"):
            global_settings.output_path = "../output/wards"
            global_settings.clustering_data_path = "../data/clustering/wards"
            sea_level_data, cluster_id_to_lat_lon_pairs, cluster_id_to_grid_point_id, tide_gauge_data = read_data(
                global_settings)
            mean_w, min_w, max_w = (
                reconstruction_per_cluster.start_reconstruction(sea_level_data, cluster_id_to_lat_lon_pairs,
                                                                cluster_id_to_grid_point_id, tide_gauge_data,
                                                                global_settings))
            # save to file
            with open(f"../output/comparison/mean_w.pkl", "wb") as outfile:
                pickle.dump(mean_w, outfile)
            with open(f"../output/comparison/min_w.pkl", "wb") as outfile:
                pickle.dump(min_w, outfile)
            with open(f"../output/comparison/max_w.pkl", "wb") as outfile:
                pickle.dump(max_w, outfile)
        else:
            with open(f"../output/comparison/mean_w.pkl", "rb") as infile:
                mean_w = pickle.load(infile)
            with open(f"../output/comparison/min_w.pkl", "rb") as infile:
                min_w = pickle.load(infile)
            with open(f"../output/comparison/max_w.pkl", "rb") as infile:
                max_w = pickle.load(infile)

        # compare the two reconstructions
        global_settings.output_path = "../output/comparison"
        plotting.plot_reconstruction_comparison(mean_sc, min_sc, max_sc, mean_w, min_w, max_w,
                                                global_settings.output_path, global_settings.timeframe)

    # calculate global mean sea level, by calculating the mean of the tide gauges in a given cluster, then weight the
    # resulting time series by the area of the cluster, check if the result is close to the global mean sea level as
    # calculated from the satellite data
    if global_settings.calc_gmsl:
        compute_global_mean_sea_level.start_calculating_gmsl(sea_level_data, cluster_id_to_lat_lon_pairs,
                                                             cluster_id_to_grid_point_id, tide_gauge_data,
                                                             global_settings)


if __name__ == '__main__':
    main()
