# TODO: for each cluster select tide gauges
# TODO: perform pca on the satellite sea level data
# TODO: fit tide gauge data to eof
# TODO: find out how to use change in sea level instead of the time series of the tide gauges. This is necessary as the
#  reference datum of the tide gauges is not uniform
# TODO: reconstruct sea level data throughout history
# TODO: plot sea level data
# TODO: compare reconstructed sea level data with observed sea level data
# TODO: compare reconstructed sea level data with regional sea level data that was not used for the reconstruction
# TODO: check if there are steps in between the different clusters
# TODO: do reconstruction on unclustered data
# TODO: compare

from src import reconstruction_per_cluster, settings

from src.preprocessing import read_data


def main():
    global_settings = settings.GlobalSettings()
    sea_level_data, clustering_data, tide_gauge_data = read_data(global_settings)

    reconstruction_per_cluster.start_reconstruction(sea_level_data, clustering_data, tide_gauge_data)
    pass


if __name__ == '__main__':
    main()
