import numpy as np
import xarray
from sklearn.decomposition import PCA

from src import plotting


def start(sea_level_data: xarray.Dataset, output_path):
    """
    Start the calculation of EOFs for the entire dataset.
    :param sea_level_data:
    :param global_settings:
    :return:
    """

    # extract the time series for the current cluster as input data for PCA
    sla_data = sea_level_data["sla"].values
    index_to_gridpoint = {}
    counter = 0
    time_dim_size, n_dim_size, m_dim_size = sla_data.shape
    grid_points = [(idx, idy) for idx in range(n_dim_size) for idy in range(m_dim_size)]
    print(len(grid_points))
    data_for_pca = np.full((len(grid_points), sla_data.shape[0]), np.nan)
    for idx, idy in grid_points:
        time_series = sla_data[:, idx, idy]
        if np.isnan(time_series).any():
            continue
        else:
            index_to_gridpoint[counter] = (idx, idy)
            data_for_pca[counter] = time_series
            counter += 1
    # remove the unused part of the data_for_pca
    data_for_pca = data_for_pca[:counter, :]
    print(data_for_pca.shape)
    # perform svd on the data for PCA
    #
    k = 20  # Number of components
    #
    pca = PCA(n_components=k)
    # # fit_transform will compute the SVD and project the data onto the k components
    pcs = pca.fit_transform(data_for_pca.T)
    eofs = pca.components_.T

    print(pcs.shape)
    print(eofs.shape)

    for i in range(k):
        current_pc = pcs[:, i]
        plotting.plot_time_series(current_pc, f"{output_path}", f"PC_{i}", sea_level_data)
        current_eof = eofs[:, i]
        current_eof_plot = np.nan * np.ones(sla_data.shape[1:])  # 2d array with shape (latitude, longitude)
        for index, grid_point in index_to_gridpoint.items():
            current_eof_plot[grid_point] = current_eof[index]
        plotting.plot_eof(current_eof_plot, f"{output_path}", f"EOF_{i}")

    # first_pc = pcs[:, 0]
    # # plot the first component
    # plotting.plot_time_series(first_pc, f"{output_path}",
    #                           f"first_PC",
    #                           sea_level_data)
    # first_EOF = eofs[:, 0]
    # # plot the first EOF, each grid point has a different value
    # eof_plot = np.nan * np.ones(sla_data.shape[1:])  # 2d array with shape (latitude, longitude)
    # for index, grid_point in index_to_gridpoint.items():
    #     eof_plot[grid_point] = first_EOF[index]
    # plotting.plot_eof(eof_plot, f"{output_path}", f"first_EOF")
    #
    # # plot second component
    # second_pc = pcs[:, 1]
    # plotting.plot_time_series(second_pc, f"{output_path}",
    #                           f"PC",
    #                           sea_level_data)
    # second_EOF = eofs[:, 1]
    # # plot the second EOF, each grid point has a different value
    # eof_plot = np.nan * np.ones(sla_data.shape[1:])  # 2d array with shape (latitude, longitude)
    # for index, grid_point in index_to_gridpoint.items():
    #     eof_plot[grid_point] = second_EOF[index]
    # plotting.plot_eof(eof_plot, f"{output_path}", f"second_EOF")

    return None
