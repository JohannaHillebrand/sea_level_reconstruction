import colorsys
import os
import random
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shapely
import xarray
import xarray as xr
from matplotlib import pyplot as plt
from shapely.geometry.point import Point
from shapely.ops import unary_union

from src.settings.settings import GlobalSettings
from src.tide_gauge_station import TideGaugeStation


def plot_xarray_dataset_on_map(xarray_dataset: xarray.Dataset, out_dir: str, name: str):
    cluster_colors = {0: "gold", 1: "yellowgreen", 2: "dodgerblue", 3: "rebeccapurple", 4: "orchid", 5: "maroon",
                      6: "darkorange", 7: "palegoldenrod", 8: "darkolivegreen", 9: "forestgreen", 10: "teal",
                      11: "darkblue", 12: "darkorchid", 13: "deeppink", 14: "red", 15: "yellow", 16: "darkseagreen",
                      17: "azure", 18: "lightsteelblue", 19: "midnightblue", 20: "plum", 21: "sienna", 22: "chartreuse",
                      23: "darkslategray", 24: "darkmagenta", 25: "crimson", 26: "cornflowerblue", 27: "chocolate",
                      28: "lemonchiffon", 29: "lavenderblush", 30: "navy", 31: "purple"}

    # Create a colormap and norm
    # Create colormap and normalization
    cmap = mcolors.ListedColormap([cluster_colors[i] for i in sorted(cluster_colors.keys())])
    cmap.set_bad(color=(1, 1, 1, 0))  # fully transparent RGBA white

    bounds = list(cluster_colors.keys()) + [max(cluster_colors.keys()) + 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    data = xarray_dataset["__xarray_dataarray_variable__"]
    fig = plt.figure(figsize=(50, 25))
    ax = plt.axes(projection=ccrs.PlateCarree())
    data.plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, add_colorbar=True)
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.savefig(os.path.join(out_dir, f"{name}.pdf"), dpi=500)
    plt.close(fig)
    return


def assign_color_to_cluster(cluster_to_grid_point_ids_dict):
    """
    Assign a color to each cluster based on the cluster id
    :param cluster_to_grid_point_ids_dict:
    :return:
    """
    cluster_colors = ["gold", "yellowgreen", "dodgerblue", "rebeccapurple", "orchid", "maroon",
                      "darkorange", "palegoldenrod", "darkolivegreen", "forestgreen", "teal", "darkblue",
                      "darkorchid",
                      "deeppink", "red", "yellow", "darkseagreen", "azure", "lightsteelblue", "midnightblue", "plum",
                      "sienna", "chartreuse", "darkslategray", "darkmagenta", "crimson", "cornflowerblue", "chocolate",
                      "lemonchiffon", "lavenderblush", "navy", "purple"]
    if not len(cluster_to_grid_point_ids_dict.keys()) < (len(cluster_colors)):
        cluster_colors = random_color_generator(len(cluster_to_grid_point_ids_dict.keys()) + 1)
    # create a dictionary that maps the cluster id to the color
    cluster_id_to_color = {cluster_id: cluster_colors[int(i)] for i, cluster_id in
                           enumerate(cluster_to_grid_point_ids_dict.keys())}
    cluster_id_to_color[-99999] = "grey"
    return cluster_id_to_color


def plot_clustering(cluster_dict, out_dir, resolution, name, cluster_id_to_color):
    """
    Plot the clustering
    :param cluster_id_to_color:
    :param cluster_dict:
    :param out_dir:
    :param resolution:
    :param name:
    :return:
    """
    # sort cluster_dict by cluster_id
    cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: item[0]))
    cluster_gdf, land_gdf = turn_dict_into_gdf(cluster_dict, resolution / 2,
                                               cluster_id_to_color)
    plot_regions(land_gdf, out_dir, cluster_gdf, name)


def turn_tide_gauge_into_gdf(cluster_id_to_tide_gauges: dict[int, list[int]], cluster_id_to_color: dict[int, str],
                             tide_gauge_data: dict[int, TideGaugeStation]):
    """
    Turn the tide gauges into a geopandas dataframe
    :param tide_gauge_data:
    :param cluster_id_to_tide_gauges:
    :param cluster_id_to_color:
    :return:
    """
    tide_gauge_data_for_gdf = {}
    tide_gauge_data_for_gdf["id"] = []
    tide_gauge_data_for_gdf["geometry"] = []
    tide_gauge_data_for_gdf["color"] = []
    for cluster_id, tide_gauges in cluster_id_to_tide_gauges.items():
        color = cluster_id_to_color[cluster_id]
        for tide_gauge in tide_gauges:
            lat = tide_gauge_data[tide_gauge].latitude
            lon = tide_gauge_data[tide_gauge].longitude
            tide_gauge_data_for_gdf["id"].append(tide_gauge)
            tide_gauge_data_for_gdf["geometry"].append(Point(lon, lat))
            tide_gauge_data_for_gdf["color"].append(color)

    tide_gauge_gdf = geopandas.GeoDataFrame(tide_gauge_data_for_gdf)
    tide_gauge_gdf.set_geometry("geometry", crs="epsg:4326")
    return tide_gauge_gdf


def plot_clustering_with_tide_gauges(cluster_dict, out_dir, name, resolution,
                                     cluster_id_to_tide_gauges: dict[int, list[int]],
                                     tide_gauge_data: dict[int, TideGaugeStation]):
    """
    Plot the clustering with tide gauges
    :param resolution:
    :param tide_gauge_data:
    :param cluster_id_to_tide_gauges:
    :param cluster_dict:
    :param out_dir:
    :param name:
    :return:
    """
    cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: item[0]))
    cluster_id_to_color = assign_color_to_cluster(cluster_dict)
    # create a dataframe for the tide gauges, where they have the same color as the cluster
    tide_gauge_gdf = turn_tide_gauge_into_gdf(cluster_id_to_tide_gauges, cluster_id_to_color, tide_gauge_data)
    cluster_gdf, land_gdf = turn_dict_into_gdf(cluster_dict, resolution / 2, cluster_id_to_color)
    plot_regions_with_tide_gauges(land_gdf, out_dir, cluster_gdf, name, tide_gauge_gdf)
    return


def plot_clustering_without_preassigned_colors(cluster_dict, out_dir, resolution, name):
    """
    Plot the clustering without preassigned colors
    :param cluster_dict: dict with cluster_id as key and list of grid points as value
    :param out_dir: the output directory to save the plot
    :param resolution: resolution of the grid in degrees
    :param name: name of the plot file (without extension)
    :return:
    """
    cluster_dict = dict(sorted(cluster_dict.items(), key=lambda item: item[0]))
    cluster_id_to_color = assign_color_to_cluster(cluster_dict)
    plot_clustering(cluster_dict, out_dir, resolution, name, cluster_id_to_color)


def turn_dict_into_gdf(cluster_dict: {float: [(float, float)]}, grid_point_area: float,
                       cluster_id_to_color: {int: str}):
    """
    Turn a dictionary into a geopandas dataframe
    :param cluster_id_to_color:
    :param grid_point_area:
    :param cluster_dict:
    :return:
    """
    land_gdf = geopandas.read_file("../data/ne_10m_land/ne_10m_land.shp")
    # print(f"number of clusters {len(cluster_dict.keys())}")
    # turn clusters into a geopandas dataframe
    cluster_ids = []
    polygons = []
    colors = []
    for cluster in cluster_dict.keys():
        # create a polygon from all grid points in the current cluster
        cluster_squares = []
        cluster_ids.append(cluster)
        # counter += 1

        for grid_point in cluster_dict[cluster]:
            square = shapely.box(
                grid_point[1] - grid_point_area,  # minx (lon)
                grid_point[0] - grid_point_area,  # miny (lat)
                grid_point[1] + grid_point_area,  # maxx
                grid_point[0] + grid_point_area  # maxy
            )

            cluster_squares.append(square)
        colors.append(cluster_id_to_color[cluster])

        # Merge all squares in the cluster using unary_union
        merged_polygon = unary_union(cluster_squares).buffer(0)
        polygons.append(merged_polygon)

    # Create GeoDataFrame with merged polygons
    cluster_gdf = geopandas.GeoDataFrame(
        {'cluster_id': cluster_ids, 'color': colors, 'geometry': polygons}
        # ,crs="EPSG:4326"  # WGS 84 coordinate system
    )
    cluster_gdf['color'] = cluster_gdf['cluster_id'].map(cluster_id_to_color)
    # print(f"number of polygons {len(cluster_gdf)}")
    return cluster_gdf, land_gdf


def random_color_generator(num_colors: int):
    """
    Generates a list of random colors
    :param num_colors:
    :return:
    """
    colors = []

    for i in range(num_colors - 1):
        h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
        r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
        colors.append('#%02x%02x%02x' % (r, g, b))
        # colors.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
    return colors


def plot_regions(land_gdf: geopandas.GeoDataFrame, output_path: str,
                 clusters_gdf: geopandas.GeoDataFrame, name: str):
    """
    Plot the regions on a map
    :param name:
    :param land_gdf:
    :param output_path:
    :param clusters_gdf:
    :return:
    """

    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    clusters_gdf.plot(ax=ax, color=clusters_gdf["color"], zorder=4, linewidth=4, edgecolor="none")

    # clusters_gdf.boundary.plot(ax=ax, color="black", zorder=5, linewidth=0.5)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    handles = [mpatches.Patch(color=color, label=f"Cluster {cluster_id}")
               for cluster_id, color in
               zip(clusters_gdf["cluster_id"].unique(), clusters_gdf["color"].unique())]
    ax.legend(handles=handles, title="Clusters")
    # plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()


def plot_regions_with_tide_gauges(land_gdf: geopandas.GeoDataFrame, output_path: str,
                                  clusters_gdf: geopandas.GeoDataFrame, name: str,
                                  tide_gauge_gdf: geopandas.GeoDataFrame):
    """
    Plot the regions on a map
    :param tide_gauge_gdf:
    :param name:
    :param land_gdf:
    :param output_path:
    :param clusters_gdf:
    :return:
    """

    ax = land_gdf.plot(color="burlywood", figsize=(20, 12), zorder=0, alpha=0.5)
    ax.set_facecolor("aliceblue")
    clusters_gdf.plot(ax=ax, color=clusters_gdf["color"], zorder=1, linewidth=4, edgecolor="none")
    tide_gauge_gdf.plot(ax=ax, marker='o', linestyle='-', color=tide_gauge_gdf["color"],
                        edgecolor='black', linewidth=2, zorder=2)
    # clusters_gdf.boundary.plot(ax=ax, color="black", zorder=5, linewidth=0.5)
    plt.xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    plt.yticks([-90, -45, 0, 45, 90])
    handles = [mpatches.Patch(color=color, label=f"Cluster {cluster_id}")
               for cluster_id, color in
               zip(clusters_gdf["cluster_id"].unique(), clusters_gdf["color"].unique())]
    ax.legend(
        handles=handles,
        title="Clusters",
        loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    # plt.savefig(os.path.join(output_path, f"{name}.svg"))
    plt.savefig(os.path.join(output_path, f"{name}.png"), bbox_inches="tight")
    plt.close()


def plot_reconstruction_error_over_time(mean_reconstruction_error_for_date: dict[datetime, float],
                                        max_reconstruction_error_for_date: dict[datetime, float],
                                        min_reconstruction_error_for_date: dict[datetime, float], cluster_id,
                                        out_dir: str, number_of_testing_tide_gauges_for_date: dict[datetime, int],
                                        number_of_training_tide_gauges_for_date: dict[datetime, int],
                                        explained_variance_ratio: float):
    """
    Plot the reconstruction error over time for a specific cluster
    :param explained_variance_ratio:
    :param number_of_training_tide_gauges_for_date:
    :param number_of_testing_tide_gauges_for_date:
    :param out_dir:
    :param mean_reconstruction_error_for_date:
    :param max_reconstruction_error_for_date:
    :param min_reconstruction_error_for_date:
    :param cluster_id:
    :return:
    """
    if (mean_reconstruction_error_for_date is not None and max_reconstruction_error_for_date is not None and
            min_reconstruction_error_for_date is not None):
        print(mean_reconstruction_error_for_date)
        print(max_reconstruction_error_for_date)
        print(min_reconstruction_error_for_date)
        dates_to_remove = []
        for date in mean_reconstruction_error_for_date.keys():
            mean_ = mean_reconstruction_error_for_date.get(date, None)
            max_ = max_reconstruction_error_for_date.get(date, None)
            min_ = min_reconstruction_error_for_date.get(date, None)
            if mean_ is None or max_ is None or min_ is None:
                dates_to_remove.append(date)
                continue
        for date in dates_to_remove:
            del mean_reconstruction_error_for_date[date]
            del max_reconstruction_error_for_date[date]
            del min_reconstruction_error_for_date[date]
            if date in number_of_testing_tide_gauges_for_date:
                del number_of_testing_tide_gauges_for_date[date]
            if date in number_of_training_tide_gauges_for_date:
                del number_of_training_tide_gauges_for_date[date]
        fig, ax1 = plt.subplots(figsize=(20, 10))  # Create figure and primary axes (ax1)

        # --- Plotting on the Left Y-axis (Reconstruction Error) ---
        ax1.plot(mean_reconstruction_error_for_date.keys(), mean_reconstruction_error_for_date.values(),
                 label="Mean Reconstruction Error", color="blue", linewidth=2)
        ax1.fill_between(mean_reconstruction_error_for_date.keys(),
                         min_reconstruction_error_for_date.values(),
                         max_reconstruction_error_for_date.values(),
                         color='lightblue', alpha=0.5,
                         label="Min/Max Reconstruction Error Range")

        ax1.set_xlabel("Date", fontsize=14)
        ax1.set_ylabel("Reconstruction Error", color="blue", fontsize=14)
        ax1.tick_params(axis='y', labelcolor="blue")  # Set tick color for ax1
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_yticks([0, 0.5, 1, 1.5, 2, 3, 5, 10])  # Set specific y-ticks

        # --- Create a Second Y-axis (Right side) for Tide Gauge Counts ---
        ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis

        ax2.plot(number_of_testing_tide_gauges_for_date.keys(), number_of_testing_tide_gauges_for_date.values(),
                 linestyle='--', color='lightgreen', label="Number of Testing Tide Gauges", linewidth=2)
        ax2.plot(number_of_training_tide_gauges_for_date.keys(), number_of_training_tide_gauges_for_date.values(),
                 linestyle='--', color='darkgreen', label="Number of Training Tide Gauges", linewidth=2)

        ax2.set_ylabel("Number of Tide Gauges", color="green", fontsize=14)
        ax2.tick_params(axis='y', labelcolor="green")  # Set tick color for ax2
        ax2.set_yticks([0, 2, 4, 8, 10, 20, 30, 40, 50])  # Set specific y-ticks for the second axis

        # --- Combine Legends from both axes ---
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)  # Place combined legend

        plt.title(
            f"Reconstruction Error and Tide Gauge Counts for Cluster {cluster_id}, explained variance: "
            f"{explained_variance_ratio}",
            fontsize=16)
        fig.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Save the figure
        file_path = f"{out_dir}/reconstruction_error_cluster_{cluster_id}.png"
        plt.savefig(file_path, dpi=600)
    return


def plot_global_mean_sea_level(global_mean_sea_level_time_series: dict[datetime, float],
                               average_tide_gauge_time_series: dict[datetime, float],
                               reference_gmsl: dict[datetime, float], global_settings: GlobalSettings) -> None:
    """
    Plot the global mean sea level time series and the average tide gauge time series and the cluster-weighted
    average tide gauge time series.
    :param global_settings:
    :param global_mean_sea_level_time_series:
    :param average_tide_gauge_time_series:
    :param reference_gmsl:
    :return:
    """
    # sort the time series by date
    global_mean_sea_level_time_series = dict(sorted(global_mean_sea_level_time_series.items()))
    average_tide_gauge_time_series = dict(sorted(average_tide_gauge_time_series.items()))
    reference_gmsl = dict(sorted(reference_gmsl.items()))
    plt.figure(figsize=(20, 10))
    plt.plot(average_tide_gauge_time_series.keys(), average_tide_gauge_time_series.values(),
             label="Average tide gauge GMSL",
             color="red", linewidth=2)
    plt.plot(global_mean_sea_level_time_series.keys(), global_mean_sea_level_time_series.values(),
             label="Cluster weighted average tide gauge GMSL", color="green", linewidth=2)
    plt.plot(reference_gmsl.keys(), reference_gmsl.values(), label="Satellite GMSL", color="blue", linewidth=2)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Global Mean Sea Level (m)", fontsize=14)
    plt.title("Global Mean Sea Level Time Series", fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig(f"{global_settings.output_path}/global_mean_sea_level.png", dpi=300)
    return None


def plot_time_series(first_component: np.array, out_dir: str, name: str, sea_level_anomaly_data: xr.Dataset):
    """
    Plot a given time series
    :param sea_level_anomaly_data:
    :param name:
    :param first_component:
    :param out_dir:
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # time_steps = sea_level_anomaly_data['time'].values
    # date_times = pd.to_datetime(time_steps)
    #
    # fig, ax = plt.subplots(figsize=(15, 6))
    # ax.plot(date_times, first_component)
    #
    # # Set the locator to show a tick for every year
    # ax.xaxis.set_major_locator(mdates.YearLocator())
    # # Set the formatter to show the full date
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Sea level in meters')
    # ax.set_title(f"{name}")
    #
    # # Rotate and align the tick labels so they don't overlap
    # fig.autofmt_xdate()
    #
    # plt.savefig(os.path.join(out_dir, f'{name}.png'))
    # plt.close(fig)

    time_steps = sea_level_anomaly_data['time'].values
    date_times = pd.to_datetime(time_steps)

    fig, ax = plt.subplots(figsize=(12, 5))
    # Plot the time series
    ax.plot(date_times, first_component, color='navy', linewidth=1.5)

    # Set the locator to show a tick for every year
    ax.xaxis.set_major_locator(mdates.YearLocator())
    # Set the formatter to show the full date
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', labelrotation=45)

    # Labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Sea Level Anomaly (m)', fontsize=12)

    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    fig.tight_layout()

    output_path = os.path.join(out_dir, f'{name}.jpg')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return None


def plot_eof(eof_to_plot: np.array, output_dir: str, name: str):
    """
    Plot a given EOF
    :param eof_to_plot:
    :param output_dir:
    :param name:
    :return:
    """
    # Define coordinate extent (longitude/latitude or other units)
    extent = [-180, 180, -90, 90]  # [xmin, xmax, ymin, ymax]

    # # Create the custom colormap
    # custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    #     "smooth_split_blue",
    #     [
    #         (0.0, "#2ca25f"),  # green
    #         (0.40, "#2ca25f"),  # mostly green
    #         (0.48, "#0570b0"),  # start blue fade
    #         (0.50, "#0570b0"),  # pure blue at center
    #         (0.52, "#0570b0"),  # end blue fade
    #         (0.60, "#de2d26"),  # mostly red
    #         (1.0, "#de2d26")  # full red
    #     ]
    # )
    # Plot using a projection (PlateCarree = regular lat/lon grid)
    # Define figure size suitable for 300–1200 DPI
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the data
    im = ax.imshow(eof_to_plot, extent=extent, origin='lower', cmap='seismic', interpolation='none')

    # Add geographic features with highest resolution
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='burlywood', zorder=0, alpha=0.5)
    # Colorbar with improved layout and font
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.85)
    cbar.set_label('Sea Level (m)', fontsize=12)

    # Improve layout and save at high DPI
    output_path = os.path.join(output_dir, f'{name}.jpg')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_reconstruction_comparison(mean_sc, min_sc, max_sc, mean_km, min_km, max_km, output_path, timeframe):
    """
    Plot the reconstruction comparison for different clustering methods
    :param mean_sc:
    :param min_sc:
    :param max_sc:
    :param mean_km:
    :param min_km:
    :param max_km:
    :param output_path:
    :param timeframe:
    :return:
    """
    mean_sc = dict(sorted(mean_sc.items()))
    min_sc = dict(sorted(min_sc.items()))
    max_sc = dict(sorted(max_sc.items()))
    mean_km = dict(sorted(mean_km.items()))
    min_km = dict(sorted(min_km.items()))
    max_km = dict(sorted(max_km.items()))

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    fig, ax = plt.subplots(figsize=(20, 10))
    # Plot the mean reconstruction for both methods
    ax.plot(mean_sc.keys(), mean_sc.values(), label="Mean Reconstruction Error Subspace Clustering", color="blue",
            linewidth=2)
    ax.fill_between(mean_sc.keys(), min_sc.values(), max_sc.values(), color='lightblue', alpha=0.5,
                    label="Min/Max Reconstruction Error Subspace Clustering")
    ax.plot(mean_km.keys(), mean_km.values(), label="Mean Reconstruction Error K-Means", color="green", linewidth=2)
    ax.fill_between(mean_km.keys(), min_km.values(), max_km.values(), color='lightgreen', alpha=0.5,
                    label="Min/Max Reconstruction Error Range K-Means")
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Reconstruction Error", fontsize=14)
    # save the plot
    ax.set_title(f"Reconstruction Comparison for {timeframe}", fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim(bottom=-1, top=20)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"reconstruction_comparison_k_means_subspace_clustering.jpg"), dpi=600)
    plt.close()

    return None
