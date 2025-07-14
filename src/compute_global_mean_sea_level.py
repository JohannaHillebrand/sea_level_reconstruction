import geopandas
import numpy as np
import shapely
import xarray
from shapely import unary_union

from src import plotting
from src.helper import prepare_tide_gauges
from src.settings.settings import GlobalSettings
from src.tide_gauge_station import TideGaugeStation


def turn_cluster_dict_to_geodataframe(cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                                      sea_level_data: xarray.Dataset):
    """
    Turn the cluster dictionary into a GeoDataFrame to determine the cluster size.
    :param cluster_id_to_lat_lon_pairs:
    :param sea_level_data:
    :return:
    """
    cluster_ids = []
    polygons = []
    grid_point_area = (sea_level_data.latitude.values[1] - sea_level_data.latitude.values[0]) / 2
    for cluster_id, lat_lon_pairs in cluster_id_to_lat_lon_pairs.items():
        cluster_squares = []
        cluster_ids.append(cluster_id)
        for lat_lon_pair in lat_lon_pairs:
            square = shapely.box(
                lat_lon_pair[1] - grid_point_area,  # minx (lon)
                lat_lon_pair[0] - grid_point_area,  # miny (lat)
                lat_lon_pair[1] + grid_point_area,  # maxx
                lat_lon_pair[0] + grid_point_area  # maxy
            )

            cluster_squares.append(square)

        # Merge all squares in the cluster using unary_union
        merged_polygon = unary_union(cluster_squares).buffer(0)
        polygons.append(merged_polygon)

        # Create GeoDataFrame with merged polygons
    cluster_gdf = geopandas.GeoDataFrame(
        {'cluster_id': cluster_ids, 'geometry': polygons}
        , crs="EPSG:4326"  # WGS 84 coordinate system
    )
    # print(f"number of polygons {len(cluster_gdf)}")
    return cluster_gdf


def start_calculating_gmsl(sea_level_data: xarray.Dataset,
                           cluster_id_to_lat_lon_pairs: dict[int, list[tuple[float, float]]],
                           cluster_id_to_grid_point_id: dict[int, list[tuple[float, float]]],
                           tide_gauge_data: dict[int, TideGaugeStation],
                           global_settings: GlobalSettings):
    """
    Start the global mean sea level calculation process.
    :param sea_level_data:
    :param cluster_id_to_lat_lon_pairs:
    :param cluster_id_to_grid_point_id:
    :param tide_gauge_data:
    :param global_settings:
    :return:
    """
    tide_gauge_data_corrected, cluster_id_to_tide_gauge, tide_gauge_to_lat_lon = prepare_tide_gauges(
        cluster_id_to_lat_lon_pairs,
        global_settings, sea_level_data,
        tide_gauge_data)

    # calculate the cluster area
    cluster_gdf = turn_cluster_dict_to_geodataframe(cluster_id_to_lat_lon_pairs, sea_level_data)
    print(cluster_gdf.crs)
    gdf_reprojected = cluster_gdf.to_crs(epsg=6933)
    print(gdf_reprojected.crs)
    area_per_cluster = {}
    total_area = 0
    for index, row in gdf_reprojected.iterrows():
        cluster_id = row['cluster_id']
        current_polygon = row['geometry']
        current_area = current_polygon.area / 10 ** 6  # converted from m^2 to km^2
        area_per_cluster[cluster_id] = current_area
        total_area += current_area

    # for each cluster, calculate the mean sea level time series from the tide gauges
    cluster_id_to_mean_sea_level = {}
    all_dates = set()
    for cluster_id, tide_gauge_ids in cluster_id_to_tide_gauge.items():
        if len(tide_gauge_ids) == 0:
            continue
        # get the tide gauge stations for the current cluster
        tide_gauge_stations = [tide_gauge_data_corrected[tide_gauge_id] for tide_gauge_id in tide_gauge_ids]
        # calculate the mean sea level time series for the current cluster
        mean_sea_level_time_series = {}
        all_dates_for_cluster = set(
            [date for station in tide_gauge_stations for date in station.timeseries_corrected_reference_datum.keys()])
        all_dates.update(all_dates_for_cluster)
        for date in all_dates_for_cluster:
            valid_stations_for_date = 0
            for station in tide_gauge_stations:
                if date not in station.timeseries_corrected_reference_datum.keys():
                    continue
                if station.timeseries_corrected_reference_datum[date] == -99999:
                    continue
                else:
                    valid_stations_for_date += 1
                    if date not in mean_sea_level_time_series:
                        mean_sea_level_time_series[date] = 0
                    mean_sea_level_time_series[date] += station.timeseries_corrected_reference_datum[date]
            if valid_stations_for_date > 0:
                mean_sea_level_time_series[date] /= valid_stations_for_date
        weight = area_per_cluster[cluster_id] / total_area
        means_sea_level_weighted = {date: 0 for date in mean_sea_level_time_series.keys()}
        for date, value in mean_sea_level_time_series.items():
            means_sea_level_weighted[date] = value * weight
        # add the weighted mean sea level time series to the cluster_id_to_mean_sea_level dictionary
        cluster_id_to_mean_sea_level[cluster_id] = mean_sea_level_time_series

    # calculate the global mean sea level time series
    global_mean_sea_level_time_series = {date: 0 for date in all_dates}
    for date in all_dates:
        number_of_clusters_for_date = 0
        for cluster_id, mean_sea_level_time_series in cluster_id_to_mean_sea_level.items():
            if date not in mean_sea_level_time_series:
                continue
            if date not in global_mean_sea_level_time_series:
                global_mean_sea_level_time_series[date] = 0
            global_mean_sea_level_time_series[date] += mean_sea_level_time_series[date]
            number_of_clusters_for_date += 1
        global_mean_sea_level_time_series[date] /= number_of_clusters_for_date

    # calculate the global mean by averaging over all tide gauges
    average_tide_gauge_time_series = {}
    for date in all_dates:
        number_of_tide_gauges_for_date = 0
        for tide_gauge_id, tide_gauge in tide_gauge_data_corrected.items():

            if date not in tide_gauge.timeseries_corrected_reference_datum.keys():
                continue
            if tide_gauge.timeseries_corrected_reference_datum[date] == -99999:
                continue
            number_of_tide_gauges_for_date += 1
            if date not in average_tide_gauge_time_series:
                average_tide_gauge_time_series[date] = 0
            average_tide_gauge_time_series[date] += tide_gauge.timeseries_corrected_reference_datum[date]
        if number_of_tide_gauges_for_date != 0:
            average_tide_gauge_time_series[date] /= number_of_tide_gauges_for_date

    # create reference_gmsl by taking the mean of the satellite sea level data
    reference_gmsl = {}
    # weight the sea level anomaly data by the cosine of the latitude
    weighted_sea_level_data = sea_level_data.copy()
    latitude_weights = np.cos(np.deg2rad(sea_level_data.latitude.values))
    # Reshape latitude weights to match the dimensions
    weights = latitude_weights[:, np.newaxis] * np.ones(sea_level_data.sla.shape[2])
    weights = weights[np.newaxis, :, :] * np.ones(sea_level_data.sla.shape[0])[:, np.newaxis, np.newaxis]
    weighted_sea_level_data["weighted_sla"] = sea_level_data.sla * weights

    for timestep in sea_level_data.time.values:
        date = timestep.astype('M8[D]').astype('O')
        mean_sea_level = weighted_sea_level_data.weighted_sla.sel(time=timestep).mean().item()
        reference_gmsl[date] = mean_sea_level

    # plot all three time series
    plotting.plot_global_mean_sea_level(global_mean_sea_level_time_series, average_tide_gauge_time_series,
                                        reference_gmsl, global_settings)

    return None
