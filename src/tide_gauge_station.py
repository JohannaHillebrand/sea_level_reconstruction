from dataclasses import dataclass
from datetime import datetime, timedelta

from loguru import logger


@dataclass
class TideGaugeStation:
    id: int
    name: str
    latitude: float
    longitude: float
    timeseries: dict
    timeseries_corrected_reference_datum: dict
    closest_grid_point: tuple[int, int] = None
    closest_lat_lon: tuple[float, float] = None

    def correct_reference_datum(self, time_series_of_closest_grid_point, time):
        """
        Correct the reference datum of the tide gauge by fitting it to the satellite sea level data
        :param time:
        :param time_series_of_closest_grid_point:
        :return:
        """
        overlapping_time_steps = []
        sla_timeseries = []
        tide_gauge_timeseries = []
        for i, sla_date in enumerate(time):
            py_sla_date = sla_date.astype('M8[D]').astype(datetime)
            for tide_gauge_date in self.timeseries.keys():
                if py_sla_date.year == tide_gauge_date.year and py_sla_date.month == tide_gauge_date.month:
                    if not self.timeseries[tide_gauge_date] == -99999:
                        overlapping_time_steps.append(tide_gauge_date)
                        sla_timeseries.append(time_series_of_closest_grid_point[i])
                        tide_gauge_timeseries.append(self.timeseries[tide_gauge_date])
        sum_of_differences = 0
        # calculate mean difference between the two time series
        # correct the entire tide gauge time series by adding the mean difference
        for i in range(len(overlapping_time_steps)):
            sum_of_differences += (sla_timeseries[i] - tide_gauge_timeseries[i])
        mean_difference = sum_of_differences / len(overlapping_time_steps)
        tide_gauge_timeseries_corrected_reference_datum = {}
        for key, value in self.timeseries.items():
            if value == -99999:
                tide_gauge_timeseries_corrected_reference_datum[key] = -99999
                continue
            tide_gauge_timeseries_corrected_reference_datum[key] = value + mean_difference
        self.timeseries_corrected_reference_datum = tide_gauge_timeseries_corrected_reference_datum


def read_and_create_stations(path: str, cutoff_date: int) -> dict[int:TideGaugeStation]:
    """
    Read filelist.txt and create a station object with the corresponding time series for each station in the file.
    :param path:
    :return:
    """
    flag_counter = 0
    no_data_values = 0
    valid_values = 0
    current_stations = {}
    with open(f"{path}/filelist.txt", "r") as file:
        for line in file:
            split_line = line.split(";")
            station_id = int(split_line[0])
            station_latitude = float(split_line[1].strip())
            station_longitude = float(split_line[2].strip())
            station_name = split_line[3].strip()
            station_timeseries = {}

            # assumption: the time series data is stored in a folder called "data" in the same directory as the
            # filelist.txt and each time series is stored in a file called <station_id>.rlrdata
            # # REPLACE the flagged values with -99999 (only flag for 011 - might be different to more than 1cm,
            # should not be used in long-term trend analysis)
            try:
                with open(f"{path}/data/{station_id}.rlrdata", "r") as rlr_file:
                    for rlr_line in rlr_file:
                        split_rlr_line = rlr_line.split(";")
                        date = float(split_rlr_line[0])
                        sea_level = float(split_rlr_line[1].strip())
                        flag = split_line[3].strip()
                        if flag == "011" or flag == "001 " or flag == "010":
                            sea_level = -99999
                            flag_counter += 1
                        if sea_level == -99999:
                            no_data_values += 1
                        else:
                            # convert sea level to meters (from mm)
                            sea_level /= 1000
                            valid_values += 1
                        real_date = year_fraction_to_date(date)
                        if real_date.year < cutoff_date:
                            continue
                        station_timeseries[real_date] = sea_level
                current_stations[station_id] = TideGaugeStation(id=station_id, name=station_name,
                                                                latitude=station_latitude,
                                                                longitude=station_longitude,
                                                                timeseries=station_timeseries,
                                                                timeseries_corrected_reference_datum={})
            except FileNotFoundError:
                logger.error(f"File not found: {path}/data/{station_id}.rlrdata")
    logger.info(f"Flag counter: {flag_counter}")
    logger.info(f"No data values: {no_data_values}")
    logger.info(f"Valid values: {valid_values}")
    logger.info(f"Number of stations: {len(current_stations)}")
    return current_stations


def year_fraction_to_date(year_fraction: float) -> datetime.date:
    year = int(year_fraction)
    start_of_year = datetime(year, 1, 1)
    days_in_year = 366 if is_leap_year(year) else 365
    fraction = year_fraction - year
    return (start_of_year + timedelta(days=fraction * days_in_year)).date()


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
