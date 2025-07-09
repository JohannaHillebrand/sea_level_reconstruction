from dataclasses import dataclass

from loguru import logger


@dataclass
class TideGaugeStation:
    id: int
    name: str
    latitude: float
    longitude: float
    timeseries: dict


def read_and_create_stations(path: str) -> dict[int:TideGaugeStation]:
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
                            valid_values += 1
                        station_timeseries[date] = sea_level
                current_stations[station_id] = TideGaugeStation(id=station_id, name=station_name,
                                                                latitude=station_latitude,
                                                                longitude=station_longitude,
                                                                timeseries=station_timeseries)
            except FileNotFoundError:
                logger.error(f"File not found: {path}/data/{station_id}.rlrdata")
    logger.info(f"Flag counter: {flag_counter}")
    logger.info(f"No data values: {no_data_values}")
    logger.info(f"Valid values: {valid_values}")
    logger.info(f"Number of stations: {len(current_stations)}")
    return current_stations
