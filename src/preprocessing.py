from typing import Any

import xarray
from xarray import Dataset

from src.settings import GlobalSettings


def read_data(global_settings: GlobalSettings) -> tuple[Dataset, Dataset, dict[Any, Any]]:
    """
    Read data from file
    :param global_settings:
    :return:
    """
    sea_level_data = xarray.open_dataset(global_settings.sea_level_data_path)
    clustering_data = xarray.open_dataset(global_settings.clustering_data_path)
    tide_gauge_data = {}
    return sea_level_data, clustering_data, tide_gauge_data
