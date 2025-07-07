from dataclasses import dataclass

from pydantic_settings import BaseSettings


@dataclass
class GlobalSettings(BaseSettings):
    """
    Global settings
    """
    clustering_data_path = "data/clustering_data.nc"
    sea_level_data_path = "data/sea_level_data.nc"
    tide_gauge_data_path = "data/tide_gauge_data.nc"
