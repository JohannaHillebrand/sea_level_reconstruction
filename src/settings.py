from pydantic_settings import BaseSettings


class GlobalSettings(BaseSettings):
    """
    Global settings
    """
    clustering_data_path: str = "../data/clustering/clustering_15.nc"
    sea_level_data_path: str = "../data/sla/sea_level_anomaly_data.nc"
    tide_gauge_data_folder: str = "../data/rlr_monthly"
