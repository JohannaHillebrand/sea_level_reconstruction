from pydantic_settings import BaseSettings


class GlobalSettings(BaseSettings):
    """
    Global settings
    """
    clustering_data_path: str = "../data/clustering/15_clusters"
    number_of_clusters: int = 15
    sea_level_data_path: str = "../data/sla/sea_level_anomaly_data.nc"
    tide_gauge_data_folder: str = "../data/rlr_monthly"
    output_path: str = "../output/15_clusters"
    timeframe: tuple[int, int] = (1993, 2024)
    number_of_principal_components: int = 10
