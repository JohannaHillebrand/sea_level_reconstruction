from pydantic_settings import BaseSettings


class GlobalSettings(BaseSettings):
    """
    Global settings
    """
    clustering_data_path: str = "../data/clustering/15_clusters"
    # clustering_data_path: str = "../data/clustering/k-means"
    number_of_clusters: int = 15
    sea_level_data_path: str = "../data/sla/sea_level_anomaly_data.nc"
    tide_gauge_data_folder: str = "../data/rlr_monthly"
    output_path: str = "../output/15_clusters"
    # output_path: str = "../output/k-means"
    cut_off_year_beginning: int = 1975
    timeframe: tuple[int, int] = (1993, 2024)
    number_of_principal_components: int = 10
    lower_bound_for_number_of_tide_gauges_per_cluster: int = 20
    reconstruction_iterations: int = 1
    percentage_of_data_used_for_reconstruction: float = 0.9
    baseline_number_of_tide_gauges_for_testing: int = 2
    reconstruction: bool = True
    calc_gmsl: bool = False
    plot_eofs_for_entire_globe: bool = False
