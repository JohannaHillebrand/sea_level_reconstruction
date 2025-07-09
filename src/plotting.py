import os

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import xarray


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
