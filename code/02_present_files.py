import geopandas as gpd
import configparser
import matplotlib.pyplot as plt
import numpy as np

def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config

def plot_geopackage_layer(gpkg_file, layer_name, output_png):
    try:
        layer = gpd.read_file(gpkg_file, layer=layer_name)

        if layer.empty or layer.is_empty.any() or not 'geometry' in layer.columns:
            print(f"Layer {layer_name} is empty, contains invalid geometries, or has no geometry column.")
            return

        fig, ax = plt.subplots(figsize=(40, 40), dpi=600)
        common_crs = 'EPSG:4326'

        if layer.crs and layer.crs.to_string() != common_crs:
            layer = layer.to_crs(common_crs)

        polygons = layer[layer.geom_type.isin(['Polygon', 'MultiPolygon'])]
        lines = layer[layer.geom_type.isin(['LineString', 'MultiLineString'])]
        points = layer[layer.geom_type.isin(['Point', 'MultiPoint'])]

        if not polygons.empty:
            polygons.plot(ax=ax, alpha=0.5, color='green')
        if not lines.empty:
            lines.plot(ax=ax, alpha=0.5, color='blue')
        if not points.empty:
            points.plot(ax=ax, alpha=0.5, color='red')

        # Default plot limits in case of invalid bounds
        default_limits = (-180, 180, -90, 90)

        try:
            total_bounds = layer.total_bounds
            if np.all(np.isfinite(total_bounds)):
                ax.set_xlim(total_bounds[0], total_bounds[2])
                ax.set_ylim(total_bounds[1], total_bounds[3])
            else:
                ax.set_xlim(default_limits[0], default_limits[1])
                ax.set_ylim(default_limits[2], default_limits[3])
        except Exception as e:
            print(f"Error setting plot limits, using default values: {e}")
            ax.set_xlim(default_limits[0], default_limits[1])
            ax.set_ylim(default_limits[2], default_limits[3])

        plt.savefig(output_png, bbox_inches='tight')
        print(f"Plot saved to {output_png}")

    except Exception as e:
        print(f"Error processing layer {layer_name}: {e}")

# Main execution
config_file = 'config.ini'
config = read_config(config_file)
gpkg_file = config['DEFAULT']['gpkg_file']
output_png = config['DEFAULT']['output_png']
asset_output_png   = 'output/asset.png'
flat_output_png    ='output/flat.png'
geocode_output_png = 'output/geocode.png'

plot_geopackage_layer(gpkg_file, 'tbl_asset_object', asset_output_png)
plot_geopackage_layer(gpkg_file, 'tbl_flat', flat_output_png)
plot_geopackage_layer(gpkg_file, 'tbl_geocode_object', geocode_output_png)
