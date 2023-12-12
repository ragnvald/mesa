import os
import glob
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon
from sqlalchemy import create_engine

def import_spatial_data(input_folder):
    asset_objects = []
    object_id_counter = 1

    file_patterns = ['*.shp', '*.gpkg']

    for pattern in file_patterns:
        for filepath in glob.glob(os.path.join(input_folder, '**', pattern), recursive=True):
            try:
                # Read spatial data
                data = gpd.read_file(filepath)
                asset_group_name = os.path.splitext(os.path.basename(filepath))[0]

                for index, row in data.iterrows():
                    # Flatten attributes into a string
                    attributes = '; '.join([f"{col}: {row[col]}" for col in data.columns if col != 'geometry'])

                    # Calculate area if geometry is polygon
                    area_m2 = row.geometry.area if isinstance(row.geometry, MultiPolygon) else 0

                    # Append asset object
                    asset_objects.append({
                        'id': object_id_counter,
                        'ref_asset_group': object_id_counter,  # Assuming each object is its own group, adjust if needed
                        'asset_group_name': asset_group_name,
                        'attributes': attributes,
                        'process': True,  # Set to False later if geometry is faulty
                        'area_m2': area_m2,
                        'geom': row.geometry
                    })
                    object_id_counter += 1

            except Exception as e:
                print(f"Error importing {filepath}: {e}")

    # Convert list to GeoDataFrame
    asset_objects_gdf = gpd.GeoDataFrame(asset_objects, geometry='geom')
    return asset_objects_gdf

def export_to_geopackage(asset_objects_gdf, output_path):
    engine = create_engine(f'sqlite:///{output_path}')

    # Export asset objects to geopackage, append if table exists
    asset_objects_gdf.to_file(output_path, layer='tbl_asset_object', driver="GPKG", if_exists='append')

    print(f"All data exported to {output_path}")

input_folder = 'input/assets'
output_path = 'output/mesa.gpkg'

asset_objects_gdf = import_spatial_data(input_folder)

# Export to geopackage
export_to_geopackage(asset_objects_gdf, output_path)
