# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#  User selects input geocode group
#  List of lines is provided (tbl_line_in)
#   1) No tbl_line_input means a line should be created based on tbl_flat.
#   2) tbl_line_input has one of more lines
#  User selects lines
#  User selects:
#     - buffered width of line in meters
#     - line segment length
#  Feedback if selected combinations will overwrite existing data.
#  Analyse overlayng geocode with line segments / polygons
#     - results in tbl_line_out_flat (we might build a stacked one at a later stage...)
#

import geopandas as gpd
import pandas as pd
import configparser
import subprocess
import datetime
import locale
import math
from shapely.geometry import box, LineString, Point, Polygon, MultiLineString, MultiPolygon
from shapely.ops import unary_union, split, polygonize, linemerge
from shapely.geometry import mapping
from fiona.crs import from_epsg
from shapely.ops import transform
import pyproj
from functools import partial

# Set locale
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.utf8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

import os
import numpy as np
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap import Style

# Read the configuration file
def read_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    return config


def read_config_classification(file_name):
    config = configparser.ConfigParser()
    config.read(file_name)
    classification = {}
    for section in config.sections():
        range_str = config[section]['range']
        start, end = map(int, range_str.split('-'))
        classification[section] = range(start, end + 1)
    return classification


# Logging function to write to the GUI log
def log_to_gui(log_widget, message):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S")
    formatted_message = f"{timestamp} - {message}"
    log_widget.insert(tk.END, formatted_message + "\n")
    log_widget.see(tk.END)
    with open("log.txt", "a") as log_file:
        log_file.write(formatted_message + "\n")

def update_progress(new_value):
    progress_var.set(new_value)
    progress_label.config(text=f"{int(new_value)}%")


def load_lines_table(gpkg_file):

    # Check if the GeoPackage file exists
    if not os.path.exists(gpkg_file):
        return None  # File not found, return None without printing an error message

    try:
        # Attempt to load 'tbl_lines' from the specified GeoPackage file
        gdf = gpd.read_file(gpkg_file, layer='tbl_lines')
        
        # Check if the loaded GeoDataFrame is empty
        if gdf.empty:
            return None  # The table 'tbl_lines' is empty, return None
        else:
            return gdf
    except ValueError:
        return None  # The table 'tbl_lines' does not exist, return None


def create_lines_table_and_lines(gpkg_file, log_widget):
    gdf_geocode_group = gpd.read_file(gpkg_file, layer='tbl_geocode_group', rows=1)
    if gdf_geocode_group.empty:
        print("The 'tbl_geocode_group' table is empty or does not exist.")
        return
    
    log_to_gui(log_widget, "Creating table and lines")

    # Extract bounding box
    minx, miny, maxx, maxy = gdf_geocode_group.total_bounds
    
    # Generate three random lines within this bounding box
    lines = []
    for i in range(3):
        # Generate random start and end points within the bounding box
        start_x = np.random.uniform(minx, maxx)
        start_y = np.random.uniform(miny, maxy)
        end_x   = np.random.uniform(minx, maxx)
        end_y   = np.random.uniform(miny, maxy)
        line    = LineString([(start_x, start_y), (end_x, end_y)])
        lines.append(line)
    
    # Create a GeoDataFrame for the lines
    gdf_lines = gpd.GeoDataFrame({
        'name_gis': [f'line_{i:03}' for i in range(1, 4)],
        'name_user': [f'line_{i:03}' for i in range(1, 4)],
        'segment_length': [15, 30, 10],
        'segment_width': [1000, 20000, 5000],
        'description': ['another line', 'another line', 'another line'],
        'geometry': lines
    }, crs=gdf_geocode_group.crs)
    
    # Save to the GeoPackage
    gdf_lines.to_file(gpkg_file, layer='tbl_lines', mode="w")


# Buffering lines so that we will get a width in meters. To do this we will have to
# reproject the data to a projection which works decently well worldwide in converting
# between degrees and meters. At a later stage one might want to facilitate for 
# the user selecting an appropriate local projection.
def process_and_buffer_lines(gpkg_file, log_widget):
    
    crs         = "EPSG:4326"
    target_crs  = "EPSG:4087"

    lines_df = load_lines_table(gpkg_file)

    if lines_df is not None:
        # Initialize an empty list for storing buffered line dictionaries
        buffered_lines_data = []

        for index, row in lines_df.iterrows():
            try:
                # Extract the geometry and attributes for the current row
                fid = index
                gdf_line = row['geometry']
                name_gis = row['name_gis']
                name_user = row['name_user']
                segment_length = int(row['segment_length'])
                segment_width = int(row['segment_width'])
                description = row['description']
                
                log_to_gui(log_widget, f"Processing line {index}, Geometry type: {type(gdf_line)}")
        
                # Create a temporary GeoDataFrame for buffering
                temp_gdf = gpd.GeoDataFrame([{'geometry': gdf_line}], geometry='geometry', crs=crs)
                
                # Reproject to target CRS for buffering
                temp_gdf_projected = temp_gdf.to_crs(target_crs)
                
                # Buffer using the specified width in meters
                temp_gdf_projected['geometry'] = temp_gdf_projected.buffer(segment_width,cap_style=2)
                
                # Reproject buffered geometries back to original CRS
                temp_gdf_buffered = temp_gdf_projected.to_crs(crs)

                if isinstance(temp_gdf_buffered.iloc[0].geometry, (Polygon, MultiPolygon)):
                    log_to_gui(log_widget, "Buffered geometry is a Polygon/MultiPolygon as expected.")
                else:
                    log_to_gui(log_widget, f"Buffered geometry is not a Polygon/MultiPolygon. Actual type: {type(temp_gdf_buffered.iloc[0].geometry)}")

                # Append the buffered geometry and attributes as a new dictionary
                buffered_lines_data.append({
                    'fid': fid, 
                    'name_gis': name_gis, 
                    'name_user': name_user, 
                    'segment_length': segment_length, 
                    'segment_width': segment_width, 
                    'description': description,
                    'geometry': temp_gdf_buffered.iloc[0].geometry  # Access the buffered geometry
                })

                log_to_gui(log_widget, f"Added a buffered version of {name_gis} to GeoPackage.")
                    
            except Exception as e:
                log_to_gui(log_widget, f"Error processing line {index}: {e}")

        # After the loop, create a GeoDataFrame from the list of dictionaries
        if buffered_lines_data:  # Check if there's any data to save
            all_buffered_lines_df = gpd.GeoDataFrame(buffered_lines_data, geometry='geometry', crs=crs)

            all_buffered_lines_df.to_file(gpkg_file, layer="tbl_lines_buffered", driver="GPKG", if_exists='replace')
            
            log_to_gui(log_widget, "All buffered lines added to the database.")

        else:
            log_to_gui(log_widget, "No lines were processed.")
      
    else:
        
        log_to_gui(log_widget, "Lines do not exist. Will create three template lines.")

        create_lines_table_and_lines(gpkg_file, log_widget)


# To cut the polygons which will be made based on a line we will need to 
# create lines which are perpendicular to the line. These lines will later
# work as "knives" to for the polygons (buffered lines). We are making
# sure that the lines are wider then the combined buffer size (length from
# the centerline to the outer buffer).
def create_perpendicular_lines(line_input, segment_width, segment_length):
    # Ensure line_input is a LineString and convert segment_width and segment_length to float
    if not isinstance(line_input, LineString):
        raise ValueError("line_input must be a LineString")
    segment_width = float(segment_width)
    segment_length = float(segment_length)

    # Define the projection transformation: EPSG:4326 to EPSG:4087 (for accurate distance calculations) and back
    transformer_to_4087 = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4087", always_xy=True)
    transformer_to_4326 = pyproj.Transformer.from_crs("EPSG:4087", "EPSG:4326", always_xy=True)
    
    # Reproject the line to EPSG:4087 for accurate distance measurements
    line_transformed = transform(transformer_to_4087.transform, line_input)
    
    # Calculate the number of segments based on the segment_length
    full_length = line_transformed.length
    num_segments = math.ceil(full_length / segment_length)  # Use ceil to ensure covering the entire line
    
    perpendicular_lines = []

    for i in range(num_segments + 1):  # +1 to include the end of the line
        d = min(i * segment_length, full_length)  # Ensure d does not exceed the length of the line
        point = line_transformed.interpolate(d)
        
        # Calculate the slope of the line at this point to find the perpendicular direction
        # This involves getting a small segment around the point and calculating its slope
        if d < segment_width:  # Handle start of line
            segment = LineString([line_transformed.interpolate(0), line_transformed.interpolate(segment_width)])
        elif d > full_length - segment_width:  # Handle end of line
            segment = LineString([line_transformed.interpolate(full_length - segment_width), line_transformed.interpolate(full_length)])
        else:
            segment = LineString([line_transformed.interpolate(d - segment_width/2), line_transformed.interpolate(d + segment_width/2)])
        
        dx = segment.coords[1][0] - segment.coords[0][0]
        dy = segment.coords[1][1] - segment.coords[0][1]
        
        # Calculate perpendicular slope; handle horizontal segments to avoid division by zero
        if dx != 0:
            angle = math.atan2(-dx, dy)  # Angle of the perpendicular to the original segment
        else:
            angle = math.pi / 2 if dy > 0 else -math.pi / 2

        # Define the extended length of the perpendicular line.
        length = (segment_width / 2) * 3  
        
        # Calculate the offsets for the perpendicular line ends
        dx_perp = math.cos(angle) * length
        dy_perp = math.sin(angle) * length
        
        # Create Points for the ends of the perpendicular line
        p1 = Point(point.x - dx_perp, point.y - dy_perp)
        p2 = Point(point.x + dx_perp, point.y + dy_perp)
        
        # Reproject points back to EPSG:4326
        p1_4326 = transform(transformer_to_4326.transform, p1)
        p2_4326 = transform(transformer_to_4326.transform, p2)
        
        # Create the perpendicular line with reprojected points
        perpendicular_line = LineString([p1_4326, p2_4326])
        perpendicular_lines.append(perpendicular_line)
    
    # Combine all perpendicular lines into a MultiLineString
    multi_line = MultiLineString(perpendicular_lines)
    
    return multi_line


# Receives a set of perpendicular lines (knives) and a polygon
# which is buffered lines. The polygon is cut into segments which
# are then returned.
def cut_into_segments(perpendicular_lines, buffered_line_geometry):

    # Ensure the geometries are of the correct type
    if not isinstance(buffered_line_geometry, Polygon):
        raise TypeError("Second call: The buffered_line_geometry must be a Polygon.")
    if not isinstance(perpendicular_lines, MultiLineString):
        raise TypeError("The perpendicular_lines must be a MultiLineString.")

    # Access the constituent LineStrings of the MultiLineString
    line_list = [line for line in perpendicular_lines.geoms]

    # Combine the boundary of the polygon with the perpendicular lines
    combined_lines = unary_union([buffered_line_geometry.boundary] + line_list)

    # Reconstruct polygons from the combined line network
    result_polygons = list(polygonize(combined_lines))

    # Create a GeoDataFrame from the resulting polygons
    segments_gdf = gpd.GeoDataFrame(geometry=result_polygons)

    return segments_gdf


def create_segments_from_buffered_lines(gpkg_file, log_widget):

    # Load clean lines from the tbl_lines-table
    lines_df = load_lines_table(gpkg_file)
    
    # Load data frame for buffered lines (tbl_lines_buffered)
    buffered_lines_gdf = gpd.read_file(gpkg_file, layer='tbl_lines_buffered')

    # Initialize an empty GeoDataFrame for accumulating segments
    all_segments_gdf = gpd.GeoDataFrame(columns=['name_gis', 'name_user', 'segment_length', 'geometry'])

    # Initialize a counter for segment_id generation
    segment_id_counter = {}

    for index, row in lines_df.iterrows():
        line_input = row.geometry
        name_gis = row['name_gis']
        name_user = row['name_user']
        segment_width = row['segment_width']
        segment_length = row['segment_length']

        # Initialize or update the counter for the current name_gis
        if name_gis not in segment_id_counter:
            segment_id_counter[name_gis] = 1

        # Generate perpendicular lines for the current line
        perpendicular_lines = create_perpendicular_lines(line_input, segment_width, segment_length)

        # Find matching buffered lines by 'name_gis'
        matches = buffered_lines_gdf[buffered_lines_gdf['name_gis'] == name_gis]

        log_to_gui(log_widget, f"Creating segments for: {name_gis}")

        # If there are any matching records, process each match
        for _, match_row in matches.iterrows():

            buffered_line_geometry = match_row.geometry
            
            # Ensure the geometries are of the correct type
            if not isinstance(buffered_line_geometry, Polygon):
                log_to_gui(log_widget, "Geometry is not a Polygon. Skipping.")
                continue  # Skip non-Polygon geometries

            # Create segments using the perpendicular lines and the matching buffered line geometry
            segments_gdf = cut_into_segments(perpendicular_lines, buffered_line_geometry)

            # Filter out invalid geometries
            valid_segments_gdf = segments_gdf[segments_gdf.is_valid]

            # Check if there are valid segments to process
            if valid_segments_gdf.empty:
                log_to_gui(log_widget, f"No valid segments created for {name_gis}. Skipping.")
                continue  # Skip if no valid segments


            # Assign segment_id and increment the counter for each segment
            valid_segments_gdf['segment_id'] = [f"{name_gis}_{segment_id_counter[name_gis]+i}" for i in range(len(valid_segments_gdf))]
            segment_id_counter[name_gis] += len(valid_segments_gdf)

             # Add attributes to the segments GeoDataFrame
            valid_segments_gdf['name_gis'] = name_gis
            valid_segments_gdf['name_user'] = name_user
            valid_segments_gdf['segment_length'] = segment_length

             # Accumulate the created segments
            all_segments_gdf = pd.concat([all_segments_gdf, valid_segments_gdf], ignore_index=True)

    # After processing all lines, check if there are any segments to save
    if not all_segments_gdf.empty:
        # Set the CRS for all_segments_gdf if it's not already set (assuming it's the same as lines_df)
        all_segments_gdf.crs = lines_df.crs

        # Export the accumulated segments to 'tbl_segments' in one batch
        all_segments_gdf.to_file(gpkg_file, layer="tbl_segments", driver="GPKG", if_exists="replace")

        log_to_gui(log_widget, "All segments have been accumulated and saved to 'tbl_segments'.")
    else:
        log_to_gui(log_widget, "No segments were created.")


def edit_lines():
    try:
        subprocess.run(["python", "08_edit_lines.py"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # If import.py fails, try running import.exe
            subprocess.run(["08_edit_lines.exe"], check=True)
        except subprocess.CalledProcessError:
            log_to_gui(log_widget, "Failed to execute edit lines script")


# Function to perform intersection with geocode data
def intersection_with_geocode_data(asset_df, segment_df, geom_type, log_widget):

    log_to_gui(log_widget, f"Processing {geom_type} intersections")

    asset_filtered = asset_df[asset_df.geometry.geom_type == geom_type]

    if asset_filtered.empty:
        return gpd.GeoDataFrame()

    return gpd.sjoin(segment_df, asset_filtered, how='inner', predicate='intersects')


# Function to perform intersection based on geometry type
def intersection_with_segments(asset_data, segment_data, log_widget):

    try:
        # Perform spatial intersection
        intersected_data = gpd.overlay(asset_data, segment_data, how='intersection')
        
        return intersected_data
    except Exception as e:

        log_to_gui(log_widget, f"Error in intersection: {str(e)}")

        return pd.DataFrame()


def build_stacked_data(gpkg_file, log_widget):

    log_to_gui(log_widget, "Building tbl_segment_stacked.")

    update_progress(10)  # Indicate start

    # Read necessary layers from the GeoPackage
    asset_data = gpd.read_file(gpkg_file, layer='tbl_asset_object')

    update_progress(15)

    asset_group_data = gpd.read_file(gpkg_file, layer='tbl_asset_group')

    update_progress(25)  # Progress after reading asset group data

    # Merge asset group data with asset data
    asset_data = asset_data.merge(asset_group_data[['id', 'name_gis_assetgroup', 'total_asset_objects', 'importance', 'susceptibility', 'sensitivity']], 
                                  left_on='ref_asset_group', right_on='id', how='left')

    lines_data = gpd.read_file(gpkg_file, layer='tbl_lines')

    update_progress(29)

    segments_data = gpd.read_file(gpkg_file, layer='tbl_segments')

    update_progress(35)

    # Ensure 'name_gis' is included from both DataFrames but rename them for clarity
    lines_data_renamed  = lines_data.rename(columns={'name_gis': 'lines_name_gis'})
    segments_related    = segments_data.merge(lines_data_renamed[['lines_name_gis']], left_on='name_gis', right_on='lines_name_gis', how='left', suffixes=('_seg', '_line'))
    
    update_progress(40)

    point_intersections = intersection_with_geocode_data(asset_data, segments_related, 'Point', log_widget)

    update_progress(45)  # Progress after point intersections

    line_intersections = intersection_with_geocode_data(asset_data, segments_related, 'LineString', log_widget)

    update_progress(49)  # Progress after line intersections

    polygon_intersections = intersection_with_geocode_data(asset_data, segments_related, 'Polygon', log_widget)

    update_progress(50)  # Progress after polygon intersections

    segment_intersections = pd.concat([point_intersections, line_intersections, polygon_intersections])

    update_progress(60)
    
    # Drop the unnecessary columns, adjust according to your final table requirements
    segment_intersections.drop(columns=['id_x', 'id_y', 'lines_name_gis'], inplace=True)
    
    # Assuming 'segment_intersections' is the GeoDataFrame you're trying to write
    segment_intersections.reset_index(drop=True, inplace=True)  # Resets the index
    segment_intersections['fid'] = segment_intersections.index  # Uses the new index as 'fid'

    # Write the intersected data to a new layer in the GeoPackage
    segment_intersections.to_file(gpkg_file, layer='tbl_segment_stacked', driver='GPKG', if_exists='replace')

    log_to_gui(log_widget, "Data processing completed.")
    
    update_progress(70)  # Final progress


# Create tbl_segment_flat by reading out values from tbl_segment_stacked
def build_flat_data(gpkg_file, log_widget):
    
    log_to_gui(log_widget, "Building tbl_segmentflat ...")

    tbl_segmemt_stacked = gpd.read_file(gpkg_file, layer='tbl_segment_stacked')

    update_progress(60)

    # Aggregation functions
    aggregation_functions = {
        'importance': ['min', 'max'],
        'sensitivity': ['min', 'max'],
        'susceptibility': ['min', 'max'],
        'name_gis': 'first',
        'segment_id': 'first',
        'asset_group_name': lambda x: ', '.join(x.unique()),  # Joining into a comma-separated string
        'geometry': 'first'
    }

    # Group by 'code' and aggregate
    tbl_segment_flat = tbl_segmemt_stacked.groupby('segment_id').agg(aggregation_functions)

    # Flatten the MultiIndex columns
    tbl_segment_flat.columns = ['_'.join(col).strip() for col in tbl_segment_flat.columns.values]

    # Rename columns after flattening
    renamed_columns = {
        'name_gis_first': 'name_gis',
        'importance_min': 'importance_min',
        'importance_max': 'importance_max',
        'sensitivity_min': 'sensitivity_min',
        'sensitivity_max': 'sensitivity_max',
        'susceptibility_min': 'susceptibility_min',
        'susceptibility_max': 'susceptibility_max',
        'asset_group_name_<lambda>': 'asset_group_names',
        'ref_asset_group_nunique': 'assets_total',
        'geometry_first': 'geometry'
    }

    tbl_segment_flat.rename(columns=renamed_columns, inplace=True)

    # Convert to GeoDataFrame
    tbl_segment_flat = gpd.GeoDataFrame(tbl_segment_flat, geometry='geometry')

    # Reset index to make 'code' a column
    tbl_segment_flat.reset_index(inplace=True)
    
    # Drop the unnecessary columns, adjust according to your final table requirements
    tbl_segment_flat.drop(columns=['segment_id_first'], inplace=True)

    # Save tbl_flat as a new layer in the GeoPackage
    tbl_segment_flat.to_file(gpkg_file, layer='tbl_segment_flat', driver='GPKG')
    
    log_to_gui(log_widget, "Completed flat segments...")


def classify_data(log_widget, gpkg_file, process_layer, column_name, config_path):
    # Load classification configuration
    classification = read_config_classification(config_path)

    # Load geopackage data
    gdf = gpd.read_file(gpkg_file, layer=process_layer)

    # Function to classify each row
    def classify_row(row):
        for label, value_range in classification.items():
            if row[column_name] in value_range:
                return label
        return 0  # or any default value

    new_column_name = column_name + "_code"
    # Apply classification
    gdf[new_column_name] = gdf.apply(lambda row: classify_row(row), axis=1)

    log_to_gui(log_widget, f"Updated codes for: {process_layer} - {column_name} ")
    update_progress(97)

    # Save the modified geopackage
    gdf.to_file(gpkg_file, layer=process_layer, driver='GPKG')


def build_flat_and_stacked(gpkg_file, log_widget):

    # Process and create tbl_stacked
    build_stacked_data(gpkg_file, log_widget)

    # Process and create tbl_flat
    build_flat_data(gpkg_file, log_widget) 
    
    classify_data(log_widget, gpkg_file, 'tbl_segment_flat', 'sensitivity_min', config_file)
    classify_data(log_widget, gpkg_file, 'tbl_segment_flat', 'sensitivity_max', config_file)
    classify_data(log_widget, gpkg_file, 'tbl_stacked', 'sensitivity', config_file)

    log_to_gui(log_widget, "Data processing and aggregation completed.")
    update_progress(100)


def process_all(gpkg_file, log_widget):

    # Process and create tbl_stacked
    process_and_buffer_lines(gpkg_file, log_widget)
    
    update_progress(30)  # Indicate start

    create_segments_from_buffered_lines(gpkg_file, log_widget)
    
    update_progress(60)  # Indicate start

    build_flat_and_stacked(gpkg_file, log_widget)
    
    update_progress(90)  # Indicate start

    log_to_gui(log_widget, "Data processing and aggregation completed.")

    update_progress(100)


def exit_program():
    root.destroy()

#####################################################################################
#  Main
#

# Load configuration settings
config_file             = 'config.ini'
config                  = read_config(config_file)
input_folder_asset      = config['DEFAULT']['input_folder_asset']
input_folder_geocode    = config['DEFAULT']['input_folder_geocode']
gpkg_file               = config['DEFAULT']['gpkg_file']
ttk_bootstrap_theme     = config['DEFAULT']['ttk_bootstrap_theme']

# Create the user interface using ttkbootstrap
root = ttk.Window(themename=ttk_bootstrap_theme)
root.title("Line processing")
root.geometry("850x700")

# Define button sizes and width
button_width = 18   
button_padx  = 7
button_pady  = 7

# Main frame defined
main_frame = tk.Frame(root)
main_frame.pack(fill='both', expand=True, pady=10)

# Create a LabelFrame for the log output
log_frame = ttk.LabelFrame(main_frame, text="Log Output", bootstyle="info") 
log_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a log widget inside the LabelFrame
log_widget = scrolledtext.ScrolledText(log_frame, height=10)
log_widget.pack(fill=tk.BOTH, expand=True)

# Create a frame to hold the progress bar and the label
progress_frame = tk.Frame(main_frame)
progress_frame.pack(pady=5)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=200, mode="determinate", variable=progress_var, bootstyle='info')
progress_bar.pack(side=tk.LEFT)

# Label for displaying the progress percentage
progress_label = tk.Label(progress_frame, text="0%", bg="light grey")
progress_label.pack(side=tk.LEFT, padx=5)

buttons_frame = tk.Frame(main_frame)
buttons_frame.pack(side='left', fill='y', padx=20, pady=5)  # Corrected this line

# Create the Initiate button
initiate_button = ttk.Button(buttons_frame, text="Initiate", command=lambda: create_lines_table_and_lines(gpkg_file, log_widget), width=button_width)
initiate_button.grid(row=0, column=0, padx=button_padx, pady=button_pady)

# Explanatory label next to the Initiate-button
explanatory_label = tk.Label(buttons_frame, text="Press this button in case you need help\nto create sample lines lines.", bg="light grey", anchor='w')
explanatory_label.grid(row=0, column=1, padx=button_padx, sticky='w')  # Align to the west (left)

# Button for editing lines. This opens a sub-process to set up the line generation.
edit_lines_button = ttk.Button(buttons_frame, text="Edit lines", command=edit_lines, width=button_width, bootstyle="secondary")
edit_lines_button.grid(row=1, column=0, padx=button_padx, pady=button_pady)

# Explanatory label next to the "Edit lines" button
explanatory_label = tk.Label(buttons_frame, text="Remember that you can import and edit\nthese lines by opening the database using QGIS.", bg="light grey", anchor='w')
explanatory_label.grid(row=1, column=1, padx=button_padx, sticky='w')  # Align to the west (left)

# Create the Process and buffer button
process_button = ttk.Button(buttons_frame, text="Process", command=lambda: process_all(gpkg_file, log_widget), width=button_width)
process_button.grid(row=2, column=0, padx=button_padx, pady=button_pady)

# Explanatory label next to the Process-button
explanatory_label = tk.Label(buttons_frame, text="Process all things related to lines.", bg="light grey", anchor='w')
explanatory_label.grid(row=2, column=1, padx=button_padx, sticky='w')  # Align to the west (left)

# Adjust the exit button to align it to the right
exit_btn = ttk.Button(buttons_frame, text="Exit", command=exit_program, width=button_width, bootstyle="warning")
exit_btn.grid(row=3, column=4, pady=button_pady, sticky='e')  # Align to the right side

root.mainloop()