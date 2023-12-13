import tkinter as tk
import geopandas as gpd
import fiona

class StatsApp:
    def __init__(self, root, geopackage):
        self.root = root
        self.geopackage = geopackage
        self.init_ui()
        self.update_stats()

    def init_ui(self):
        self.root.title("Statistics Panel")
        self.label = tk.Label(self.root, text="Initializing...", font=("Helvetica", 12))
        self.label.pack(pady=20)

    def update_stats(self):
        # List available layers in the geopackage
        layer_names = fiona.listlayers(self.geopackage)
        print("Available layers in the geopackage:", layer_names)

        # Load different tables from the geopackage
        # Make sure the layer names match with those in the list
        tbl_asset_group = gpd.read_file(self.geopackage, layer='tbl_asset_group')
        tbl_asset_object = gpd.read_file(self.geopackage, layer='tbl_asset_object')
        tbl_geocode_group = gpd.read_file(self.geopackage, layer='tbl_geocode_group')
        tbl_geocode_object = gpd.read_file(self.geopackage, layer='tbl_geocode_object')

        # Calculate the required statistics
        asset_layer_count = len(tbl_asset_group)
        asset_feature_count = len(tbl_asset_object)
        geocode_layer_count = len(tbl_geocode_group)
        geocode_object_count = len(tbl_geocode_object)

        # Update the label text with the new statistics
        stats_text = (f"Asset Layers: {asset_layer_count}, Total Features: {asset_feature_count}\n"
                      f"Geocode Layers: {geocode_layer_count}, Total Geocode Objects: {geocode_object_count}")

        self.label.config(text=stats_text)
        # Schedule the next update
        self.root.after(20000, self.update_stats)

# Load the geopackage file
geopackage = "output/mesa.gpkg"  # Ensure this path is correct

# Create the Tkinter window
root = tk.Tk()
app = StatsApp(root, geopackage)

# Run the application
root.mainloop()
