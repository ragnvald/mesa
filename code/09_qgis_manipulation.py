import sys

# Path to the QGIS installation
# These paths will vary depending on where and how QGIS is installed on your system
qgis_path = "C:/OSGeo4W/apps/qgis/python"
qgis_plugins_path = "C:/OSGeo4W/apps/qgis/python/plugins"

# Add paths to sys.path
sys.path.append(qgis_path)
sys.path.append(qgis_plugins_path)

from qgis.core import QgsProject, QgsApplication



from qgis.core import QgsProject, QgsApplication

# Supply path to qgis install location
QgsApplication.setPrefixPath("C:\OSGeo4W\bin\qgis-bin.exe", True)

# Create a reference to the QgsApplication
app = QgsApplication([], False)

# Initialize the application
app.initQgis()

# Load and manipulate your project
project = QgsProject.instance()
project.read('/qgis/mesa.qgz')

# Perform your operations here
# e.g., adding layers, changing layer properties, etc.

# Save changes to the project
project.write()

# Exit the application
app.exitQgis()
app.exit()