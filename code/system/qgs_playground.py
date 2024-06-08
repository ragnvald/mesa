import xml.etree.ElementTree as ET
import uuid

def add_layer_to_qgs(qgs_file, layer_name, gpkg_path, filter_query=None):
    tree = ET.parse(qgs_file)
    root = tree.getroot()

    layer_tree_group = root.find('.//layer-tree-group')
    custom_order = layer_tree_group.find('.//custom-order')
    snapping_settings = root.find('.//snapping-settings/individual-layer-settings')

    existing_layer_id = None
    for layer in layer_tree_group.findall('layer-tree-layer'):
        if layer.attrib['name'] == f"{layer_name}":
            existing_layer_id = layer.attrib['id']
            break

    if not existing_layer_id:
        layer_id = f"{layer_name}_{uuid.uuid4()}"
        
        # Define the source string with an optional filter
        source_str = f"{gpkg_path}|layername={layer_name}"
        if filter_query:
            source_str += f"|subset=&quot;{filter_query}&quot;"

        # Define new layer
        new_layer = ET.SubElement(layer_tree_group, 'layer-tree-layer', {
            'checked': 'Qt::Unchecked',
            'legend_split_behavior': '0',
            'legend_exp': '',
            'expanded': '1',
            'patch_size': '-1,-1',
            'source': source_str,
            'providerKey': 'ogr',
            'id': layer_id,
            'name': f"{layer_name}"
        })

        if custom_order is not None:
            custom_order_item = ET.Element('item')
            custom_order_item.text = layer_id
            custom_order.insert(0, custom_order_item)
            custom_order_item.tail = "\n"  # Add a newline after the item element


        # Add new layer to snapping settings
        new_snapping_setting = ET.SubElement(snapping_settings, 'layer-setting', {
            'tolerance': '12',
            'units': '1',
            'type': '1',
            'maxScale': '0',
            'id': layer_id,
            'enabled': '0',
            'minScale': '0'
        })

        new_snapping_setting.tail = "\n"  # Add a newline for layout   
        
        tree.write(qgs_file)
        return layer_id
    else:
        # Check if snapping setting exists for this layer
        setting_exists = any(setting.attrib.get('id') == existing_layer_id for setting in snapping_settings.findall('layer-setting'))
        if not setting_exists:
            # Add new layer to snapping settings
            new_snapping_setting = ET.SubElement(snapping_settings, 'layer-setting', {
                'tolerance': '12',
                'units': '1',
                'type': '1',
                'maxScale': '0',
                'id': existing_layer_id,
                'enabled': '0',
                'minScale': '0'
            })

            tree.write(qgs_file)

        return existing_layer_id

# Example usage
qgs_file_path = 'qgis/mesa/mesa.qgs'
layer_name = 'tbl_stackeiii'
gpkg_file_path = 'output/mesa.gpkg'
filter_query = "ref_geocodegroup = 1"
layer_id = add_layer_to_qgs(qgs_file_path, layer_name, gpkg_file_path, filter_query)
print("Layer ID:", layer_id)