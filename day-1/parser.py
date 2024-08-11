# parser.py
import os

from json_handler import read_json, modify_json, write_json
from yaml_handler import read_yaml, modify_yaml, write_yaml
from xml_handler import read_xml, modify_xml, write_xml





readable_json = read_json('conf.json')
print(readable_json)

readable_xml = read_xml('conf.xml')
print(readable_xml)

xml_file = 'conf.xml'

# Use the function
modify_xml(xml_file, 'create', 'randon', 'new_setting')  # Create a setting
modify_xml(xml_file, 'create', 'teet', '323')  # Create a setting
modify_xml(xml_file, 'create', 'teet2', '45354')  # Create a setting
modify_xml(xml_file, 'modify', 'randon', 'modified')  # Modify the newly created setting
modify_xml(xml_file, 'delete', 'teet')  # Delete the modified setting


