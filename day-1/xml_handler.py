import xml.etree.ElementTree as ET

def read_xml(file_path):
    # Parse XML file
    tree = ET.parse(file_path)

    # Get the root element of the XML document
    root = tree.getroot()

    # Iterate over child elements of root
    for child in root:
        print(f"Child tag: {child.tag}, Child attribute: {child.attrib}")

        # Iterate over grandchild elements
        for grandchild in child:
            print(f"Grandchild tag: {grandchild.tag}, Grandchild text: {grandchild.text}")



def modify_xml(file_path, action, setting_tag, new_value=None):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Find the 'settings' element
    settings = root.find('settings')

    if action == 'create':
        # Create a new setting element
        new_element = ET.Element(setting_tag)
        # Set the text of the new_element
        new_element.text = new_value
        # Attach new_element to settings
        settings.append(new_element)

    else:
        for element in settings:
            # If the action is 'delete' and the element tag matches, remove it
            if action == 'delete' and element.tag == setting_tag:
                settings.remove(element)

            # If the action is 'modify' and the element tag matches, modify the element text
            if action == 'modify' and element.tag == setting_tag:
                element.text = new_value

    # Write the modifications back to the file
    tree.write(file_path)


def write_xml(file_path, data):
    # code to write json back to file
    pass
