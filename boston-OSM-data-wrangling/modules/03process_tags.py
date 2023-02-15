import xml.etree.cElementTree as ET
import re

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')


def key_type(element, keys):
    """
    Takes in an XML element (<node />, <way />) and a dictionary of keys.
    If the element is a <tag />, its ['k'] attribute is analyzed,
    the dictionary keys and lists are incremented accordingly,
    - lower: valid tags that only contain lowercase letters
    - lower_colon: valid tags that contain lowercase letter with one or more colons in their name
    - problemchars: tags with problematic characters
    - other: tags that don't fall in the previous three categories
    """
    
    if element.tag == 'tag':
        if lower.search(element.attrib['k']):
            keys['lower'] +=1
        elif lower_colon.search(element.attrib['k']):
            keys['lower_colon'] += 1
        elif problemchars.search(element.attrib['k']):
            keys['problemchars'] += 1
        else:
            keys['other'] += 1
    return keys


def process_tags(filename):
    """
    Takes in a dataset in XML format, parses it, and executes the function key_type() for each element.
    Returns a dictionary with the count of different types of keys.
    """
    keys = {"lower": 0, "lower_colon": 0, "problemchars": 0, "other": 0}
    for _, element in ET.iterparse(filename):
        keys = key_type(element, keys)

    return keys

tag_type_dict = process_tags(dataset)
print (tag_type_dict)