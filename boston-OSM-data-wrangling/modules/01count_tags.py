import xml.etree.cElementTree as ET
import pprint

dataset = "data/boston_massachusetts.osm"

def count_tags(filename):
    """
    Takes in a dataset in XML format.
    Returns a dictionary of the tags and the number of each tag.
    """
    tag_dict = {}
    for event, elem in ET.iterparse(filename):
        if elem.tag not in tag_dict:
            tag_dict[elem.tag] = 1
        else:
            tag_dict[elem.tag] += 1

    return tag_dict

print(count_tags(dataset))