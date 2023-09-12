import xml.etree.cElementTree as ET
import re

name_problem_chars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\t\r\n]')

def get_problem_names(filename):
    """
    Takes in a dataset in XML format, parses it and returns a list with the values of tags with problematic characters.
    """
    problemchars_list = []
    for _, element in ET.iterparse(filename):
        if is_street_name(element):
            if name_problem_chars.search(element.attrib['v']):
                problemchars_list.append(element.attrib['v'])
    return problemchars_list

print(get_problem_names(dataset))