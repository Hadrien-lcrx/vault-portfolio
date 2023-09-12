def get_problemkeys(filename):
    """
    Takes in a dataset in XML format, parses it and returns a list with the values of tags with problematic characters.
    """
    problemchars_list = []
    for _, element in ET.iterparse(filename):
        if element.tag == 'tag':
            if problemchars.search(element.attrib['k']):
                problemchars_list.append(element.attrib['k'])
    return problemchars_list

print(get_problemkeys(dataset))