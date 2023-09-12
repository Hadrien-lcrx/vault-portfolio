import xml.etree.cElementTree as ET

from 06abbr_mapping import expected, abbr_mapping
from 08rest_mapping import typo_mapping, numbers_mapping, po_box_mapping
from 10char_mapping import char_mapping

expected.extend(['Center', 'Circle', 'Driveway', 'Mall'])


def typo_correct(street_name, street_type):
    if type(typo_mapping[street_type]) == type('string'):
            name = typo_mapping[street_type]
    elif type(typo_mapping[street_type]) == type({}):
        if '2nd' in street_name:
            name = 'Sidney Street'
            # add attribute addr:floor '2nd Floor' typo_mapping[street_type]['Sidney Street']
        elif '18' in street_name:
            name = 'First Street'
            # add attribute addr:floor '18th Floor' typo_mapping[street_type]['First Street']
        elif '5th' in street_name:
            name = 'Boylston Street'
            # add attribute addr:floor '5th Floor' typo_mapping[street_type]['Boylston Street']
        elif street_type == 'LEVEL':
            name = 'Lomasney Way'
            # add attribute addr:floor 'Roof Level' typo_mapping[street_type]['Lomasney Way']
        elif 'Two Center' in street_name:
            name = 'Center Plaza'
            # add attribute addr:housenumber '2' typo_mapping[street_type]['Two Center Plaza']['Center Plaza']
        else:
            for key in typo_mapping[street_type]:
                name = key
                # add attribute addr:housenumber = value
    return name
        
def numbers_correct(street_name, street_type):
    if 'Suite' in street_name:
        for key in numbers_mapping[street_type]:
            name = key
            # print (name)
            # add attribute addr:suitenumber = value
    else:
        for key in numbers_mapping[street_type]:
            name = key
            # add attribute addr:housenumber = value
    return name

def char_correct(street_name, street_type):
    print (street_name, street_type)
    if street_name + ' ' + street_type == 'Church Street, Harvard Square':
        for key in char_mapping[street_name + ' ' + street_type]:
            name = key
    elif street_name + ' ' + street_type == 'Massachusetts Ave; Mass Ave':
        name = char_mapping[street_name + ' ' + street_type]
    return name

def audit_abbreviations(filename):
    problem_street_types = defaultdict(set)
    for event, elem in ET.iterparse(filename):
        if is_street_name(elem):
            expected_street_type(problem_street_types, elem.attrib['v'])
    return problem_street_types

def expected_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)
        
def update_name(name):
    street_type = name.split(' ')[-1]
    street_name = name.rsplit(' ', 1)[0]
    if (street_name + ' ' + street_type) in char_mapping:
        name = char_correct(street_name, street_type)
    elif street_type in abbr_mapping:
        name = street_name + ' ' + abbr_mapping[street_type]
    elif street_type in typo_mapping:
        name = typo_correct(street_name, street_type)
    elif street_type in numbers_mapping:
        name = numbers_correct(street_name, street_type)
    
    elif street_type in po_box_mapping:
        name = po_box_mapping[street_type]
    return name
    
def run_updates(filename):
    st_types = audit_abbreviations(dataset)
    for st_type, ways in st_types.items():
        for name in ways:
            better_name = update_name(name)
            if better_name != name:
                corrected_names[name] = better_name
    return corrected_names
            
corrected_names = {}           
corrected_names = run_updates(dataset)
print_sorted_dict(corrected_names, "%s: %s")