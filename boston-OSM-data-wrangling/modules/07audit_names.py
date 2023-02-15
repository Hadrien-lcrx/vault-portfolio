typo_full_names = {}

def audit_street_name(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if (all_types[street_type] < 20) and (street_type not in expected) and (street_type not in abbr_mapping):
            if street_type in typo_full_names:
                typo_full_names[street_type].append(street_name)
            else:
                typo_full_names.update({ street_type:[street_name] })

def audit_name(filename):
    for event, elem in ET.iterparse(filename):
        if is_street_name(elem):
            audit_street_name(street_types, elem.attrib['v'])    
    # print_sorted_dict(street_types)
    return typo_full_names

audit_name(dataset)