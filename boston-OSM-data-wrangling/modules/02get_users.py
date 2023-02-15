def get_users(filename):
    users = set()
    for _, element in ET.iterparse(filename):
        for att in element.attrib:
            if att == 'uid':
                if element.attrib['uid'] not in users:
                    users.add(element.attrib['uid'])

    return users

print(len(get_users(dataset)))