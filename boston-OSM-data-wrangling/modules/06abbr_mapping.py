expected = ['Artery', 'Avenue', 'Boulevard', 'Broadway', 'Commons', 'Court', 'Drive', 'Lane', 'Park', 'Parkway',
            'Place', 'Road', 'Square', 'Street', 'Terrace', 'Trail', 'Turnpike', 'Wharf',
            'Yard']

abbr_mapping = { 'Ave': 'Avenue',
                  'Ave.': 'Avenue',
                  'Ct': 'Court',
                  'Dr': 'Drive',
                  'HIghway': 'Highway',
                  'Hwy': 'Highway',
                  'Pl': 'Place',
                  'place': 'Place',
                  'Pkwy': 'Parkway',
                  'Rd': 'Road',
                  'rd.': 'Road',
                  'Sq.': 'Square',
                  'St': 'Street',
                  'st': 'Street',
                  'ST': 'Street',
                  'St,': 'Street',
                  'St.': 'Street',
                  'street': 'Street',
                  'Street.': 'Street'
                }