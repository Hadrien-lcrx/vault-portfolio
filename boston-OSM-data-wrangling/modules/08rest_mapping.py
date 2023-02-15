expected.extend(['Center', 'Circle', 'Driveway', 'Mall'])

typo_mapping = {  'Albany': 'Albany Street',
                  'Boylston': 'Boylston Street',
                  'Cambrdige': 'Cambridge Street',
                  'Dartmouth': 'Dartmouth Street',
                  'Elm': 'Elm Street',
                  'Hampshire': 'Hampshire Street',
                  'Holland': 'Holland Street',
                  'Newbury': 'Newbury Street',
                  'Lafayette': 'Lafayette Avenue',
                  'Longwood': 'Longwood Avenue',
                  'Winsor': 'Winsor Avenue',
                  'Pasteur': 'Avenue Louis Pasteur',
                  'Corner': 'Webster Street',
                  'Building': {'South Market Street': 4},
                  'B Street Ext': 'B Street',
                  'Fellsway': 'Fellsway Parkway',
                  'Fenway': 'Fenway Park',
                  'Floor': {'Boylston Street': '5th Floor'},
                  'Garage': 'Stillings Street',
                  'H': 'Hancock Street',
                  'Hall': {'Faneuil Hall Square': 1},
                  'LEVEL': {'Lomasney Way': 'Roof Level'},
                  'Market': 'Faneuil Hall Square',
                  'Plaza': { 'Two Center Plaza': {'Center Plaza': 2}},
                  'South': 'Charles Street South',
                  'floor': { 'Sidney Street': '2nd Floor',
                             'First Street': '18th Floor'},
                  'Windsor': 'Windsor Place'
               }

numbers_mapping = { '#12': {'Harvard Street': 12},
                    '#1302': {'Cambridge Street': 1302},
                    '#501': {'Bromfield Street': 501},
                    '104': {'Mill Street': 'Suite 104'},
                    '1100': {'First Street': 'Suite 1100'},
                    '1702': { 'Franklin Street': 'Suite 1702'},
                    '3': {'Kendall Square': 'Suite B3201'},
                    '303': {'First Street': 'Suite 303'},
                    '6': {'Atlantic Avenue': 700 }
                  }

po_box_mapping = { '846028': 'Albany Street'}

expected = sorted(expected)
expected