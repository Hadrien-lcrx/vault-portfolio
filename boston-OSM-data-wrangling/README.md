## Introduction

This is a data wrangling project cleaning the names of Boston Metropolitan Area's streets:
- [Boston on OpenStreetMap](https://www.openstreetmap.org/relation/2315704)
- [Boston Metro extract](https://mapzen.com/data/metro-extracts/metro/boston_massachusetts/)

**This notebook can be read more comfortably [here](https://hadrien-lcrx.github.io/notebooks/Boston_Data_Wrangling.html).**

**You can also access my portfolio and other projects [here](https://hadrien-lcrx.github.io).**


## Source files

- `Data Wrangling | Boston, Massachusetts - OpenStreetMap.ipynb` -- The notebook containing the analysis
- `data/boston_massachusetts.osm` -- The original dataset downloaded from OpenStreetMap
- `*.csv` -- The .csv files created during the data wrangling process
- `boston.db` -- The SQL database created during the data wrangling process
- - `schema.py` — Defines the format of the final dictionaries (that will define the structure of our final csvs)
- `modules` -- The folder containing the whole notebook presented as Python 14 Python modules to execute.


## Modules files

This Data Wrangling project was written and thought for Jupyter.
It's been turned into scripts and modules for submission requirements purposes.

Each module starts with a number, indicating the order in which to execute them.

- `01count_tags.py` returns a dictionary indicating the number of each tags
- `02get_users.py` returns an integer, the number of unique users who built the dataset
- `03process_tags.py` returns a dictionary indicating the number of tax corresponding to 4 pre-defined regex
- `04get_problemkeys.py` returns the tag keys that might prove problematic
- `05audit.py` returns a dictionary containing all the street types in the data set
- `06abbr_ampping.py` defines dictionaries with valid street types, and expected corrections for invalid street types
- `07audit_names.py` returns a dictionary witg problematic street names that are not in the expected dictionary and lists their occurences, to help decide if they should be valid or invalid
- `08rest_mapping.py` defines new mapping dictionaries with valid names and expected corrections, based on the findings of `07audit_names.py`
- `09get_problem_names.py` makes one last run at identifying incorrect names with special characters and returns a list of potentially problematic names
- `10char_mapping.py` builds one last mapping dictionnary based on the findings of `09get_problem_names.py`
- `11update_names` cleans the data and returns all corrected street names in a dictionary mapping the new correct version to the previous incorrect version
- `12schema.py` defines the format of the final dictionaries that will define the structure of our final csvs
- `13process_map.py` creates 5 csv files with corrected data, that can be used to build a SQL database
- `14creating_database.py`creates the database with 5 tables corresponding to our 5 csv files
- `OpenStreetMap Data Wrangling | Case Study - Auditing & Cleaning.pdf` details the auditing and cleaning process
- `OpenStreetMap Data Wrangling | Case Study - Querying.pdf` details the querying process and analyzes the results 


## Requirements

This notebok was written in Python 3 and used the following libraries:
- xml
- pprint
- re
- collections
- csv
- codecs
- cerberus
- schema
- sqlite3