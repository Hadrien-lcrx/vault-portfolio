## Introduction

This is a machine learning project trying to predict Persons of Interest in the Enron fraud case based on financial and email correspondence data.

**This notebook can be read more comfortably [here](https://hadrien-lcrx.github.io/notebooks/Enron_POI_Identification.html).**

**You can also access my portfolio and other projects [here](https://hadrien-lcrx.github.io).**


## Files

- `Enron POI Identification.ipynb` -- The notebook containing the analysis
- `enron61702insiderpay.pdf` -- Table containing financial information for each person in the dataset
- `final_project_dataset.pkl` -- The investigated dataset
- `my_*.pkl` -- the classifier, dataset and feature list dumped at the end of the analysis
- `poi_email_addresses.py` -- List of email addresses for each person in the dataset
- `poi_id.py` -- A python file version of the notebook
- `poi_names.txt` —- A list of possible Persons of Interest identified from press, and assessment of their presence in our dataset
- `tester.py` -- A python module testing the final selected machine learning algorithm's performance


## Requirements

This notebok was written in Python 3 and used the following libraries:
- sklearn
- pandas
- numpy
- matplotlib
- seaborn
