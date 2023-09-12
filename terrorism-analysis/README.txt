CSE 6242 – Team #45: Team GTD
===

1. Description

This package includes all code and software used to create the Global Terrorism Database Explorer project. It also includes notebooks used to pre-process and predict the number of casualties per incident.

2. Installation

2.1. Pre-Requisites

	1. Get NPM here – https://www.npmjs.com/get-npm: (On Debian, `curl -fsSL https://deb.nodesource.com/setup_19.x | bash - && apt-get install -y nodejs`)
	2. Required Python Libraries:
		1. pandas
		2. numpy
		3. matplotlib
		4. seaborn
		5. sklearn
		6. jupyter
	3. Data Files
		1. GTD – https://www.start.umd.edu/data-tools/global-terrorism-database-gtd
			1. Save the XLSX file as "globalterrorismdb_0919dist.xlsx" in this directory.
		2. GTD JSON – https://gtd-public.s3.amazonaws.com/gtd.json
			1. Save the converted JSON file in gtdex-api/resources/gtd.json

2.2. ElasticNet Variable Selection & Linear Regression
	
	1. Navigate to the CODE folder and start Jupyter notebook
		
		```
		cd ./CODE
		jupyter notebook .
		```
	2. When the notebook is started, navigate to Elastic net for variable selection-4.ipynb
	3. Step through the notebook

2.3. GTD Explorer API

	1. In terminal or command prompt, navigate to gtdex-api
	
		```
		cd CODE/gtdex-api
		```
	2. Install all packages

		```
		npm install
		```
	3. Start the API Server

		```
		npm run start
		```
	4. Test the API Server by navigating to http://localhost:3000/api/country

2.4 GTD Explorer

	1. Start the API Server in section 2.3
	2. In terminal or command prompt, navigate to gtdex

		```
		cd CODE/gtdex
		```
	3. Install all packages

		```
		npm install
		```
	4. Start the Front End Server

		```
		npm run serve
		```
	5. Test the Server by navigating to http://localhost:8080

3. Execution

	1. Complete all pre-requisites and ensure that the API and Front End server are running
	2. Open a browser and navigate to http://localhost:8080
	3. In the Left Panel, change the title of "Query 1" to "US Fatal Incidents – 2001"
	3. From the country drop down, select "United States"
	4. Using the Incident Year slider, set the minimum and maximum years to 2001
	5. Click "Query"
	6. The map should zoom in on the United States with a dot representing each incident.
	7. Hover over any incident. A panel should slide down to show incident details.
	8. In the left panel below the query, a box will appear with high level statistics for the query.
	9. In the top navigation, click on "Bloom"
	10. Scroll down and click "Graph Database Bloom" to navigate to the Graph Database

4. Demo

	1. You can view the project online without needing any installation. 
		1. http://ec2-13-58-123-188.us-east-2.compute.amazonaws.com:8080/