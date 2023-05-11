## Model Training

### Ancillary data manually downloading.
Land Cover: https://www.mrlc.gov/data?f%5B0%5D=year%3A2019 \
Population Density: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11 \
Soil Type (link is not available now): https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/use/?cid=nrcs142p2_054013 \
Lithology: https://doi.pangaea.de/10.1594/PANGAEA.788537 \
Elevation: https://www.gebco.net/data_and_products/gridded_bathymetry_data/ \
Cropland: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php \
Building Footprint: https://github.com/microsoft/USBuildingFootprints 

### EPA in-situ PM2.5 data downloading
Historical EPA PM2.5 data downloading using <code>EPA_pm_down.py</code>. Set up the time duration. The outputs are monthly dataframes with hourly data.

### MINTS's data downloading
Historical MINTS data from all the Nodes can be downloaded using <code>/MINTS_scripts/pythonSyncher.py</code>. This required <code>mintsDefinitionsBirdCall.yaml</code> file to be in the same directory (which not included). The outputs are daily CSV files 10 seconds temporal fequency.

### OpenAQ PM2.5, Meteorological data, AOD data downloading.
Historical OpenAQ PM2.5 data, meteorological data with humidity data and AOD data can be downloaded using <code>meteo_AOD_openAQ_down.py</code>. The outputs are hourly CSV files.

### Averaging MINTS’s PM2.5 data hourly and combine with EPA and OpenAQ PM2.5 data.
The python script <code>pm_clean.py</code> will average the PM2.5 data from MINTS sensors and the combine those hourly data with EPA and OpenAQ PM2.5 data. The outputs are monthly CSV files of PM2.5 data from all the sources.

### Finding all the in-situ locations
Combine all the monthly CSV of PM2.5 data together and find all the PM ground observation locations using <code>findPM_sensorCoord.py</code>. The output is a CSV with all the ground sensors locations.

### Matching ancillary data (without building footprint) with sensor locations.
Pre-required ground sensor location CSV file from the previous step. Match the ancillary data with sensor coordiantes using <code>ancillary_data_match.py</code>. The output is a CSV file of sensor location with ancillary data (without building footprint data). 

### Pre-process of building footprint dataset.
Convert building footprint data from geojson to CSV and save state CSVs. Creating a CSV file of State boundary coordinates. The main python script for this is <code>/Building_footprint/main_building.py</code> and it is linked to <code>read_building.py</code> and <code>cal_building.py</code> scripts. 

### Calculating building footprint information at the ground sensor locations.
This required, building footprint data in each states from CSV format from the previous process and the State boundary coordinates CSV file. This also required ancillary data at senosr location CSV file. \
The script <code>add_building_to_ancillary.py</code> has two parts and the first part was written to execute on parallel computing. The outputs of the first part are seperate CSV files of ground sensor locations with ancillary and building footprint data. \
The second part is for combining those seperate CSV files created by the first part together and save a one CSV file for all the ancillary data including building footprint data.

### Matching meteorological, humidity and AOD data and calculate solar angles at ground sensor locations.
Required ancillary data at ground sensor locations CSV file from the previous step. The script <code>match_dynamic.py</code> match all the historical data (meteorological, humidity and AOD) and calculte the solar angles at the senosr locations. The outputs are daily CSV files with all the matched data with senosr locations.

### Combine daily dataframes into monthly dataframes and add monthly PM dataframes.
The daily CSV files are combined together and PM2.5 data CSV file combine with matched datetime using <code>combine_monthlyDF.py</code> script. The outputs are monthly CSV files with all the matched data with senosr locations.

### Combine monthly dataframes and Train a model and evaluate it.
Using <code>build_model.py</code> script, combine all those monthly CSV files and define two groups - historical and forecast - based on selected variables. Then it trains models for each groups and evaluate them. The outputs are the trained models and plots of evaluations.

