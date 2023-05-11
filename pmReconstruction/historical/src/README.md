## Historical PM2.5 Estimation

### Making the US region reference coordinate dataframe with all the ancillary data.
This process has three parts. \
Part 1, pre-required one raw meteorological dataset, the ancillary datasets (except raw building footprint data). Execute the <code>USA_coord_ancillary.py</code> script with commenting the last part (adding building footprint data). The output is dataframe of continuous 10km x 10km grid coordinates of US with matching ancillary data except building footprint. \
Part 2, pre-required the CSV file from the first part and analyzed building footprint data CSV file created from the model training. The script <code>building.py</code> create a CSV file with only building footprint data over the US region. \
Part 3, pre-required the CSV files created from part 1 and part 2. Execute the same script <code>USA_coord_ancillary.py</code> again but with commenting the first part and uncommenting the last part. The output is a complete ancillary data with US coordinates.

### Downloading historical data for a given datetime.
Pre-required ancillary data in US coordinates CSV file <code>USA_latlon_ancillary_df_final.csv</code>. Downloading historical meteorological, humidity and AOD data for a given datetime using <code>data_down.py</code>. Then match the data with US coordinates and combine with corresponding ancillary data using parallel computings. The outputs are separated CSV files for one hour.

### Combine separated dataframes and PM estimation.
The script <code>pm_estimate.py</code> combine the separated CSV files created from the previous step and estimate the PM2.5 concentration over the US region. The outputs are a dataframe with estimated PM2.5 values and a PM2.5 reconstructed US map.


