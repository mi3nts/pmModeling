## Forecast PM2.5 Estimation

### Downloading forecast data for a given datetime.
Pre-required ancillary data in US coordinates CSV file <code>USA_latlon_ancillary_df_final.csv</code>. Downloading forecast meteorological, humidity and AOD data for a given datetime using <code>data_down.py</code>. Then match the data with US coordinates and combine with corresponding ancillary data using grid interpolation. The outputs is a CSV file for one hour.

### PM estimation.
Estimate the PM2.5 concentration over the US region using <code>pm_estimate.py</code>. The outputs are a dataframe with estimated PM2.5 values and a PM2.5 reconstructed US map.
