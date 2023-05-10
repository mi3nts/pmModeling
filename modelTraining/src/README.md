## Model Training

### step 1: Ancillary data manually downloading.
Land Cover: https://www.mrlc.gov/data?f%5B0%5D=year%3A2019 \
Population Density: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11 \
Soil Type (link is not available now): https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/use/?cid=nrcs142p2_054013 \
Lithology: https://doi.pangaea.de/10.1594/PANGAEA.788537 \
Elevation: https://www.gebco.net/data_and_products/gridded_bathymetry_data/ \
Cropland: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php \
Building Footprint: https://github.com/microsoft/USBuildingFootprints 

### step 2: EPA in-situ PM2.5 data downloading
Historical PM2.5 data downloading using <code>EPA_pm_down.py</code>. Set up the time duration. The outputs are monthly dataframes with hourly data.

### step 3: MINTS's data downloading
Historical MINTS data from all the Nodes can be downloaded using <code>/MINTS_scripts/pythonSyncher.py</code>. This required <code>mintsDefinitionsBirdCall.yaml</code> file to be in the same directory (which not included).
