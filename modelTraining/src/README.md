# ModelTrainingSteps

## step 1: Ancillary data downloading manually.
Land Cover: https://www.mrlc.gov/data?f%5B0%5D=year%3A2019 \
Population Density: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11 \
Soil Type (link is not available now): https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/use/?cid=nrcs142p2_054013 \
Lithology: https://doi.pangaea.de/10.1594/PANGAEA.788537 \
Elevation: https://www.gebco.net/data_and_products/gridded_bathymetry_data/ \
Cropland: https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php \
Building Footprint: https://github.com/microsoft/USBuildingFootprints 

## step 2: EPA in-situ PM2.5 data downloading
Historical PM2.5 data downloading using $\color{green}EPA\_pm\_down.py$. The outputs are monthly dataframes. 
