
######  activate myenv5 environment
# required to install rasterio using pip: pip install rasterio

'''
Make a dataFrame of latitudes and longitudes only within the USA.
The gap between latitudes and longitudes is 0.1
Add ancillary data (Statistic data) to the dataFrame.
'''

import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Proj
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import os
import pygrib
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable


# First, making latitudes and longitudes dataFrame only within north american reagion.
# Metero data set has values only on north american reagion, so latitude and longitude can be extract from one metero data file.  
meteroDF = xr.open_dataset('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/ref_data/00', engine='cfgrib')

metero_lat = meteroDF.latitude
metero_lon = meteroDF.longitude
metero_val = meteroDF.t2m.values

# Finding latitudes and longitudes of numerical values of temperature
metero_row = []
for n in range(len(metero_val)):
    for m in range(len(metero_val[n])):
        if np.isnan(metero_val[n][m]):
            print()
        else:
            metero_row.append([metero_lat.values[n], metero_lon.values[m]])

final_df = pd.DataFrame(metero_row, columns=["latitude", "longitude"])




# Cropland dataset has non-zero values only on USA region.
# So this dataset can be used to extract the latitudes and longitudes only on USA.
crop_data = xr.open_rasterio("/Users/prabu/Desktop/Research/Satellite_Proj/Data/CropLand/2021_30m_cdls_USA/2021_30m_cdls.tif")
# To converting latitude and longitude into array indexes corresponding to cropland data sent. 
# Cropland data set uses the same convertion.
pro = Proj('+init=epsg:5070')


# Finding latitudes and longitudes of non zero values of crop data and adding cropland data.
crop_row = []
for index, row in final_df.iterrows():
    latitude = final_df.latitude[index]
    longitude = final_df.longitude[index]

    crop_lon,crop_lat = pro(longitude,latitude)
    try:
        crop_value_location = crop_data.sel(y = crop_lat,x = crop_lon, method='nearest',tolerance = 50)
        crop_value = crop_value_location.values[0]
        if crop_value != 0:
            crop_row.append([latitude, longitude, crop_value])
    except:
        ex = "no values"

    
final_df2 = pd.DataFrame(crop_row, columns=["latitude", "longitude", "cropland"])





#####################################################################################################
# Match and add landcover data
landcover = xr.open_rasterio("/Users/prabu/Desktop/Research/Satellite_Proj/Data/Landcover/nlcd_2019_land_cover_l48_20210604/nlcd_2019_land_cover_l48_20210604.img")
pro_land = Proj('+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True')

land_row = []
for index, row in final_df2.iterrows():
    latitude = final_df2.latitude[index]
    longitude = final_df2.longitude[index]
    land_lon,land_lat = pro_land(longitude,latitude)
    try:
        land_value_location = landcover.sel(y = land_lat, x = land_lon, method='nearest',tolerance=1000)
        land_value = land_value_location.values[0]
        land_row.append([latitude, longitude, land_value])
    except:
        ex = "no values"

land_df = pd.DataFrame(land_row, columns=["latitude", "longitude", "landcover"])

merged_df = pd.merge(final_df2, land_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])



####################################################################################################
# Match and add soil type
soil_data = xr.open_rasterio("/Users/prabu/Desktop/SatPy/DataSource/global_soil_regions_geoTIFF/so2015v2.tif")
soil_data = soil_data.where(soil_data.y<50,drop=True).where(soil_data.y>25,drop=True).where(soil_data.x>-125,drop=True).where(soil_data.x<-65,drop=True)
#print(soil_data.values[0])

soil_row = []
for index, row in final_df2.iterrows():
    latitude = final_df2.latitude[index]
    longitude = final_df2.longitude[index]
    #land_lon,land_lat = pro_land(longitude,latitude)
    try:
        soil_value_location = soil_data.sel(y = latitude, x = longitude, method='nearest',tolerance=0.5)
        soil_value = soil_value_location.values[0]
        soil_row.append([latitude, longitude, soil_value])
    except:
        ex = "no values"

soil_df = pd.DataFrame(soil_row, columns=["latitude", "longitude", "soil"])

merged_df2 = pd.merge(merged_df, soil_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])


#####################################################################################################
# Match and add Lithology data
lithology = xr.open_rasterio("/Users/prabu/Desktop/SatPy/DataSource/hartmann-moosdorf_2012/glim_wgs84_0point5deg.txt.asc")
#print("Triming to US extent")
lithology = lithology.where(lithology.y<50,drop=True).where(lithology.y>25,drop=True).where(lithology.x>-125,drop=True).where(lithology.x<-65,drop=True)
#print(lithology.values[0])
litho_row = []
for index, row in final_df2.iterrows():
    latitude = final_df2.latitude[index]
    longitude = final_df2.longitude[index]
    #land_lon,land_lat = pro_land(longitude,latitude)
    try:
        litho_value_location = lithology.sel(y = latitude, x = longitude, method='nearest',tolerance=0.5)
        litho_value = litho_value_location.values[0]
        litho_row.append([latitude, longitude, litho_value])
    except:
        ex = "no values"

litho_df = pd.DataFrame(litho_row, columns=["latitude", "longitude", "lithology"])

merged_df3 = pd.merge(merged_df2, litho_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])



################################################################################################
# Match and add elevation data
population = xr.open_rasterio('/Users/prabu/Desktop/Satellite_Project/Data/Data/gpw-v4-population-density-rev11_2020_30_sec_tif/gpw_v4_population_density_rev11_2020_30_sec.tif')
population = population.where(population.y<50,drop=True).where(population.y>25,drop=True).where(population.x>-125,drop=True).where(population.x<-65,drop=True)
#print(elevation)
pop_row = []
for index, row in final_df2.iterrows():
    latitude = final_df2.latitude[index]
    longitude = final_df2.longitude[index]
    #land_lon,land_lat = pro_land(longitude,latitude)
    try:
        pop_value_location = population.sel(y = latitude, x = longitude, method='nearest',tolerance=0.5)
        pop_value = pop_value_location.values[0]
        pop_row.append([latitude, longitude, pop_value])
    except:
        ex = "no values"

pop_df = pd.DataFrame(pop_row, columns=["latitude", "longitude", "population"])

merged_df4 = pd.merge(merged_df3, pop_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])


#########################################################################################
# Match and add population data
elevation = xr.open_rasterio('/Users/prabu/Desktop/Satellite_Project/Data/Data/GEBCO_02_Aug_2022_c2119f267c0d/gebco_2022_n50.0_s25.0_w-125.0_e-65.0.tif')
elevation = elevation.where(elevation.y<50,drop=True).where(elevation.y>25,drop=True).where(elevation.x>-125,drop=True).where(elevation.x<-65,drop=True)
#print(elevation)
elev_row = []
for index, row in final_df2.iterrows():
    latitude = final_df2.latitude[index]
    longitude = final_df2.longitude[index]
    #land_lon,land_lat = pro_land(longitude,latitude)
    try:
        elev_value_location = elevation.sel(y = latitude, x = longitude, method='nearest',tolerance=0.5)
        elev_value = elev_value_location.values[0]
        elev_row.append([latitude, longitude, elev_value])
    except:
        ex = "no values"

elev_df = pd.DataFrame(elev_row, columns=["latitude", "longitude", "elevation"])

merged_df5 = pd.merge(merged_df4, elev_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])

merged_df5.to_csv('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/USA_latlon_ancillary_df.csv')


#########################################################################################
# Add building footprint data

other_ancillary = pd.read_csv('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/USA_latlon_ancillary_df.csv')
building = pd.read_csv('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/building_footprint.csv')

merged_df6 = pd.merge(other_ancillary, building,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])
merged_df6 = merged_df6.drop(columns=['Unnamed: 0_x', 'Unnamed: 0_y'])
merged_df6 = merged_df6.replace(np.nan,0)
merged_df6.to_csv('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/USA_latlon_ancillary_df_final.csv')





'''
# Scatter plot of latitudes and longitudes only in USA
fname = '/Users/prabu/Desktop/Research/Satellite_Proj/Data/USBoundry/cb_2018_us_state_500k.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='none')

fig=plt.figure(figsize=[30,18])
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.add_feature(shape_feature,edgecolor='blue')
ax.gridlines(draw_labels=True)
ax.coastlines()
#c = ax.pcolor(final_df2["longitude"], final_df2["latitude"],final_df2["cropland"],cmap='Blues',vmin=-300,vmax=300)
#c = ax.scatter(lons,lats, s=1)
c = ax.scatter(final_df2["longitude"], final_df2["latitude"], c = final_df2["cropland"], s=1, cmap='gist_ncar',vmin=0,vmax=250)
fig.colorbar(c, ax=ax)
plt.savefig("/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/USA_lat_lon2.png")
'''