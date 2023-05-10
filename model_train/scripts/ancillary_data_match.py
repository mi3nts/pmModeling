# conda env = myenv1

import xarray as xr
import pandas as pd
from pyproj import Proj



# Open pm sensor coordinate dataFrame
coord_df = pd.read_csv("/Users/prabu/Desktop/Europa_scripts/Ancillary_data_match/pmSensorLocationRound.csv")
#coord_df = pd.read_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/coordinates/coordinatesUS.csv")


# Cropland  
da_crop = xr.open_rasterio("/Users/prabu/Desktop/Research/Satellite_Proj/Data/CropLand/2021_30m_cdls_USA/2021_30m_cdls.tif")
pro_crop = Proj('+init=epsg:5070')
# Landcover  
da_land = xr.open_rasterio("/Users/prabu/Desktop/Research/Satellite_Proj/Data/Landcover/nlcd_2019_land_cover_l48_20210604/nlcd_2019_land_cover_l48_20210604.img")
pro_land = Proj('+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs=True')
# Population density
da_pop = xr.open_rasterio("/Users/prabu/Desktop/Satellite_Project/Data/Data/gpw-v4-population-density-rev11_2020_30_sec_tif/gpw_v4_population_density_rev11_2020_30_sec.tif")
# Soil type
da_soil = xr.open_rasterio("/Users/prabu/Desktop/Satellite_Project/Data/Data/global_soil_regions_geoTIFF/so2015v2.tif")
# Lithology
da_lith = xr.open_rasterio("/Users/prabu/Desktop/Satellite_Project/Data/Data/hartmann-moosdorf_2012/glim_wgs84_0point5deg.txt.asc")
# Elevation
da_elev = xr.open_rasterio("/Users/prabu/Desktop/Satellite_Project/Data/Data/GEBCO_02_Aug_2022_c2119f267c0d/gebco_2022_n50.0_s25.0_w-125.0_e-65.0.tif")




#coord_df = coord_df.loc[0:50]
rows = []
for index, row in coord_df.iterrows():
    try:
        latitude = coord_df.latitude[index]
        longitude = coord_df.longitude[index]

        lon_crop,lat_crop = pro_crop(longitude,latitude)
        value_location_crop = da_crop.sel(y = lat_crop,x = lon_crop, method='nearest',tolerance = 50)
        value_crop = value_location_crop.values[0]

        lon_land,lat_land = pro_land(longitude,latitude)
        value_location_land = da_land.sel(y = lat_land,x = lon_land, method='nearest',tolerance=990)
        value_land = value_location_land.values[0]

        value_location_pop = da_pop.sel(y = latitude,x = longitude, method='nearest',tolerance=0.01)
        value_pop = value_location_pop.values[0]

        value_location_soil = da_soil.sel(y = latitude,x = longitude, method='nearest',tolerance=0.5)
        value_soil = value_location_soil.values[0]

        value_location_lith = da_lith.sel(y = latitude,x = longitude, method='nearest',tolerance=0.5)
        value_lith = value_location_lith.values[0]

        value_location_elev = da_elev.sel(y = latitude,x = longitude, method='nearest',tolerance=0.5)
        value_elev = value_location_elev.values[0]

        rows.append([latitude, longitude, value_crop, value_land, value_pop, value_soil, value_lith, value_elev])
        
    except:
        print('no cropland data on lat = ' + str(latitude) + ' lon = ' + str(longitude))


df = pd.DataFrame(rows, columns=["latitude", "longitude", "cropland", "landcover", "population_density", "soiltype", "lithology", "elevation"])
df.to_csv('/Users/prabu/Desktop/Europa_scripts/Ancillary_data_match/pm_sensor_loc_round_and_ancillary.csv')

print("done")


################################################### ADDING BUILDING FOOTPRINT DATA ####################################################################
'''
#from read_building import *
from cal_building import *

coord_df = pd.read_csv("/Users/prabu/Desktop/Europa_scripts/Ancillary_data_match/pm_sensor_loc_and_ancillary.csv")
# building state boundry
input_dataFrame = "/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/"
state_boundary = pd.read_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/state_boundary_coordinates.csv")

coord_df = coord_df.loc[0:5]

#print(coord_df)

rows = []
for index, row in coord_df.iterrows():
    try:
        latitude = coord_df.latitude[index]
        longitude = coord_df.longitude[index]
        cropland = coord_df.cropland[index]
        landcover = coord_df.landcover[index]
        population_density = coord_df.population_density[index]
        soiltype = coord_df.soiltype[index]
        lithology = coord_df.lithology[index]
        elevation = coord_df.elevation[index]

        for s in range(len(state_boundary)):
            if (state_boundary.lon_1[s] < longitude < state_boundary.lon_2[s]) & (state_boundary.lat_1[s] < latitude < state_boundary.lat_2[s]):
                state_file = state_boundary.State[s] + ".csv"
                state, num_build, avg_dist, tot_area =near_building(input_dataFrame, state_file, latitude, longitude)
                #print(state)
                rows.append([latitude, longitude, cropland, landcover, population_density, soiltype, lithology, elevation, state, num_build, avg_dist, tot_area])
    except:
        print('no cropland data on lat = ' + str(latitude) + ' lon = ' + str(longitude))

df = pd.DataFrame(rows, columns=["latitude", "longitude", "cropland", "landcover", "population_density", "soiltype", "lithology", "elevation", "state", "num_building", "avg_distance", "tot_area"])
print(df)

'''


        
        
        

