######  activate myenv5 environment

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
import math


coord = pd.read_csv('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/USA_latlon_ancillary_df.csv')

Alabama = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Alabama.csv')
Alaska = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Alaska.csv')
Arizona = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Arizona.csv')
Arkansas = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Arkansas.csv')
California = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/California.csv')
Colorado = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Colorado.csv')
Connecticut = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Connecticut.csv')
Delaware = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Delaware.csv')
DistrictofColumbia = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/DistrictofColumbia.csv')
Florida = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Florida.csv')
Georgia = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Georgia.csv')
Hawaii = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Hawaii.csv')
Idaho = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Idaho.csv')
Illinois = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Illinois.csv')
Indiana = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Indiana.csv')
Iowa = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Iowa.csv')
Kansas = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Kansas.csv')
Kentucky = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Kentucky.csv')
Louisiana = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Louisiana.csv')
Maine = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Maine.csv')
Maryland = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Maryland.csv')
Massachusetts = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Massachusetts.csv')
Michigan = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Michigan.csv')
Minnesota = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Minnesota.csv')
Mississippi = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Mississippi.csv')
Missouri = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Missouri.csv')
Montana = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Montana.csv')
Nebraska = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Nebraska.csv')
Nevada = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Nevada.csv')
NewHampshire = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/NewHampshire.csv')
NewJersey = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/NewJersey.csv')
NewMexico = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/NewMexico.csv')
NewYork = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/NewYork.csv')
NorthCarolina = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/NorthCarolina.csv')
NorthDakota = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/NorthDakota.csv')
Ohio = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Ohio.csv')
Oklahoma = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Oklahoma.csv')
Oregon = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Oregon.csv')
Pennsylvania = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Pennsylvania.csv')
RhodeIsland = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/RhodeIsland.csv')
SouthCarolina = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/SouthCarolina.csv')
SouthDakota = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/SouthDakota.csv')
Tennessee = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Tennessee.csv')
Texas = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Texas.csv')
Utah = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Utah.csv')
Vermont = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Vermont.csv')
Virginia = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Virginia.csv')
Washington = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Washington.csv')
WestVirginia = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/WestVirginia.csv')
Wisconsin = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Wisconsin.csv')
Wyoming = pd.read_csv('/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/Wyoming.csv')


df1 = pd.DataFrame(Alabama)
df2 = pd.DataFrame(Alaska)
df3 = pd.DataFrame(Arizona)
df4 = pd.DataFrame(Arkansas)
df5 = pd.DataFrame(California)
df6 = pd.DataFrame(Colorado)
df7 = pd.DataFrame(Connecticut)
df8 = pd.DataFrame(Delaware)
df9 = pd.DataFrame(DistrictofColumbia)
df10 = pd.DataFrame(Florida)
df11 = pd.DataFrame(Georgia)
df12 = pd.DataFrame(Hawaii)
df13 = pd.DataFrame(Idaho)
df14 = pd.DataFrame(Illinois)
df15 = pd.DataFrame(Indiana)
df16 = pd.DataFrame(Iowa)
df17 = pd.DataFrame(Kansas)
df18 = pd.DataFrame(Kentucky)
df19 = pd.DataFrame(Louisiana)
df20 = pd.DataFrame(Maine)
df21 = pd.DataFrame(Maryland)
df22 = pd.DataFrame(Massachusetts)
df23 = pd.DataFrame(Michigan)
df24 = pd.DataFrame(Minnesota)
df25 = pd.DataFrame(Mississippi)
df26 = pd.DataFrame(Missouri)
df27 = pd.DataFrame(Montana)
df28 = pd.DataFrame(Nebraska)
df29 = pd.DataFrame(Nevada)
df30 = pd.DataFrame(NewHampshire)
df31 = pd.DataFrame(NewJersey)
df32 = pd.DataFrame(NewMexico)
df33 = pd.DataFrame(NewYork)
df34 = pd.DataFrame(NorthCarolina)
df35 = pd.DataFrame(NorthDakota)
df36 = pd.DataFrame(Ohio)
df37 = pd.DataFrame(Oklahoma)
df38 = pd.DataFrame(Oregon)
df39 = pd.DataFrame(Pennsylvania)
df40 = pd.DataFrame(RhodeIsland)
df41 = pd.DataFrame(SouthCarolina)
df42 = pd.DataFrame(SouthDakota)
df43 = pd.DataFrame(Tennessee)
df44 = pd.DataFrame(Texas)
df45 = pd.DataFrame(Utah)
df46 = pd.DataFrame(Vermont)
df47 = pd.DataFrame(Virginia)
df48 = pd.DataFrame(Washington)
df49 = pd.DataFrame(WestVirginia)
df50 = pd.DataFrame(Wisconsin)
df51 = pd.DataFrame(Wyoming)

combine_df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10,
    df11, df12, df13, df14, df15, df16, df17, df18, df19, df20,
    df21, df22, df23, df24, df25, df26, df27, df28, df29, df30,
    df31, df32, df33, df34, df35, df36, df37, df38, df39, df40,
    df41, df42, df43, df44, df45, df46, df47, df48, df49, df50, df51], ignore_index=True, sort=False)
#print(len(df1))
#print(len(df2))
#print(len(combine_df))



rows = []
for index, row in coord.iterrows():
    latitude = coord.latitude[index]
    longitude = coord.longitude[index]

    print(index)
   
    try:
        near_build = combine_df.loc[(combine_df['mid_lat'].sub(latitude)*combine_df['mid_lat'].sub(latitude) + combine_df['mid_lon'].sub(longitude)*combine_df['mid_lon'].sub(longitude))**(1/2) <= 0.001]
        nearest_index = ((near_build['mid_lat'].sub(latitude)*near_build['mid_lat'].sub(latitude) + near_build['mid_lon'].sub(longitude)*near_build['mid_lon'].sub(longitude))**(1/2)).idxmin()

        if (near_build.bbox_lon_1[nearest_index] < longitude < near_build.bbox_lon_2[nearest_index]) & (near_build.bbox_lat_1[nearest_index] < latitude < near_build.bbox_lat_2[nearest_index]):
            update_df = near_build.drop(nearest_index)
        else:
            update_df = near_build

        num_build = len(update_df)
        #print("Number of near buildings " + str(num_build))
        tot_dist = sum((update_df['mid_lat'].sub(latitude)*update_df['mid_lat'].sub(latitude) + update_df['mid_lon'].sub(longitude)*update_df['mid_lon'].sub(longitude))**(1/2))
        avg_dist = tot_dist/num_build
        tot_area = sum(update_df.area)

        rows.append([latitude, longitude, num_build, avg_dist, tot_area])
    except:
        #print("No data found on " + str(latitude) + "& " + str(longitude))
        aa = 0


final_df = pd.DataFrame(rows, columns=["latitude", "longitude", "num_build", "avg_dist", "tot_area"])
final_df.to_csv("/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/USA_lat_lon_ancillary/building_footprint.csv") 



#latitude = coord.latitude[1]
#longitude = coord.longitude[1]
#latitude = 38.875161
#longitude = -77.012816

#print(latitude)
#print(longitude)

'''
#near_index = min((combine_df['mid_lat'].sub(latitude)*combine_df['mid_lat'].sub(latitude) + combine_df['mid_lon'].sub(longitude)*combine_df['mid_lon'].sub(longitude))**(1/2))
near_build = combine_df.loc[(combine_df['mid_lat'].sub(latitude)*combine_df['mid_lat'].sub(latitude) + combine_df['mid_lon'].sub(longitude)*combine_df['mid_lon'].sub(longitude))**(1/2) <= 0.001]
#print(near_build)
nearest_index = ((near_build['mid_lat'].sub(latitude)*near_build['mid_lat'].sub(latitude) + near_build['mid_lon'].sub(longitude)*near_build['mid_lon'].sub(longitude))**(1/2)).idxmin()
#print("nearest index " + str(nearest_index))

if (near_build.bbox_lon_1[nearest_index] < longitude < near_build.bbox_lon_2[nearest_index]) & (near_build.bbox_lat_1[nearest_index] < latitude < near_build.bbox_lat_2[nearest_index]):
    #print("sensor is inside the building")
    update_df = near_build.drop(nearest_index)
    #print(update_df)
else:
    #print("Sensor is outside")
    update_df = near_build

num_build = len(update_df)
#print("Number of near buildings " + str(num_build))
tot_dist = sum((update_df['mid_lat'].sub(latitude)*update_df['mid_lat'].sub(latitude) + update_df['mid_lon'].sub(longitude)*update_df['mid_lon'].sub(longitude))**(1/2))
avg_dist = tot_dist/num_build
#print("avg dist " + str(avg_dist))
tot_area = sum(update_df.area)
#print("total area " + str(tot_area))

row = []
row.append([latitude, longitude, num_build, avg_dist, tot_area])
final_df = pd.DataFrame(row, columns=["latitude", "longitude", "num_build", "avg_dist", "tot_area"])
print(final_df)
'''
#location = state.index.get_loc(state['mid_lon'] = latitude, method='nearest')
#location = state.index.get_loc(latitude)
#location = state.index.get_indexer([latitude, longitude], method="nearest")
#print(location)
#print(min(range(len(state)), key=lambda i: (abs(state.mid_lat[i]- latitude) & abs(state.mid_lon[i]- longitude))))

#result_index = (state['mid_lat'].sub(latitude)*state['mid_lat'].sub(latitude) + state['mid_lon'].sub(longitude)*state['mid_lon'].sub(longitude)).abs().idxmin()
#result_index = (combine_df['mid_lat'].sub(latitude)*combine_df['mid_lat'].sub(latitude) + combine_df['mid_lon'].sub(longitude)*combine_df['mid_lon'].sub(longitude)).abs().idxmin()
#result_index = min((combine_df['mid_lat'].sub(latitude)*combine_df['mid_lat'].sub(latitude) + combine_df['mid_lon'].sub(longitude)*combine_df['mid_lon'].sub(longitude))**(1/2))
#result_index2 = state['mid_lon'].sub(longitude).abs().idxmin()
#result_index2 = math.sqrt(state['mid_lat'] - latitude).idxmin()
#print(result_index)
#build_lat = combine_df.mid_lat[result_index]
#build_lon = combine_df.mid_lon[result_index]
#print(build_lat)
#print(build_lon)

#dist = (build_lat.sub(latitude)*build_lat.sub(latitude) + build_lon.sub(longitude)*build_lon.sub(longitude))**(1/2)
#area = 
#print(dist)
#print(result_index2)

#print(state.mid_lat[result_index])
#print(state.mid_lon[result_index])







'''
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

    
final_df2 = pd.DataFrame(crop_row, columns=["latitude", "longitude", "cropland"]
'''