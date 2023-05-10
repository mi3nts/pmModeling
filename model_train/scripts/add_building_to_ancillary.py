
#####################################################################################################################################################################
#   This script has two parts.
#   [Comment the second part] The first part for add building footprint data into the ancillary dataFrame - parallel jobs used to create seperate dataFrames.
#   [Comment the first part] The second part combine the seperated dataFrames created by the first part.
#####################################################################################################################################################################


# conda env = pmtest

import xarray as xr
import pandas as pd
from pyproj import Proj
import math
import sys

'''
################################# TO CREATE ONLY SEPERATE DFS ##########################################################################################################
coord_df = pd.read_csv("/scratch/prabuddha/2yrHist_train/pmSensorCoordDF/pm_sensor_loc_round_and_ancillary.csv")
# building state boundry
input_dataFrame = "/scratch/prabuddha/2yrHist_train/rawData/buildingFootprintDF/stateDF/"
state_boundary = pd.read_csv("/scratch/prabuddha/2yrHist_train/rawData/buildingFootprintDF/state_boundary_coordinates.csv")

#coord_df = coord_df.loc[0:5]



start = 0
end  = len(coord_df)
step = round(end/25)
arr = [coord_df[start:step], coord_df[step:2*step], coord_df[2*step:3*step], coord_df[3*step:4*step], coord_df[4*step:5*step], coord_df[5*step:6*step], 
    coord_df[6*step:7*step], coord_df[7*step:8*step], coord_df[8*step:9*step], coord_df[9*step:10*step], coord_df[10*step:11*step], coord_df[11*step:12*step], 
    coord_df[12*step:13*step], coord_df[13*step:14*step], coord_df[14*step:15*step], coord_df[15*step:16*step], coord_df[16*step:17*step], coord_df[17*step:18*step], 
    coord_df[18*step:19*step], coord_df[19*step:20*step], coord_df[20*step:21*step], coord_df[21*step:22*step], coord_df[22*step:23*step], coord_df[23*step:24*step],
    coord_df[24*step:25*step]]
ind = int(sys.argv[1])
coord = arr[ind]



# Finding details about building footprint
def near_building(input_df_dir, state_file, lat, lon):

    num_near_build = 0
    avg_distance = 0
    tot_area = 0
    state_name = "--"

    state = pd.read_csv(input_df_dir + state_file)


    for n in range(len(state)):
        dist = math.sqrt((state.mid_lon[n] - lon)**2 + (state.mid_lat[n] - lat)**2)

        if dist < 0.001:
            if (state.bbox_lon_1[n] < lon < state.bbox_lon_2[n]) & (state.bbox_lat_1[n] < lat < state.bbox_lat_2[n]):
                print("inside")
            else:
                num_near_build = num_near_build + 1
                avg_distance = (avg_distance + dist)/num_near_build
                tot_area = tot_area + state.area[n]
                state_name = state_file[0:len(state_file)-4]
            
    return(state_name, num_near_build, avg_distance, tot_area)



rows = []
for index, row in coord.iterrows():
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
        print('no data on lat = ' + str(latitude) + ' lon = ' + str(longitude))

df = pd.DataFrame(rows, columns=["latitude", "longitude", "cropland", "landcover", "population_density", "soiltype", "lithology", "elevation", "state", "num_building", "avg_distance", "tot_area"])
df.to_csv("/scratch/prabuddha/2yrHist_train/pmSensorCoordDF/building_ancillary_dfs/coord_build_ancillary_part_" + str(ind) + ".csv")
#print(df)
######################################################################################################################################################################
'''

################################# TO COMBINE SEPARATED DFS INTO A SINGLE DF ###########################################################################################

df_path = "/scratch/prabuddha/2yrHist_train/pmSensorCoordDF/building_ancillary_dfs/"
df0 = pd.read_csv(df_path + "coord_build_ancillary_part_0.csv")
df1 = pd.read_csv(df_path + "coord_build_ancillary_part_1.csv")
df2 = pd.read_csv(df_path + "coord_build_ancillary_part_2.csv")
df3 = pd.read_csv(df_path + "coord_build_ancillary_part_3.csv")
df4 = pd.read_csv(df_path + "coord_build_ancillary_part_4.csv")
df5 = pd.read_csv(df_path + "coord_build_ancillary_part_5.csv")
df6 = pd.read_csv(df_path + "coord_build_ancillary_part_6.csv")
df7 = pd.read_csv(df_path + "coord_build_ancillary_part_7.csv")
df8 = pd.read_csv(df_path + "coord_build_ancillary_part_8.csv")
df9 = pd.read_csv(df_path + "coord_build_ancillary_part_9.csv")
df10 = pd.read_csv(df_path + "coord_build_ancillary_part_10.csv")
df11 = pd.read_csv(df_path + "coord_build_ancillary_part_11.csv")
df12 = pd.read_csv(df_path + "coord_build_ancillary_part_12.csv")
df13 = pd.read_csv(df_path + "coord_build_ancillary_part_13.csv")
df14 = pd.read_csv(df_path + "coord_build_ancillary_part_14.csv")
df15 = pd.read_csv(df_path + "coord_build_ancillary_part_15.csv")
df16 = pd.read_csv(df_path + "coord_build_ancillary_part_16.csv")
df17 = pd.read_csv(df_path + "coord_build_ancillary_part_17.csv")
df18 = pd.read_csv(df_path + "coord_build_ancillary_part_18.csv")
df19 = pd.read_csv(df_path + "coord_build_ancillary_part_19.csv")
df20 = pd.read_csv(df_path + "coord_build_ancillary_part_20.csv")
df21 = pd.read_csv(df_path + "coord_build_ancillary_part_21.csv")
df22 = pd.read_csv(df_path + "coord_build_ancillary_part_22.csv")
df23 = pd.read_csv(df_path + "coord_build_ancillary_part_23.csv")
df24 = pd.read_csv(df_path + "coord_build_ancillary_part_24.csv")


df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24])
df = df[df["state"].str.contains("--")==False]
#print(df)
df.to_csv("/scratch/prabuddha/2yrHist_train/pmSensorCoordDF/pm_sensor_loc_round_and_ancillary_and_building.csv")










