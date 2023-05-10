
from multiprocessing.connection import wait
from time import sleep
import math
import pandas as pd
import geopandas as gpd
import json



##############################################################################
# This is not the main script. Main script is "main_building.py"

# python environment = building_foot

# Data download: https://github.com/microsoft/USBuildingFootprints
##############################################################################


import geopy.distance


# Calculating distance between one EPA sensor and one of closest building in Km.
def find_radius():
    sensor_loc = (38.875161, -77.012816) 
    near_build_loc = (38.8751455, -77.0137945) 
    return(geopy.distance.geodesic(sensor_loc, near_build_loc).km)



def find_state(data_path, list, lat, lon):
    for s in list:
        print(s)
        input_path = data_path + s
        states = gpd.read_file(input_path)
        data = states.geometry
        jason_data = data.to_json()
        aDict = json.loads(jason_data)
        coord = aDict['bbox']
        print(coord)

        if (coord[0] < lon < coord[2]) & (coord[1] < lat < coord[3]):
            return(str(s))





    #print()



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


