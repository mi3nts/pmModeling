
##############################################################################
# python environment = building_foot

# Data download: https://github.com/microsoft/USBuildingFootprints

# Written: Prabuddha Madusanka
##############################################################################


from read_building import *
from cal_building import *
import os

# Step 1
# The directory to all States data in geojson format.
data_path = "/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/raw/"
output_dataFrame = "/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/"
file_names = os.listdir(data_path) 
# Reading only geojason files
State_list = list(filter(lambda k: 'geo' in k, file_names))

# Step 2
# Access the CSV and 
input_dataFrame = "/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/"
CSV_names = os.listdir(input_dataFrame) 
df_list = list(filter(lambda k: 'csv' in k, CSV_names))
#print(df_list)


'''
#-----------------------------------------------------------------------------------------
# Visualizing footprint data of given state.

# Here index 23 is Distric of Columbia (DC)
state = State_list[22]
# Image will be saved in this directory.
save_location = "/Users/prabu/Desktop/code_testing/Dist_Columbia3.png"

visualize_state(state, save_location)

print("")
print("Footprint image of ", state, " saved")
print("")
#-----------------------------------------------------------------------------------------
'''

'''
#-----------------------------------------------------------------------------------------
# Converting geojson data to CSV dataFrame
for i in range(len(State_list)):
    state = State_list[i]
    input_path = data_path + state 
    output_path = output_dataFrame + state[0:len(state)-8] + ".csv"
    # Function is in read_building.py  
    geo_to_csv(input_path, output_path)

    print("")
    print("CSV file of ", state[0:len(state)-8], "is saved")
    print("")
#-----------------------------------------------------------------------------------------
'''


'''
#-----------------------------------------------------------------------------------------
# Finding appropreate radius to find nearest buildings to a sensor.
# Calculating distance between one EPA sensor and one of closest building in Km.

# Function is in "cal_building.py"
print(find_radius())
# The calculated distance is 84 m. 
# Therefore 100 m distance selected as radius
#-----------------------------------------------------------------------------------------
'''


'''
#-----------------------------------------------------------------------------------------
# finding boundary coordinates of all the States and save in a CSV file.
coord_row = []
for state in State_list:
    input_path = data_path + state
    states = gpd.read_file(input_path)
    data = states.geometry
    jason_data = data.to_json()
    aDict = json.loads(jason_data)
    coord = aDict['bbox']

    stateName = str(state[0:len(state)-8])
    coord_lon_1 = coord[0]
    coord_lat_1 = coord[1]
    coord_lon_2 = coord[2]
    coord_lat_2 = coord[3]

    coord_row.append([stateName, coord_lon_1, coord_lat_1, coord_lon_2, coord_lat_2])

    print("done ", stateName)

df = pd.DataFrame(coord_row, columns=["State", "lon_1", "lat_1", "lon_2", "lat_2"])
df.to_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/state_boundary_coordinates.csv")  
#-----------------------------------------------------------------------------------------
'''


'''
#-----------------------------------------------------------------------------------------
# Get the EPA ground observations coordinate list.
EPA_sensor_list = pd.read_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/coordinates/coordinatesUS.csv")
state_boundary = pd.read_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/state_boundary_coordinates.csv")

rows = []
for i in range(len(EPA_sensor_list)):
    lat = EPA_sensor_list.Latitude[i]
    lon = EPA_sensor_list.Longitude[i]
    print("sensor number ", i)

    for s in range(len(state_boundary)):
        if (state_boundary.lon_1[s] < lon < state_boundary.lon_2[s]) & (state_boundary.lat_1[s] < lat < state_boundary.lat_2[s]):
            print("State is ", state_boundary.State[s])
            state_file = state_boundary.State[s] + ".csv"
            state, num_build, avg_dist, tot_area =near_building(input_dataFrame, state_file, lat, lon)
            rows.append([lat, lon, state, num_build, avg_dist, tot_area])

df = pd.DataFrame(rows, columns=["sensor_lat", "sensor_lon", "state", "num_building", "avg_distance", "tot_area"])
df.to_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/building.csv")  

#-----------------------------------------------------------------------------------------
'''


'''
#-----------------------------------------------------------------------------------------
# Cleaning building footprint dataframe
df = pd.read_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/building.csv")
df= df[df['num_building'] != 0]
df.to_csv("/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/final_building.csv")  
#-----------------------------------------------------------------------------------------
'''