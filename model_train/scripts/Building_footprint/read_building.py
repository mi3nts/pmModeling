
##############################################################################
# This is not the main script. Main script is "main_building.py"

# python environment = building_foot

# Data download: https://github.com/microsoft/USBuildingFootprints
##############################################################################

import geopandas as gpd
import geoplot
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import json
import pandas as pd


# Visualizing footprint data of given state.
def visualize_state(file, save_location):
    states = gpd.read_file(file)
    geoplot.polyplot(states, projection=gcrs.AlbersEqualArea(), edgecolor='darkgrey', 
                    facecolor='grey', linewidth=.3, figsize=(50, 50))
    plt.savefig(save_location)
    return("Footprint image of ", file, " saved in ", save_location)


# # Converting geojson data to CSV dataFrame
def geo_to_csv(state, output_path):
    states = gpd.read_file(state)
    data = states.geometry
    jason_data = data.to_json()

    # json data in string format, so making a dictionary
    aDict = json.loads(jason_data)

    # get main keys in dictionary 
    '''
    for key in aDict:
        print(key)
    '''

    # There are three main keys - type, features, bbox
    # All the data available in feature key
    feature_data = aDict['features']
    # Features has sub-catagories - id, type, properties, geometry, bbox
    # bbox has the margin coordinates of building footprints

    rows = []
    for i in range(len(feature_data)):
        coordinateData = feature_data[i]['geometry']['coordinates']
        num_points = len(coordinateData[0])

        bbox_lon_1 = feature_data[i]['bbox'][0]
        bbox_lat_1 = feature_data[i]['bbox'][1]
        bbox_lon_2 = feature_data[i]['bbox'][2]
        bbox_lat_2 = feature_data[i]['bbox'][3]

        mid_lon = (bbox_lon_1 + bbox_lon_2)/2
        mid_lat = (bbox_lat_1 + bbox_lat_2)/2

        area = data[i].area

        rows.append([num_points, coordinateData[0], bbox_lon_1, bbox_lat_1, bbox_lon_2, bbox_lat_2, mid_lon,
                    mid_lat, area])

    df = pd.DataFrame(rows, columns=["num_points", "coordinates", "bbox_lon_1", "bbox_lat_1", 
                                    "bbox_lon_2", "bbox_lat_2", "mid_lon", "mid_lat", "area"])
    #print(df)
    df.to_csv(output_path)  