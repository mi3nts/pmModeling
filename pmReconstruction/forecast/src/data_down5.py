# install netCDF4 using: conda install -c conda-forge netcdf4
# install ECMWF API: pip install ecmwf-opendata==0.0.14
# conda activate pmtest

import s3fs
from ecmwf.opendata import Client
import os
from os.path import exists
#from datetime import datetime
import datetime
import pandas as pd
import pygrib
import xarray as xr
from pyproj import Proj
from pysolar.solar import *
import sys
#import cProfile
import time as ct
from scipy.spatial.distance import cdist
import numpy as np


start_time = ct.time()

save_dir = '/scratch/prabuddha/pm_est_fc/data/raw_data/'
# where the ancillary dataFrame saved
ancillary_df = '/scratch/prabuddha/pm_est/ref_data/USA_latlon_ancillary_df_final.csv'
save_raw_df = '/scratch/prabuddha/pm_est_fc/data/raw_df/'


fs = s3fs.S3FileSystem(anon=True)
client = Client(source="ecmwf")

# crnt_time is current UTC time
crnt_time = datetime.datetime.now() + datetime.timedelta(hours=5)
str_crnt_time = str(crnt_time)
year, month, day, hour = str_crnt_time[0:4], str_crnt_time[5:7], str_crnt_time[8:10], str_crnt_time[11:13]
print("Current UTC time " + str_crnt_time)


# meteoro data has two key words - time and step
# time is when the forecast start (normally forecast start 10 or 12 hrs behind the current UTC).
# time keyword can have only 00, 06, 12, 18.
# to get the current UTC data, we have to find the closest value for time by substracting 10 hrs from current UTC.
# Then add multiplication of 3 until close to the current UTC.
# this multiplication value is the step.

if 0 <= int(hour) < 6:
    time = "12"
    if int(hour) < 3:
        step = 12
        new_hour = "00"
    else:
        step = 15
        new_hour = "03"

elif 6 <= int(hour) < 12:
    time = "18"
    if int(hour) < 9:
        step = 12
        new_hour = "06"
    else:
        step = 15
        new_hour = "09"

elif 12 <= int(hour) < 18:
    time = "00"
    if int(hour) < 15:
        step = 12
        new_hour = "12"
    else:
        step = 15
        new_hour = "15"

else:
    time = "06"
    if int(hour) < 21:
        step = 12
        new_hour = "18"
    else:
        step = 15
        new_hour = "21"

print("downloading data for " + year + "-" + month + "-" + day + " " + new_hour + ":00")


try:
    os.mkdir(save_dir + year)
except FileExistsError:
    print("year folder exists")

try:
    os.mkdir(save_dir + year + '/' + month)
except FileExistsError:
    print("month folder exists")

try:
    os.mkdir(save_dir + year + '/' + month + '/' + day)
except FileExistsError:
    print("day folder exists")

try:
    os.mkdir(save_dir + year + '/' + month + '/' + day + '/' + new_hour)
except FileExistsError:
    print("hour folder exists")


meteo_dir = save_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + "meteo_" + new_hour + '.grib2'
AOD_dir = save_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + "AOD_" + new_hour


def down_meteo():
    result = client.retrieve(
        time=time,
        stream = "oper",
        type = "fc",
        #step=[3,6],
        step = step,
        param = ["2t", "r", "sp", "10u", "10v", "tp", "q"],
        #param = ["tp"],
        target = meteo_dir,
    )
    print(result.datetime)


def down_AOD():
    date_time = year + "-" + month + "-" + day + " " + new_hour + ":00:00"
    datetime_object = datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')
    julian_day = datetime_object.strftime('%j')
    digit = len(str(julian_day))
    if digit == 1:
        julian_day = '0' + '0' + str(julian_day)
    elif digit == 2:
        julian_day = '0' + str(julian_day)
    else:
        julian_day = str(julian_day)
    AWS_path = 'noaa-goes16/ABI-L2-AODC/' + str(year) + '/' + julian_day + '/' + new_hour + '/*'
    files = fs.glob(AWS_path)
    fs.get(files[0], AOD_dir)

stage1_time = ct.time()
execution1_time = stage1_time - start_time
print(f"Setting up time: {execution1_time} seconds")


if exists(meteo_dir):
    print("meteorogical data exist")
else:
    down_meteo()

if exists(AOD_dir):
    print("AOD data exist")
else:
    down_AOD()

stage2_time = ct.time()
execution2_time = stage2_time - stage1_time
print(f"Downloading time: {execution2_time} seconds")

#import time
ct.sleep(30)

###############################################################
# Make a DataFrame
coord_df = pd.read_csv(ancillary_df)
#print(len(coord_df))
#coord = coord_df[0:10]
#coord = coord_df
#print(coord)


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



stage3_time = ct.time()
execution3_time = stage3_time - stage2_time
print(f"Variable assign time: {execution3_time} seconds")

######### matching meteorological data #############################
meteo_grbs = pygrib.open(meteo_dir)
detail = meteo_grbs.select()

grb_temp = meteo_grbs.select(name='2 metre temperature')[0]
grb_rhum = meteo_grbs.select(name='Relative humidity')[0]
grb_pres = meteo_grbs.select(name='Surface pressure')[0]
grb_uwind = meteo_grbs.select(name='10 metre U wind component')[0]
grb_vwind = meteo_grbs.select(name='10 metre V wind component')[0]
grb_shum = meteo_grbs.select(name='Specific humidity')[0]

mete_lats, mete_lons = grb_temp.latlons()
mete_grid_points = np.column_stack((mete_lats.ravel(), mete_lons.ravel()))
mete_df_points = coord[['latitude', 'longitude']].values
mete_distances = cdist(mete_df_points, mete_grid_points, 'euclidean')
mete_nearest_indexes = np.argmin(mete_distances, axis=1)

var_temp = grb_temp.values.ravel()[mete_nearest_indexes]
var_rhum = grb_rhum.values.ravel()[mete_nearest_indexes]
var_pres = grb_pres.values.ravel()[mete_nearest_indexes]
var_uwind = grb_uwind.values.ravel()[mete_nearest_indexes]
var_vwind = grb_vwind.values.ravel()[mete_nearest_indexes]
var_shum = grb_shum.values.ravel()[mete_nearest_indexes]

coord['temperature'] = var_temp
coord['rel_humidity'] = var_rhum
coord['spec_humidity'] = var_shum
coord['pressure'] = var_pres
coord['u_wind'] = var_uwind
coord['v_wind'] = var_vwind

# total precipitation data not always available
try:
    grb_preci = meteo_grbs.select(name='Total Precipitation')[0]
    var_preci = grb_temp.values.ravel()[mete_nearest_indexes]
    coord['precipitation'] = var_preci
except:
    print('no precipitation data')
    coord['precipitation'] = 0


################ Matching AOD data ###############################
AOD_data = xr.open_dataset(AOD_dir, engine='netcdf4')
sat_sweep =  AOD_data.variables['goes_imager_projection'].attrs["sweep_angle_axis"]
sat_h = AOD_data.variables['goes_imager_projection'].attrs["perspective_point_height"]
xrcor = AOD_data.assign_coords(x=(AOD_data.x)*sat_h,y=(AOD_data.y)*sat_h)
#pro = Proj("+proj=geos +lon_0=-75 +h=" +str(round(sat_h)) + " +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True")
pro = Proj("+proj=geos +lon_0=-75 +h=35786023 + x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True") 

df_row = []
for index, row in coord.iterrows():
    longitude, latitude = coord.longitude[index], coord.latitude[index]
    long,lati = pro(longitude,latitude)
    value_location = xrcor.sel(y = lati,x = long, method='nearest', tolerance = 2000)
    AOD_value = value_location.AOD.data
    DQF_value = value_location.DQF.data
    df_row.append([latitude, longitude, AOD_value, DQF_value])
aod_df = pd.DataFrame(df_row, columns=["latitude", "longitude", "AOD", "DQF"])


################ calculate the solar azimuthal and zenith angles for each row in the DataFrame
SA_date = datetime.datetime(int(year), int(month), int(day), int(new_hour), tzinfo=datetime.timezone.utc)
coord['SZA'] = [float(90) - get_altitude(lat, lon, SA_date) for lat, lon in zip(coord['latitude'], coord['longitude'])]
coord['SAA'] = [get_azimuth(lat, lon, SA_date) for lat, lon in zip(coord['latitude'], coord['longitude'])]
coord = pd.merge(coord, aod_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])
print(coord)


down_dir = save_raw_df
try:
    os.mkdir(down_dir + year)
except FileExistsError:
    print("year folder exists")
try:
    os.mkdir(down_dir + year + '/' + month)
except FileExistsError:
    print("month folder exists")
try:
    os.mkdir(down_dir + year + '/' + month + '/' + day)
except FileExistsError:
    print("day folder exists")
try:
    os.mkdir(down_dir + year + '/' + month + '/' + day + '/' + new_hour)
except FileExistsError:
    print("hour folder exists")
out_df_path = down_dir + year + '/' + month + '/' + day + '/' + new_hour + '/' + 'df_part_' + str(ind) + '.csv'

coord.to_csv(out_df_path)


stage4_time = ct.time()
execution4_time = stage4_time - stage3_time
print(f"Matching time: {execution4_time} seconds")

