
# environment is pmtest

import s3fs
import cdsapi
import os
from datetime import datetime
import datetime
from os.path import exists
import xarray as xr
import pygrib
import pandas as pd
import sys
from pyproj import Proj
from pysolar.solar import *


fs = s3fs.S3FileSystem(anon=True)
c = cdsapi.Client()



# where the data download
directory = '/scratch/prabuddha/pm_est/raw_data/'

# where the ancillary dataFrame saved
ancillary_df = '/scratch/prabuddha/pm_est/ref_data/USA_latlon_ancillary_df_final.csv'



# Data available upto 7 days behind current day
# Go 7 days back and add 5 hours to get UTC time
week_ago = datetime.datetime.now() - datetime.timedelta(days=7) + datetime.timedelta(hours=5)
week_ago_str = str(week_ago)
year, month, day, hour = week_ago_str[0:4], week_ago_str[5:7], week_ago_str[8:10], week_ago_str[11:13]
time = hour + ':00'

#year = '2022'
#month = '12'
#day = '02'
#hour = '15'
#time = hour + ':00'






try: 
    os.mkdir(directory + year)
except FileExistsError:
    print("year folder exists")

try: 
    os.mkdir(directory + year + '/' + month)
except FileExistsError:
    print("month folder exists")

try: 
    os.mkdir(directory + year + '/' + month + '/' + day)
except FileExistsError:
    print("day folder exists")

try: 
    os.mkdir(directory + year + '/' + month + '/' + day + '/' + hour)
except FileExistsError:
    print("hour folder exists")





meteo_path = directory + year + '/' + month + '/' + day + '/' + hour + '/' + hour + '_meteoro'
humid_path = directory + year + '/' + month + '/' + day + '/' + hour + '/' + hour + '_humidity'
AOD_path = directory + year + '/' + month + '/' + day + '/' + hour + '/' + hour + '_AOD'

area = [50, -125, 25, -65,]  # US 
grid = [0.1, 0.1]




# download recent data (7 day ago) - Not real time
def down_meteo():
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            #'variable': '2m_temperature',
            'variable': ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 
            'surface_pressure', 'total_precipitation', 'skin_reservoir_content', 'evaporation', 'boundary_layer_height'],
            'year': year,
            'month': month,
            'day': day,
            'time': time,
            'area' : area,
            'grid' : grid,
        }, meteo_path)



def down_humid():
    data = c.retrieve('reanalysis-era5-pressure-levels',
        {
        'product_type': 'reanalysis',
        'variable': ['relative_humidity', 'specific_humidity', 'specific_rain_water_content'], 
        'pressure_level': '1000',
        'year': year,
        'month': month,
        'day': day,
        'time': time,
        #'format': 'grib',			# Supported format: grib and netcdf. Default: grib
        'area' : area, 
        'grid' : grid,		# Latitude/longitude grid.  (one latitude devided into 10 parts)         Default: 0.25 x 0.25  
        }, humid_path)



def down_AOD():
    julian_day = week_ago.strftime('%j')
    digit = len(str(julian_day))

    if digit == 1:
    	    julian_day = '0' + '0' + str(julian_day)
    elif digit == 2:
  	    julian_day = '0' + str(julian_day)
    else:
  	    julian_day = str(julian_day)

    AWS_path = 'noaa-goes16/ABI-L2-AODC/' + str(year) + '/' + julian_day + '/' + hour + '/*'
    files = fs.glob(AWS_path)
    fs.get(files[0], AOD_path)
 




if exists(meteo_path):
    print("meteorogical data exist")
else:
    down_meteo()

if exists(humid_path):
    print("humidity data exist")
else:
    down_humid()

if exists(AOD_path):
    print("AOD data exist")
else:
    down_AOD()





import time
#time.sleep(5)

###############################################################
# Make a DataFrame
coord_df = pd.read_csv(ancillary_df)
#coord = coord_df

start = 0
end  = len(coord_df)
step = round(end/20)
arr = [coord_df[start:step], coord_df[step:2*step], coord_df[2*step:3*step], coord_df[3*step:4*step], coord_df[4*step:5*step], coord_df[5*step:6*step], 
    coord_df[6*step:7*step], coord_df[7*step:8*step], coord_df[8*step:9*step], coord_df[9*step:10*step], coord_df[10*step:11*step], coord_df[11*step:12*step], 
    coord_df[12*step:13*step], coord_df[13*step:14*step], coord_df[14*step:15*step], coord_df[15*step:16*step], coord_df[16*step:17*step], coord_df[17*step:18*step], 
    coord_df[18*step:19*step], coord_df[19*step:20*step]]
ind = int(sys.argv[1])
coord = arr[ind]


meteo_grbs = pygrib.open(meteo_path)
temperature = meteo_grbs.select()[0]
dewpoint_temperature = meteo_grbs.select()[1]
u_component_of_wind = meteo_grbs.select()[2]
v_component_of_wind = meteo_grbs.select()[3]
pressure = meteo_grbs.select()[4]
precipitation = meteo_grbs.select()[5]
skin_reservoir_content = meteo_grbs.select()[6]
evaporation = meteo_grbs.select()[7]
blh = meteo_grbs.select()[8]


humid_grbs = pygrib.open(humid_path)
relative_humidity = humid_grbs.select()[0]

AOD_data = xr.open_dataset(AOD_path, engine='netcdf4')
sat_sweep =  AOD_data.variables['goes_imager_projection'].attrs["sweep_angle_axis"]
sat_h = AOD_data.variables['goes_imager_projection'].attrs["perspective_point_height"]
xrcor = AOD_data.assign_coords(x=(AOD_data.x)*sat_h,y=(AOD_data.y)*sat_h)

#pro = Proj("+proj=geos +lon_0=-75 +h=35786023 + x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True") 
pro = Proj("+proj=geos +lon_0=-75 +h=" +str(round(sat_h)) + " +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True") 
#AOD = AOD_data.AOD.data
#DQF = AOD_data.DQF.data
#xx_goes, yy_goes = np.meshgrid(AOD_data.x.data*sat_h, AOD_data.y.data*sat_h)
# Convert projected coordinates system to geographic coordinates system
#lons, lats = pro(xx_goes, yy_goes, inverse=True)

SA_date = datetime.datetime(int(year), int(month), int(day), int(hour), tzinfo=datetime.timezone.utc)



df_row = []
for index, row in coord.iterrows():
    longitude, latitude = coord.longitude[index], coord.latitude[index]
    tolerence = 0.05

    temp = temperature.data(lat1=latitude-tolerence, lat2=latitude+tolerence, lon1=longitude-tolerence, lon2=longitude+tolerence)
    dew = dewpoint_temperature.data(lat1=latitude-tolerence, lat2=latitude+tolerence, lon1=longitude-tolerence, lon2=longitude+tolerence)
    u_wind = u_component_of_wind.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
    v_wind = v_component_of_wind.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
    pres = pressure.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
    precip = precipitation.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
    skin = skin_reservoir_content.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
    evap = evaporation.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
    boundry = blh.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)

    r_humid = relative_humidity.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)

    lon,lat = pro(longitude,latitude)
    value_location = xrcor.sel(y = lat,x = lon, method='nearest', tolerance = 2000)
    AOD_value = value_location.AOD.data
    DQF_value = value_location.DQF.data

    SAA = float(90) - get_altitude(latitude, longitude, SA_date)
    SZA = get_azimuth(latitude, longitude, SA_date)

    df_row.append([latitude, longitude, temp[0][0][0], dew[0][0][0], u_wind[0][0][0], v_wind[0][0][0], pres[0][0][0], precip[0][0][0], skin[0][0][0], evap[0][0][0], boundry[0][0][0], r_humid[0][0][0], AOD_value, DQF_value, SAA, SZA])
    #df_row.append([latitude, longitude, temp[0][0][0], r_humid[0][0][0], AOD_value, DQF_value])
    print(index)




df = pd.DataFrame(df_row, columns=["latitude", "longitude", "temperature", "dew_temp", "u_wind", "v_wind", "pressure", "precipitation", "skin_reservoir", "evaporation", "blh", "rel_humidity", "AOD", "DQF", "SSA", "SZA"])
#df = pd.DataFrame(df_row, columns=["latitude", "longitude", "temperature", "rel_humidity", "AOD", "DQF"])
#df.to_csv('/Users/prabu/Desktop/MyGitHub/Satellite/Satellite-Project/py_ml/realTimeData/dynamic_df_on_' + year + '_' + month + '_' + day + '_' + hour + '.csv')

merged_df = pd.merge(coord, df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])

down_dir = '/scratch/prabuddha/pm_est/raw_df/'
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
    os.mkdir(down_dir + year + '/' + month + '/' + day + '/' + hour)
except FileExistsError:
    print("hour folder exists")

out_df_path = down_dir + year + '/' + month + '/' + day + '/' + hour + '/' + 'df_part_' + str(ind) + '.csv'
merged_df.to_csv(out_df_path)

#merged_df.to_csv('/scratch/prabuddha/pm_est/raw_df/complete_df_on_' + year + '_' + month + '_' + day + '_' + hour + '_test' + '.csv')



print("done")
