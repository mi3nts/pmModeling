# conda environment pmtest

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
from os.path import exists
import xarray as xr
import datetime



# where the input dataframes saved.
dfs_path = '/scratch/prabuddha/pm_est_fc/data/raw_df/'
# trained model direction
#model_dir = "/scratch/prabuddha/pm_est_fc/model/new_ETreg_random.joblib"
model_dir = "/scratch/prabuddha/pm_est_fc/model/fc_est100_r1_ETreg_random.joblib"
# where the US boundary shape folder
USboundary_shape = '/scratch/prabuddha/pm_est/USBoundry/cb_2018_us_state_500k.shp'
# saving location of output pm estimated dataframe
output_df_dir = '/scratch/prabuddha/pm_est_fc/data/fc_df/test2_'
# saving location of output pm scatter map
output_map = "/scratch/prabuddha/pm_est_fc/data/fc_plot/test2_"



#week_ago = datetime.datetime.now() - datetime.timedelta(days=7) + datetime.timedelta(hours=5)
#week_ago_str = str(week_ago)
#year, month, day, hour = week_ago_str[0:4], week_ago_str[5:7], week_ago_str[8:10], week_ago_str[11:13]
#time = hour + ':00'
#year = '2023'
#month = '05'
#day = '17'
#hour = '12'
#time = hour + ':00'


# crnt_time is current UTC time
crnt_time = datetime.datetime.now() + datetime.timedelta(hours=5)
# this is the time last data set downloaded when this code run. 
last_data_time_utc = datetime.datetime.now() + datetime.timedelta(hours=5)
last_data_time_utc = str(last_data_time_utc)
#str_crnt_time = str(crnt_time)
year, month, day, hour = last_data_time_utc[0:4], last_data_time_utc[5:7], last_data_time_utc[8:10], last_data_time_utc[11:13]
print("Current UTC time " + last_data_time_utc)
time = hour + ':00'



#down_dir = '/scratch/prabuddha/pm_est/raw_df/'
path = dfs_path + year + '/' + month + '/' + day + '/' + hour + '/'
csv_file = 'df_' + year + '_' + month + '_' + day + '_' + hour + '.csv'
#path = '/scratch/prabuddha/pm_est_fc/data/raw_df/2023/04/15/15/'
#csv_file = 'df_2023_04_15_15.csv'
#files = os.listdir(path) 
#print(files)
#files = os.listdir(dfs_path) 
"""
if exists(path):
    print("Data exist")
    est()
else:
    #down_meteo()
    print('No data')
"""

predictors = ['Cropland', 'Landcover', 'Soil type', 'Lithology', 'Population density', 'Elevation','Number of building', 'Avgerage building distance', 'Total building area',
        'Temperature','Pressure','Total precipitation', 'Relative humidity', 'Specific humidity', 'AOD - Aerosol optical depth','DQF - AOD Data quality flags',
        'Solar azimuth angle', 'Solar zenith angle', 'Wind speed', 'Boundary layer height', 'Month']

def make_prediction2(modelName, est_df):
        predictions_train = pd.DataFrame(modelName.predict(est_df),columns=["Predictions"])
        return predictions_train


def est():
    #df0 = pd.read_csv(path + files[0])
    #df1 = pd.read_csv(path + files[1])
    #df2 = pd.read_csv(path + files[2])
    #df3 = pd.read_csv(path + files[3])
    #df4 = pd.read_csv(path + files[4])
    #df5 = pd.read_csv(path + files[5])
    #df6 = pd.read_csv(path + files[6])
    #df7 = pd.read_csv(path + files[7])
    #df8 = pd.read_csv(path + files[8])
    #df9 = pd.read_csv(path + files[9])
    #df10 = pd.read_csv(path + files[10])
    #df11 = pd.read_csv(path + files[11])
    #df12 = pd.read_csv(path + files[12])
    #df13 = pd.read_csv(path + files[13])
    #df14 = pd.read_csv(path + files[14])
    #df15 = pd.read_csv(path + files[15])
    #df16 = pd.read_csv(path + files[16])
    #df17 = pd.read_csv(path + files[17])
    #df18 = pd.read_csv(path + files[18])
    #df19 = pd.read_csv(path + files[19])
    #df20 = pd.read_csv(path + files[20])
    #df21 = pd.read_csv(path + files[21])
    #df22 = pd.read_csv(path + files[22])
    #df23 = pd.read_csv(path + files[23])
    #df24 = pd.read_csv(path + files[24])
    #df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20, df21, df22, df23, df24], ignore_index=True)
    df = pd.read_csv(path + csv_file)
    df = df.drop('Unnamed: 0.1', axis=1)
    df = df.drop('Unnamed: 0', axis=1)
    df['DQF'] = df['DQF'].fillna(0)
    df["uv10"]= np.sqrt(df["u_wind"].values*df["u_wind"].values +df["v_wind"].values*df["v_wind"].values)
    df['month'] = month
    df = df.rename(columns={'cropland':'Cropland', 'landcover':'Landcover', 'soil':'Soil type', 'lithology':'Lithology', 'population':'Population density', 'elevation':'Elevation',
        'num_build':'Number of building', 'avg_dist':'Avgerage building distance', 'tot_area':'Total building area', 'temperature':'Temperature','pressure':'Pressure',
        'precipitation':'Total precipitation', 'rel_humidity':'Relative humidity', 'spec_humidity':'Specific humidity', 'AOD':'AOD - Aerosol optical depth', 
        'DQF':'DQF - AOD Data quality flags','SAA':'Solar azimuth angle', 'SZA':'Solar zenith angle', 'uv10':'Wind speed', 'pblh':'Boundary layer height', 'month':'Month'})


    df_noAOD = df[df['AOD - Aerosol optical depth'].isna()]
    df_noAOD["pm_est"] = 0.0


    df_AOD = df.dropna(subset=['AOD - Aerosol optical depth'])
    df_AOD = df_AOD.reset_index()
    df_AOD = df_AOD.drop('index', axis=1)
    est_df = pd.DataFrame(df_AOD, columns = predictors)

    loaded_rf = joblib.load(model_dir)
    predicted_train_valid2 = make_prediction2(loaded_rf,est_df)
    df_AOD["pm_est"] = predicted_train_valid2
    final_df = pd.concat([df_AOD, df_noAOD], ignore_index=True)
    final_df['pm_est'] = final_df['pm_est'].replace(0, np.nan)

    date_time = year + "-" + month + "-" + day + " " + hour + ":00:00"
    final_df['time'] = date_time
    final_df['time'] = pd.to_datetime(final_df['time'])
    final_df['time'] = final_df['time'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    final_df['time'] = pd.to_datetime(final_df['time'])

    final_df.to_csv(output_df_dir + 'df_' + year + '_' + month + '_' + day + '_' + hour + '.csv')


    ######## make netCDF file #################################
    nc_df = final_df[['time', 'latitude', 'longitude', 'pm_est']]
    nc_df = nc_df.set_index(['time', 'latitude', 'longitude'])

    # Convert the DataFrame to an xarray dataset
    ds = nc_df.to_xarray()

    units = 'hours since 1970-01-01 00:00:00'
    calendar = 'standard'
    times = pd.to_datetime(ds['time'].values)
    ds['time'] = xr.DataArray([t.year for t in times], dims='time', attrs={'units': 'year'})

    # Add units and attributes to the coordinate variables
    ds['latitude'].attrs['units'] = 'degrees_north'
    ds['longitude'].attrs['units'] = 'degrees_east'

    # Add attributes to the data variable
    ds['pm_est'].attrs['long_name'] = 'PM concentration'
    ds['pm_est'].attrs['units'] = 'micrograms/m^3'

    # Add global attributes
    ds.attrs['Conventions'] = 'CF-1.6'
    ds.attrs['history'] = 'Converted from pandas dataframe to NetCDF using xarray'
    # Save the dataset to a NetCDF file
    ds.to_netcdf(output_df_dir + 'df_' + year + '_' + month + '_' + day + '_' + hour + '.nc')



    '''
    # Convert the pandas dataframe to an xarray dataset
    nc_ds = xr.Dataset.from_dataframe(final_df.set_index(['time', 'latitude', 'longitude']))
    # Write the xarray dataset to a NetCDF file
    nc_ds.to_netcdf(output_df_dir + 'df_' + year + '_' + month + '_' + day + '_' + hour + 'test.nc')
    '''

    #df_pv = final_df.pivot(index="latitude", columns="longitude", values='pm_est')
    #da = xr.DataArray(data=df_pv.values, name="pm", dims=["lat", "lon"], coords=[df_pv.index ,df_pv.columns])
    #da.to_netcdf(output_df_dir + 'df_' + year + '_' + month + '_' + day + '_' + hour + '.nc')
    '''
    nc_lat = df["latitude"]
    nc_lon = df["longitude"]
    nc_pm = df["pm_est"]
    nc_pm, nc_lat , nc_lon = np.around([nc_pm, nc_lat, nc_lon], decimals=2)
    nc_df = pd.DataFrame({"pm2_5":nc_pm, "latitude":nc_lat, "longitude":nc_lon})
    idx = pd.MultiIndex.from_arrays(arrays=[nc_lat,nc_lon], names=["latitude","longitude"])
    s = pd.Series(data=nc_pm, index=idx)
    nc_da = xr.DataArray.from_series(s)
    #nc_da.to_netcdf(output_df_dir + 'df_' + year + '_' + month + '_' + day + '_' + hour + '.nc')
    '''
    fname = USboundary_shape
    shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='none')
    fig=plt.figure(figsize=[50,30])
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.add_feature(shape_feature,edgecolor='blue')
    ax.coastlines()
    c = ax.scatter(final_df["longitude"], final_df["latitude"], c = final_df["pm_est"], s=15, cmap='turbo',vmin=0,vmax=35, marker='s', linewidth=0)
    fig.colorbar(c, ax=ax, shrink=0.55)
    plt.title("PM2.5 Estimation " + year + "-" + month + "-" + day + " " + time, fontsize=50)
    plt.savefig(output_map + "pm_est_" + year + "_" + month + "_" + day + "_" + hour + ".png")
    print("done")
    



if exists(path):
    print("Data exist")
    est()
else:
    #down_meteo()
    print('No data')

