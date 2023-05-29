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
from scipy.interpolate import griddata
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import cartopy.crs as ccrs
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
import matplotlib.pyplot as plt
import time as ct

stage_time = ct.time()

# where the data download
directory = '/scratch/prabuddha/2yrHist_train/rawData/meteo_AOD/'
# where the ancillary dataFrame saved
ancillary_df = '/scratch/prabuddha/pm_est/ref_data/USA_latlon_ancillary_df_final.csv'
# Save output results
down_dir = '/scratch/prabuddha/dust_events/Dixie_Fire_2021/output/'
down_dir_plot = '/scratch/prabuddha/dust_events/Dixie_Fire_2021/output_plots/'


year = '2021'
month = '07'
day = '16'
hour = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
#hour = ['04', '05', '06', '07', '08', '09']

# Ancillary dataframe
#coord_df = pd.read_csv(ancillary_df)
#print(len(coord_df))
#coord = coord_df[0:10]
#coord = coord_df
#print(coord)


# Interpolate the selected variables onto the latitude and longitude coordinates in the dataframe
# Select the meteorological data
'''
selected_vars = ['2 metre temperature', '2 metre dewpoint temperature', '10 metre U wind component', '10 metre V wind component', 'Surface pressure', 'Total precipitati
on', 'Skin reservoir content', 'Evaporation', 'Boundary layer height', 'Lake cover', 'Leaf area index, high vegetation', 'Leaf area index, low vegetation', 'Snowfall', 
'Surface net solar radiation', 'Total cloud cover']
'''
selected_vars = ['2 metre temperature', '2 metre dewpoint temperature', '10 metre U wind component', '10 metre V wind component', 'Surface pressure', 
        'Total precipitation', 'Skin reservoir content', 'Evaporation', 'Boundary layer height', 'Lake cover', 'Leaf area index, high vegetation', 
        'Leaf area index, low vegetation', 'Total cloud cover']
# Select the himidity data
selected_vars2 = ['Relative humidity', 'Specific humidity', 'Specific rain water content']

# Select the variables for PM estimation
predictors = ['cropland', 'landcover', 'soil', 'lithology', 'population', 'elevation', 'num_build', 'avg_dist', 'tot_area', '2 metre temperature', 
        '2 metre dewpoint temperature', 'Surface pressure', 'Total precipitation', 'Skin reservoir content', 'Evaporation', 'Boundary layer height', 
        'Lake cover', 'Leaf area index, high vegetation', 'Leaf area index, low vegetation', 'Total cloud cover', 'Relative humidity', 'Specific humidity', 
        'Specific rain water content', 'AOD', 'DQF', 'SAA', 'SZA', 'uv10', 'Month']

# US map
fname = '/scratch/prabuddha/pm_est/USBoundry/cb_2018_us_state_500k.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='none')





def dataMatch(Hour):
    time_ = Hour + ':00'

    meteo_path = directory + year + '/' + month + '/' + day + '/' + Hour + '/' + 'meteoro'
    humid_path = directory + year + '/' + month + '/' + day + '/' + Hour + '/' + 'humidity'
    AOD_path = directory + year + '/' + month + '/' + day + '/' + Hour + '/' + 'AOD'

    ######### matching meteorological data #############################
    coord_df = pd.read_csv(ancillary_df)
    coord = coord_df
    #coord = coord_df[0:100]

    meteo_grbs = pygrib.open(meteo_path)
    detail = meteo_grbs.select()
    grb_temp = meteo_grbs.select(name='2 metre temperature')[0]
    mete_lats, mete_lons = grb_temp.latlons()

    interpolated_data = dict((var, griddata((mete_lats.ravel(), mete_lons.ravel()), meteo_grbs.select(name=var)[0].values.ravel(), 
        (coord['latitude'], coord['longitude']), method='linear')) for var in selected_vars)
    coord.reset_index(drop=True, inplace=True)
    coord = pd.concat([coord, pd.DataFrame(interpolated_data)], axis=1)

    humid_grbs = pygrib.open(humid_path)
    humid_detail = humid_grbs.select()
    #print(humid_detail)

    interpolated_data2 = dict((var, griddata((mete_lats.ravel(), mete_lons.ravel()), humid_grbs.select(name=var)[0].values.ravel(), 
        (coord['latitude'], coord['longitude']),method='linear')) for var in selected_vars2)
    coord.reset_index(drop=True, inplace=True)
    coord = pd.concat([coord, pd.DataFrame(interpolated_data2)], axis=1)
    #print(coord)

    ################ Matching AOD data ###############################
    AOD_data = xr.open_dataset(AOD_path, engine='netcdf4')
    sat_sweep =  AOD_data.variables['goes_imager_projection'].attrs["sweep_angle_axis"]
    sat_h = AOD_data.variables['goes_imager_projection'].attrs["perspective_point_height"]
    xrcor = AOD_data.assign_coords(x=(AOD_data.x)*sat_h,y=(AOD_data.y)*sat_h)
    #pro = Proj("+proj=geos +lon_0=-75 +h=" +str(round(sat_h)) + " +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True")
    pro = Proj("+proj=geos +lon_0=-75 +h=35786023 + x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True") 
    AOD_val = AOD_data.AOD.data
    DQF_val = AOD_data.DQF.data
    xx_goes, yy_goes = np.meshgrid(AOD_data.x.data*sat_h, AOD_data.y.data*sat_h)
    aod_lons, aod_lats = pro(xx_goes, yy_goes, inverse=True)
    # removing inf from the coordinates
    lon_inf_mask = np.isinf(aod_lons)
    aod_lons[lon_inf_mask] = np.nan
    lat_inf_mask = np.isinf(aod_lats)
    aod_lats[lat_inf_mask] = np.nan
    valid_indices = ~np.isnan(aod_lons) & ~np.isnan(aod_lats)
    # remove NaN values and their corresponding t values
    aod_lon_clean = aod_lons[valid_indices]
    aod_lat_clean = aod_lats[valid_indices]
    AOD_val_clean = AOD_val[valid_indices]
    DQF_val_clean = DQF_val[valid_indices]
    xx = coord['longitude']
    yy = coord['latitude']
    AOD_interp = griddata((aod_lon_clean, aod_lat_clean), AOD_val_clean, (xx, yy), method='linear')
    DQF_interp = griddata((aod_lon_clean, aod_lat_clean), DQF_val_clean, (xx, yy), method='nearest')
    coord['AOD'] = AOD_interp
    coord['DQF'] = DQF_interp
    #print(coord)

    ################ calculate the solar azimuthal and zenith angles for each row in the DataFrame
    SA_date = datetime.datetime(int(year), int(month), int(day), int(Hour), tzinfo=datetime.timezone.utc)
    coord['SZA'] = [float(90) - get_altitude(lat, lon, SA_date) for lat, lon in zip(coord['latitude'], coord['longitude'])]
    coord['SAA'] = [get_azimuth(lat, lon, SA_date) for lat, lon in zip(coord['latitude'], coord['longitude'])]
    #coord = pd.merge(coord, aod_df,  how='left', left_on=["latitude","longitude"], right_on = ["latitude","longitude"])
    coord['Month'] = month
    coord['datetime'] = year + '-' + month + '-' + day + ' ' + time_
    #print(coord)
    #print(coord.columns.tolist())
    return coord


def pmest(df):
    df = df.drop('Unnamed: 0', axis=1)
    #df['AOD'] = df['AOD'].fillna(0)
    df['DQF'] = df['DQF'].fillna(0)
    #print(df)
    df["uv10"]= np.sqrt(df["10 metre U wind component"].values*df["10 metre U wind component"].values +df["10 metre V wind component"].values*df["10 metre V wind component"].values)
    #print(df)
    #print(list(df.columns))
    #est_df = pd.DataFrame(df, columns = predictors)
    
    df_noAOD = df[df['AOD'].isna()]
    df_noAOD["pm_est"] = 0.0
    df_AOD = df.dropna(subset=['AOD'])
    df_AOD = df_AOD.reset_index()
    df_AOD = df_AOD.drop('index', axis=1)

    est_df = pd.DataFrame(df_AOD, columns = predictors)

    loaded_rf = joblib.load("/scratch/prabuddha/dust_events/hist_model/hist_est100_r1_ETreg_random.joblib")
    predictions_train = pd.DataFrame(loaded_rf.predict(est_df),columns=["Predictions"])
    df_AOD["pm_est"] = predictions_train
    final_df = pd.concat([df_AOD, df_noAOD], ignore_index=True)
    final_df['pm_est'] = final_df['pm_est'].replace(0, np.nan)
    return final_df

''' TO BE DELETE
    # Get the size of the DataFrame
    rows, columns = est_df.shape
    # Print the size
    #print("Number of rows:", rows)
    #print("Number of columns:", columns)
    if rows == 0:
        print("Number of rows:", rows)
        df_AOD["pm_est"] = 0.0
        final_df = pd.concat([df_AOD, df_noAOD], ignore_index=True)
        final_df['pm_est'] = final_df['pm_est'].replace(0, np.nan)
        return final_df
    else:
        loaded_rf = joblib.load("/scratch/prabuddha/dust_events/hist_model/hist_est100_r1_ETreg_random.joblib")
        print("PM estimating")
        predictions_train = pd.DataFrame(loaded_rf.predict(est_df),columns=["Predictions"])
        df_AOD["pm_est"] = predictions_train
        final_df = pd.concat([df_AOD, df_noAOD], ignore_index=True)
        final_df['pm_est'] = final_df['pm_est'].replace(0, np.nan)
        return final_df
'''


for h in hour:
    matched_df = dataMatch(h)
    df_AOD = matched_df.dropna(subset=['AOD'])
    rows, columns = df_AOD.shape
    if rows == 0:
        print(year + "-" + month + "-" + day + " " + h + " dataframe has no AOD")
    else:
        final_df = pmest(matched_df)
        print(final_df)

        try:
            os.mkdir(down_dir + year)
        except FileExistsError:
            print(year + " year folder exists")
        try:
            os.mkdir(down_dir + year + '/' + month)
        except FileExistsError:
            print(month + " month folder exists")
        try:
            os.mkdir(down_dir + year + '/' + month + '/' + day)
        except FileExistsError:
            print(day + " day folder exists")
        '''
        try:
            os.mkdir(down_dir + year + '/' + month + '/' + day + '/' + hour)
        except FileExistsError:
            print("hour folder exists")
        '''
        out_df_path = down_dir + year + '/' + month + '/' + day + '/' + 'df_' + h + '.csv'
        final_df.to_csv(out_df_path)

        fig=plt.figure(figsize=[50,30])
        ax = plt.subplot(projection=ccrs.PlateCarree())
        ax.add_feature(shape_feature,edgecolor='blue')
        #ax.gridlines(draw_labels=True)
        ax.coastlines()
        #c = ax.pcolor(final_df2["longitude"], final_df2["latitude"],final_df2["cropland"],cmap='Blues',vmin=-300,vmax=300)
        #c = ax.scatter(building_coord["longitude"], building_coord["latitude"], s=3)
        c = ax.scatter(final_df["longitude"], final_df["latitude"], c = final_df["pm_est"], s=15, cmap='turbo',vmin=0,vmax=25, marker='s', linewidth=0)
        fig.colorbar(c, ax=ax, shrink=0.55)
        plt.title("PM2.5 Estimation " + year + "-" + month + "-" + day + " " + h, fontsize=50)
        plt.savefig(down_dir_plot + year + "_" + month + "_" + day + "_" + h + ".png")




stage2_time = ct.time()
execution3_time = stage2_time - stage_time
print(f"Variable assign time: {execution3_time} seconds")

