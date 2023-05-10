

# conda env = pmtest

import pandas as pd
from os.path import exists
import xarray as xr
import pygrib
from datetime import datetime
import datetime
from pysolar.solar import *
from pyproj import Proj
import sys


#year = ['2020']
#month = ['11']
#day = ['1']
#hour = ['00']


pmMonthlyPath = '/scratch/prabuddha/2yrHist_train/monthlyDF/pm2_5/'
meteoPath = '/scratch/prabuddha/2yrHist_train/rawData/meteo_AOD/'
dailyOut = '/scratch/prabuddha/2yrHist_train/dailyDF/dynamic/'
ancillary_df = '/scratch/prabuddha/2yrHist_train/pmSensorCoordDF/pm_sensor_loc_round_and_ancillary_and_building.csv'

arr = [['2022','05','14'],['2022','05','25'],['2022','06','11'],['2022','06','22'],['2022','07','9'],['2022','07','23'],['2022','08','6'],['2022','08','20'],['2022','09','3'],['2022','09','17'],['2022','09','26'],['2022','10','1'],['2022','10','15'],['2022','10','29']]

hour = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']


ind = int(sys.argv[1])
date = arr[ind]


y = date[0]
m = date[1]
d = date[2]

coord_df = pd.read_csv(ancillary_df)


df_row = []
for h in hour:
    meteoCSV = meteoPath + y + '/' + m + '/' + d + '/' + h + '/meteoro'
    humidCSV = meteoPath + y + '/' + m + '/' + d + '/' + h + '/humidity'
    AODcsv = meteoPath + y + '/' + m + '/' + d + '/' + h + '/AOD'
    if exists(meteoCSV) and exists(humidCSV) and exists(AODcsv):
        meteoDf = pygrib.open(meteoCSV)
        humidDf = pygrib.open(humidCSV)
        AODdf = xr.open_dataset(AODcsv, engine='netcdf4')

        temperature = meteoDf.select()[0]
        dewpoint_temperature = meteoDf.select()[1]
        u_component_of_wind = meteoDf.select()[2]
        v_component_of_wind = meteoDf.select()[3]
        pressure = meteoDf.select()[4]
        precipitation = meteoDf.select()[5]
        skin_reservoir_content = meteoDf.select()[6]
        evaporation = meteoDf.select()[7]
        blh = meteoDf.select()[8]
        lake_cover = meteoDf.select()[9]
        leaf_indx_high_veg = meteoDf.select()[10]
        leaf_indx_low_veg = meteoDf.select()[11]
        snowfall = meteoDf.select()[13]
        surface_net_solar_radi = meteoDf.select()[14]
        tot_cloud_cover = meteoDf.select()[15]

        rel_humid = humidDf.select()[0]
        spe_humid = humidDf.select()[1]
        specific_rain_water_content = humidDf.select()[2]

        sat_sweep =  AODdf.variables['goes_imager_projection'].attrs["sweep_angle_axis"]
        sat_h = AODdf.variables['goes_imager_projection'].attrs["perspective_point_height"]
        xrcor = AODdf.assign_coords(x=(AODdf.x)*sat_h,y=(AODdf.y)*sat_h)
        pro = Proj("+proj=geos +lon_0=-75 +h=" +str(round(sat_h)) + " +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs=True")

        SA_date = datetime.datetime(int(y), int(m), int(d), int(h), tzinfo=datetime.timezone.utc)

        #df_row = []
        for index, row in coord_df.iterrows():
            longitude, latitude = coord_df.longitude[index], coord_df.latitude[index]
            cropland = coord_df.cropland[index]
            landcover = coord_df.landcover[index]
            population_density = coord_df.population_density[index]
            soiltype = coord_df.soiltype[index]
            lithology = coord_df.lithology[index]
            elevation = coord_df.elevation[index]
            num_building = coord_df.num_building[index]
            avg_distance = coord_df.avg_distance[index]
            tot_area = coord_df.tot_area[index]
            tolerence = 0.05

            try:
                temp = temperature.data(lat1=latitude-tolerence, lat2=latitude+tolerence, lon1=longitude-tolerence, lon2=longitude+tolerence)
                dew = dewpoint_temperature.data(lat1=latitude-tolerence, lat2=latitude+tolerence, lon1=longitude-tolerence, lon2=longitude+tolerence)
                u_wind = u_component_of_wind.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                v_wind = v_component_of_wind.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                pres = pressure.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                precip = precipitation.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                skin = skin_reservoir_content.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                evap = evaporation.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                boundry = blh.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                lake = lake_cover.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                high_veg = leaf_indx_high_veg.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                low_veg = leaf_indx_low_veg.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                snow = snowfall.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                solr_radi = surface_net_solar_radi.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                cloud = tot_cloud_cover.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)

                r_humid = rel_humid.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                s_humid = spe_humid.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)
                rain = specific_rain_water_content.data(lat1=latitude-tolerence,lat2=latitude+tolerence,lon1=longitude-tolerence,lon2=longitude+tolerence)

                lon,lat = pro(longitude,latitude)
                value_location = xrcor.sel(y = lat,x = lon, method='nearest', tolerance = 2000)
                AOD_value = value_location.AOD.data
                DQF_value = value_location.DQF.data

                SAA = float(90) - get_altitude(latitude, longitude, SA_date)
                SZA = get_azimuth(latitude, longitude, SA_date)

                df_row.append([SA_date, latitude, longitude, cropland, landcover, population_density, soiltype, lithology, elevation, num_building, avg_distance, tot_area, temp[0][0][0], dew[0][0][0], u_wind[0][0][0], v_wind[0][0][0], pres[0][0][0], precip[0][0][0], skin[0][0][0], evap[0][0][0], boundry[0][0][0], lake[0][0][0], high_veg[0][0][0], low_veg[0][0][0], snow[0][0][0], solr_radi[0][0][0],cloud[0][0][0], r_humid[0][0][0], s_humid[0][0][0], rain[0][0][0], AOD_value, DQF_value, SAA, SZA])

            except:
                print('problem on ' + y + '-' + m + '-' + d + '-' + h)

df = pd.DataFrame(df_row, columns=["dateTime", "latitude", "longitude", "cropland", "landcover", "population_density", "soiltype", "lithology", "elevation", "num_building", "avg_distance", "tot_area", "temperature", "dew_temp", "u_wind", "v_wind", "pressure", "precipitation", "skin_reservoir", "evaporation", "blh", "lake_cover", "high_veg", "low_veg", "snowfall", "solar_radiation", "cloud_cover", "rel_humidity", "speci_humidity", "rain_content", "AOD", "DQF", "SSA", "SZA"])
df.to_csv('/scratch/prabuddha/2yrHist_train/dailyDF/dynamic/df_' + y + '_' + m + '_' + d + '.csv')




