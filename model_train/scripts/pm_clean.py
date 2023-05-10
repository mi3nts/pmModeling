
# UTD pm data in daily files.
# EPA pm data in monthly files.
# OpenAQ data in hourly files.
# read those file seperatly and clean and combine and save monthly files.
# make coordinate monthly files.
# make total coordinate file.

# --------------------------------------------------------------------------------

# conda env = pmtest

from os.path import exists
import pandas as pd
from datetime import datetime

year = ['2022']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
#month = ['01']
day = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
hour = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
#month = ['01']
#day = ['1']
#hour = ['00', '01']



UTDrawDataPath = "/scratch/prabuddha/2yrHist_train/rawData/MINTS_pm/raw/"
EPArawDataPath = "/scratch/prabuddha/2yrHist_train/rawData/EPA_pm/"
OpenAQrawDataPath = "/scratch/prabuddha/2yrHist_train/rawData/meteo_AOD/"

monthlyOutPutPath = "/scratch/prabuddha/2yrHist_train/monthlyDF/pm2_5_with_source/"

UTD_ids = ['001e06318c91', '001e06373724', '001e0637371e', '001e06373996', '001e063739c7', '001e063739cd', '001e0636e547', '001e06323a37', '001e0637372e',
        '001e06318c9d', '001e06305a12', '001e06323a12', '001e06318cd1', '001e06305a61', '001e06323a05', '001e06305a57', '001e063059c2', '001e06318c28',
        '001e06305a6b', '001e063239e3', '001e06305a6c', '001e063239e6', '001e06305a0a', '001e06318cee', '001e06318cf1', '001e063059c1', '001e063239e4',
        '001e06323952', '001e06305a2c', '001e06323a06', '001e06305a12', '001e06303b4f', '001e06318c3c']
#UTD_ids = ['001e06318c91', '001e063059c2']
ips_var = ['dateTime', 'pm2_5']
gps_var = ['dateTime', 'latitudeCoordinate', 'longitudeCoordinate']
EPA_var = ['UTC', 'Value', 'Latitude', 'Longitude']
OpenAQ_var = ['value', 'date.utc', 'coordinates.latitude', 'coordinates.longitude']


def UTD(Y, M):
    monthly_df = pd.DataFrame()
    for i in UTD_ids:
        ddf = pd.DataFrame()
        for d in day:
            digit = len(d)
            if digit == 1:
                d = '0' + d
            else:
                d = d
            IPS = 'MINTS_' + i + '_IPS7100_' + Y + '_' + M + '_' + d + '.csv'
            GPS = 'MINTS_' + i + '_GPSGPGGA2_' + Y + '_' + M + '_' + d + '.csv'
            IPS_link = UTDrawDataPath + i + '/' + y + '/' + '/' + m + '/' + d + '/' + IPS
            GPS_link = UTDrawDataPath + i + '/' + y + '/' + '/' + m + '/' + d + '/' + GPS
            if exists(IPS_link) and exists(GPS_link):
                try:
                    ips_file = pd.read_csv(IPS_link)
                    ips_df = pd.DataFrame(ips_file, columns = ips_var)
                    ips_df['dateTime'] = ips_df['dateTime'].str.slice(0, 18)
                    ips_df['dateTime'] = pd.to_datetime(ips_df['dateTime'])
                    ipsDF = ips_df.groupby(pd.Grouper(key='dateTime', freq='1h')).mean()
                    gps_file = pd.read_csv(GPS_link)
                    gps_df = pd.DataFrame(gps_file, columns = gps_var)
                    gps_df['dateTime'] = gps_df['dateTime'].str.slice(0, 18)
                    gps_df['dateTime'] = pd.to_datetime(gps_df['dateTime'])
                    gpsDF = gps_df.groupby(pd.Grouper(key='dateTime', freq='1h')).mean()
                    gpsDF = gpsDF.rename({'latitudeCoordinate': 'latitude', 'longitudeCoordinate': 'longitude'}, axis=1)
                    final_df = pd.merge(ipsDF, gpsDF, how='outer', on='dateTime')
                    
                    ddf = pd.concat([ddf, final_df])
                    #print(ddf)
                except:
                    print('problem UTD data on ' + Y + '-' + M + '-' + d)

        monthly_df = pd.concat([monthly_df, ddf])
        monthly_df.index.name='dateTime'
        #monthly_df = monthly_df.rename_axis(index='foo', columns="dateTime")
        #print(monthly_df)
        monthly_df['pm_source'] = 2
        #monthly_df = monthly_df.sort_values(by='dateTime')
        #print(monthly_df)
    
    try:
        monthly_df = monthly_df.sort_values(by='dateTime')
        #print(monthly_df)
        #monthly_df.to_csv(monthlyOutPutPath + 'UTD_df_' + Y + '_' + M + '.csv')
        return monthly_df
    except:
        print("no UTD data on " + Y + "-" + M)
    print("utd " + Y + "-" + M + " done")
    


def EPA(Y, M):
    EPA_link = EPArawDataPath + Y + "_" + M + ".csv"
    if exists(EPA_link):
        EPA_file = pd.read_csv(EPA_link)
        EPA_df = pd.DataFrame(EPA_file, columns = EPA_var)
        EPA_df = EPA_df.rename({'Latitude':'latitude', 'Longitude':'longitude', 'UTC':'dateTime', 'Value':'pm2_5'}, axis=1)
        EPA_df['dateTime'] = pd.to_datetime(EPA_df['dateTime'])
        EPA_df = EPA_df.set_index(['dateTime'])
        EPA_df = EPA_df.sort_values(by='dateTime')
        EPA_df['pm_source'] = 0
        #EPA_df.to_csv(monthlyOutPutPath + 'EPA_df_' + Y + '_' + M + '.csv')
        return EPA_df
        #print(EPA_df)
    else:
        print('no EPA data on' + Y + '-' + M)
    print("epa " + Y + "-" + M + " done")


def OpenAQ(Y, M):
    Ddf = pd.DataFrame()
    for d in day:
        Hdf = pd.DataFrame()
        for h in hour:
            OpenAQ_link = OpenAQrawDataPath + Y + '/' + M + '/' + d + '/' + h + '/OpenAQ_pm.csv'
            if exists(OpenAQ_link):
                OpenAQ_file = pd.read_csv(OpenAQ_link)
                OpenAQ_df = pd.DataFrame(OpenAQ_file, columns = OpenAQ_var)
                OpenAQ_df = OpenAQ_df.rename({'coordinates.latitude':'latitude', 'coordinates.longitude':'longitude', 'date.utc':'dateTime', 'value':'pm2_5'}, axis=1)
                OpenAQ_df['dateTime'] = OpenAQ_df['dateTime'].str.slice(0, 18)
                OpenAQ_df['dateTime'] = pd.to_datetime(OpenAQ_df['dateTime'])
                OpenAQ_df = OpenAQ_df.set_index(['dateTime'])
                Hdf = pd.concat([Hdf, OpenAQ_df])
        Ddf = pd.concat([Ddf, Hdf])
        Ddf['pm_source'] = 1
    try:
        Ddf = Ddf.sort_values(by='dateTime')
        #Ddf.to_csv(monthlyOutPutPath + 'OpenAQ_df_' + Y + '_' + M + '.csv')
        return Ddf
    except:
        print('no OpenAQ data on ' + Y + '-' + M)
    print("openaq " + Y + "-" + M + " done")



for y in year:
    for m in month:
        UTDdf = UTD(y, m)
        EPAdf = EPA(y, m)
        OpenAQdf = OpenAQ(y, m)
        
        try:
            monthlyDf = pd.concat([UTDdf, EPAdf, OpenAQdf])
            monthlyDf = monthlyDf.sort_values(by='dateTime')
            monthlyDf.to_csv(monthlyOutPutPath + 'new_pm_df_' + y + '_' + m + '.csv')
            #print(monthlyDf)
        except:
            print('no data on ' + y + '-' + m)
        #print(OpenAQdf)
        
        #print(UTDdf)
        #print(EPAdf)









'''
raw_data_path = '/Users/prabu/Desktop/Dounload_Central_data/JoppaCentralRawData/raw/'

ids = ['001e06318c91', '001e06373724', '001e0637371e', '001e06373996', '001e063739c7', '001e063739cd', '001e0636e547', '001e06323a37', '001e0637372e',
        '001e06318c9d', '001e06305a12', '001e06323a12', '001e06318cd1', '001e06305a61', '001e06323a05', '001e06305a57', '001e063059c2', '001e06318c28',
        '001e06305a6b', '001e063239e3', '001e06305a6c', '001e063239e6', '001e06305a0a', '001e06318cee', '001e06318cf1', '001e063059c1', '001e063239e4',
        '001e06323952', '001e06305a2c', '001e06323a06', '001e06305a12', '001e06303b4f', '001e06318c3c']

ips_predictors = ['dateTime', 'pm2_5']
gps_predictors = ['dateTime', 'latitudeCoordinate', 'longitudeCoordinate']



monthly_df = pd.DataFrame()

for i in ids:
    
    for y in year:
        ddf = pd.DataFrame()
        for m in month:
            #row = []
            for d in day:
                IPS = 'MINTS_' + i + '_IPS7100_' + y + '_' + m + '_' + d + '.csv'
                GPS = 'MINTS_' + i + '_GPSGPGGA2_' + y + '_' + m + '_' + d + '.csv'
                IPS_link = raw_data_path + i + '/' + y + '/' + '/' + m + '/' + d + '/' + IPS
                GPS_link = raw_data_path + i + '/' + y + '/' + '/' + m + '/' + d + '/' + GPS

                if exists(IPS_link):
                    ips_file = pd.read_csv(IPS_link)
                    ips_df = pd.DataFrame(ips_file, columns = ips_predictors)
                    ips_df['dateTime'] = pd.to_datetime(ips_df['dateTime'])
                    ipsDF = ips_df.groupby(pd.Grouper(key='dateTime', freq='1h')).mean()

                    gps_file = pd.read_csv(GPS_link)
                    gps_df = pd.DataFrame(gps_file, columns = gps_predictors)
                    gps_df['dateTime'] = pd.to_datetime(gps_df['dateTime'])
                    gpsDF = gps_df.groupby(pd.Grouper(key='dateTime', freq='1h')).mean()
                    gpsDF = gpsDF.rename({'latitudeCoordinate': 'latitude', 'longitudeCoordinate': 'longitude'}, axis=1)

                    final_df = pd.merge(ipsDF, gpsDF, how='outer', on='dateTime')
                    #row.append(final_df)
                    ddf = pd.concat([ddf, final_df])

            #ddf.to_csv('/Users/prabu/Desktop/Dounload_Central_data/analyzing/df_' + i + '_' + y + '_' + m + '.csv')
    monthly_df = pd.concat([monthly_df, ddf])

monthly_df = monthly_df.sort_values(by='dateTime')
monthly_df.to_csv('/Users/prabu/Desktop/Dounload_Central_data/analyzing/df_' + y + '_' + m + '.csv')
'''
