
# conda environment = pmtest


import requests
import pandas as pd
import os
import json



#statelist = ['48']      # Texas
statelist = ['01','04','05','06','07','08','09','10','11','12','13','16','17','18','19','20','21','22','23',
             '24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','44',
             '45','46','47','48','49','50','51','53','54','55','56']

bedate = 20201201
edate = 20201231

download_directory = "/scratch/prabuddha/2yrHist_train/rawData/EPA_pm/2020_12.csv"


concat_df = pd.DataFrame()
for state in statelist:
    print("Downloading state"+ ' ' + state )
    link = 'https://aqs.epa.gov/data/api/sampleData/byState?email=prabuddhamail@gmail.com&key=goldgoose94&param=88101&bdate=' 
    r = requests.get(link + str(bedate) + '&edate=' + str(edate) + '&state='+state)
    try:
        j=r.json()
        print(state + ' ' + 'Download Complete')
        j = j['Data']
        raw_df = pd.DataFrame.from_dict(j)
        AQS_df = pd.DataFrame(raw_df,columns=['latitude','longitude','date_gmt','time_gmt','units_of_measure','sample_measurement'])
        AQS_df["UTC"] = AQS_df['date_gmt'] +"T"+AQS_df['time_gmt']
        AQS_df.rename({'latitude':'Latitude', 'longitude':'Longitude', 'sample_measurement':'Value'}, axis=1,inplace=True)
        AQS_df.drop(columns=['date_gmt','time_gmt'],inplace=True)
        if concat_df.empty:
            concat_df = AQS_df
        else:
            concat_df = pd.concat([concat_df,AQS_df])
    except:
        print(state + ' ' + 'Failed')
# Merge states together
#concat_df = concat_df.sort_values(by=['UTC'], ascending=True)
concat_df.to_csv(download_directory)

print("done")
