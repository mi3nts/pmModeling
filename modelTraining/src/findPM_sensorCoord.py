
# conda env = pmtest

import pandas as pd
from os.path import exists


year = ['2020', '2021', '2022']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

pmMonthlyPath = '/scratch/prabuddha/2yrHist_train/monthlyDF/pm2_5/'
outputPath = '/scratch/prabuddha/2yrHist_train/pmSensorCoordDF/new_pmSensorLocationRound.csv'

coord_var = ['latitude', 'longitude']


df = pd.DataFrame()
for y in year:
    for m in month:
        pmPath = pmMonthlyPath + 'pm_df_' + y + '_' + m + '.csv'
        if exists(pmPath):
            pmDf_file = pd.read_csv(pmPath)
            coord_df = pd.DataFrame(pmDf_file, columns = coord_var)
            #print(y + " " + m + " " + str(len(coord_df)))
            #print(y)
            #print(m)

            df = pd.concat([df, coord_df])
            #print(df)

#print(len(df))
Cdf = df.groupby(["latitude","longitude"],as_index=False).count()
print('before round' + str(len(Cdf)))
Rdf = Cdf.round({'latitude': 4, 'longitude': 4}) 
GRdf = Rdf.groupby(["latitude","longitude"],as_index=False).count()
print('after round' +str(len(GRdf)))
#GRdf.to_csv(outputPath)
#print(Cdf)
