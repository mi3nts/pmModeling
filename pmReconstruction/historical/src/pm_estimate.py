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
from datetime import datetime
import datetime



# Data available upto 7 days behind current day
# Go 7 days back and add 5 hours to get UTC time
week_ago = datetime.datetime.now() - datetime.timedelta(days=7) + datetime.timedelta(hours=5)
week_ago_str = str(week_ago)
year, month, day, hour = week_ago_str[0:4], week_ago_str[5:7], week_ago_str[8:10], week_ago_str[11:13]
time = hour + ':00'

day = '03'
hour = '15'
time = hour + ':00'
down_dir = '/scratch/prabuddha/pm_est/raw_df/'
path = down_dir + year + '/' + month + '/' + day + '/' + hour + '/'
files = os.listdir(path) 


df0 = pd.read_csv(path + files[0])
df1 = pd.read_csv(path + files[1])
df2 = pd.read_csv(path + files[2])
df3 = pd.read_csv(path + files[3])
df4 = pd.read_csv(path + files[4])
df5 = pd.read_csv(path + files[5])
df6 = pd.read_csv(path + files[6])
df7 = pd.read_csv(path + files[7])
df8 = pd.read_csv(path + files[8])
df9 = pd.read_csv(path + files[9])
df10 = pd.read_csv(path + files[10])
df11 = pd.read_csv(path + files[11])
df12 = pd.read_csv(path + files[12])
df13 = pd.read_csv(path + files[13])
df14 = pd.read_csv(path + files[14])
df15 = pd.read_csv(path + files[15])
df16 = pd.read_csv(path + files[16])
df17 = pd.read_csv(path + files[17])
df18 = pd.read_csv(path + files[18])
df19 = pd.read_csv(path + files[19])

df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19], ignore_index=True)
df = df.drop('Unnamed: 0.1', axis=1)
df = df.drop('Unnamed: 0', axis=1)


#df = pd.read_csv("/scratch/prabuddha/pm_est/raw_df/complete_df_on_2022_12_02_15_test.csv")
df['AOD'] = df['AOD'].fillna(0)
df['DQF'] = df['DQF'].fillna(0)


df["uv10"]= np.sqrt(df["u_wind"].values*df["u_wind"].values +df["v_wind"].values*df["v_wind"].values)
print(list(df.columns))

predictors = ['cropland', 'landcover', 'soil', 'lithology', 'population', 'elevation', 'num_build', 'avg_dist', 'tot_area', 'temperature', 'dew_temp', 
        'pressure', 'precipitation', 'skin_reservoir', 'evaporation', 'blh', 'rel_humidity', 'AOD', 'DQF', 'SSA', 'SZA', 'uv10']


est_df = pd.DataFrame(df, columns = predictors)
#print(est_df)

loaded_rf = joblib.load("/scratch/prabuddha/pm_est/ExtraTreeReg_model/ETreg_random.joblib")
print(loaded_rf)


print(est_df.isnull().values.any())


def make_prediction2(modelName, est_df):
        predictions_train = pd.DataFrame(modelName.predict(est_df),columns=["Predictions"])
        return predictions_train

predicted_train_valid2 = make_prediction2(loaded_rf,est_df)

df["pm_est"] = predicted_train_valid2
save_path = '/scratch/prabuddha/pm_est/final_df/' + 'df_' + year + '_' + month + '_' + day + '_' + hour + '.csv'
#save_path = '/scratch/prabuddha/pm_est/final_df/test.csv'
df.to_csv(save_path)



fname = '/scratch/prabuddha/pm_est/USBoundry/cb_2018_us_state_500k.shp'
shape_feature = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor='none')


fig=plt.figure(figsize=[50,30])
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.add_feature(shape_feature,edgecolor='blue')
#ax.gridlines(draw_labels=True)
ax.coastlines()
#c = ax.pcolor(final_df2["longitude"], final_df2["latitude"],final_df2["cropland"],cmap='Blues',vmin=-300,vmax=300)
#c = ax.scatter(building_coord["longitude"], building_coord["latitude"], s=3)
c = ax.scatter(df["longitude"], df["latitude"], c = df["pm_est"], s=15, cmap='turbo',vmin=10,vmax=25, marker='s', linewidth=0)
fig.colorbar(c, ax=ax, shrink=0.55)
plt.title("PM2.5 Estimation " + year + "-" + month + "-" + day + " " + time, fontsize=50)
plt.savefig("/scratch/prabuddha/pm_est/output_plot/pm_est_" + year + "_" + month + "_" + day + "_" + hour + ".png")
#plt.savefig("/scratch/prabuddha/pm_est/output_plot/pm_est_test.png")



