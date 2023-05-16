# conda env = model

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score



df1 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2020_10.csv")
df2 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2020_11.csv")
df3 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2020_12.csv")
df4 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_01.csv")
df5 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_02.csv")
df6 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_03.csv")
df7 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_04.csv")
df8 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_05.csv")
df9 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_06.csv")
df10 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_07.csv")
df11 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_08.csv")
df12 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_09.csv")
df13 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_10.csv")
df14 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_11.csv")
df15 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2021_12.csv")
df16 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_01.csv")
df17 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_02.csv")
df18 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_03.csv")
df19 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_04.csv")
df20 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_05.csv")
df21 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_06.csv")
df22 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_07.csv")
df23 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_08.csv")
df24 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_09.csv")
df25 = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/new2_final_df_2022_10.csv")
df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18,
df19, df20, df21, df22, df23, df24, df25])
#df = df1


df_epa = df[df['pm_source'] == 0]
df_openAQ = df[df['pm_source'] == 1]
df_mints = df[df['pm_source'] == 2]
df = pd.concat([df_epa, df_openAQ, df_mints])
#df = df_epa
#df.to_csv("/scratch/prabuddha/2yrHist_train/model_evaluation/only_EPA.csv")
df = df.drop_duplicates(subset=["latitude", "longitude", "dateTime", "pm2_5"])



Outdir = "/scratch/prabuddha/2yrHist_train/model_evaluation/fc2_blh_est100_r1_"
Outdir_model = "/scratch/prabuddha/2yrHist_train/ML_model/fc2_blh_est100_r1_"
#Outdir_model = "/scratch/prabuddha/pm_est_fc/model/new_"

#df = pd.read_csv("/scratch/prabuddha/2yrHist_train/monthlyDF/final_df/final_df_2020_10.csv")
#df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})

# Remove empty cell
df = df.dropna()
df.reset_index(drop=True)

# Skip negative pm2.5 value
df = df[df['pm2_5'] > 0]
# Skip low quality AOD value
df = df[df['DQF'] < 2]

# Add speed of the wind to the dataFrame
df["uv10"]= np.sqrt(df["u_wind"].values*df["u_wind"].values +df["v_wind"].values*df["v_wind"].values)
#print(list(df.columns))
df['dateTime'] = pd.to_datetime(df['dateTime'])
df['Month'] = df['dateTime'].dt.month
#print(df[0:10])


df = df.rename(columns={'cropland':'Cropland', 'landcover':'Landcover', 'population_density':'Population density', 'soiltype':'Soil type',
        'lithology':'Lithology', 'elevation':'Elevation', 'num_building':'Number of building', 'avg_distance':'Avgerage building distance',
        'tot_area':'Total building area', 'temperature':'Temperature', 'dew_temp':'Dewpoint temperature', 'u_wind':'Wind U component',
        'v_wind':'Wind V component', 'pressure':'Pressure', 'precipitation':'Total precipitation', 'skin_reservoir':'Skin reservoir',
        'evaporation':'Evaporation', 'blh':'Boundary layer height', 'lake_cover':'Lake cover', 'high_veg': 'Leaf area index - high vegetation',
        'low_veg': 'Leaf area index - low vegetation', 'snowfall':'Snowfall', 'solar_radiation':'Solar radiation', 'cloud_cover':'Cloud cover',
        'rel_humidity':'Relative humidity', 'speci_humidity':'Specific humidity', 'rain_content':'Rain content', 'AOD':'AOD - Aerosol optical depth',
        'DQF':'DQF - AOD Data quality flags', 'SSA':'Solar azimuth angle', 'SZA':'Solar zenith angle', 'uv10':'Wind speed'})

'''
# All the input variables
predictors_all = ['Cropland', 'Landcover', 'Population density', 'Soil type', 'Lithology', 'Elevation', 'Number of building', 'Avgerage building distance',
        'Total building area', 'Temperature', 'Dewpoint temperature', 'Wind U component', 'Wind V component', 'Pressure', 'Total precipitation',
        'Skin reservoir', 'Evaporation', 'Boundary layer height', 'Lake cover', 'Leaf area index - high vegetation', 'Leaf area index - low vegetation',
        'Snowfall', 'Solar radiation', 'Cloud cover', 'Relative humidity', 'Specific humidity', 'Rain content', 'AOD - Aerosol optical depth',
        'DQF - AOD Data quality flags', 'Solar azimuth angle', 'Solar zenith angle', 'Wind speed', 'Month']
'''

'''
# all input variables - no wind components
predictors_all = ['Cropland', 'Landcover', 'Population density', 'Soil type', 'Lithology', 'Elevation', 'Number of building', 'Avgerage building distance',
        'Total building area', 'Temperature', 'Dewpoint temperature', 'Pressure', 'Total precipitation',
        'Skin reservoir', 'Evaporation', 'Boundary layer height', 'Lake cover', 'Leaf area index - high vegetation', 'Leaf area index - low vegetation',
        'Snowfall', 'Solar radiation', 'Cloud cover', 'Relative humidity', 'Specific humidity', 'Rain content', 'AOD - Aerosol optical depth',
        'DQF - AOD Data quality flags', 'Solar azimuth angle', 'Solar zenith angle', 'Wind speed', 'Month']
'''

'''
predictors_all = ['Cropland', 'Landcover', 'Population density', 'Soil type', 'Lithology', 'Elevation',
        'Temperature', 'Dewpoint temperature', 'Pressure', 'Total precipitation',
        'Skin reservoir', 'Evaporation', 'Boundary layer height', 'Leaf area index - high vegetation', 'Leaf area index - low vegetation',
        'Relative humidity', 'AOD - Aerosol optical depth', 'DQF - AOD Data quality flags', 'Solar azimuth angle', 'Solar zenith angle', 'Wind speed', 'Month']
'''


#forecast
predictors_all = ['Cropland', 'Landcover', 'Population density', 'Soil type', 'Lithology', 'Elevation', 'Number of building', 'Avgerage building distance',
        'Total building area', 'Temperature', 'Pressure', 'Total precipitation', 'Boundary layer height',
        'Relative humidity', 'Specific humidity', 'AOD - Aerosol optical depth',
        'DQF - AOD Data quality flags', 'Solar azimuth angle', 'Solar zenith angle', 'Wind speed', 'Month']



# Feature and label separation
#train_X = pd.DataFrame(train, columns = predictors_all)
#train_Y = pd.DataFrame(train, columns = ['pm2_5'])
#test_X = pd.DataFrame(test, columns = predictors_all)
#test_Y = pd.DataFrame(test, columns = ['pm2_5'])

X = pd.DataFrame(df, columns = predictors_all)
y = pd.DataFrame(df, columns = ['pm2_5'])


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#train, test = train_test_split(df, test_size=0.1, random_state=3)
#print(X_train)
#print(y_train)

# Define the Extra Tree Regression model
etr = ExtraTreesRegressor()

# Define the hyperparameters to be optimized
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Define the performance metric
scoring = 'r2'

# Define the Grid Search object
grid_search = GridSearchCV(estimator=etr, param_grid=params, scoring=scoring, cv=5)

# Perform Grid Search to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print('Best hyperparameters:', best_params)

# Fit the model with the best hyperparameters on the entire training set
etr = ExtraTreesRegressor(**best_params)
etr.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = etr.predict(X_val)
print(y_pred)





