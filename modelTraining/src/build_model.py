
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



Outdir = "/scratch/prabuddha/2yrHist_train/model_evaluation/fc_est200_r1_"
#Outdir_model = "/scratch/prabuddha/2yrHist_train/ML_model/fc2_blh_est100_r1_"
Outdir_model = "/scratch/prabuddha/pm_est_fc/model/fc_est200_r1_"

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

df = df.rename(columns={'cropland':'Cropland', 'landcover':'Landcover', 'population_density':'Population density', 'soiltype':'Soil type',
        'lithology':'Lithology', 'elevation':'Elevation', 'num_building':'Number of building', 'avg_distance':'Avgerage building distance',
        'tot_area':'Total building area', 'temperature':'Temperature', 'dew_temp':'Dewpoint temperature', 'u_wind':'Wind U component',
        'v_wind':'Wind V component', 'pressure':'Pressure', 'precipitation':'Total precipitation', 'skin_reservoir':'Skin reservoir',
        'evaporation':'Evaporation', 'blh':'Boundary layer height', 'lake_cover':'Lake cover', 'high_veg': 'Leaf area index - high vegetation',
        'low_veg': 'Leaf area index - low vegetation', 'snowfall':'Snowfall', 'solar_radiation':'Solar radiation', 'cloud_cover':'Cloud cover',
        'rel_humidity':'Relative humidity', 'speci_humidity':'Specific humidity', 'rain_content':'Rain content', 'AOD':'AOD - Aerosol optical depth',
        'DQF':'DQF - AOD Data quality flags', 'SSA':'Solar azimuth angle', 'SZA':'Solar zenith angle', 'uv10':'Wind speed'})


# Split data for Training (80%) and Testing (20%)
train, test = train_test_split(df, test_size=0.1, random_state=3)

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
train_X = pd.DataFrame(train, columns = predictors_all)
train_Y = pd.DataFrame(train, columns = ['pm2_5'])
test_X = pd.DataFrame(test, columns = predictors_all)
test_Y = pd.DataFrame(test, columns = ['pm2_5'])


# Train the model
ETreg = ExtraTreesRegressor(n_estimators=200, random_state=1).fit(train_X, np.ravel(train_Y))


#Outdir = "model_evalu_4"

def make_prediction(modelName, trainset, testset,train,test):
        predictions_train = pd.DataFrame(modelName.predict(trainset),columns=["Predictions"])
        predictions_test = pd.DataFrame(modelName.predict(testset),columns=["Predictions"])

        predictions_train_valid = pd.concat([train["pm2_5"].reset_index(drop=True), predictions_train.reset_index(drop=True)],axis=1)
        predictions_test_valid = pd.concat([test["pm2_5"].reset_index(drop=True), predictions_test.reset_index(drop=True)],axis=1)

        predictions_train_valid['Residuals'] = predictions_train_valid['pm2_5']  - predictions_train_valid['Predictions']
        predictions_test_valid['Residuals'] = predictions_test_valid['pm2_5']  - predictions_test_valid['Predictions']
        return predictions_train_valid, predictions_test_valid

def residual_plot(predictionFile,outname):
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing+0.02, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        plt.figure(figsize=(10, 10))
        ax_scatter = plt.axes(rect_scatter)

        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labeltop=False,labelsize = 20)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False,labelright=True,labelsize = 20)

        # fig, ax_scatter = plt.subplots(figsize=[15, 10])

#         smoothed = lowess(predictionFile['Predictions'],predictionFile['Value'])
        ax_scatter.scatter(predictionFile['Predictions'], predictionFile['Residuals'], s=5, edgecolors = 'k', facecolors = 'none')
        # ax_scatter.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax_scatter.set_ylabel('Residuals', fontsize=20)
        ax_scatter.set_xlabel('Fitted Values',fontsize=20)
        ax_histx.set_title('Residuals vs Fitted n=%d' %(len(predictionFile)),fontsize=30)
        ax_scatter.tick_params(labelsize = 20)
        # ax_scatter.plot([min(predictions_train_valid['Value']),max(predictions_train_valid['Value'])],[0,0],color = 'k',linestyle = ':', alpha = .3)



        binwidth = 1
        lim = np.ceil(np.abs([predictionFile['Predictions'],predictionFile['pm2_5']]).max() / binwidth) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)

        ax_histy.hist(predictionFile['Residuals'], bins=bins, orientation='horizontal')
        ax_histy.set_ylim(ax_scatter.get_ylim())

        ax_histx.hist(predictionFile['Predictions'], bins=bins, orientation='vertical')
        ax_histx.set_xlim(ax_scatter.get_xlim())
        plt.savefig(Outdir + outname +'_Residual.jpg', bbox_inches='tight', dpi=300)
        #plt.savefig(Outdir + '_Residual.jpg', bbox_inches='tight', dpi=300)


def model_scatterplot_together(trainFile,testFile,outname):
    rmse_train = mean_squared_error(trainFile['pm2_5'], trainFile['Predictions'],squared=False)
    rmse_test = mean_squared_error(testFile['pm2_5'], testFile['Predictions'],squared=False)

    rmse_train_ = round(rmse_train, 3)
    rmse_test_ = round(rmse_test, 3)

    R_train = trainFile['pm2_5'].corr(trainFile['Predictions'])
    R_test = testFile['pm2_5'].corr(testFile['Predictions'])

    R_train_ = round(R_train, 3)
    R_test_ = round(R_test, 3)

    x1 = trainFile['Predictions']
    y1 = trainFile['pm2_5']

    x2 = testFile['Predictions']
    y2 = testFile['pm2_5']

    #x1 = np.random.normal(0, 1, 100)
    #y1 = np.random.normal(0, 1, 100)
    #x2 = np.random.normal(2, 1, 100)
    #y2 = np.random.normal(2, 1, 100)

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 4)
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_xhist = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_yhist = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    #ax_main.scatter(x1, y1, color='blue', alpha=0.5)
    s1 = ax_main.scatter(x2, y2, color='red', label='Testing', alpha=0.5, s=10)
    s2 = ax_main.scatter(x1, y1, color='blue', label='Training', alpha=0.5, s=10)

    x1_dens = np.linspace(x1.min(), x1.max(), 100)
    y1_dens = np.linspace(y1.min(), y1.max(), 100)
    x2_dens = np.linspace(x2.min(), x2.max(), 100)
    y2_dens = np.linspace(y2.min(), y2.max(), 100)
    #ax_xhist.plot(x1_dens, gaussian_kde(x1)(x1_dens), color='blue')
    #ax_xhist.fill_between(x1_dens, gaussian_kde(x1)(x1_dens), alpha=0.5, color='blue')
    ax_xhist.plot(x2_dens, gaussian_kde(x2)(x2_dens), color='red')
    ax_xhist.fill_between(x2_dens, gaussian_kde(x2)(x2_dens), alpha=0.5, color='red')
    ax_xhist.plot(x1_dens, gaussian_kde(x1)(x1_dens), color='blue')
    ax_xhist.fill_between(x1_dens, gaussian_kde(x1)(x1_dens), alpha=0.5, color='blue')
    #ax_yhist.plot(gaussian_kde(y1)(y1_dens), y1_dens, color='blue')
    #ax_yhist.fill_betweenx(y1_dens, gaussian_kde(y1)(y1_dens), alpha=0.5, color='blue')
    ax_yhist.plot(gaussian_kde(y2)(y2_dens), y2_dens, color='red')
    ax_yhist.fill_betweenx(y2_dens, gaussian_kde(y2)(y2_dens), alpha=0.5, color='red')
    ax_yhist.plot(gaussian_kde(y1)(y1_dens), y1_dens, color='blue')
    ax_yhist.fill_betweenx(y1_dens, gaussian_kde(y1)(y1_dens), alpha=0.5, color='blue')

    ax_main.set_xlabel('Fitted Values')
    ax_main.set_ylabel('Observed Values')
    ax_main.set_title('Train_RMSE = ' + str(rmse_train_) + ' Train_R = ' + str(R_train_) + ' Test_RMSE = ' + str(rmse_test_) + ' Test_R = ' + str(R_test_))
    ax_xhist.set_ylabel('Density')
    ax_yhist.set_xlabel('Density')

    ax_main.legend(handles=[s1, s2], loc='upper right')

    fig.tight_layout()

    plt.savefig(Outdir + outname + '_scatter_plot_together.jpg', bbox_inches='tight',dpi=300)


def model_scatterplot(predictionFile,outname):
        rmse = mean_squared_error(predictionFile['pm2_5'], predictionFile['Predictions'],squared=False)
        R = predictionFile['pm2_5'].corr(predictionFile['Predictions'])
        print(outname)
        print("rmse = %f" %(rmse))
        print('Correlation Coefficient %f \n' %R)

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.01

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing+0.02, width, 0.2]
        rect_histy = [left + width + spacing+0.03, bottom, 0.2, height]

        plt.figure(figsize=(10, 10))
        ax_scatter = plt.axes(rect_scatter)

        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labeltop=False,labelsize = 20)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False,labelright=True,labelsize = 20)

        # fig, ax_scatter = plt.subplots(figsize=[15, 10])

        smoothed = lowess(predictionFile['Predictions'],predictionFile['pm2_5'])
        ax_scatter.scatter(predictionFile['Predictions'], predictionFile['pm2_5'], s=5, edgecolors = 'k', facecolors = 'none')
        # ax_scatter.plot(smoothed[:,0],smoothed[:,1],color = 'r')
        ax_scatter.set_ylabel('Observed Values', fontsize=20)
        xticks = np.arange(0, 150, 15)
        yticks = np.arange(0, 150, 15)

#         xticks = np.linspace(0, np.abs([predictionFile['Predictions'],predictionFile['Value']]).max(), 10)
#         yticks = np.linspace(0, np.abs([predictionFile['Predictions'],predictionFile['Value']]).max(), 10)

#         ax_scatter.set_ylim(ax_scatter.get_ylim())
        ax_scatter.set_ylim(0,150)
        ax_scatter.set_xlabel('Fitted Values',fontsize=20)
#         ax_scatter.set_xlim(ax_scatter.get_xlim())
        ax_scatter.set_xlim(0,150)
        ax_histx.set_title('Observed vs. Fitted n=%d RMSE=%.3f R=%.3f' %(len(predictionFile),rmse,R),fontsize=30)
        ax_scatter.tick_params(labelsize = 20)
        # ax_scatter.plot([min(predictions_train_valid['Value']),max(predictions_train_valid['Value'])],[0,0],color = 'k',linestyle = ':', alpha = .3)
        ax_scatter.set_xticks(xticks)
        ax_scatter.set_yticks(yticks)


        binwidth = 1
        lim = np.ceil(np.abs([predictionFile['Predictions'],predictionFile['pm2_5']]).max() / binwidth) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)

        ax_histy.hist(predictionFile['pm2_5'], bins=bins, orientation='horizontal')
        ax_histy.set_ylim(ax_scatter.get_ylim())

        ax_histx.hist(predictionFile['Predictions'], bins=bins, orientation='vertical')
        ax_histx.set_xlim(ax_scatter.get_xlim())

        plt.savefig(Outdir + outname + '_Observed_vs_Fitted.jpg', bbox_inches='tight',dpi=300)

def importance_plot(model,features,outname):
    important_features_dict = {}
    for idx, val in enumerate(model.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=False)
    rankfeature =[]
    rankvalue = []
    for i in important_features_list:
        rankvalue.append(model.feature_importances_[i])
        rankfeature.append(features[i])

    fig = plt.figure(figsize=(12,8))
    plt.ion()
#     plt.show()
    barlist= plt.barh(rankfeature, rankvalue)
    for i in range(3):
        barlist[-(i+1)].set_color('r')
    for i in range(3,6):
        barlist[-(i+1)].set_color('orange')
    plt.close(fig)
    fig.savefig(Outdir + outname+".png", bbox_inches='tight', dpi=300)



predicted_train_valid, predicted_test_valid = make_prediction(ETreg,train_X,test_X,train,test)
residual_plot(predicted_train_valid,"ETreg_random_train")
model_scatterplot(predicted_train_valid,"ETreg_random_train")
residual_plot(predicted_test_valid,"ETreg_random_test")
model_scatterplot(predicted_test_valid,"ETreg_random_test")
importance_plot(ETreg,predictors_all,outname="Importance_ETreg_random")
model_scatterplot_together(predicted_train_valid, predicted_test_valid, "ETreg")

#Outdir_model = "2y_model_4"
#joblib.dump(ETreg, Outdir_model + "ETreg_random.joblib")
