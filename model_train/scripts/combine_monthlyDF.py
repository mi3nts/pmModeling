
# conda env = pmtest

import pandas as pd

pmPath = "/scratch/prabuddha/2yrHist_train/monthlyDF/pm2_5_with_source/"
inputPath = "/scratch/prabuddha/2yrHist_train/dailyDF/dynamic/"
savePath = "/scratch/prabuddha/2yrHist_train/monthlyDF/finalDF_PMsource_withDuplicates/"

year = "2022"
month = "02"
day = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28"]
#day = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31"]

pmLink = pmPath + "new_pm_df_" + year + "_" + month + ".csv"



df = pd.DataFrame()
for d in day:
    try:
        inputLink = inputPath + "df_" + year + "_" + month + "_" + d + ".csv"
        inputdf = pd.read_csv(inputLink)
        inputdf = inputdf.dropna()
        df = pd.concat([df, inputdf])
    except:
        print("no dataframe on " + year + "_" + month + "_" + d)

df['dateTime'] = df['dateTime'].str.slice(0, 18)
df['dateTime'] = pd.to_datetime(df['dateTime'])

pmdf = pd.read_csv(pmLink)
Rpmdf = pmdf.round({'latitude': 4, 'longitude': 4})
Rpmdf['dateTime'] = pd.to_datetime(Rpmdf['dateTime'])
#pmDF = Rpmdf.drop_duplicates()
pmDF = Rpmdf


final_df = pd.merge(df, pmDF,  how='left', left_on=["dateTime","latitude","longitude"], right_on = ["dateTime","latitude","longitude"])
final_df = final_df.dropna()

final_df.to_csv(savePath + "new2_final_df_" + year + "_" + month + ".csv")
