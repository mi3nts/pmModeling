
# conda env = myenv1

import pandas as pd



input_dataFrame = "/Users/prabu/Desktop/Research/Satellite_Proj/Data/BuildingFootprint/dataFrames/"
state_name = "Georgia.csv"

state_file = pd.read_csv(input_dataFrame + state_name)
#print(len(state_file))

df = state_file.loc[1990897:3981792]
df.to_csv("georgia_2.csv")
#print(df_2)

print("done")

'''

cal_1 = pd.read_csv("california_1.csv")
cal_2 = pd.read_csv("california_2.csv")
cal_3 = pd.read_csv("california_3.csv")
cal_4 = pd.read_csv("california_4.csv")

df = pd.concat([cal_1, cal_2, cal_3, cal_4])
print(df)

'''