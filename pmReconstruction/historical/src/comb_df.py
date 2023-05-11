
# python environment = pmtest

import os
import pandas as pd

path = '/scratch/prabuddha/pm_est/raw_df'
files = os.listdir(path)

df = pd.DataFrame(files, columns=["file"])
df.to_csv(path + '/files.csv')
print("done")

