import xarray as xr
import pandas as pd
import numpy as np
import cftime

# Load the dataframe from the CSV file
df = pd.read_csv('/scratch/prabuddha/pm_est_fc/data/fc_df/df_2023_04_14_18.csv')
df = df[['time', 'latitude', 'longitude', 'pm_est']]

df = df.set_index(['time', 'latitude', 'longitude'])

# Convert the DataFrame to an xarray dataset
ds = df.to_xarray()

units = 'hours since 1970-01-01 00:00:00'
calendar = 'standard'
times = pd.to_datetime(ds['time'].values)
ds['time'] = xr.DataArray([t.year for t in times], dims='time', attrs={'units': 'year'})

# Add units and attributes to the coordinate variables
ds['latitude'].attrs['units'] = 'degrees_north'
ds['longitude'].attrs['units'] = 'degrees_east'

# Add attributes to the data variable
ds['pm_est'].attrs['long_name'] = 'PM concentration'
ds['pm_est'].attrs['units'] = 'micrograms/m^3'

# Add global attributes
ds.attrs['Conventions'] = 'CF-1.6'
ds.attrs['history'] = 'Converted from pandas dataframe to NetCDF using xarray'

# Save the dataset to a NetCDF file
ds.to_netcdf('/scratch/prabuddha/pm_est_fc/data/fc_df/z4.nc')

