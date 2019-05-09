# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas,
Institut fur Wasser- und Umweltsystemmodellierung - IWS
"""
import os
import time
import timeit

import netCDF4 as nc
import numpy as np


print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

main_dir = r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management'
os.chdir(main_dir)

data_ppt_loc = r'X:\exchange\ElHachem\PPT_data'
nc_f = os.path.join(data_ppt_loc, 'precip.mon.mean.0.5x0.5.nc')

in_nc = nc.Dataset(nc_f)
lon0 = in_nc.variables['lon']
lat0 = in_nc.variables['lat']
time_var = in_nc.variables['time']

#         lon2 = lon
time_arr = nc.num2date(time_var[:],
                       time_var.units,
                       calendar='standard')

outfile = r'new_ppt_data_transf.nc'
dsout = nc.Dataset(outfile, 'w', clobber=True)

time = dsout.createDimension('time', None)

lats = np.linspace(-89.75, 89.75, lat0.shape[0], True)[::-1]
lons = np.linspace(-180, 180, lon0.shape[0], True)
cols = len(lons)
rows = len(lats)
lat = dsout.createDimension('lat', rows)
lat = dsout.createVariable('lat', 'f4', ('lat',))
lat.standard_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = "Y"
lat[:] = lats

lon = dsout.createDimension('lon', cols)
lon = dsout.createVariable('lon', 'f4', ('lon',))
lon.standard_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = "X"
lon[:] = lons


times = dsout.createVariable('time', 'double', ('time',))
times.units = "hours since 1800-1-1 00:00:0.0"

times.long_name = 'Time'
times[:] = time_var[:]
times.delta_t = "0000-01-00 00:00:00"
times.avg_period = "0000-01-00 00:00:00"
times.standard_name = 'time'
times.axis = "T"
times.coordinate_defines = "start"
times.actual_range = 1297320.0, 1858344.0
times._ChunkSizes = 1


acc_precip = dsout.createVariable(
    'precip',
    'f4',
    ('time', 'lat', 'lon'),
    zlib=True,
    complevel=9,
    least_significant_digit=1,
    fill_value=-9999)

orig_idx = [ix for ix, _ in enumerate(lon0)]
# nw_ix = np.linspace()
new_idx = [ix0 + 360 if ix0 < 360. else ix0 - 360 for ix0 in orig_idx]
# print(orig_idx, new_idx)
# np.array(in_nc.variables['precip'][0, 85, new_idx], dtype='float64')
# print(in_nc.variables['precip'][10, 10, new_idx].data)
acc_precip[:, :, :] = np.array(in_nc.variables['precip']
                               [:, :, new_idx], dtype='float64')
acc_precip[:][acc_precip[:] < 0] = np.nan  # need fixing
acc_precip.long_name = "Average Monthly Rate of Precipitation"
acc_precip.units = 'mm/day'
acc_precip.setncattr('grid_mapping', 'spatial_ref')
# #
crs = dsout.createVariable('spatial_ref', 'i4')
crs.spatial_ref = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
print('dsout')
# STOP = timeit.default_timer()  # Ending time
# print(('\n\a\a\a Done with everything on %s. Total run time was'
#        ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
