# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
import os
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.ioff()

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

main_dir = os.path.join(
    r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management')
os.chdir(main_dir)
in_csv_file = os.path.join(main_dir, r'irrigation_nile.csv')
assert in_csv_file

in_df_irrigation = pd.read_csv(in_csv_file, sep=',', index_col=0,
                               header=[0, 1, 2], skiprows=[3])

# %%
x_rows = in_df_irrigation.index[:-2].values

y_arr = in_df_irrigation['Irrigation potential'].values[:-2]
y_arr_list = np.array([float(y_arr[i][0].replace(' ', ''))
                       for i in range(len(y_arr))])
y_arr_list2 = np.array([2220000., 80000.,   150000., 202000.,
                        4420000., 30000., 150000., 180000., 2750000., 10000.])
x_rows2 = ['Ethiopia', 'Burundi',   'Eritrea', 'Uganda', 'Egypt', 'Tanzania',
           'Rwanda', 'Kenya', 'Sudan', 'Zaire']
colors = ['gold', 'yellowgreen', 'brown', 'grey',
          'c', 'lightblue', 'lightcoral', 'm', 'teal', 'salmon']
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
plt.figure(figsize=(16, 12))
plt.pie(y_arr_list2, labels=x_rows2, autopct='%1.1f%%', colors=colors,
        shadow=False, startangle=120, explode=explode, textprops={'fontsize': 12})

#plt.title('Irrigation Potential for the Nile Basin', fontsize=12)
plt.axis('equal')
plt.savefig(os.path.join(
            main_dir, 'irrigation_potential.pdf'),
            frameon=True, papertype='a4', bbox_inches='tight')

# %% plt water requirement
y_demand = np.array(in_df_irrigation[
    'Gross irrigation water requirement 2'].values[:-2],
    dtype='float64')
y_demand = y_demand.reshape(10,)
x_deman = in_df_irrigation['Gross irrigation water requirement 2'].index[:-2].values

colors = ['gold', 'yellowgreen', 'brown', 'grey',
          'c', 'lightblue', 'lightcoral', 'm', 'teal', 'darkblue']

plt.figure(figsize=(12, 12))
x = np.arange(0, len(y_demand), 1)
plt.bar(x, y_demand, color=colors, width=0.5)

plt.xlim([-0.20, 10])
plt.xticks(x, x_deman, fontsize=12)
plt.ylabel('Irrigation Water Requirement (km3/yr)', fontsize=12)
#plt.title('Gross irrigation water requirement (km3/yr)')
# plt.axis('equal')
plt.grid(axis='y', alpha=0.5)
plt.show()
plt.savefig(os.path.join(
            main_dir, 'irrigation_demand.pdf'),
            frameon=True, papertype='a4', bbox_inches='tight')

# %% read and plot discharge data Aswan Damm
in_disch_max_csv = os.path.join(main_dir, 'disch_data',
                                r'max_q_aswan_dam.csv')
in_disch_min_csv = os.path.join(main_dir, 'disch_data',
                                r'min_q_aswan_dam.csv')
in_disch_mean_csv = os.path.join(main_dir, 'disch_data',
                                 r'mean_q_aswan_dam.csv')
plt.figure(figsize=(16, 12))
in_disch_df_max = pd.read_csv(in_disch_max_csv, index_col=2, sep=';')
in_disch_df_max.index = pd.to_datetime(in_disch_df_max.index, format='%Y-%m')
in_disch_df_min = pd.read_csv(in_disch_min_csv, index_col=2, sep=';')
in_disch_df_mean = pd.read_csv(in_disch_mean_csv, index_col=2, sep=';')

disch_data_max = in_disch_df_max[' Value'][in_disch_df_max[' Value'] >= 0].values
disch_data_min = in_disch_df_min[' Value'][in_disch_df_min[' Value'] >= 0].values
disch_data_mean = in_disch_df_mean[' Value'][in_disch_df_mean[' Value'] >= 0].values
plt.plot(in_disch_df_max.index[1:], disch_data_max,
         c='r', label='Q max', alpha=0.75)

plt.plot(in_disch_df_max.index[1:], disch_data_mean,
         c='g', label='Q mean', alpha=0.75)
plt.plot(in_disch_df_max.index[1:], disch_data_min,
         c='b', label='Q min', alpha=0.75)
plt.legend(loc=0)
plt.grid(alpha=0.5)
plt.yticks(np.linspace(0, disch_data_max.max(), 30))
plt.ylabel('Q (m3/s)')
plt.xlabel('Time')
plt.title('Monthly Discharge Values at Aswan Dam (m³/s)')
plt.show()
plt.savefig(os.path.join(
            main_dir, 'aswan_dam_discharge.png'),
            frameon=True, papertype='a4', bbox_inches='tight')

# %% plot SWRO cost
y_arr_list2 = np.array([11, 12, 7, 6, 9, 31, 26])
x_rows2 = ['Intake and \n Discharge \n Construction', 'Pretreatment \n Construction',
           'Project Design \n and Permitting', 'SWRO \n Replacement', 'Other', 'SWRO System \n Construction', 'Power']
colors = ['gold', 'yellowgreen', 'salmon', 'grey', 'm', 'teal', 'red']
explode = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
plt.figure(figsize=(16, 12))
plt.pie(y_arr_list2, labels=x_rows2, autopct='%1.1f%%', colors=colors,
        shadow=True, startangle=120, explode=explode, textprops={'fontsize': 12})

plt.title('Typical SWRO Plant Construction Cost Breakdown', fontsize=12)
plt.axis('equal')

plt.savefig(os.path.join(
            main_dir, 'swro.png'),
            frameon=True, papertype='a4', bbox_inches='tight')
STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
