# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
import os
import time
import timeit

from matplotlib import cm, colors
from scipy.interpolate import griddata

from fluxes_spatial_dist_functions import agg_max_mean_min_cycle
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile as shp

plt.ioff()

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

main_dir = os.path.join(
    r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\data\gridded_ppt_monthly_data')
os.chdir(main_dir)

shp_loc = r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\Nile_qgis\nile_basin\nile_basin.shp'
nile_river_shp = r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\Nile_qgis\nile_river_6ca\nile_river_6ca.shp'
in_ppt_df_file_gpcc = os.path.join(main_dir,
                                   'cat_nbr_1020034170_gridded_M_ppt_vals_nc_f_ly_v2018_025.csv')

in_ppt_df_gpcp = pd.read_csv(in_ppt_df_file_gpcc,
                             sep=';',
                             index_col=0)
in_ppt_df_gpcp.index = pd.to_datetime(in_ppt_df_gpcp.index, format='%Y-%m-%d')
in_ppt_df_gpcp_res = in_ppt_df_gpcp.resample('M', label='right', closed='right',
                                             base=0).sum() / 1
in_ppt_df_gpcp_res.iloc[in_ppt_df_gpcp_res.index.month == 1, :].mean()
df_max_val, df_min_val, df_mean, df_dev = agg_max_mean_min_cycle(
    in_ppt_df_gpcp_res)
x_vls = []
y_vls = []
for v, _ in enumerate(df_max_val.index):
    x_vls.append(float(df_max_val.index[v].split(',')[0].replace('(', '')))
    y_vls.append(float(df_max_val.index[v].split(',')[1].replace(')', '')))
x_vls_arr = np.array(x_vls)
y_vls_arr = np.array(y_vls)

sf = shp.Reader(shp_loc)
shp_xx = []
shp_yy = []
# plt.figure()
for shape in sf.shapeRecords():
    shp_xx.append([i[0] for i in shape.shape.points[:]])
    shp_yy.append([i[1] for i in shape.shape.points[:]])

shp_riv = shp.Reader(nile_river_shp)
shp_xx_rv = []
shp_yy_rv = []
# plt.figure()
for shape_rv in shp_riv.shapeRecords():
    shp_xx_rv.append([i[0] for i in shape_rv.shape.points[:]])
    shp_yy_rv.append([i[1] for i in shape_rv.shape.points[:]])

x, y = np.meshgrid(np.linspace(x_vls_arr.min(), x_vls_arr.max(), 30),
                   np.linspace(y_vls_arr.min(), y_vls_arr.max(), 30))
plot_avg = False
if plot_avg:
    fig, ax0 = plt.subplots(nrows=1, figsize=(16, 14), dpi=200)
    ax0.set_xlim(x_vls_arr.min() - 0.25, x_vls_arr.max() + 0.25)
    ax0.set_ylim(y_vls_arr.min() - 0.25, y_vls_arr.max() + 0.25)
    cmap = plt.get_cmap('viridis_r')
    zi = griddata((x_vls_arr, y_vls_arr), df_mean.values,
                  (x, y), method='linear')
    # norm = colors.BoundaryNorm(boundaries=np.array([0, 180, 60]), ncolors=256)
    im = ax0.contourf(x, y, zi, 1000, cmap=cmap, extend='max',
                      vmin=0, vmax=165, origin='lower')

    for k in range(len(shp_xx)):
        ax0.plot(shp_xx[k], shp_yy[k], c='k')

    # for k_rv in range(len(shp_xx_rv)):
    #     ax0.scatter(shp_xx_rv[k_rv], shp_yy_rv[k_rv], c='darkblue', s=14, marker='^')
    cb = fig.colorbar(im, ax=ax0, orientation='vertical',
                      ticks=[0, 20, 40, 60, 80, 100, 120, 140, 160.0])
    cb.ax.tick_params(labelsize=18)
    cb.ax.set_yticklabels([0, 20, 40, 60, 80, 100, 120, 140, 160.0])
    ax0.locator_params(nbins=4)
    cb.ax.set_ylabel('Average Monthly Rainfall (mm)', fontsize=20,
                     fontweight="bold",)
    cb.set_alpha(1)
    cb.draw_all()
    im.cmap.set_over('navy')
    plt.grid(alpha=0.25, linestyle='--')

    plt.xlabel('Longitude', fontsize=16)

    plt.ylabel('Latitude', fontsize=16)
    plt.title('')
    plt.savefig(os.path.join(
        main_dir, 'gpcc_nile_mean_ppt.png'),
        frameon=True, papertype='a4', bbox_inches='tight')

plt_orig_monthly_vls = True
if plt_orig_monthly_vls:
    var = 'df_average_monthly_vals'
    var_bounds_dict = {var: [0, 301]}
    bounds_mean = [0, 25, 50, 75, 100., 150, 200, 250, 300, 300.01]

    color_bounds = {var: bounds_mean}
    x_vals = x
    y_vals = y
    fig = plt.figure(figsize=(23, 15), dpi=200)  # (20, 24)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
#     up_lim = 400.01
#     colors = [(0, "indianred"), (10 / up_lim, "orange"),
#               (25 / up_lim, "y"), (50 / up_lim, "limegreen"),
#               (100 / up_lim, "green"), (150 / up_lim, "darkgreen"),
#               (200 / up_lim, 'pink'), (250 / up_lim, 'violet'),
#               (300 / up_lim, 'lightblue'),
#               (350 / up_lim, 'blue'),
#               (400.01 / up_lim, 'darkblue')]
#     colors = [(0.0, "darkblue"), (0.0625, "lightblue"), (0.125, 'yellowgreen'),
#               (0.1825, 'limegreen'),
#               (0.25, "g"), (0.375, "darkgreen"),
#               (0.5, "violet"), (.875, 'orange'), (0.99, 'r'), (1, 'darkred')]
#     cmap = mcolors.LinearSegmentedColormap.from_list(
#         'my_colormap', colors)
    interval_ppt = np.linspace(0.1, 0.95)
    colors_ppt = plt.get_cmap('Blues')(interval_ppt)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'name', colors_ppt)

    #cmap_ppt = plt.get_cmap('jet_r')
    cmap.set_over('darkblue')
    norm_ppt = mcolors.BoundaryNorm(bounds_mean, cmap.N)
    # cmap = plt.get_cmap('Blues')
#             cbar_label = 'Average monthly rainfall values based on the data from 1950 till 2016 (mm/month)'
    cbar_label = '(mm/month)'
    monthes = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May',
               6: 'June', 7: 'July', 8: 'August', 9: 'September',
               10: 'October', 11: 'November', 12: 'December'}

    from matplotlib import rc
    from matplotlib import rcParams
    rc('font', size=16)
    rc('font', family='serif')
    rc('axes', labelsize=20)
    rc('xtick', labelsize=14)
    rc('ytick', labelsize=14)
    rcParams['axes.labelpad'] = 30

    xticks_label = np.round(np.linspace(
        x_vls_arr.min(), x_vls_arr.max(), 3, endpoint=True), 2)
    yticks_label = np.round(np.linspace(
        y_vls_arr.min(), y_vls_arr.max(), 3, endpoint=True), 2)
    for i in range(1, 13):
        print('plotting for var: %s and month nbr: %d' % (var, i))

        #ax = fig.add_subplot(3, 4, i)
        ax = fig.add_subplot(2, 6, i)
        ax.set_xlim(x_vls_arr.min(), x_vls_arr.max())
        ax.set_ylim(y_vls_arr.min(), y_vls_arr.max())
        ax.set_title('%s' %
                     monthes[i], fontweight="bold")  # Title

        grid_vals = in_ppt_df_gpcp_res.iloc[in_ppt_df_gpcp_res.index.month == i, :].mean(
        )

        zi = griddata((x_vls_arr, y_vls_arr), grid_vals,
                      (x, y), method='linear')
        print(zi.max())
        zi[zi < 0] = np.nan

    # norm = colors.BoundaryNorm(boundaries=np.array([0, 180, 60]), ncolors=256)
        im = ax.contourf(x, y, zi, 1000, cmap=cmap, norm=norm_ppt,
                         extend='max', origin='lower',
                         vmin=0, vmax=300)

        ax.grid(True, alpha=0.5, linestyle='--')
#         ax.tick_params(axis='x', labelsize=10)
#         ax.tick_params(axis='y', labelsize=10)

#         if i in [2, 3, 4, 6, 7, 8]:
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#         if i == 9:
#             #             plt.xlabel('Longitude')
#
#             ax.set_xticks(xticks_label)
# #             plt.ylabel('Latitude')
#             ax.set_xticklabels(xticks_label)
# #             ax.tick_params(axis='x', labelsize=10)
# #             ax.tick_params(axis='y', labelsize=10)
#         if i == 10 or i == 11 or i == 12:
#             #             plt.xlabel('Longitude')
#             ax.set_xticks(xticks_label)
#             ax.set_yticklabels([])
# #             ax.tick_params(axis='x', labelsize=10)
# #             ax.tick_params(axis='y', labelsize=10)
#         if i == 1 or i == 5:  # i == 1 or
#             #             plt.ylabel('Latitude')
#             ax.set_yticks(yticks_label)
#             ax.set_xticklabels([])
        if i in [2, 3, 4, 5, 6]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if i == 7:
            #             plt.xlabel('Longitude')

            ax.set_xticks(xticks_label)
            ax.set_yticks(yticks_label)
            ax.set_yticklabels(yticks_label)
#             plt.ylabel('Latitude')
            ax.set_xticklabels(xticks_label)
#             ax.tick_params(axis='x', labelsize=10)
#             ax.tick_params(axis='y', labelsize=10)
        if i in [8, 9, 10, 11, 12]:
            #             plt.xlabel('Longitude')
            ax.set_xticks(xticks_label)
            ax.set_yticklabels([])
#             ax.tick_params(axis='x', labelsize=10)
#             ax.tick_params(axis='y', labelsize=10)
        if i == 1:  # i == 1 or
            #             plt.ylabel('Latitude')
            ax.set_yticks(yticks_label)
            ax.set_xticklabels([])
#             ax.tick_params(axis='x', labelsize=10)
#             ax.tick_params(axis='y', labelsize=10)
        for k in range(len(shp_xx)):
            ax.plot(shp_xx[k], shp_yy[k], c='k', alpha=0.5)
            ax.axis('equal')
    fig.text(0.09, 0.5, 'Latitude', ha='center',
             va='center', rotation='vertical', fontsize=16)

    fig.text(0.5, 0.07, 'Longitude', ha='center',
             va='center', rotation='horizontal', fontsize=16)

    ax.tick_params(axis='both', which='major')
    ax_legend = fig.add_axes([0.1225, 0.02725, 0.78, 0.022], zorder=3)

    cbar = fig.colorbar(im, cax=ax_legend,  # spacing='popotional',
                        # boundaries=color_bounds[var],
                        ticks=color_bounds[var],
                        # fraction=0.034, pad=0.03, aspect=30,
                        orientation='horizontal')
    cbar.ax.tick_params(width=1.5)
    cbar.set_label('Rainfall average values from 1950-2016 (mm/month)',
                   fontweight="bold")

    plt.savefig(os.path.join(
        main_dir + '_' + 'vals_per_months' + '_ppt_values_5.png'),
        bbox_inches='tight')
    plt.close()

STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
