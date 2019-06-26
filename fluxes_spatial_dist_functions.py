# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
import gc
import os
from time import time as timing
import time
import timeit

from matplotlib import style

import _pickle as cPickle
# import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd

plt.ioff()
style.use('seaborn-bright')


def timer(func):
    ''' a decorater to wrap a fct and time it'''
    def fct(*args, **kwargs):
        before = timing()
        rv = func(*args, **kwargs)
        after = timing()
        print('elapsed', after - before)
        return rv
    return fct


def getFiles(data_dir_, file_ext_str):
    ''' create function to get files based on dir '''
    dfs_files = []
    for r, _, f in os.walk(data_dir_):
        for fs in f:
            if fs.endswith(file_ext_str):
                dfs_files.append(os.path.join(r, fs))
    return sorted(dfs_files)


def create_out_dir(save_dir, fldr_name, freq_name, month_nbr):
    '''fct to create folders for output df for every month'''
    final_dir_mean = (save_dir + fldr_name % freq_name)
    out_m_mean = os.path.join(final_dir_mean, month_nbr)
    if not os.path.exists(out_m_mean):
        os.mkdir(out_m_mean)
    return out_m_mean


def create_df_fr_other_df(df_cycle, month_idx_nbr):
    '''fct to create df using other df and a month nbr from idx '''
    df_out = pd.DataFrame(data=df_cycle.loc[month_idx_nbr, :].values,
                          index=df_cycle.columns)
    return df_out


def resampleDf(df_, temp_freq, temp_shift, unit_fct=1):
    ''' resample DF based on freq and time shift and the sum '''
    assert isinstance(df_, pd.DataFrame), 'data is not a df'
    return df_.resample(temp_freq, label='right', closed='right',
                        base=temp_shift).sum() / unit_fct


def save_df(df_tosave, out_dir, df_name):
    ''' fct to save df file based on name and out dir'''
    save_name = os.path.join(out_dir, df_name) + '.csv'
    return df_tosave.to_csv(save_name, sep=';', header=None)


def df_max_mean_min_cycle(data_frame):
    ''' group a Dataframe monthly, to find maxn min and avg yearly cycle'''
    data_frame = data_frame[data_frame >= 0]
    df_cycle_max = data_frame.groupby([data_frame.index.month]).max()
    df_cycle_mean = data_frame.groupby([data_frame.index.month]).mean()
    df_cycle_min = data_frame.groupby([data_frame.index.month]).min()
    idx = df_cycle_max.index
    return idx, df_cycle_max, df_cycle_mean, df_cycle_min


def do_line_plot_x_y(subplot, xvals, yvals, label_, color_):
    '''plot line plot of x and y and define label and color'''
    subplot.plot(xvals, yvals, label=label_, color=color_, alpha=0.7)
    subplot.grid(True)
    subplot.set_xticks([i for i in xvals])
    subplot.set_xlabel("month")
    subplot.set_ylabel('mm')
    plt.legend(loc=0)
    return subplot


def plot_ppt_anunal_cycle(in_ppt_df, out_dir):
    '''plot max, min, and mean annual cycle'''
    ix_ppt, ppt_max, ppt_mean, ppt_min = df_max_mean_min_cycle(
        in_ppt_df)
    _, ax = plt.subplots(figsize=(20, 10))
    do_line_plot_x_y(ax, ix_ppt, ppt_max, 'PPT Max', 'r')
    do_line_plot_x_y(ax, ix_ppt, ppt_mean, 'PPT Mean', 'b')
    do_line_plot_x_y(ax, ix_ppt, ppt_min, 'PPT Min', 'c')

    plt.savefig(os.path.join(out_dir, 'annual_ppt_pet_cycle_SA.png'),
                frameon=True, papertype='a4', bbox_inches='tight')
    return


def agg_max_mean_min_cycle(df_row):
    '''fct to resample and calculate max, min and mean cycles'''
    (_, df_cycle_max, df_cycle_mean,
     df_cycle_min) = df_max_mean_min_cycle(df_row)
    _, df_max_val = df_cycle_max.idxmax(
        axis=0), df_cycle_max.max(axis=0)
    _, df_min_val = df_cycle_min.idxmin(
        axis=0), df_cycle_min.min(axis=0)
    df_mean = df_cycle_mean.mean(axis=0)
    df_dev = (df_max_val - df_min_val) / df_mean
    return df_max_val, df_min_val, df_mean, df_dev


def read_nc_grid_file(nc_file, nc_var_list,
                      time_name, lon_name, lat_name,
                      cut_idx):
    '''fct to read nc file and extract ppt data, time, lat, lon'''
    in_nc = nc.Dataset(nc_file)
    # print('extracting lon, lat and data from nc_file: %s' % nc_file)
    for nc_var_name in nc_var_list:
        if nc_var_name in in_nc.variables:
            # print('reading var: %s' % nc_var_name)
            lon = in_nc.variables[lon_name]
            lat = in_nc.variables[lat_name]
            in_var_data = in_nc.variables[nc_var_name]
            time_var = in_nc.variables[time_name]
            try:
                time_arr = nc.num2date(time_var[:],
                                       time_var.units,
                                       calendar='standard')
            except Exception:
                time_arr = nc.num2date(time_var[:],
                                       time_var.units,
                                       calendar=time_var.calendar)
            time_idx = pd.DatetimeIndex(time_arr)

            if cut_idx:  # use it to match time from two NC files
                time_idx_df = pd.DataFrame(index=time_idx)
                time_idx_df = time_idx_df.iloc[cut_fr_idx:cut_to_idx, :]
                in_var_data = in_var_data[cut_fr_idx:cut_to_idx, :, :]
                time_idx = time_idx_df.index
            # print('done getting the values for all the grid')

            return lat, lon, in_var_data, time_idx, in_nc


def calc_cycle_grid_vals(nc_file, nc_var_name,
                         time_name, lon_name, lat_name,
                         temp_freq, out_freq_name, save_dir,
                         use_fctr=False, unit_fctr=1):
    '''fct to calculate and save cycles  per grid point'''
    lat, _, in_var_data, time_idx, nc_dataset = read_nc_grid_file(
        nc_file, nc_var_name, time_name, lon_name, lat_name, cut_idx)
    out_var_name = nc_f[-15:-3]
    # print('out var name is: ', out_var_name)
    for ix, _ in enumerate(lat):
        print('going through latitude idx: %d' % ix)
        try:
            ppt_row = np.array(in_var_data[:, ix, :].data, dtype='float64')
        except Exception:
            ppt_row = np.array(in_var_data[:, ix, :], dtype='float64')
        df_ = pd.DataFrame(index=time_idx, data=ppt_row)
        if use_fctr:
            for m, idx in zip(df_.index.daysinmonth, df_.index):
                df_.loc[idx, :] = df_.loc[idx, :].values * m

        df_.dropna(inplace=True)
        df_idx = resampleDf(df_, temp_freq, 0, unit_fctr)
        df_max, df_min, df_mean, df_dev = agg_max_mean_min_cycle(
            df_idx)

        save_df(df_max, save_dir,
                'df_maximum_%s_vals\df_max_vals_%s_for_row_idx_%d'
                % (out_freq_name, out_var_name, ix))
        save_df(df_min, save_dir,
                'df_minimum_%s_vals\df_min_vals_%s_for_row_idx_%d'
                % (out_freq_name, out_var_name, ix))
        save_df(df_mean, save_dir,
                'df_average_%s_vals\df_avg_%s_for_row_idx_%d'
                % (out_freq_name, out_var_name, ix))
        save_df(df_dev, save_dir,
                'deviation_max_min_%s\df_%s_dev_for_row_idx_%d'
                % (out_freq_name, out_var_name, ix))

        del (df_, df_idx, df_max, df_min, df_mean, df_dev, ppt_row, ix)
    # print('done extracting df cycle for every cell in the grid')
    nc_dataset.close()
    return


def fill_arr_for_plt_maps(df_files_dir, nc_file, nc_var_list,
                          lon_name, lat_name, what_to_plt):
    ''' fct that reads df cycle per cell and constructs total grid'''

    in_monthly_vals = getFiles(df_files_dir, '.csv')
#     print(in_monthly_vals)
    lat, lon, _, _, _ = read_nc_grid_file(nc_file,
                                          nc_var_list,
                                          'time',
                                          lon_name,
                                          lat_name,
                                          cut_idx)
    fill_data_month_max_arr = np.zeros(
        shape=(1, lat.shape[0], lon.shape[0]))
#         lon2 = np.array([(l - 360) if l >= 180 else l for l in lon],
#                         dtype='float64')

    x, y = np.meshgrid(lon, lat)

    print('constructing grid for netcdf file: ', nc_file)
    vals_per_sr = [f for f in in_monthly_vals
                   if nc_file[-15:-3] in f]
    assert len(vals_per_sr) > 0, 'error getting the files'
    for _, df_monthly_row_vls in zip(lat, vals_per_sr):
        row_idx = int(df_monthly_row_vls.split(
            sep='_')[-1].split('.')[0])
        df_vls = pd.read_csv(df_monthly_row_vls, sep=';',
                             index_col=0, header=None)
        print(df_vls)

        if what_to_plt == 'deviation':
            df_vls[df_vls <= 0] = np.nan

        if what_to_plt != 'deviation':
            df_vls[df_vls < 0] = np.nan

        fill_data_month_max_arr[:, row_idx,
                                :] = df_vls.values.ravel()

    return x, y, fill_data_month_max_arr


def find_similarity_old_mtx(df_files_dir, nc_file, nc_var_list,
                            lon_name, lat_name, what_to_plt):
    '''fct to calculate std between each cell and surrounding 8 cells'''
    _, _, fill_data_month_max_arr = fill_arr_for_plt_maps(df_files_dir,
                                                          nc_file,
                                                          nc_var_list,
                                                          lon_name,
                                                          lat_name,
                                                          what_to_plt)

    lat, lon, _, _, _ = read_nc_grid_file(
        nc_file, nc_var_list, 'time',
        lon_name, lat_name, cut_idx)
    fill_data_month_var_arr = np.zeros(shape=(1, lat[1:-1].shape[0],
                                              lon[1:-1].shape[0]))
    xdev, ydev = np.meshgrid(lon[1:-1], lat[1:-1])
    for latx, _ in enumerate(lat[1:-1]):
        for lonx, _ in enumerate(lon[1:-1]):
            print('going through lat idx: ', latx, 'lon idx: ', lonx)
            point_of_int = fill_data_month_max_arr[:, latx, lonx][:]
            point_a = fill_data_month_max_arr[:, latx, lonx - 1][:]
            point_b = fill_data_month_max_arr[:, latx + 1, lonx - 1][:]
            point_c = fill_data_month_max_arr[:, latx + 1, lonx][:]
            point_d = fill_data_month_max_arr[:, latx + 1, lonx + 1][:]
            point_e = fill_data_month_max_arr[:, latx, lonx + 1][:]
            point_f = fill_data_month_max_arr[:, latx - 1, lonx + 1][:]
            point_g = fill_data_month_max_arr[:, latx - 1, lonx][:]
            point_h = fill_data_month_max_arr[:, latx - 1, lonx - 1][:]

            var_0_max = np.std([point_of_int, point_a,
                                point_b, point_c, point_d, point_e,
                                point_f, point_g, point_h])
            fill_data_month_var_arr[:, latx, lonx] = var_0_max

    return xdev, ydev, fill_data_month_var_arr


def create_dir_and_df_and_save_df(save_dir, fldr_name, freq_name, month_nbr,
                                  df_name, out_var_name, idx,
                                  df_to_save, df_cols=None):
    ''' fct to create out dir and out folder and df for saving'''
    out_m_dev = create_out_dir(save_dir, fldr_name, freq_name, month_nbr)
    if df_cols is None:
        df_idx = df_to_save.columns
    elif df_cols is not None:
        df_idx = df_cols.columns
    df_out_dev = pd.DataFrame(data=df_to_save, index=df_idx)
    save_df(df_out_dev, out_m_dev, df_name % (out_var_name, idx))
    return


def calc_monthly_grid_vals(nc_file, nc_var_name,
                           time_name, lon_name, lat_name,
                           temp_freq, out_freq_name, save_dir,
                           use_fctr=False, unit_fctr=1):
    '''fct to calculate and save cycles  per grid point'''

    lat, _, in_var_data, time_idx, _ = read_nc_grid_file(nc_file,
                                                         nc_var_name,
                                                         time_name,
                                                         lon_name,
                                                         lat_name,
                                                         cut_idx)
    out_var_name = nc_file[-15:-3]
    print('out var name in file is: ', out_var_name)

    for ix, _ in enumerate(lat):
        print('going through latitude idx: %d' % ix)
        ppt_row = np.array(in_var_data[:, ix, :].data, dtype='float64')
        try:
            df_ = pd.DataFrame(index=time_idx, data=ppt_row)
        except Exception:
            ppt_row = np.array(in_var_data[:, ix, :], dtype='float64')
            df_ = pd.DataFrame(index=time_idx, data=ppt_row)
        if use_fctr:
            for m, idx in zip(df_.index.daysinmonth, df_.index):
                df_.loc[idx, :] = df_.loc[idx, :].values * m

        df_.dropna(inplace=True)
        df_idx = resampleDf(df_, temp_freq, 0, unit_fctr)
        idx, df_cycle_max, df_cycle_mean, df_cycle_min = df_max_mean_min_cycle(
            df_idx)

        for month_nbr in idx:
            print('extracting data for month nbr: ', month_nbr)
            mt_nb = 'month_2_nbr_%d' % month_nbr
#
#             df_dev_m = ((df_cycle_max.loc[month_nbr, :].values -
#                          df_cycle_min.loc[month_nbr, :].values)
#                         / df_cycle_mean.loc[month_nbr, :].values)
#
#             create_dir_and_df_and_save_df(save_dir,
#                                           r'\deviation_max_min_%s_per_month',
#                                           out_freq_name, mt_nb,
#                                           'df_%s_dev_for_row_idx_%d',
#                                           out_var_name, ix,
#                                           df_dev_m, df_cycle_max)
#
#             create_dir_and_df_and_save_df(save_dir,
#                                           r'\df_minimum_%s_vals_per_month',
#                                           out_freq_name, mt_nb,
#                                           'df_%s_min_for_row_idx_%d',
#                                           out_var_name, ix,
#                                           df_cycle_min.loc[month_nbr,
#                                                            :].values,
#                                           df_cycle_min)
#
#             create_dir_and_df_and_save_df(save_dir,
#                                           r'\df_maximum_%s_vals_per_month',
#                                           out_freq_name, mt_nb,
#                                           'df_%s_max_for_row_idx_%d',
#                                           out_var_name, ix,
#                                           df_cycle_max.loc[month_nbr,
#                                                            :].values,
#                                           df_cycle_max)
            create_dir_and_df_and_save_df(save_dir,
                                          r'\df_average_%s_vals_per_month',
                                          out_freq_name, mt_nb,
                                          'df_%s_avg_for_row_idx_%d',
                                          out_var_name, ix,
                                          (df_cycle_mean.loc[month_nbr,
                                                             :].values) / df_cycle_mean.sum(),
                                          df_cycle_mean)

            gc.collect()
        del (df_idx, df_, ppt_row, idx, df_cycle_max,
             df_cycle_mean, df_cycle_min)
    return


def cal_coeff_of_var_and_max_min_temp_var(nc_file, nc_var_name,
                                          time_name, lon_name, lat_name,
                                          temp_freq, use_fctr=False, unit_fctr=1,
                                          coeff_var=False, min_max_var=False):
    '''fct to calculate and save cycles  per grid point'''
    lat, lon, in_var_data, time_idx, _ = read_nc_grid_file(
        nc_file, nc_var_name, time_name, lon_name, lat_name, cut_idx)
    if coeff_var:
        fill_data_coeff_of_var = np.zeros(shape=(1, lat.shape[0],
                                                 lon.shape[0]))
    if min_max_var:
        fill_data_min_temp_var = np.zeros(shape=(1, lat.shape[0],
                                                 lon.shape[0]))
        fill_data_max_temp_var = np.zeros(shape=(1, lat.shape[0],
                                                 lon.shape[0]))
    xdev, ydev = np.meshgrid(lon, lat)

    # out_var_name = nc_f[-15:-3]
    print('Calculating Coefficient of Variation')
    # print('out var name is: ', out_var_name)
    for latx, _ in enumerate(lat):
        print('going through latitude idx: %d' % latx)
        for lonx, _ in enumerate(lon):
            print('going through Longitude idx: %d' % lonx)
            try:
                ppt_row = np.array(
                    in_var_data[:, latx, lonx].data, dtype='float64')
            except Exception:
                ppt_row = np.array(in_var_data[:, latx, lonx], dtype='float64')
            df_ = pd.DataFrame(index=time_idx, data=ppt_row)
            df_.dropna(inplace=True)
            df_idx = resampleDf(df_, temp_freq, 0, unit_fctr)
            # print(df_idx)
            if use_fctr:
                for m, idx in zip(df_.index.daysinmonth, df_.index):
                    df_.loc[idx, :] = df_.loc[idx, :].values * m

            if coeff_var:
                std_ppt, mean_ppt = np.std(df_idx), np.mean(df_idx)
                coeff_of_var = std_ppt / mean_ppt
                fill_data_coeff_of_var[:, latx, lonx] = coeff_of_var

            if min_max_var:
                idx, _, df_cycle_mean, _ = df_max_mean_min_cycle(
                    df_idx)

                min_temp_var = (
                    df_cycle_mean.min() /
                    df_cycle_mean.mean()).values
                max_temp_var = (
                    df_cycle_mean.max() /
                    df_cycle_mean.mean()).values
                if max_temp_var > 0:
                    print(max_temp_var)
                fill_data_min_temp_var[:, latx, lonx] = min_temp_var
                fill_data_max_temp_var[:, latx, lonx] = max_temp_var

    if coeff_var:
        return xdev, ydev, fill_data_coeff_of_var
    if min_max_var:
        return xdev, ydev, fill_data_min_temp_var, fill_data_max_temp_var


def plot_colormesh(var, x_vals, y_vals, grid_vals, color_bounds,
                   out_save_dir, plot_title, var_bounds_dict,
                   pref, data_source):
    print('plotting for: %s' % var)
#     up_lim = 4000.01
    up_lim = 500.01
    my_dpi = 100
    fig = plt.figure(figsize=(16, 8), dpi=my_dpi)

    ax0 = fig.add_subplot(111, frame_on=True)

    # for yearrly values
#     colors = [(0 / up_lim, "r"), (10 / up_lim, "salmon"), (50 / up_lim, "orangered"),
#               (100 / up_lim, "goldenrod"), (200 / up_lim, "yellow"),
#               (400 / up_lim, "lightgreen"), (800 / up_lim, "olive"),
#               (1000 / up_lim, "darkolivegreen"),
#               (1500 / up_lim, "aqua"), (2000 / up_lim, "dodgerblue"),
#               (2500 / up_lim, "b"), (4000 / up_lim, 'darkblue'), (4000.01 / up_lim, 'fuchsia')]
    # for monthly values
#     if 'monthly_var' in var:
#         colors = [(0, "red"), (0.05, "orangered"),
#                   (.100, "goldenrod"), (.15, "yellow"),
#                   (0.2, "lightgreen"), (.3, "aqua"),
#                   (.4, 'dodgerblue'), (.5, 'lightblue'),
#                   (.6, 'b'), (.7, 'darkblue'), (0.8, 'fuchsia')]
#     cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
#     # for rainfall std deviation
    colors = [(0.0, "darkblue"), (.01, "blue"), (.05, "lightblue"),
              (.10, "darkgreen"),
              (.25, "olive"), (.5, "lightgreen"),
              (.8, "gold"), (.9, "red"),
              (1, 'darkred')]  # ,(5000 / up_lim, 'darkblue')]
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
#     cmap = plt.get_cmap('jet')
    cmap.set_bad('snow', 1.)
    cbar_label = 'Rainfall (mm/year)'

    if 'Coeff' in plot_title:
        up_lim = 1

        colors = [(0, "w"),
                  (.05, 'darkblue'),
                  (.25, "green"),
                  (.35, "yellow"),
                  (.50, "darkorange"),
                  (.75, "r"), (1, "darkred"),
                  (1.0, "darkred")]
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        cbar_label = ''
    if 'Maximum Annual' in plot_title:
        print('maximum var')
        up_lim = 2.51
        colors = [(0 / up_lim, "darkred"),
                  (1 / up_lim, "salmon"),
                  (1.25 / up_lim, "orange"),
                  (1.5 / up_lim, "yellow"), (1.75 / up_lim, "c"),
                  (2.51 / up_lim, 'darkblue')]
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        cbar_label = 'Ratio between Maximum annual and Mean of annual Precipitation values'

    if 'Contribution' in plot_title:
        up_lim = 1
        print('contribution var')
        colors = [(0, "w"), (0.001, 'darkred'),
                  (.05, 'r'),
                  (.25, "gold"),
                  (.3, "lightgreen"),
                  (1, "darkblue")]
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

        cbar_label = r'$\frac{Wettest\ month\ in\ the\ yearly\ cycle}{Total\ yearly\ rainfall\ sum}$'

    if 'Minimal Annual' in plot_title:
        up_lim = 1
        colors = [(0, "salmon"),
                  (0.1 / up_lim, "salmon"),
                  (0.2 / up_lim, "orange"),
                  (0.3 / up_lim, "yellow"), (0.5 / up_lim, "c"),
                  (1 / up_lim, 'darkblue')]
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        cbar_label = 'Ratio between Minimum annual and Mean of annual Precipitation values'
    print(grid_vals[0].shape)
    im = ax0.pcolormesh(x_vals, y_vals, grid_vals[0], cmap=cmap,  # cmap,
                        snap=True, alpha=1, shading='gouraud',
                        vmin=var_bounds_dict[var][0],
                        vmax=var_bounds_dict[var][1])

    cbar = fig.colorbar(im, ax=ax0,  # spacing='popotional',
                        boundaries=color_bounds[var],
                        extend='max',
                        ticks=color_bounds[var],
                        norm=mcolors.BoundaryNorm(color_bounds[var], cmap.N),
                        fraction=0.024, pad=0.02,
                        aspect=30)
    cbar.ax.tick_params(labelsize=10)
    plt.grid(alpha=0.5, linestyle='--')

    plt.xlabel('Longitude', fontsize=13), plt.ylabel('Latitude', fontsize=13)
    plt.axis('equal')
    if plot_fig_titles:
        fig.suptitle(plot_title, fontsize=12, y=.92)
    if data_source == 'gpcp':
        plt.xticks(np.arange(0, 361, 30), ('0', '30E', '60E', '90E',
                                           '120E', '150E', '180',
                                           '150W', '120W', '90W',
                                           '60W', '30W', '0'))
        plt.yticks(np.arange(-90, 91, 30), ('90S', '60S', '30S', '0',
                                            '30N', '60N', '90N'))

    cbar.set_label(cbar_label, fontsize=13)
    plt.savefig(os.path.join(
        out_save_dir, var + data_source + '_' + pref + '_ppt_values_2.png'),
        frameon=True, papertype='a4', bbox_inches='tight')
    plt.close()
    return


@timer
def plot_global_maps(var_to_plt, nc_files, nc_var_list, long_name,
                     lat_name, out_save_dir, data_source, plt_orig_vls=False,
                     plt_var_mtx=False, plt_orig_monthly_vls=False,
                     plt_coeff_of_var=False, plt_max_temp_dev=False,
                     plt_min_temp_dev=False):

    #     vars_fldr = ['deviation_max_min_%s' % var_to_plt,
    #                  'df_maximum_%s_vals' % var_to_plt,
    #                  'df_minimum_%s_vals' % var_to_plt,
    #                  'df_average_%s_vals' % var_to_plt]
    vars_fldr = ['df_average_%s_vals' % var_to_plt]

    if var_to_plt == 'yearly':
        if plt_orig_vls:
            #             var_bounds_dict = {vars_fldr[0]: [0, 5],
            #                                vars_fldr[1]: [0, 6000],
            #                                vars_fldr[2]: [0, 1000],
            #                                vars_fldr[3]: [0, 4000]}
            var_bounds_dict = {vars_fldr[0]: [0, 4000]}
            bounds_mean = [0, 10, 50, 100, 200,
                           400, 800, 1000, 1500, 2000, 2500, 4000, 4000.01]
            color_bounds = {vars_fldr[0]: bounds_mean}

        if plt_var_mtx:
            var_bounds_dict = {vars_fldr[0]: [0, 500.01]}
            bounds_mean = [0, 5, 50, 125, 250, 450, 500, 500.01]
#             var_bounds_dict = {vars_fldr[0]: [0, 1.01]}
#             bounds_mean = [0, .1, .2, .5, .75, 1]
            color_bounds = {vars_fldr[0]: bounds_mean}
    if var_to_plt == 'monthly':
        if plt_orig_vls:
            #             var_bounds_dict = {vars_fldr[0]: [0, 100],
            #                                vars_fldr[1]: [0, 2000],
            #                                vars_fldr[2]: [0, 500],
            #                                vars_fldr[3]: [0, 1200]}
            var_bounds_dict = {vars_fldr[0]: [0, None]}
#             bounds_mean = [10, 25, 50, 75, 100, 150, 200,
#                            300, 400, 600, 800, 1000, 2000]
            color_bounds = {vars_fldr[0]: None}
        if plt_var_mtx:
            color_bounds = {vars_fldr[0]: [0, None]}
            var_bounds_dict = {vars_fldr[0]: [0, 20],
                               vars_fldr[1]: [0, 300],
                               vars_fldr[2]: [0, 20],
                               vars_fldr[3]: [0, 50]}
    for var in vars_fldr:
        if 'deviation' in var:
            #             what_to_plt = 'deviation'
            pass
        if 'deviation' not in var:
            #             what_to_plt = 'min and max'
            pass
        if plt_orig_vls:
            #             df_fldr = os.path.join(data_dir_gpc, var)

            #             x_vals, y_vals, grid_vals = fill_arr_for_plt_maps(df_fldr,
            #                                                               nc_files,
            #                                                               nc_var_list,
            #                                                               long_name,
            #                                                               lat_name,
            # what_to_plt)
            x_vals = cPickle.load(open("grid_x_vls.pkl", 'rb'))
            y_vals = cPickle.load(open("grid_y_vls.pkl", 'rb'))
            grid_vals = cPickle.load(open("grid_orig_vls.pkl", "rb"))
            plot_title = ('Mean Annual Precipitation (1950 - 2016)')
            plot_colormesh(var, x_vals, y_vals, grid_vals, color_bounds,
                           out_save_dir, plot_title, var_bounds_dict,
                           'ppt_annual_', data_source,)
        if plt_var_mtx:

            #             df_fldr = os.path.join(data_dir_gpc, var)

            #             x_vals, y_vals, grid_vals = find_similarity_old_mtx(df_fldr,
            #                                                                 nc_files,
            #                                                                 nc_var_list,
            #                                                                 long_name,
            #                                                                 lat_name,
            #                                                                 what_to_plt)
            #             cPickle.dump(x_vals, open("grid_std_x_vals_.pkl", "wb"))
            #             cPickle.dump(y_vals, open("grid_std_y_vals_.pkl", "wb"))
            #             cPickle.dump(grid_vals, open("grid_std_var_.pkl", "wb"))
            x_vals = cPickle.load(open("grid_std_x_vals_.pkl", 'rb'))
            y_vals = cPickle.load(open("grid_std_y_vals_.pkl", 'rb'))

            grid_vals = cPickle.load(open("grid_std_var_.pkl", "rb"))
#             import pdb
#             pdb.set_trace()
            plot_title = (('Standard deviation between each grid cell'
                           ' and surounding 8 cells based on average yearly values'
                           ' from 1950 till 2016'))
            plot_colormesh(var, x_vals, y_vals, grid_vals, color_bounds,
                           out_save_dir, plot_title, var_bounds_dict,
                           'std_dev', data_source)
        if plt_orig_monthly_vls:
            var = 'df_average_monthly_vals'
            var_bounds_dict = {var: [0, 800.01]}
            bounds_mean = [0, 50, 100, 150, 200, 300, 400, 500,
                           600, 700, 800, 800.01]
#             var_bounds_dict = {var: [0, .51]}
#             bounds_mean = [0, 0.050, 0.100, 0.150, 0.200, 0.300, 0.400, 0.500]
            # 0.600, 0.700, 0.800]
            color_bounds = {var: bounds_mean}
            x_vals = cPickle.load(open("grid_x_vls.pkl", 'rb'))
            y_vals = cPickle.load(open("grid_y_vls.pkl", 'rb'))
            fig = plt.figure(figsize=(26, 13), dpi=200)
            fig.subplots_adjust(hspace=0.1, wspace=0.1)
#             up_lim = 800.01
#             colors = [(0, "indianred"), (50 / up_lim, "orange"),
#                       (100 / up_lim, "y"), (150 / up_lim, "limegreen"),
#                       (200 / up_lim, "green"), (400 / up_lim, "darkgreen"),
#                       (600 / up_lim, 'lightblue'),
#                       (800 / up_lim, 'blue'),
#                       (800.01 / up_lim, 'darkblue')]
            colors = [(0.0, "darkblue"), (0.0625, "lightblue"), (0.125, 'yellowgreen'),
                      (0.1825, 'limegreen'),
                      (0.25, "g"), (0.375, "darkgreen"),
                      (0.5, "violet"), (.875, 'orange'), (0.99, 'r'), (1, 'darkred')]
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'my_colormap', colors)
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
            rcParams['axes.labelpad'] = 15
            for i in range(1, 13):
                print('plotting for var: %s and month nbr: %d' % (var, i))
#                 df_fldr = os.path.join(data_dir_gpc, var + '_per_month',
#                                        'month_2_nbr_%d' % i)
#                 x_vals, y_vals, grid_vals = fill_arr_for_plt_maps(df_fldr,
#                                                                   nc_files,
#                                                                   nc_var_list,
#                                                                   long_name,
#                                                                   lat_name,
#                                                                   what_to_plt)
#
#                 cPickle.dump(grid_vals, open(
#                     "grid_vls_2_per_mth_%d.pkl" %
#                     i, "wb"))
#                 cPickle.dump(x_vals, open(
#                     "grid_x_vls.pkl", "wb"))
#                 cPickle.dump(y_vals, open(
#                     "grid_y_vls.pkl", "wb"))

                ax = fig.add_subplot(3, 4, i)
                ax.set_title('%s' %
                             monthes[i], fontweight="bold")  # , size=fontsize)  # Title

                grid_vals = cPickle.load(open("grid_vls_per_mth_%d.pkl"
                                              % i, 'rb'))

                im = ax.pcolormesh(x_vals, y_vals, grid_vals[0], cmap=cmap,  # cmap,
                                   snap=True, alpha=1, shading='gouraud',
                                   vmin=var_bounds_dict[var][0],
                                   vmax=var_bounds_dict[var][1])

                ax.grid(True, alpha=0.5, linestyle='--')

                if i in [2, 3, 4, 6, 7, 8]:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                if i == 9:
                    plt.xlabel(
                        'Longitude')
                    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
                    plt.ylabel(
                        'Latitude')
                    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
                    ax.set_yticks([-80, -40, 0, 40, 80])
                if i == 10 or i == 11 or i == 12:
                    plt.xlabel('Longitude')
                    ax.set_xticks([-150, -100, -50, 0, 50, 100, 150])
                    ax.set_yticklabels([])
                if i == 1 or i == 5:
                    plt.ylabel(
                        'Latitude')
                    ax.set_yticks([-80, -40, 0, 40, 80])
                    ax.set_xticklabels([])
            ax.tick_params(axis='both', which='major')
            ax_legend = fig.add_axes([0.1225, 0.004725, 0.78, 0.022], zorder=3)
            cbar = fig.colorbar(im, cax=ax_legend,  # spacing='popotional',
                                boundaries=color_bounds[var],
                                extend='max',
                                ticks=color_bounds[var],
                                norm=mcolors.BoundaryNorm(
                                    color_bounds[var], cmap.N),
                                cmap=cmap,
                                # fraction=0.034, pad=0.03, aspect=30,
                                orientation='horizontal')
            cbar.ax.tick_params(width=1.5)
            cbar.set_label(cbar_label, fontweight="bold")

            plt.savefig(os.path.join(
                out_save_dir, var + data_source + '_' + 'vals_per_months' + '_ppt_values_.png'),
                frameon=True, papertype='a4', bbox_inches='tight')
            plt.close()
#==============================================================
#
#==============================================================
    x_vals = cPickle.load(open("grid_x_vls.pkl", 'rb'))
    y_vals = cPickle.load(open("grid_y_vls.pkl", 'rb'))

    if plt_coeff_of_var:
        var = 'Coff_of_Variation'
        var_bounds_dict = {var: [0, 1.01]}
        color_bounds = {var: [0, 0.05, 0.1, .25, .35, .5, 0.75, 1, 1.01]}
        grid_vals = cPickle.load(open("grid_coeff_of_var.pkl", 'rb'))
#         x_vals, y_vals, grid_vals = cal_coeff_of_var_and_max_min_temp_var(
#             nc_files, nc_var_list, 'time', long_name, lat_name, 'Y',
#             use_fctr=False, unit_fctr=1, coeff_var=True, min_max_var=False)
#         cPickle.dump(grid_vals, open("grid_coeff_of_var.pkl", "wb"))
        plot_title = 'Coefficient of Variation (1950 - 2016)'
        plot_colormesh(var, x_vals, y_vals, grid_vals, color_bounds,
                       out_save_dir, plot_title, var_bounds_dict,
                       'Coeff_of_variation', data_source)

    if plt_max_temp_dev:
        var = 'df_temp_dev_max_%s_vals' % var_to_plt
        var_bounds_dict = {var: [1, 2.51]}
        # {var: [0, .501]}
#         color_bounds = {var: [0, .05, .10, .20, .5, .501]}
        color_bounds = {var: [1, 1.25, 1.5, 1.75, 2, 2.5, 2.51]}
#         df_fldr = os.path.join(data_dir_gpc, var)
        grid_vals_max = cPickle.load(open("grid_max_temp_dev_vls.pkl", 'rb'))

#         x_vals, y_vals, grid_vals_min, grid_vals_max = cal_coeff_of_var_and_max_min_temp_var(
#             nc_files, nc_var_list, 'time', long_name, lat_name, 'M',
#             use_fctr=False, unit_fctr=1, coeff_var=False, min_max_var=True)
#         cPickle.dump(grid_vals_max, open("grid_max_temp_dev_vls.pkl", "wb"))
#         cPickle.dump(grid_vals_min, open("grid_min_temp_dev_vls.pkl", "wb"))
        plot_title = (
            'Maximum Annual Temporal Precipitation Variability (1950 - 2016)')
#         plot_title = (
#             'Contribution of the rainiest month to the annual precipiation sum')
        plot_colormesh(var, x_vals, y_vals, grid_vals_max, color_bounds,
                       out_save_dir, plot_title, var_bounds_dict,
                       'ppt_max_mean_variability', data_source)

    if plt_min_temp_dev:
        var = 'df_temp_dev_min_%s_vals' % var_to_plt
        var_bounds_dict = {var: [0, 1]}
        color_bounds = {var: [0, 0.1, 0.2, .3, .50, .75, 1]}
#         df_fldr = os.path.join(data_dir_gpc, var)
        grid_vals_min = cPickle.load(open("grid_min_temp_dev.pkl", 'rb'))
        plot_title = (
            'Minimal Annual Temporal Precipitation Variability (1950 - 2016)')
        plot_colormesh(var, x_vals, y_vals, grid_vals_min, color_bounds,
                       out_save_dir, plot_title, var_bounds_dict,
                       'ppt_min_variability', data_source)
    return


#==============================================================================
#
#==============================================================================
if __name__ == '__main__':

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management'
    os.chdir(main_dir)
    # thing_to_save = cPickle.load( open( "filename.pkl", "rb" ) )
    data_dir_gpc = os.path.join(main_dir, 'data')
    if not os.path.exists(data_dir_gpc):
        os.mkdir(data_dir_gpc)

    data_ppt_loc = r'X:\exchange\ElHachem\PPT_data'
    shp_dir_loc = os.path.join(main_dir,
                               r'continents_shp\continent shapefile\continent.shp')
    nc_var_lst = ['PREC', 'precip']

    nc_f = os.path.join(data_ppt_loc, 'full_data_monthly_v2018_025.nc')
#     nc_f = os.path.join(data_ppt_loc, 'full_data_monthly_size25_v2018_25.nc')

#     nc_f = os.path.join(data_ppt_loc, 'precip.mon.mean.0.5x0.5.nc')
    print('reading file:', nc_f)
    cut_idx = True
    cut_fr_idx = 709
    cut_to_idx = 1512

    plot_fig_titles = False
#     calc_cycle_grid_vals(nc_f, nc_var_lst, 'time', 'lon',
#                          'lat', 'M', 'monthly', data_dir_gpc,
#                          use_fctr=False, unit_fctr=1)
#     calc_cycle_grid_vals(nc_f, nc_var_lst, 'time', 'lon',
#                          'lat', 'Y', 'yearly', data_dir_gpc,
#                          use_fctr=False, unit_fctr=1)
#
#     calc_monthly_grid_vals(nc_f, nc_var_lst, 'time', 'lon',
#                            'lat', 'M', 'monthly', data_dir_gpc,
#                            use_fctr=False, unit_fctr=1)
#

    plot_global_maps('yearly', nc_f, nc_var_lst,
                     'lon', 'lat', main_dir, 'gpcc',
                     False, False, True, False, False, False)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
