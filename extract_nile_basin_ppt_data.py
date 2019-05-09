# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % EL Hachem Abbas,
Institut für Wasser- und Umweltsystemmodellierung - IWS
"""
# %%

import os
import time
import timeit

import ogr
import osr

import netCDF4 as nc
import numpy as np
import pandas as pd
import pickle as pk


ogr.UseExceptions()
osr.UseExceptions()


def get_all_files(files_ext, files_dir):
    new_list = []
    for elm in os.listdir(files_dir):
        if elm[-len(files_ext):] == files_ext:
            new_list.append(files_dir + '\\' + elm)
    return(sorted(new_list))


def save_obj(obj, name):
    ''' saves a dict as pickle file'''
    with open(name + '.pkl', 'wb+') as f:
        pk.dump(obj, f, protocol=0)
# %%


def extract_grid_ppt_shpfile(_,
                             in_net_cdf_files_list, in_cat_shp_files,
                             sub_cat_out_ppt_data_df,
                             dict_for_gridded_data_,
                             _dict_for_coords_int_, time_freq,
                             other_args):
    '''
     Extract precipitation from a given list of netCDf files
     The catchments shapefile can have one more catchment polygons.
    # a list of some arguments
    # [field name to use as catchment names / numbers,
    # netCDF EPSG code,
    # netCDF time name,
    # netCDF X coords name,
    # netCDF Y coords name,
    # netCDF variable to read,
    # apply cell corner correction]
    other_args = ['DN', 32718, 'time', 'lat', 'long', 'ppt', True]
    '''

    for _, in_cat_shp_file in enumerate(in_cat_shp_files):
        cat_vec = ogr.Open(in_cat_shp_file)
        lyr = cat_vec.GetLayer(0)

        spt_ref = lyr.GetSpatialRef()
        # print(spt_ref)
        trgt = osr.SpatialReference()
        trgt.ImportFromEPSG(other_args[1])
        tfm = osr.CreateCoordinateTransformation(spt_ref, trgt)
        back_tfm = osr.CreateCoordinateTransformation(trgt, spt_ref)
        feat_dict = {}
        feat_area_dict = {}
        cat_area_ratios_dict = {}
        cat_envel_dict = {}

        feat = lyr.GetNextFeature()

        while feat:
            geom = feat.GetGeometryRef()
            f_val = feat.GetFieldAsString(str(other_args[0]))
            if f_val is None:
                raise RuntimeError

            feat_area_dict[f_val] = geom.Area()  # do before transform

            geom.Transform(tfm)
            feat_dict[f_val] = feat

            cat_envel_dict[f_val] = geom.GetEnvelope()  # do after transform
            feat = lyr.GetNextFeature()

        for in_net_cdf in in_net_cdf_files_list:
            print('Going through: %s' % in_net_cdf)
            in_nc = nc.Dataset(in_net_cdf)
            lat_arr = in_nc.variables[other_args[4]][:]
            lon_arr = in_nc.variables[other_args[3]][:]

            apply_cell_correc = other_args[6]

            # convert the netCDF time to regular time
            time_var = in_nc.variables[other_args[2]]

            time_arr = nc.num2date(in_nc.variables[other_args[2]][:],
                                   time_var.units,
                                   calendar='standard')   # time_var.calendar)
            time_index = pd.date_range(start=time_arr[0], end=time_arr[-1],
                                       freq=time_freq)
            ppt_var = in_nc.variables[other_args[5]]

            print('Counting time from (in the netCDF file):', time_var.units)
            print('Start date in the netCDF: ', time_arr[0])
            print('End date in the netCDF: ', time_arr[-1])
            print('Total time steps in the netCDF: ', len(time_arr))

            cell_size = round(lon_arr[1] - lon_arr[0], 3)
            x_l_c = lon_arr[0]
            x_u_c = lon_arr[-1]
            y_l_c = lat_arr[0]
            y_u_c = lat_arr[-1]

            flip_lr = False
            flip_ud = False

            if x_l_c > x_u_c:
                x_l_c, x_u_c = x_u_c, x_l_c
                flip_lr = True

            if y_l_c > y_u_c:
                y_l_c, y_u_c = y_u_c, y_l_c
                flip_ud = True

            if apply_cell_correc:
                x_l_c -= (cell_size / 2.)
                x_u_c -= (cell_size / 2.)
                y_l_c -= (cell_size / 2.)
                y_u_c -= (cell_size / 2.)

            x_coords = np.arange(x_l_c, x_u_c * 1.00000001, cell_size)
            y_coords = np.arange(y_l_c, y_u_c * 1.00000001, cell_size)
            cat_x_idxs_dict = {}
            cat_y_idxs_dict = {}

            print(feat_dict.keys())

            for cat_no in feat_dict.keys():
                print('Cat no:', cat_no)
                geom = feat_dict[cat_no].GetGeometryRef()

                extents = cat_envel_dict[cat_no]
                cat_area = feat_area_dict[cat_no]

                inter_areas = []
                x_low, x_hi, y_low, y_hi = extents

                # adjustment to get all cells intersecting the polygon
                x_low = x_low - cell_size
                x_hi = x_hi + cell_size
                y_low = y_low - cell_size
                y_hi = y_hi + cell_size
                # print(x_coords, x_low, x_hi)
                x_cors_idxs = np.where(np.logical_and(x_coords >= x_low,
                                                      x_coords <= x_hi))[0]
                y_cors_idxs = np.where(np.logical_and(y_coords >= y_low,
                                                      y_coords <= y_hi))[0]

                x_cors = x_coords[x_cors_idxs]
                y_cors = y_coords[y_cors_idxs]
                # print(x_cors_idxs)
                cat_x_idxs = []
                cat_y_idxs = []

                for x_idx in range(x_cors.shape[0] - 1):
                    for y_idx in range(y_cors.shape[0] - 1):
                        ring = ogr.Geometry(ogr.wkbLinearRing)

                        ring.AddPoint(x_cors[x_idx], y_cors[y_idx])
                        ring.AddPoint(x_cors[x_idx + 1], y_cors[y_idx])
                        ring.AddPoint(x_cors[x_idx + 1], y_cors[y_idx + 1])
                        ring.AddPoint(x_cors[x_idx], y_cors[y_idx + 1])
                        ring.AddPoint(x_cors[x_idx], y_cors[y_idx])

                        poly = ogr.Geometry(ogr.wkbPolygon)
                        poly.AddGeometry(ring)

                        inter_poly = poly.Intersection(geom)

                        # to get the area, I convert it to coordinate sys of
                        # the shapefile that is hopefully in linear units
                        inter_poly.Transform(back_tfm)
                        inter_area = inter_poly.Area()

                        inter_areas.append(inter_area)

                        cat_x_idxs.append((x_cors[x_idx] - x_l_c) / cell_size)
                        cat_y_idxs.append((y_cors[y_idx] - y_l_c) / cell_size)
                        _dict_for_coords_int_[cat_no].append((x_cors[x_idx],
                                                              y_cors[y_idx]))
                cat_area_ratios_dict[cat_no] = np.divide(inter_areas, cat_area)

                cat_x_idxs_dict[cat_no] = np.int64(np.round(cat_x_idxs, 6))
                cat_y_idxs_dict[cat_no] = np.int64(np.round(cat_y_idxs, 6))

            for idx, date in enumerate(time_index):
                if date in sub_cat_out_ppt_data_df.index:
                    all_ppt_vals = np.array(ppt_var[idx].data, dtype='float64')
                if flip_lr:
                    all_ppt_vals = np.fliplr(all_ppt_vals)
                if flip_ud:
                    all_ppt_vals = np.flipud(all_ppt_vals)

                for cat_no in feat_dict.keys():

                    ppt_vals = all_ppt_vals[cat_y_idxs_dict[cat_no],
                                            cat_x_idxs_dict[cat_no]]

                    fin_ppt_vals = np.multiply(ppt_vals,
                                               cat_area_ratios_dict[cat_no])
                    sub_cat_out_ppt_data_df.loc[date][cat_no] = \
                        round(np.sum(fin_ppt_vals), 2)

                    dict_for_gridded_data_[cat_no][date].append(ppt_vals)
                print('done getting data for', date)

            in_nc.close()
            cat_vec.Destroy()

            break
    return dict_for_gridded_data_, _dict_for_coords_int_


def read_dict_and_make_nc_files(dict_file):
    dict_obcts_in_list = []
    with (open(dict_file, "rb")) as openfile:
        while True:
            try:
                dict_obcts_in_list.append(pk.load(openfile))
            except EOFError:
                break
    return dict_obcts_in_list


def build_df_from_dict(in_cat_dict, in_crd_dict, in_nc_ext,
                       out_sub_dir, time_freq):
    out_dir = os.path.join(main_dir, out_sub_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    assert in_cat_dict.keys() == in_crd_dict.keys()
    for cat in in_cat_dict.keys():
        print('building df for cat nbr:', cat)
        df_all_vals = pd.DataFrame.from_dict(in_cat_dict[cat],
                                             orient='index')
        df_all_vals.dropna(axis=0, inplace=True)
        df_all = pd.DataFrame(data=[val[0] for val in df_all_vals.values],
                              index=df_all_vals.index,
                              columns=[coords for coords in in_crd_dict[cat]])
        df_all.to_csv(os.path.join(out_dir,
                                   'cat_nbr_%s_gridded_%s_ppt_vals_nc_f_%s.csv'
                                   % (cat, time_freq, in_nc_ext)),
                      sep=';', float_format='%.3f')

    return df_all


# %%
if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = os.path.join(
        r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management')
    os.chdir(main_dir)

    # specify which shp file
    in_cat_shpfile = [os.path.join(main_dir,
                                   r'Nile_qgis\nile_basin\nile_basin.shp')]
    assert in_cat_shpfile
    in_data_dir = os.path.join(r'X:\exchange\ElHachem\PPT_data')
    in_nc_files = [os.path.join(in_data_dir, 'precip.mon.mean.0.5x0.5.nc')]
#     in_nc_files = [os.path.join(in_data_dir, 'full_data_monthly_v2018_025.nc')]
    assert in_nc_files

    in_nc_ext = in_nc_files[0][-15:-3]
    out_dict_vals_name = ('out_nc_gridded_monthly_ppt_data_nc_f_%s'
                          % in_nc_ext)
    out_dict_coords_name = ('out_nc_gridded_monthly_ppt_coords_nc_f_%s'
                            % in_nc_ext)

    other_args = ['HYBAS_ID', 4326, 'time', 'lon',
                  'lat', 'precip', True]

    start_date = '1891-01-01'
    end_date = '2016-12-01'

    freq = 'M'

    for in_cat_shp in in_cat_shpfile:
        cat_vec = ogr.Open(in_cat_shp)
        lyr = cat_vec.GetLayer(0)

        feat_area_dict = {}

        for feat in lyr:  # just to get the names of the catchments
            geom = feat.GetGeometryRef()
            f_val = feat.GetFieldAsString(str(other_args[0]))
            if f_val is None:
                raise RuntimeError
            feat_area_dict[f_val] = geom.Area()
    cat_vec.Destroy()

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    out_data_df = pd.DataFrame(index=date_range,
                               columns=feat_area_dict.keys(),
                               dtype=float)

    dict_for_coords_int = {d: [] for d in feat_area_dict.keys()}
    dict_for_gridded_data = {c: {d: [] for d in out_data_df.index}
                             for c in feat_area_dict.keys()}

    out_dict_data, out_dict_coords = \
        extract_grid_ppt_shpfile(date_range,
                                 in_nc_files, in_cat_shpfile,
                                 out_data_df,
                                 dict_for_gridded_data,
                                 dict_for_coords_int,
                                 freq,
                                 other_args)
    save_obj(out_dict_data, out_dict_vals_name)
    save_obj(out_dict_coords, out_dict_coords_name)

    our_dict_ = read_dict_and_make_nc_files(
        os.path.join(main_dir, 'out_nc_gridded_monthly_ppt_data_nc_f_%s.pkl'
                     % in_nc_ext))[0]
    coord_dict = read_dict_and_make_nc_files(
        os.path.join(main_dir, 'out_nc_gridded_monthly_ppt_coords_nc_f_%s.pkl'
                     % in_nc_ext))[0]

    build_df_from_dict(our_dict_, coord_dict, in_nc_ext,
                       os.path.join(main_dir,
                                    r'data\gridded_ppt_monthly_data'), freq)
STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
