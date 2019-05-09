def extract_continents_coords(in_cat_shp_files, in_netcdf_files_list,
                              nc_var):
    import ogr
#         import osr
    global ppt_var, ppt_var_edit

    cat_vec = ogr.Open(in_cat_shp_files)
    lyr = cat_vec.GetLayer(0)

    feat_dict = {}
    feat_area_dict = {}
    cat_envel_dict = {}

    feat = lyr.GetNextFeature()

    while feat:
        geom = feat.GetGeometryRef()
        f_val = feat.GetFieldAsString('CONTINENT')

        if f_val is None:
            raise RuntimeError

        feat_area_dict[f_val] = geom.Area()  # do before transform

        feat_dict[f_val] = feat

        # do after transform
        cat_envel_dict[f_val] = geom.GetEnvelope()
        feat = lyr.GetNextFeature()

    for in_net_cdf in in_netcdf_files_list:
        print('Going through: %s' % in_net_cdf)
        in_nc = nc.Dataset(in_net_cdf)
        lat_arr = in_nc.variables['lat'][:]
        lon_arr = in_nc.variables['lon'][:]

        apply_cell_correc = True

        ppt_var = in_nc.variables[nc_var]
        ppt_var_edit = np.full(ppt_var.shape, -9999., dtype=np.float32)

        cell_size = round(lon_arr[1] - lon_arr[0], 3)
        x_l_c = lon_arr[0]
        x_u_c = lon_arr[-1]
        y_l_c = lat_arr[0]
        y_u_c = lat_arr[-1]

        if x_l_c > x_u_c:
            x_l_c, x_u_c = x_u_c, x_l_c

        if y_l_c > y_u_c:
            y_l_c, y_u_c = y_u_c, y_l_c

        if apply_cell_correc:
            x_l_c -= (cell_size / 2.)
            x_u_c -= (cell_size / 2.)
            y_l_c -= (cell_size / 2.)
            y_u_c -= (cell_size / 2.)

        x_coords = np.arange(x_l_c, x_u_c * 1.00000001, cell_size)
        y_coords = np.arange(y_l_c, y_u_c * 1.00000001, cell_size)

        print(feat_dict.keys())

        for cat_no in feat_dict.keys():
            print('Cat no:', cat_no)
            geom = feat_dict[cat_no].GetGeometryRef()

            extents = cat_envel_dict[cat_no]

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

            # x_cors = x_coords[x_cors_idxs]
            # y_cors = y_coords[y_cors_idxs]

            try:
                ppt_var_edit[:,
                             y_cors_idxs,
                             x_cors_idxs] = ppt_var[:,
                                                    y_cors_idxs,
                                                    x_cors_idxs].data

            except Exception:
                print('changing coord type')

                x_coords = np.sort(np.mod(x_coords - 180.0, 360.0) - 180.0)
                y_coords = np.sort(np.mod(y_coords - 180.0, 360.0) - 180.0)
                # adjustment to get all cells intersecting the polygon
                x_low = x_low - cell_size
                x_hi = x_hi + cell_size
                y_low = y_low - cell_size
                y_hi = y_hi + cell_size

                x_cors_idxs = np.where(
                    np.logical_and(
                        x_coords >= x_low,
                        x_coords <= x_hi))[0]
                y_cors_idxs = np.where(
                    np.logical_and(
                        y_coords >= y_low,
                        y_coords <= y_hi))[0]

                ppt_var_edit[:, y_cors_idxs[:, np.newaxis],
                             x_cors_idxs] = np.asarray(
                    ppt_var[:, y_cors_idxs, x_cors_idxs].data)

        break
    return ppt_var_edit

def intersect_dfs_to_nc_sources(df1_loc, df2_loc):
    dfs_1 = getFiles(df1_loc, '.csv')
    dfs_2 = getFiles(df2_loc, '.csv')

    for _, (df1, df2) in enumerate(zip(dfs_1, dfs_2)):
        row_idx1 = int(df1.split(
            sep='_')[-1].split('.')[0])
        row_idx2 = int(df2.split(
            sep='_')[-1].split('.')[0])

        if row_idx1 == row_idx2:
            in_df1 = pd.read_csv(
                df1, sep=';', header=None, index_col=0)
            in_df2 = pd.read_csv(
                df2, sep=';', header=None, index_col=0)
            print('intersecting dfs:', df1, df2)

            df2_mod = pd.DataFrame(index=in_df2.index)

            for ix, val in zip(in_df1.index, in_df1.values):

                if val[0] < 0:
                    df2_mod.loc[ix, 'Ppt_mod'] = val[0]
#                         print(val)
                if val[0] >= 0:
                    df2_mod.loc[ix, 'Ppt_mod'] = in_df2.loc[ix,
                                                            :].values[0]

            df2_mod.to_csv(os.path.join(data_dir_gpc, 'mod_final',
                                        'new_df_row_idx_%d.csv'
                                        % row_idx1), sep=';')

def fill_df_based_on_idx(data_mtx, datetime_arr, row_ix, col_ix):
    return pd.DataFrame(index=datetime_arr, data=np.array(
        data_mtx[:, row_ix, col_ix].data, dtype='float64'))


def save_obj(obj, name):
    ''' saves a dict as pickle file'''
    import pickle as pk
    with open(name + '.pkl', 'wb+') as f:
        pk.dump(obj, f, protocol=0)


def find_similarity_grid(var_to_plt, nc_file, nc_var_list, time_name,
                         lon_name, lat_name, cut_idx):

    lat, lon, in_var_data, time_idx, _ = read_nc_grid_file(nc_file,
                                                           nc_var_list,
                                                           time_name,
                                                           lon_name,
                                                           lat_name,
                                                           cut_idx)
    fill_data_month_max_arr = np.empty(
        shape=(1, lat[1:-1].shape[0], lon[1:-1].shape[0]))
    out_data_dict = {row: {col: [] for col, _ in enumerate(lon[1:-1])}
                     for row, _ in enumerate(lat[1:-1])}
    for row_idx, lat_val in enumerate(lat[1:-1]):
        print('going through row idx: ', row_idx,
              'with latitude value: ', lat_val)
        for col_idx, lon_val in enumerate(lon[1:-1]):
            print('going through col idx: ', col_idx,
                  'with Longitude value: ', lon_val)

            cnt_point = fill_df_based_on_idx(in_var_data, time_idx,
                                             row_idx, col_idx)
            pnt_a = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx, col_idx - 1)
            pnt_e = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx, col_idx + 1)
            pnt_b = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx + 1, col_idx - 1)
            pnt_c = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx + 1, col_idx)
            pnt_d = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx + 1, col_idx + 1)
            pnt_f = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx - 1, col_idx + 1)
            pnt_g = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx - 1, col_idx)
            pnt_h = fill_df_based_on_idx(in_var_data, time_idx,
                                         row_idx - 1, col_idx - 1)
            #
            if var_to_plt != 'monthly':
                cnt_point = resampleDf(
                    cnt_point, var_to_plt, 0, 1)
                pnt_a = resampleDf(
                    pnt_a, var_to_plt, 0, 1)
                pnt_e = resampleDf(
                    pnt_e, var_to_plt, 0, 1)
                pnt_b = resampleDf(
                    pnt_b, var_to_plt, 0, 1)
                pnt_c = resampleDf(
                    pnt_c, var_to_plt, 0, 1)
                pnt_d = resampleDf(
                    pnt_d, var_to_plt, 0, 1)
                pnt_f = resampleDf(
                    pnt_f, var_to_plt, 0, 1)
                pnt_g = resampleDf(
                    pnt_g, var_to_plt, 0, 1)
                pnt_h = resampleDf(
                    pnt_h, var_to_plt, 0, 1)
#
            df_max_val0, df_min_val0, df_mean0, df_dev0 = cal_cycle_and_values(
                cnt_point)
            df_max_vala, df_min_vala, df_meana, df_deva = cal_cycle_and_values(
                pnt_a)
            df_max_vale, df_min_vale, df_meane, df_deve = cal_cycle_and_values(
                pnt_e)
            df_max_valb, df_min_valb, df_meanb, df_devb = cal_cycle_and_values(
                pnt_b)
            df_max_valc, df_min_valc, df_meanc, df_devc = cal_cycle_and_values(
                pnt_c)
            df_max_vald, df_min_vald, df_meand, df_devd = cal_cycle_and_values(
                pnt_d)
            df_max_valf, df_min_valf, df_meanf, df_devf = cal_cycle_and_values(
                pnt_f)
            df_max_valg, df_min_valg, df_meang, df_devg = cal_cycle_and_values(
                pnt_g)
            df_max_valh, df_min_valh, df_meanh, df_devh = cal_cycle_and_values(
                pnt_h)

            mean_max_var_value = np.mean([df_max_vala, df_max_vale, df_max_valb,
                                          df_max_valc, df_max_vald, df_max_valf,
                                          df_max_valg, df_max_valh])
            var_0_max = np.var([df_max_val0.values, mean_max_var_value])

            mean_min_var_value = np.mean([df_min_vala, df_min_vale, df_min_valb,
                                          df_min_valc, df_min_vald, df_min_valf,
                                          df_min_valg, df_min_valh])
            var_0_min = np.var([df_min_val0.values, mean_min_var_value])

            mean_mean_var_value = np.mean([df_meana, df_meane, df_meanb,
                                           df_meanc, df_meand, df_meanf,
                                           df_meang, df_meanh])
            var_0_mean = np.var([df_mean0.values, mean_mean_var_value])

            mean_dev_var_value = np.mean([df_deva, df_deve, df_devb,
                                          df_devc, df_devd, df_devf,
                                          df_devg, df_devh])
            var_0_dev = np.var([df_dev0.values, mean_dev_var_value])

            out_data_dict[row_idx][col_idx].append([var_0_max, var_0_min,
                                                    var_0_mean, var_0_dev])
            fill_data_month_max_arr[:, row_idx, col_idx] = var_0_max
    save_obj(
        out_data_dict,
        'out_dict_gpcc_variances_all_max_min_mean_dev')

    return out_data_dict


def save_final_mtx(mtx):  # need fixing
    try:
        np.savetxt('out_mtx_arr_freq_%s_.txt'
                   % 'monthly', mtx,
                   delimiter=';', fmt='%0.2f',
                   newline='\n')
    except Exception:
        print('error while saving mtx')
        import pdb
        pdb.set_trace()
    return