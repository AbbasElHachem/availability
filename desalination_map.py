# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
import os
import time
import timeit

from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import Basemap

import geopandas as gp
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile as shp_lib

plt.ioff()
print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program
#==============================================================================
#
#==============================================================================
desalination_plants = False
water_scarcity = False
inflow_outflow = False
lorenz_curves = True
cultivation_area_ratio = False
dependency_ratio = False
groudnwater_availability = False
surface_water_availability = False
water_stress = False
seasonal_variability_idx = False
world_population = False
ice_sheets = False
global_reservoirs_per_country = False
global_reservoirs_distribution = False
global_reservoirs_capacity_per_ctr = False
water_demand_sector = False

make_plot_titles = False
#==============================================================================
#
#==============================================================================
main_dir = os.path.join(
    r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management')
assert main_dir
os.chdir(main_dir)

shp = r'ne_10m_admin_0_countries\ne_10m_admin_0_countries'
assert shp
glaciers_shp = r'glaciers_shp\10m_physical\ne_10m_glaciated_areas.shp'
assert glaciers_shp
# in_shp_reservoirs = r'global_reservoirs_dams\dams-rev01-global\GRanD_dams_v1_1'

in_shp_reservoirs = r"X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\GRanD_Version_1_3\GRanD_dams_v1_3"
assert in_shp_reservoirs

fontsize = 16
colorbar_label_fontsize = 16
#==============================================================================
#
#==============================================================================


def do_colorbar(fig, cmap, bins, label, extent):
    norm = mcolors.BoundaryNorm(bins, cmap.N)
    ax_legend = fig.add_axes([0.1725, 0.02525, 0.68, 0.0225], zorder=3)
    cb = mpl.colorbar.ColorbarBase(ax_legend, extend=extent, ticks=bins,
                                   boundaries=bins, norm=norm, cmap=cmap,
                                   orientation='horizontal')
    cb.ax.tick_params(labelsize=colorbar_label_fontsize)
    cb.set_label(label, fontsize=colorbar_label_fontsize)
    return cb
#==============================================================================
#
#==============================================================================


def savefig(fig_name):
    return plt.savefig('%s.png' % fig_name, frameon=True,
                       papertype='a4', bbox_inches='tight', pad_inches=.2)


#==============================================================================
#
#==============================================================================
if desalination_plants:
    location_plants = r'desalination_plants_location.csv'
    data = pd.read_csv(location_plants, sep=';', index_col=2)
    title = 'Location of main Desalination plants worldwide'

    my_dpi = 100
    plt.figure(figsize=(20, 12), dpi=my_dpi)
    # Make the background map
    m = Basemap(llcrnrlon=-180, llcrnrlat=-66, urcrnrlon=180, urcrnrlat=90)
    m.drawcoastlines(color='k', linewidth=0.4)

    m.plot(data['lat'], data['lon'], linestyle='none',
           marker="o", markersize=5, alpha=1, c="r",
           markeredgecolor="black", markeredgewidth=1)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    if make_plot_titles:
        plt.title(title, fontsize=12)
    savefig('Desalination2')
#==============================================================================
#
#==============================================================================
if water_scarcity:
    year = '2014'

    title = 'Water availabilty per inhabitant in {} (m3/Capita)'.format(year)
    data_availability = r'countries_isocode_water_availability_m3_per_inhabitant.csv'

    in_df = pd.read_csv(data_availability, sep=',', index_col=1, header=0)
    in_df = in_df['2013-2017']
    in_df[in_df.values > 2000] = 2000

    new_df = pd.DataFrame(index=in_df.index, data=in_df.values)

#     clrs = ['darkred', 'red', 'indianred', 'darkorange', 'gold',
#             'greenyellow', 'g', 'darkgreen', 'b', 'darkblue']
    clrs = ['darkred', 'darkorange', 'darkgreen', 'darkblue']

    cmap = mcolors.ListedColormap(clrs)
#     bounds = {0: [0, 50], 1: [50, 100], 2: [100., 250.], 3: [250., 500],
#               4: [500, 750],
#               5: [750, 1000], 6: [1000, 1250], 7: [1250, 1500],
#               8: [1500, 1750], 9: [1750, 2000.01]}
    bounds = {0: [0, 500], 1: [500, 1000], 2: [1000., 1700.]}
    for ix, val in zip(new_df[0].index,
                       new_df[0].values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                new_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            new_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[2][1]:
            new_df.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            'Water availability per capita in m3 for year {}'.format(year),
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):
        iso3 = info['ADM0_A3']

        if iso3 not in new_df.index:
            color = '#dddddd'
        else:

            try:
                color = new_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
#     bins = [0., 50, 100, 250, 500., 750, 1000, 1250, 1500, 1750, 2000, 2000.01]
    bins = [0., 500, 1000, 1700, 1700.1]
    cb = do_colorbar(fig, cmap, bins, '(m3 / capita / year)', 'max')

    savefig('water_scarcity')

#==============================================================================
#
#==============================================================================
if inflow_outflow:
    year = '2014'

    data_availability = r'inflow_outflow_per_countrycode.csv'
#     country_area = r'countries_population_area_iso_code.csv'
    country_area = r'coutries_area_km2.csv'

    in_df = pd.read_csv(data_availability, sep=',', index_col=0)
    in_df.dropna(inplace=True)
    values = in_df['outflow 10^9 m3/year'].values - \
        in_df['inflow 10^9 m3/year'].values

    in_df['needed_values'] = values

    in_df_area_ctr = pd.read_csv(country_area, sep=',',
                                 index_col=0, usecols=[1, 2])
    new_idx = in_df.index.intersection(in_df_area_ctr.index)
    vls_per_mm_per_area = (in_df.loc[new_idx, 'needed_values'] /
                           in_df_area_ctr.loc[new_idx, '2014']) * 1e6
    new_df = pd.DataFrame(index=new_idx, data=vls_per_mm_per_area)

    clrs = ['fuchsia', 'm', 'darkred', 'red', 'indianred',
            'darkorange', 'gold', 'yellowgreen',
            'g', 'darkgreen', 'lightblue', 'royalblue', 'b', 'darkblue']

    cmap = mcolors.ListedColormap(clrs)

    bounds = {0: [-2000, -1500], 1: [-1500., -1000],
              2: [-1000., -750.], 3: [-750, -500],
              4: [-500, -250], 5: [-250, -100.1], 6: [-100, -0.],
              7: [0, 100], 8: [100.1, 250],
              9: [250, 500], 10: [500, 750], 11: [750, 1000],
              12: [1000, 1500], 13: [1500, 2000]}

    for ix, val in zip(new_df[0].index,
                       new_df[0].values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                new_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            new_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[13][1]:
            new_df.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 12))

    ax = fig.add_subplot(111, frame_on=True)
    if make_plot_titles:
        fig.suptitle('Water Outflow minus Inflow for year {} (mm/y)'.format(year),
                     fontsize=14, y=.82)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=90)
    m.drawmapboundary(fill_color='w', linewidth=0)
    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for i, (info, shape) in enumerate(zip(m.units_info, m.units)):
        iso3 = info['ADM0_A3']
        if iso3 not in new_df.index:
            color = '#dddddd'
        else:
            try:
                color = new_df.loc[iso3, 'colors']
            except Exception:
                print('changing color')
                color = '#dddddd'
        if not isinstance(color, str):
            color = '#dddddd'
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [-2000, -1500, -1000, -750, -500, -250, -100,
            0, 100.1, 250, 500, 750, 1000, 1500, 2000]

    do_colorbar(fig, cmap, bins, 'mm / year', 'neither')
    savefig('surface_water_outflow_minus_inflow_new')
#==============================================================================
#
#==============================================================================
if lorenz_curves:

    year = '2014'
    cols = ['Country Name', 'Country Code', year]
    title = 'Lorenz Curves: Worldwide Water availability compared to Global Population (year 2014)'

    data_availability = r'countries_isocode_water_availability_m3_per_inhabitant.csv'
    population_df = r'population_per_country.csv'

    in_df = pd.read_csv(data_availability, sep=',', index_col=1)
    in_df.dropna(inplace=True)
    in_df['Water Availability'] = in_df['2013-2017'].values

    in_df_pop = pd.read_csv(population_df, sep=',', index_col=0)
    new_idx = in_df.index.intersection(in_df_pop.index)

    new_df = pd.DataFrame(index=new_idx,
                          data=in_df.loc[new_idx, 'Water Availability'])
    new_df['population'] = in_df_pop.loc[new_idx, '2014'] / 10e5
    new_df.sort_values('Water Availability', ascending=True, inplace=True,
                       kind='quick')

    new_df['pop*water'] = (new_df['Water Availability'].values *
                           new_df['population'].values)
    sum_pop = new_df['population'].values.sum()
    sum_pop_wat = new_df['pop*water'].values.sum()
    new_df['ratio_pop'] = np.cumsum(new_df['population'] / sum_pop)
    new_df['ratio_pop*water'] = np.cumsum(new_df['pop*water'] / sum_pop_wat)

    d = pd.DataFrame(
        index=new_df['ratio_pop'].values,
        data=new_df['ratio_pop*water'].values)
# #     d_slope = d.copy()
#
# #     d_slope['population'] = d_slope.index
# #     d_slope = d_slope.diff()
# #     d_slope['slope'] = d_slope[0].values / d_slope['population'].values
# #     pop_diff = [j - i for i, j in zip(d_slope['population'].values[:-1],
# #                                       d_slope['population'].values[1:])]
# #     water_diff = [j - i for i, j in zip(d_slope[0].values[:-1],
# #                                         d_slope[0].values[1:])]
# #     df_slope = pd.DataFrame(index=pop_diff, data=water_diff)
#
# #     df_slope['population'] = df_slope.index
    plt.figure(figsize=(22, 12))
    # this is another inset axes over the main axes

    plt.plot(d.index, d.values, c='b', label='Observed Values',
             linewidth=2)

    plt.plot([0., 1.], [0., 1.], c='r', alpha=0.95,
             linestyle=':', label='Line of Perfect Equality')

    plt.legend(loc=0, fontsize=fontsize)
    if make_plot_titles:
        plt.title(title, fontsize=fontsize)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.yticks(np.arange(0.0, 1.01, 0.2), fontsize=fontsize)
    plt.xticks(np.arange(0.0, 1.01, 0.2), fontsize=fontsize)
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlabel("Fraction of Global Population (capita)", fontsize=fontsize + 3,
               labelpad=14)
    plt.ylabel(
        r"Fraction of Global Water Availability (m$^3$/capita)",
        fontsize=fontsize + 3, labelpad=14)
#     a = plt.axes([0.185, 0.65, .2, .2], facecolor='w')
#     plt.plot(d_slope.index, d_slope['slope'].values, c='g',
#              label='Slope', linewidth=1)

#     plt.title('Slope')
#     plt.xlabel("Fraction of Global Population (capita)", fontsize=10)
#     plt.ylabel(
#         r"$\frac{Water \ availability \ difference}{Population \ difference}$",
#         fontsize=14)

    savefig('Lorenz_curves_pop_availability')
#==============================================================================
#
#==============================================================================

if cultivation_area_ratio:
    data_availability = r'% of total country area cultivated.csv'
    in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=None)
    in_df.dropna(how='all', inplace=True)

    clrs = ['darkred', 'r', 'darkorange', 'gold',
            'limegreen', 'darkgreen', 'b', 'darkblue']

    cmap = mcolors.ListedColormap(clrs)

    bounds = {0: [0, 5], 1: [5, 10.], 2: [10., 20], 3: [20, 30],
              4: [30, 40], 5: [40, 50], 6: [50, 60.1]}
    # 7: [70, 80], 8: [80, 90], 9: [90, 100]}

    for ix, val in zip(in_df.index, in_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[6][1]:
            in_df.loc[ix, 'colors'] = clrs[-1]
    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(('Country Cultivated Area as (%) of Total Area'),
                     fontsize=12, y=.92)

    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)
    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in in_df.index:
            color = '#dddddd'
        else:
            try:
                color = in_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [0, 5, 10, 20, 30, 40, 50, 60, 60.01]
    do_colorbar(fig, cmap, bins, 'Cultivated Area (%)', 'max')
    savefig('Cultivated_Area')

#==============================================================================
#
#==============================================================================
if dependency_ratio:

    data_availability = r'dependancy_ratio.csv'
    in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=0)
    in_df.dropna(how='all', inplace=True)
    clrs = ['darkblue', 'b', 'c', 'lightblue',
            'darkgreen', 'g', 'limegreen',
            'greenyellow', 'gold', 'darkorange',
            'pink', 'violet', 'm',
            'indianred', 'red', 'darkred']

    cmap = mcolors.ListedColormap(clrs)
    bounds = {0: [0, 5], 1: [5., 10.], 2: [10., 15],
              3: [15, 20], 4: [20, 25], 5: [25, 30], 6: [30, 35], 7: [35, 40],
              8: [40, 45], 9: [45, 50], 10: [50, 55], 11: [55, 60],
              12: [60, 65], 13: [65, 70], 14: [70, 80], 15: [80, 100]}

    for ix, val in zip(in_df.index,
                       in_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[15][1]:
            in_df.loc[ix, 'colors'] = clrs[-1]

    fig = plt.figure(figsize=(22, 8), dpi=100)
    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            ('Indicator expressing the percent of total renewable water'
             ' resources originating outside the country.'),
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in in_df.index:
            color = '#dddddd'
        else:
            try:
                color = in_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
            50, 55, 60, 65, 70.0, 80.0, 100]
    do_colorbar(fig, cmap, bins, 'Dependancy Ratio (%)', 'neither')
    savefig('dependancy_ratio')

#==============================================================================
#
#==============================================================================

if groudnwater_availability:

    year = '2014'

    data_availability = r'total_reneable_groundwater.csv'

    in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=0)
    in_df.dropna(how='all', inplace=True)
#     in_df[in_df > 500] = 500
    country_area = r'coutries_area_km2.csv'
    in_df_area_ctr = pd.read_csv(country_area, sep=',', index_col=0,
                                 header=0, encoding='latin1')
    in_df_area_ctr.dropna(how='all', inplace=True)
    new_idx = in_df_area_ctr.index.intersection(in_df.index)

    d = in_df.loc[new_idx, 'Total renewable groundwater 10e9m3/year']
    d2 = in_df_area_ctr.loc[new_idx, '2014']

    new_df = pd.DataFrame(index=new_idx, data=(d / d2) * 1e7)

    clrs = ['darkred', 'red', 'orange', 'y', 'limegreen',
            'green', 'darkgreen', 'aqua', 'royalblue', 'b', 'darkblue']
    cmap = mcolors.ListedColormap(clrs)

    bounds = {0: [0, 25], 1: [25, 50], 2: [50, 100],
              3: [100, 200], 4: [200, 300], 5: [300, 400],
              6: [400, 500], 7: [500, 600], 8: [600, 700], 9: [700, 800]}

    for ix, val in zip(new_df.index,
                       new_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                new_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            new_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[9][1]:
            new_df.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(('Total Renewable Groundwater mm/year.'),
                     fontsize=12, y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in new_df.index:
            color = '#dddddd'
        else:

            try:
                color = new_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'
        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [0, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 800.01]
    do_colorbar(fig, cmap, bins, 'mm / y', 'max')
    savefig('total_renewable_groundwater_mm_per_y')

#==============================================================================
#
#==============================================================================
if surface_water_availability:

    year = '2014'

    data_availability = r'renewable_surface_water.csv'
#     data_availability = r'Total_renewable_water_resources.csv'

    in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=0)
    in_df.dropna(how='all', inplace=True)

    country_area = r'coutries_area_km2.csv'
    in_df_area_ctr = pd.read_csv(country_area, sep=',', index_col=0, header=0,
                                 encoding='latin1')
    in_df_area_ctr.dropna(how='all', inplace=True)

    new_idx = in_df_area_ctr.index.intersection(in_df.index)

    d = in_df.loc[new_idx, 'Total renewable surface water (10^9 m3/year)']
#     d = in_df.loc[new_idx, 'Total renewable water resources (10^9 m3/year)']
    d2 = in_df_area_ctr.loc[new_idx, '2014']
    new_df = pd.DataFrame(index=new_idx, data=(d / d2) * 1e7)

    clrs = ['darkred', 'red', 'orange', 'y', 'limegreen',
            'green', 'darkgreen', 'aqua', 'royalblue', 'b', 'darkblue']
    cmap = mcolors.ListedColormap(clrs)

    bounds = {0: [0, 25], 1: [25, 50], 2: [50, 100],
              3: [100, 250], 4: [250, 500], 5: [500, 750],
              6: [750, 1000], 7: [1000, 1250], 8: [1250, 1500],
              9: [1500, 2000]}

    for ix, val in zip(new_df.index,
                       new_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                new_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            new_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[9][1]:
            new_df.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            ('Total Renewable Surface Water defined as:'
             ' the sum of internal and total external'
             ' renewable surface water resources (mm/year).'),
            fontsize=12,
            y=.92)
#     fig.suptitle(
#         ('Total Renewable Water Resources: The sum of internal renewable water resources and external renewable water resources \n'
#          ' It corresponds to the maximum theoretical yearly amount of water available for a country at a given moment (mm/year).'),
#         fontsize=12,
#         y=.95)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in new_df.index:
            color = '#dddddd'
        else:

            try:
                color = new_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
    bins = [0, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 2000, 2000.01]
    do_colorbar(fig, cmap, bins, 'mm / y', 'max')

    savefig('total_renewable_surfacewater_mm_per_y')
#     savefig('total_renewable_total_water_mm_per_y')
#==============================================================================
#
#==============================================================================
if water_stress:
    #     data_availability = r'water_stress.csv'
    data_availability = r'water_stress_world_bank.csv'
#     data_availability = r'Freshwater withdrawal as % of total renewable water resources (%).csv'
#     in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=0)
    in_df = pd.read_csv(data_availability, sep='\t', index_col=0, header=None)
    in_df.dropna(how='all', inplace=True)
#     raise Exception
    clrs = ['darkblue', 'b', 'lightblue', 'darkgreen', 'g', 'greenyellow', 'gold',
            'darkorange', 'pink', 'indianred', 'red', 'darkred']

    cmap = mcolors.ListedColormap(clrs)

    bounds = {0: [0, 10], 1: [10., 20.], 2: [20., 30],
              3: [30, 40],
              4: [40, 50], 5: [50, 60], 6: [60, 70],
              7: [70, 80], 8: [80, 90], 9: [90, 100],
              10: [100, 500], 11: [500, 1000]}

    for ix, val in zip(in_df.index,
                       in_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[11][1]:
            in_df.loc[ix, 'colors'] = clrs[-1]

    fig = plt.figure(figsize=(22, 8), dpi=600)
    ax = fig.add_subplot(111, frame_on=False)
#     fig.suptitle(
#         ('Total freshwater withdrawn in a given year, expressed in percentage of the total renewable water resources. This parameter is an indication of the pressure on the renewable water resources. '),
#         fontsize=12,
#         y=.92)
    if make_plot_titles:
        fig.suptitle(
            ('Water Stress: an indicator expressing'
             ' the ratio of total withdrawals to total renewable supply. '),
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in in_df.index:
            color = '#dddddd'
        else:

            try:
                color = in_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000]
    do_colorbar(fig, cmap, bins, 'Water Stress (%)', 'neither')

    savefig('Water Stress World Bank')

#=============================================================================
#
#=============================================================================
if seasonal_variability_idx:

    data_availability = r'seasonal_variability_idx.csv'
    in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=0)
    in_df.dropna(how='all', inplace=True)
    clrs = ['darkblue', 'b', 'c', 'lightblue', 'greenyellow',
            'darkgreen', 'orange',
            'm', 'red', 'darkred']

    cmap = mcolors.ListedColormap(clrs)
    bounds = {0: [0, 0.5], 1: [.5, 1], 2: [1, 1.5], 3: [1.5, 2.0], 4: [2.0, 2.5],
              5: [2.5, 3.0], 6: [3.0, 3.5], 7: [3.5, 4.0], 8: [4., 4.5], 9: [4.5, 5.0]}

    for ix, val in zip(in_df.index,
                       in_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[9][1]:
            in_df.loc[ix, 'colors'] = clrs[-1]

    fig = plt.figure(figsize=(22, 8), dpi=600)
    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            ('Normalized indicator of the variation in water supply between months of the year.'),
            fontsize=fontsize,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in in_df.index:
            color = '#dddddd'
        else:
            try:
                color = in_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
    bins = [0, .5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    do_colorbar(
        fig,
        cmap,
        bins,
        'Seasonal Variability Index (Very low to Very High)',
        'neither')
    savefig('seasonality_idx')
#==============================================================================
#
#==============================================================================
if ice_sheets:

    fig = plt.figure(figsize=(22, 12), dpi=100)

    ax = fig.add_subplot(111, frame_on=False)
    sf = shp_lib.Reader(glaciers_shp)
    glaciers = gp.GeoDataFrame.from_file(glaciers_shp)
#     glaciers.plot(color='darkblue')
    m = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)
    m.drawmapboundary(fill_color='w', linewidth=0.2)
    m.fillcontinents(color='whitesmoke', alpha=0.3)
    m.drawcoastlines(color='k', linewidth=0.4)
#     m.bluemarble()
    m.drawmeridians(range(-180, 180, 30), linewidth=0.15, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-90, 91, 30), linewidth=0.15, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)

    BLUE = 'b'

    for poly in glaciers.geometry:

        ax.add_patch(PolygonPatch(poly, fc=BLUE, ec=BLUE, alpha=1,
                                  zorder=2))
    if make_plot_titles:
        plt.title('Major Glaciers and Ice Sheefts Worldwide', fontsize=12)
    savefig('ice_sheets2')
#==============================================================================
#
#==============================================================================
if world_population:
    year = '2015'
    data_availability = r'world_population.csv'

    in_df = pd.read_csv(data_availability, sep=',', index_col=0, header=0)
    in_df['Population in Millions'] = (
        in_df['Total population (1000 inhab)'].values * 1000) / 1e6

    clrs = ['darkblue', 'darkgreen', 'lightgreen',
            'gold', 'darkorange', 'salmon', 'red', 'darkred']

    cmap = mcolors.ListedColormap(clrs)
    bounds = {0: [0, 10], 1: [10., 25.], 2: [25., 50],
              3: [50, 100],
              4: [100, 150], 5: [150, 300], 6: [300, 500],
              7: [500, 1000]}

    for ix, val in zip(in_df.index,
                       in_df['Population in Millions'].values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[7][1]:
            in_df.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            'Population per country in millions in year {} '.format(year),
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):
        iso3 = info['ADMIN']
        if iso3 not in in_df.index:
            color = '#dddddd'
        else:

            try:
                color = in_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [0., 10, 25, 50., 100, 150, 300, 500, 1000]
#     bins = [0., 500, 1000, 1700, 1700.1]
    cb = do_colorbar(fig, cmap, bins, 'Population in Million', 'max')

    savefig('world_population')
#==============================================================================
#
#==============================================================================
if global_reservoirs_per_country:
    my_dpi = 300
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)

    fig.suptitle(
        'Global distribution of main reservoirs (Capacity > $10Mm^{3})$',
        fontsize=12,
        y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    # m.drawrivers(linewidth=0.1, color='navy')

    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    countries = []
    for info_ctr, shape_ctr in zip(m.units_info, m.units):

        country = info_ctr['ADMIN']
        if country not in countries:
            countries.append(country)

    res_per_ctr = {ctr: [] for ctr in set(countries)}

    m.readshapefile(in_shp_reservoirs, 'units', color='#444444', linewidth=.2)
    s0 = 0.5
    for info_res, shape_res in zip(m.units_info, m.units):
        cap_mcm = info_res['CAP_MCM']
        ctr_res = info_res['COUNTRY']
#         if ctr_res == 'Germany':
#             break
        if ctr_res == 'United States':
            ctr_res = 'United States of America'
#         try:
#
        if ctr_res in res_per_ctr.keys():
            res_per_ctr[ctr_res].append(1)
#             else:
#                 res_per_ctr[ctr_res].append(0)
#         except Exception:
#             break
    res_per_ctr_sum = {k: sum(res_per_ctr[k]) for k in res_per_ctr.keys()}
    df_res = pd.DataFrame.from_dict(data=res_per_ctr_sum, orient='index',
                                    columns=['Nbr of Reservoirs'])
    clrs = ['darkred', 'red', 'salmon', 'darkorange',
            'gold', 'lightgreen', 'darkgreen', 'darkblue']

    cmap = mcolors.ListedColormap(clrs)
    bounds = {0: [0, 10], 1: [10., 25.], 2: [25., 50],
              3: [50, 100],
              4: [100, 150], 5: [150, 300], 6: [300, 500],
              7: [500, 1000]}

    for ix, val in zip(df_res.index,
                       df_res['Nbr of Reservoirs'].values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                df_res.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            df_res.loc[ix, 'colors'] = clrs[0]
        if val > bounds[7][1]:
            df_res.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            'Population per country in millions in year {} '.format(year),
            fontsize=12,
            y=.92)

    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):
        iso3 = info['ADMIN']
        if iso3 not in df_res.index:
            color = '#dddddd'
        else:

            try:
                color = df_res.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [0., 10, 25, 50., 100, 150, 300, 500, 1000]
#     bins = [0., 500, 1000, 1700, 1700.1]
    cb = do_colorbar(
        fig,
        cmap,
        bins,
        'Global distribution (by country) of large reservoirs in GRanD database',
        'max')

    savefig('world_reservoirs')

#==============================================================================
#
#==============================================================================

if global_reservoirs_distribution:
    my_dpi = 300

    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            'Global distribution of main reservoirs',
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)
    m.drawcoastlines(color='k', linewidth=0.3)
    m.drawcountries(color='k', linewidth=0.25)
    m.drawmeridians(range(-180, 180, 30), linewidth=0.15, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.drawparallels(range(-65, 90, 30), linewidth=0.15, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    # m.drawrivers(linewidth=0.1, color='navy')
    m.readshapefile(in_shp_reservoirs, 'units', color='#444444', linewidth=.2)

    capacities = []
    scatters_lst = []
    for i, (info_res, shape_res) in enumerate(zip(m.units_info, m.units)):
        cap_mcm = info_res['CAP_MCM']

        if cap_mcm < 0:
            cap_mcm = 0

        capacities.append(cap_mcm)

#         if (cap_mcm / 1000) > 50:
#             c = 'g'
        if (cap_mcm / 1000) > 100:
            c = 'r'
        else:
            c = 'b'
#         if 0 < cap_mcm <= 10:
#             label = '<10'
# #                         s = s0
# #             c = 'darkblue'
#         if 10 < cap_mcm <= 50:
#             label = '10-50'
#             #             cap_mcm = 50
#             # #             s = 2 * s0
# #             c = 'blue'
#         if 50 < cap_mcm <= 100:
#             label = '50-100'
#             #             cap_mcm = 100
#             # #             s = 3 * s0
# #             c = 'darkgreen'
#         if 100 < cap_mcm <= 200:
#             label = '100-200'
#             #             cap_mcm = 200
#             # #             s = 4 * s0
# #             c = 'green'
#         if 200 < cap_mcm <= 300:
#             label = '200-300'
#             #             cap_mcm = 300
#             #             s = 5 * s0
# #             c = 'pink'
# #         if 300 < cap_mcm <= 400:
#             #             s = 6 * s0
# #             c = 'm'
#         if 300 < cap_mcm <= 500:
#             label = '300-500'
#             #             s = 7 * s0
# #             c = 'salmon'
#         if 500 > cap_mcm:
#             label = '>500'
        #             s = 8 * s0
#             c = 'darkred'

#         print(cap_mcm)
        ax.scatter(shape_res[0], shape_res[1],
                   s=cap_mcm / 10000,
                   c=c, marker='d', alpha=0.75)

    l1 = ax.scatter([], [], s=0.1,  c='b', marker='d')
    l2 = ax.scatter([], [], s=1,  c='b', marker='d')
    l3 = ax.scatter([], [], s=3, c='b', marker='d')
    l4 = ax.scatter([], [], s=10,  c='g', marker='d')
    l5 = ax.scatter([], [], s=20, c='r', marker='d')


#     labels = list(np.unique(df_.cat.values))
    labels = ['<1', '1-5', '5-10', '10-100', '>100']
    leg = ax.legend([l1, l2, l3, l4, l5], labels, ncol=10,
                    frameon=True, fontsize=12,
                    handlelength=1, loc=8, borderpad=.5,
                    handletextpad=0.5,
                    title=' Reservoir Storage Capacity $Km^{3}$',
                    scatterpoints=1)
    ax.set_xlabel('Longitude', fontsize=12)
#     ax.legend(loc=0)
    ax.xaxis.set_label_coords(.5, -0.065)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.yaxis.set_label_coords(-0.045, 0.5)

    savefig('reservoirs2')
#==============================================================================
#
#==============================================================================
if global_reservoirs_capacity_per_ctr:

    my_dpi = 300
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            'Global distribution of main reservoirs (Capacity > $10Mm^{3})$',
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    # m.drawrivers(linewidth=0.1, color='navy')

    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    countries = []
    for info_ctr, shape_ctr in zip(m.units_info, m.units):

        country = info_ctr['ADMIN']
        if country not in countries:
            countries.append(country)

    res_per_ctr = {ctr: [] for ctr in set(countries)}

    m.readshapefile(in_shp_reservoirs, 'units', color='#444444', linewidth=.2)
    s0 = 0.5
    for info_res, shape_res in zip(m.units_info, m.units):
        cap_mcm = info_res['CAP_MCM']
        ctr_res = info_res['COUNTRY']
#         if ctr_res == 'Germany':
#             break
        if ctr_res == 'United States':
            ctr_res = 'United States of America'
#         try:
#
        if ctr_res in res_per_ctr.keys():
            res_per_ctr[ctr_res].append(cap_mcm)
#             else:
#                 res_per_ctr[ctr_res].append(0)
#         except Exception:
#             break
    res_per_ctr_sum = {k: sum(res_per_ctr[k]) for k in res_per_ctr.keys()}
    df_res = pd.DataFrame.from_dict(data=res_per_ctr_sum, orient='index',
                                    columns=['Nbr of Reservoirs'])
    clrs = ['darkred', 'red', 'salmon', 'darkorange',
            'gold', 'm', 'pink', 'lightgreen', 'darkgreen', 'lightblue', 'darkblue']

    cmap = mcolors.ListedColormap(clrs)
    bounds = {0: [0, 100], 1: [100., 250.], 2: [250., 500],
              3: [500, 1000],
              4: [1000, 1500], 5: [1500, 3000], 6: [3000, 5000],
              7: [5000, 10000], 8: [10000, 20000], 9: [20000, 40000]}

    for ix, val in zip(df_res.index,
                       df_res['Nbr of Reservoirs'].values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                df_res.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            df_res.loc[ix, 'colors'] = clrs[0]
        if val > bounds[9][1]:
            df_res.loc[ix, 'colors'] = clrs[-1]

    my_dpi = 100
    fig = plt.figure(figsize=(22, 8), dpi=my_dpi)

    ax = fig.add_subplot(111, frame_on=False)

#     fig.suptitle(
#         'Population per country in millions in year {} '.format(year),
#         fontsize=12,
#         y=.92)

    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1,
                    fontsize=fontsize)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):
        iso3 = info['ADMIN']
        if iso3 not in df_res.index:
            color = '#dddddd'
        else:

            try:
                color = df_res.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)

    bins = [0., 100, 250, 500., 1000, 1500, 3000, 5000, 10000,
            20000, 40000, 40000.01]
#     bins = [0., 500, 1000, 1700, 1700.1]
    cb = do_colorbar(fig, cmap,
                     bins,
                     'Estimated Storage Capacity per Country (Millions of m3)',
                     'max')

    savefig('capacity')

#==============================================================================
#
#==============================================================================

if water_demand_sector:
    in_df = pd.read_csv(r'X:\hiwi\ElHachem\Prof_Bardossy\Handook_Water_resources_Management\agriculture_industry_municipal_water_demand_original_sorted.csv', index_col=0,
                        usecols=[0, 1])
    in_df.fillna(0, inplace=True)
    in_df[in_df.values < 60] = 59
    print(in_df)
#     clrs = ['darkblue', 'b', 'c', 'lightblue',
#             'olive', 'darkgreen', 'g', 'limegreen',
#             'greenyellow', 'gold', 'darkorange',
#             'pink', 'violet', 'm',
#             'indianred', 'red', 'darkred'][::-1]
    clrs = ['darkblue', 'b', 'c', 'lightblue',
            'olive'][::-1]

    cmap = mcolors.ListedColormap(clrs)
#     bounds = {0: [0, 5], 1: [5., 10.], 2: [10., 15],
#               3: [15, 20], 4: [20, 25], 5: [25, 30], 6: [30, 35], 7: [35, 40],
#               8: [40, 45], 9: [45, 50], 10: [50, 55], 11: [55, 60],
#               12: [60, 65], 13: [65, 70], 14: [70, 80], 15: [80, 90],
#               16: [90, 100]}
    bounds = {0: [0, 59], 1: [60, 70], 2: [70, 80], 3: [80, 90],
              4: [90, 100]}
    for ix, val in zip(in_df.index,
                       in_df.values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[4][1]:
            in_df.loc[ix, 'colors'] = clrs[-1]

    fig = plt.figure(figsize=(22, 8), dpi=100)
    ax = fig.add_subplot(111, frame_on=False)
    if make_plot_titles:
        fig.suptitle(
            ('Agriculture water withdrawal as % of total water withdrawal (%)'),
            fontsize=12,
            y=.92)
    m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=85)
    m.drawmapboundary(fill_color='w', linewidth=0)

    m.drawmeridians(range(-180, 180, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.drawparallels(range(-65, 90, 30), linewidth=0.25, dashes=[4, 2],
                    labels=[1, 0, 0, 1], color='gray', zorder=1)
    m.readshapefile(shp, 'units', color='#444444', linewidth=.2)
    for info, shape in zip(m.units_info, m.units):

        iso3 = info['ADMIN']
        if iso3 not in in_df.index:
            color = '#dddddd'
        else:
            try:
                color = in_df.loc[iso3, 'colors']
            except Exception:
                color = 'darkblue'

        patches = [Polygon(np.array(shape), True)]
        pc = PatchCollection(patches)
        pc.set_facecolor(color)
        ax.add_collection(pc)
        bins = [0, 60, 70.0, 80.0, 90, 100]
#     bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
#             50, 55, 60, 65, 70.0, 80.0, 90, 100]
    do_colorbar(fig, cmap, bins,
                'Agriculture water withdrawal as % of total water withdrawal (%)', 'neither')
    savefig('Agriculture_ratio2')

STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
