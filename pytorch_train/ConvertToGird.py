# -*- coding: utf-8 -*-
import numpy as np
import datetime
import math
import os
from netCDF4 import Dataset

class LightingToGird(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        mn = config_dict['GridRowColNum']
        latlon_nc = Dataset('latlon.nc')
        lat_ = latlon_nc.variables['lat'][:, :]
        lon_ = latlon_nc.variables['lon'][:, :]
        latlon_nc.close()

        latlon = np.zeros(shape=[mn * mn, 2], dtype=float)
        latlon[:, 0] = lat_.reshape(mn * mn)
        latlon[:, 1] = lon_.reshape(mn * mn)

        self.lat_max = np.max(latlon[:, 0], keepdims=False)
        self.lat_min = np.min(latlon[:, 0], keepdims=False)
        self.lon_max = np.max(latlon[:, 1], keepdims=False)
        self.lon_min = np.min(latlon[:, 1], keepdims=False)

        # forecast region to scan, rough estimation
        # self.lat_max = self.lat_max + (self.lat_max - self.lat_min) / mn
        # self.lat_min = self.lat_min - (self.lat_max - self.lat_min) / mn
        # self.lon_max = self.lon_max + (self.lon_max - self.lon_min) / mn
        # self.lon_min = self.lon_min - (self.lon_max - self.lon_min) / mn
        # single grid point range, rough estimation
        self.sin_distance = (self._cal_distance(latlon[0][0], latlon[0][1], latlon[1][0], latlon[1][1]) +
                             self._cal_distance(latlon[-1][0], latlon[-1][1], latlon[-2][0], latlon[-2][1])) / 2
        self.latlon = latlon

    def _cal_distance(self, la1, lo1, la2, lo2):
        ER = 6370.8
        radLat1 = (np.pi / 180.0) * la1
        radLat2 = (np.pi / 180.0) * la2
        radLng1 = (np.pi / 180.0) * lo1
        radLng2 = (np.pi / 180.0) * lo2
        d = 2.0 * np.arcsin(np.sqrt(
            np.power(np.sin((radLat1 - radLat2) / 2.0), 2) + np.cos(radLat1) * np.cos(radLat2) * np.power(
                np.sin((radLng1 - radLng2) / 2.0), 2))) * ER
        return d

    def _lalo_to_grid_new(self, la, lo):
        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max:
            return -1
        d = self._cal_distance(la, lo, self.latlon[:, 0], self.latlon[:, 1])
        if np.min(d) > self.sin_distance * np.sqrt(2):
            return -1
        idx = np.argmin(d)
        return idx

    def getPeroid1HourGridFromFile(self, tFilePath, t1):
        mn = self.config_dict['GridRowColNum']
        grid = np.zeros(mn * mn, dtype=int)
        t2 = t1 + datetime.timedelta(hours=1)
        with open(tFilePath, 'r', encoding='GBK') as tfile:
            for line in tfile:
                linedata = line.split()
                temp_date = linedata[1]
                temp_time = linedata[2]
                temp_dt = datetime.datetime.strptime(temp_date + ' ' + temp_time[0:8], "%Y-%m-%d %H:%M:%S")
                if not t1 <= temp_dt <= t2:
                    continue
                linedata[3] = linedata[3].lstrip('纬度=')
                linedata[4] = linedata[4].lstrip('经度=')
                la = float(linedata[3])
                lo = float(linedata[4])
                idx = self._lalo_to_grid_new(la, lo)
                if idx == -1:
                    continue
                grid[idx] += 1
        return grid





