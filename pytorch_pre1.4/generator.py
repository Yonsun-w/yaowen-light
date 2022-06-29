import numpy as np
import torch
from torch.utils.data import Dataset as py_Dataset
import datetime
import os
from netCDF4 import Dataset


class LightingToGird(object):
    def __init__(self, config_dict):
        mn = config_dict['GridRowColNum']

        # netCDF4 read .nc file split to Longitude, latitude
        # todo 读取经纬度信息？？ 那既然读取的是一个文件的一列 怎么设置为手动？？
        latlon_nc = Dataset(config_dict['LatlonFilePath'])
        # 取nc文件里所有的经纬度讯息
        lat_ = latlon_nc.variables['lat'][:, :]
        lon_ = latlon_nc.variables['lon'][:, :]
        latlon_nc.close()

        latlon = np.zeros(shape=[mn * mn, 2], dtype=float)
        # Longitude, latitude to grid
        # 给所有行的第一个数据 第二个数据赋值  每一个实体为(lat,lon)
        latlon[:, 0] = lat_.reshape(mn * mn)
        latlon[:, 1] = lon_.reshape(mn * mn)

        # a mistake: lat->longitude lon->latitude
        # 获取最大的经纬度讯息
        self.lat_max = np.max(latlon[:, 0], keepdims=False)
        self.lat_min = np.min(latlon[:, 0], keepdims=False)
        self.lon_max = np.max(latlon[:, 1], keepdims=False)
        self.lon_min = np.min(latlon[:, 1], keepdims=False)

        # forecast region to scan, rough estimation
        # self.lat_max = self.lat_max + (self.lat_max - self.lat_min) / mn
        # self.lat_min = self.lat_min - (self.lat_max - self.lat_min) / mn
        # self.lon_max = self.lon_max + (self.lon_max - self.lon_min) / mn
        # self.lon_min = self.lon_min - (self.lon_max - self.lon_min) / mn
        # print(lat_max, lat_min, lon_max, lon_min, (lat_max - lat_min) / 0.04, (lon_max - lon_min) / 0.04)
        # print(self.lat_max_, self.lat_min_, self.lon_max_, self.lon_min_, (self.lat_max_ - self.lat_min_) / 0.04, (self.lon_max_ - self.lon_min_) / 0.04)

        # single grid point range, rough estimation
        # todo 这是干嘛的
        self.sin_distance = (self._cal_distance(latlon[0][0], latlon[0][1], latlon[1][0], latlon[1][1]) +
                             self._cal_distance(latlon[-1][0], latlon[-1][1], latlon[-2][0], latlon[-2][1])) / 2
        self.latlon = latlon

    #Calculate distance by  latitude and longitude coordinates
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

    def getPeroid1HourGridFromFile(self, tFilePath, t1, mn):
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
        return grid.reshape([mn, mn])


class Dataloader(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict

        self.light_grid_generator = LightingToGird(config_dict)

        self.is_wrffile_exist, self.is_obsfile_exist, self.WRFFileName, self.WRFFileDeltaHour = self._checkData()

    def _checkData(self):
        # 首先获取日期
        ddt = datetime.datetime.strptime(self.config_dict['Datetime'], '%Y%m%d%H%M')
        # 将日期向后推8小时转换为世界时？ 再推6小时(貌似是为了增加鲁棒性) 得到wrf的时间
        wrf_time = ddt + datetime.timedelta(hours=-8) + datetime.timedelta(hours=(-6))
        #_getTimePeriod这里貌似链接上一步 也是增加鲁棒性
        wrf_time_hour, delta_hour = self._getTimePeriod(wrf_time)
        # delta_hour是推迟的时间？
        delta_hour += 6

        # 获取nc文件路径
        nc_filename = self.config_dict['WRFFileDir'] + wrf_time.strftime("%Y-%m-%d") + '_' + wrf_time_hour + '.wrfvar.nc'
        wrf_time2 = wrf_time + datetime.timedelta(hours=(-6))
        wrf_time_hour2, delta_hour2 = self._getTimePeriod(wrf_time2)
        nc_filename2 = self.config_dict['WRFFileDir'] + wrf_time2.strftime("%Y-%m-%d") + '_' + wrf_time_hour2 + '.wrfvar.nc'
        if os.path.exists(nc_filename):
            print('WRF file exist! (input file:{})'.format(nc_filename))
            is_wrffile_exist = True
        elif os.path.exists(nc_filename2):
            print('WRF file exist! (input file:{})'.format(nc_filename2))
            is_wrffile_exist = True
            nc_filename = nc_filename2
            delta_hour = delta_hour + 6
        else:
            is_wrffile_exist = False
            nc_filename = None
            delta_hour = None
        is_obsfile_exist = True
        for hour_plus in range(config_dict['TruthHistoryHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus - config_dict['TruthHistoryHourNum'])
            # if not os.path.exists(config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth.npy'):
            if not os.path.exists(config_dict['TruthFileDirGrid'] + dt.strftime('%Y_%m_%d') + '.txt'):
                is_obsfile_exist = False
                break
        if is_obsfile_exist:
            print('History Light file exist! (input time:{})'.format(datetime.datetime.strptime(config_dict['Datetime'], '%Y%m%d%H%M')))
        return is_wrffile_exist, is_obsfile_exist, nc_filename, delta_hour

    def _getTimePeriod(self, dt):
        time = dt.strftime("%H:%M:%S")
        hour = int(time[0:2])
        if 0 <= hour < 6:
            nchour = '00'
        elif 6 <= hour < 12:
            nchour = '06'
        elif 12 <= hour < 18:
            nchour = '12'
        elif 18 <= hour <= 23:
            nchour = '18'
        else:
            print('error')
        delta_hour = hour - int(nchour)
        return nchour, delta_hour

    # todo wrf就是nc文件吧？
    def _loadWRFData(self):  # 20200619
        ncfilepath = self.WRFFileName
        delta_hour = self.WRFFileDeltaHour
        config_dict = self.config_dict
        variables3d = ['QICE', 'QGRAUP', 'QSNOW']
        variables2d = ['W_max']
        sumVariables2d = ['RAINNC']
        param_list = ['QICE', 'QSNOW', 'QGRAUP', 'W_max', 'RAINNC']
        m = config_dict['GridRowColNum']
        n = config_dict['GridRowColNum']
        grid_list = []
        if config_dict['WRFChannelNum'] == 217:
            with Dataset(ncfilepath) as nc:
                grid = nc.variables['varone'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
                grid = np.transpose(grid, (0, 2, 3, 1))  # (12, 159, 159, n)
        else:
            with Dataset(ncfilepath) as nc:
                for s in param_list:
                    if s in variables3d:
                        temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m,
                               0:n]  # (12, 27, 159, 159)
                        temp[temp < 0] = 0
                        if config_dict['WRFChannelNum'] == 29:
                            ave_3 = np.zeros((config_dict['ForecastHourNum'], m, n, 9))
                            for i in range(9):
                                ave_3[:, :, :, i] = np.mean(temp[:, 3 * i:3 * (i + 1), :, :], axis=1)  # (12, 159, 159, 9)
                            grid_list.append(ave_3)
                        else:
                            temp = np.transpose(temp, (0, 2, 3, 1))  # (12, 159, 159, 27)
                            grid_list.append(temp)
                    elif s in variables2d:
                        if s == 'W_max':
                            tmp = nc.variables['W'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
                            tmp = np.transpose(tmp, (0, 2, 3, 1))
                            temp = np.max(tmp, axis=-1, keepdims=True)
                        else:
                            temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                        grid_list.append(temp)
                    elif s in sumVariables2d:
                        temp = nc.variables[s][delta_hour + 1:delta_hour + config_dict['ForecastHourNum'] + 1, 0:m,
                               0:n] - \
                               nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                        temp = temp[:, :, :, np.newaxis]
                        grid_list.append(temp)
            grid = np.concatenate(grid_list, axis=-1)
        grid = torch.tensor(grid, dtype=torch.float32, device=self.config_dict['Device'])
        return grid

    def _loadObsData(self):
        m = self.config_dict['GridRowColNum']
        n = self.config_dict['GridRowColNum']
        history_batch = np.zeros(shape=[self.config_dict['TruthHistoryHourNum'], m, n, 1], dtype=np.float32)
        ddt = datetime.datetime.strptime(self.config_dict['Datetime'], '%Y%m%d%H%M')
        for hour_plus in range(self.config_dict['TruthHistoryHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus - self.config_dict['TruthHistoryHourNum'])
            tFilePath = self.config_dict['TruthFileDirGrid'] + dt.strftime('%Y_%m_%d') + '.txt'
            truth_grid = self.light_grid_generator.getPeroid1HourGridFromFile(tFilePath, dt, self.config_dict['GridRowColNum'])
            history_batch[hour_plus, :, :, :] = truth_grid[:, :, np.newaxis]
        history_batch = torch.tensor(history_batch, device=self.config_dict['Device'])
        return history_batch

    def getDataStates(self):
        return self.is_wrffile_exist, self.is_obsfile_exist

    def getData(self):
        if self.is_wrffile_exist:
            wrf = self._loadWRFData()
            wrf = wrf.unsqueeze(dim=0)
        else:
            wrf = torch.tensor([-1])
        if self.is_obsfile_exist:
            obs = self._loadObsData()
            obs = obs.unsqueeze(dim=0)
        else:
            obs = torch.tensor([-1])
        return wrf, obs


if __name__ == "__main__":
    from config import read_config
    from torch.utils.data import DataLoader
    config_dict = read_config()

