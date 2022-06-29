import numpy as np
import datetime
import struct
import os
from netCDF4 import Dataset

class Writefile(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        mn = config_dict['GridRowColNum']

        latlon_nc = Dataset(config_dict['LatlonFilePath'])

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
        self.latlon = latlon.reshape((config_dict['GridRowColNum'], config_dict['GridRowColNum'], 2))

        gap_x = 0.04
        gap_y = 0.04
        self.grid_transformer_near4, row, col = self._getLatlonTransformer(gap_x, gap_y)

        self.writeinfo_dict = {}
        self.writeinfo_dict['fGridDataLeftLongitude'] = float(self.lon_min)
        self.writeinfo_dict['fGridDataTopLatitude'] = float(self.lat_max)
        # self.writeinfo_dict['fGridDataCenterLongitude'] = float((self.lon_min + self.lon_max) / 2)
        # self.writeinfo_dict['fGridDataCenterLatitude'] = float((self.lat_min + self.lat_max) / 2)
        self.writeinfo_dict['fGridDataCenterLongitude'] = float(self.lon_min) + gap_x * (col // 2)
        self.writeinfo_dict['fGridDataCenterLatitude'] = float(self.lat_max) - gap_y * (row // 2)
        self.writeinfo_dict['fGridDataDeltaX'] = gap_x
        self.writeinfo_dict['fGridDataDeltaY'] = gap_y
        self.writeinfo_dict['nGridDataXNum'] = col
        self.writeinfo_dict['nGridDataYNum'] = row
        #
        for key, value in self.writeinfo_dict.items():
            print(key+'\t:'+str(value))
        #

    # 获取经纬度边界
    def getEdge(self):
        return self.lon_min, self.lon_max, self.lat_min, self.lat_max

    # get every grid point (equal-distance grid) nearest 4 points in equal-latlon grid
    def _getLatlonTransformer(self, gap_x, gap_y):
        row = int(np.ceil((self.lat_max - self.lat_min) / gap_x))
        col = int(np.ceil((self.lon_max - self.lon_min) / gap_y))
        output_lat = np.zeros((row,))
        for i in range(row):
            output_lat[i] = self.lat_min + gap_x * i
        output_lon = np.zeros((col,))
        for i in range(col):
            output_lon[i] = self.lon_min + gap_y * i
        # grid_transformer = np.zeros((self.config_dict['GridRowColNum'], self.config_dict['GridRowColNum'], 2))
        grid_transformer_near4 = np.zeros((self.config_dict['GridRowColNum'], self.config_dict['GridRowColNum'], 2, 2))
        for i in range(self.config_dict['GridRowColNum']):
            for j in range(self.config_dict['GridRowColNum']):
                # binary search
                halve_l = 0
                halve_r = row - 1
                halve_mid = (halve_r + halve_l) // 2
                while halve_r - halve_l > 1:
                    if self.latlon[i, j, 0] > output_lat[halve_mid]:
                        halve_l = halve_mid + 1
                    else:
                        halve_r = halve_mid - 1
                    if self.latlon[i, j, 0] < output_lat[halve_l] and halve_l > 0:
                        halve_l -= 1
                    if self.latlon[i, j, 0] > output_lat[halve_r] and halve_r < row - 1:
                        halve_r += 1
                    halve_mid = (halve_r + halve_l) // 2
                if self.latlon[i, j, 0] - output_lat[halve_l] < output_lat[halve_r] - self.latlon[i, j, 0]:
                    # grid_transformer[i, j, 0] = halve_l
                    grid_transformer_near4[i, j, 0, 0] = halve_l
                    grid_transformer_near4[i, j, 0, 1] = halve_r
                else:
                    # grid_transformer[i, j, 0] = halve_r
                    grid_transformer_near4[i, j, 0, 0] = halve_r
                    grid_transformer_near4[i, j, 0, 1] = halve_l

                halve_l = 0
                halve_r = col - 1
                halve_mid = (halve_r + halve_l) // 2
                while halve_r - halve_l > 1:
                    if self.latlon[i, j, 1] > output_lon[halve_mid]:
                        halve_l = halve_mid + 1
                    else:
                        halve_r = halve_mid - 1
                    if self.latlon[i, j, 1] < output_lon[halve_l] and halve_l > 0:
                        halve_l -= 1
                    if self.latlon[i, j, 1] > output_lon[halve_r] and halve_r < col - 1:
                        halve_r += 1
                    halve_mid = (halve_r + halve_l) // 2
                if self.latlon[i, j, 1] - output_lon[halve_l] < output_lon[halve_r] - self.latlon[i, j, 1]:
                    # grid_transformer[i, j, 1] = halve_l
                    grid_transformer_near4[i, j, 1, 0] = halve_l
                    grid_transformer_near4[i, j, 1, 1] = halve_r
                else:
                    # grid_transformer[i, j, 1] = halve_r
                    grid_transformer_near4[i, j, 1, 0] = halve_r
                    grid_transformer_near4[i, j, 1, 1] = halve_l
        #
        # print(grid_transformer_near4,row,col)
        #
        return grid_transformer_near4, row, col

    def writeResultFile(self, pre_grid, hour_plus):

        row = self.writeinfo_dict['nGridDataYNum']
        col = self.writeinfo_dict['nGridDataXNum']
        grid = -np.ones((row, col))
        for i in range(self.config_dict['GridRowColNum']):
            for j in range(self.config_dict['GridRowColNum']):
                grid[int(self.grid_transformer_near4[i, j, 0, 0]), int(self.grid_transformer_near4[i, j, 1, 0])] = pre_grid[i, j]
        for i in range(self.config_dict['GridRowColNum']):
            for j in range(self.config_dict['GridRowColNum']):
                if grid[int(self.grid_transformer_near4[i, j, 0, 1]), int(self.grid_transformer_near4[i, j, 1, 0])] == -1:
                    grid[int(self.grid_transformer_near4[i, j, 0, 1]), int(self.grid_transformer_near4[i, j, 1, 0])] = pre_grid[i, j]
                if grid[int(self.grid_transformer_near4[i, j, 0, 0]), int(self.grid_transformer_near4[i, j, 1, 1])] == -1:
                    grid[int(self.grid_transformer_near4[i, j, 0, 0]), int(self.grid_transformer_near4[i, j, 1, 1])] = pre_grid[i, j]
                if grid[int(self.grid_transformer_near4[i, j, 0, 1]), int(self.grid_transformer_near4[i, j, 1, 1])] == -1:
                    grid[int(self.grid_transformer_near4[i, j, 0, 1]), int(self.grid_transformer_near4[i, j, 1, 1])] = pre_grid[i, j]

        grid = np.flip(grid, axis=0)
        #todo 此时grid就是等经纬度了 可以直接输出就行
        #test
        dt_d = datetime.datetime.strptime(self.config_dict['Datetime'], '%Y%m%d%H%M') + datetime.timedelta(hours=hour_plus)
        # grid【1，1】 = 1
        np.save(os.path.join(self.config_dict['ResultDistanceSavePath'], '{}_h{}.npy'.format(dt_d.strftime('%Y%m%d%H%M'), hour_plus)), pre_grid.cpu().detach().numpy())


        # np.savetxt(self.config_dict['ResultDistanceSavePath']+'grid.txt',grid,delimiter='\t')
        # with open(self.config_dict['ResultDistanceSavePath']+'grid_test.txt', 'wb') as file:
        #     for i in range(row*col):
        #         temp = int(grid[i])
        #         if temp < 0:
        #             temp = 0
        #         file.write(temp+'\t')

        #
        print(grid.shape)
        grid = grid.flatten()
        print(grid.shape)
        st = datetime.datetime.strptime(self.config_dict['Datetime'], '%Y%m%d%H%M')
        dt = datetime.datetime.strptime(self.config_dict['Datetime'], '%Y%m%d%H%M') + datetime.timedelta(hours=hour_plus)
        savepath = os.path.join(self.config_dict['ResultSavePath'], '{}_h{}.dat'.format(dt.strftime('%Y%m%d%H%M'), hour_plus))
        with open(savepath, 'wb') as file:
            ## write file header
            # nDataType
            temp = struct.pack('h', 8)
            file.write(temp)
            temp = struct.pack('h', 0)
            file.write(temp)
            # tSourceDataStartTime
            elapse = (st + datetime.timedelta(hours=-8) - datetime.datetime(1970, 1, 1, 0, 0, 0, 0)).total_seconds()
            temp = struct.pack('i', int(elapse))
            file.write(temp)

            # tSourceDataEndTime
            elapse = (st + datetime.timedelta(hours=-8) + datetime.timedelta(hours=hour_plus+1) - datetime.datetime(1970, 1, 1, 0, 0, 0, 0)).total_seconds()
            temp = struct.pack('i', int(elapse))
            file.write(temp)

            # nMinutes
            temp = struct.pack('i', 60 * self.config_dict['ForecastHourNum'])
            file.write(temp)

            # fGridDataLeftLongitude
            temp = struct.pack('f', self.writeinfo_dict['fGridDataLeftLongitude'])
            file.write(temp)

            # fGridDataTopLatitude
            temp = struct.pack('f', self.writeinfo_dict['fGridDataTopLatitude'])
            file.write(temp)

            # fGridDataCenterLongitude
            temp = struct.pack('f', self.writeinfo_dict['fGridDataCenterLongitude'])
            file.write(temp)

            # fGridDataCenterLatitude
            temp = struct.pack('f', self.writeinfo_dict['fGridDataCenterLatitude'])
            file.write(temp)

            # fGridDataDeltaX
            temp = struct.pack('f', self.writeinfo_dict['fGridDataDeltaX'])
            file.write(temp)

            # fGridDataDeltaY
            temp = struct.pack('f', self.writeinfo_dict['fGridDataDeltaY'])
            file.write(temp)

            # nGridDataXNum
            temp = struct.pack('i', self.writeinfo_dict['nGridDataXNum'])
            file.write(temp)

            # nGridDataYNum
            temp = struct.pack('i', self.writeinfo_dict['nGridDataYNum'])
            file.write(temp)

            # nType
            temp = struct.pack('h', 8)
            file.write(temp)
            temp = struct.pack('h', 0)
            file.write(temp)

            # nLevel
            temp = struct.pack('i', 0)
            file.write(temp)

            # print(file.tell())
            # Version
            temp = struct.pack('B', 1)
            file.write(temp)
            # print(file.tell())

            # ForecastStepsCount
            # print(file.tell())
            temp = struct.pack('B', self.config_dict['ForecastHourNum'])
            file.write(temp)
            # print(file.tell())

            # ForecastStepIndex
            # print(file.tell())
            temp = struct.pack('B', hour_plus)
            file.write(temp)
            # print(file.tell())

            # ForecastStepMinutes
            # print(file.tell())
            temp = struct.pack('B', 60)
            file.write(temp)
            # print(file.tell())

            # DataStatus
            temp = struct.pack('i', 0)
            file.write(temp)

            # fTemp[7]
            for i in range(7):
                temp = struct.pack('f', 0)
                file.write(temp)

            # write data
            for i in range(row*col):
                temp = int(grid[i])
                if temp < 0:
                    temp = 0
                data = struct.pack('B', temp)
                file.write(data)


if __name__ == "__main__":
    pass



