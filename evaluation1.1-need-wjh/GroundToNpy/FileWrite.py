# -*- coding: utf-8 -*-
import numpy as np
import datetime
import os
import config
import netCDF4 as nc
from draw import drawImage


class TxtTrueFiletoGrid(object):
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.lat_max = float(config_dict['latEnd'])
        self.lat_min = float(config_dict['latBegin'])
        self.lon_max = float(config_dict['lonEnd'])
        self.lon_min = float(config_dict['lonBegin'])
        # 单元格为 m km * n km
        # m代表经度上的单元格 也就是一个单元格 经度上的km数
        self.lon_gap = config_dict['lonGap']
        # n代表纬度上的
        self.lat_gap = config_dict['latGap']
        # 计算横着的长度 也就是根据经度计算 由于中国在上半球 所以我们直接计算纬度最大的那部分就好 也就是下边那条线
        distance_row = self._cal_distance(self.lat_max, self.lon_min, self.lat_max, self.lon_max)
        # 计算竖着的长度
        distance_col = self._cal_distance(self.lat_min, self.lon_min, self.lat_max, self.lon_min)
        # 矩阵的行 为
        self.row = int(distance_row / self.lat_gap)
        # 矩阵的列
        self.col = int(distance_col / self.lon_gap)


        # 如果是等经纬度 直接算就可以了
        if config_dict['equal_dis'] == 0:
            self.row = int((self.lat_max - self.lat_min )/ self.lat_gap)
            self.col = int((self.lon_max - self.lon_min )/ self.lon_gap)

        # 返回的矩阵
        self.grid = np.zeros(shape=[self.row, self.col], dtype=float)


        # 开始扫描的时间段
        self.start_time = datetime.datetime.strptime(config_dict['startTime'], "%Y%m%d%H%M%S")
        self.end_time = datetime.datetime.strptime(config_dict['endTime'], "%Y%m%d%H%M%S")

        # 代表时间纬度 用于nc文件存储上
        self.time = 0
        # 时间分辨率 存储了每个要输出的时间段，每个时间段对应一个nc文件
        self.time_slot = []
        st = self.start_time
        while st <= self.end_time:
            self.time += 1
            self.time_slot.append(st)
            # 时间分辨率
            st += datetime.timedelta(hours=config_dict['timeGap'])


        dataTime = self.start_time
        # 存在的日期文件列表 注意 这里存放的文件是不带父目录的 父目录为 config_dict['TruthFileDir']
        # 这里只保存日期时间 不保留具体的小时
        self.txtFiles = []
        while (dataTime <= self.end_time):
            truthfilepath = os.path.join(self.config_dict['TruthFileDir'],  dataTime.strftime('_%Y_%m_%d') + '.txt')
            if not os.path.exists(truthfilepath):
                print('Lighting data file `{}` not exist!'.format(truthfilepath))
            else:
                self.txtFiles.append(truthfilepath)
            dataTime += datetime.timedelta(days=1)

        if not os.path.isdir(self.config_dict['output']):
            print('您的文件输出目录指定不存在或错误，请检查')

        if not os.path.isdir(self.config_dict['TruthFileDir']):
            print('您的txt文件输入目录指定不存在，请检查')


        print('-----txt数据格点化文件初始化完毕，等待写入中----------')




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

    # 将文件夹下所有的txt记录数据转换为png图像
    def all_trueFile_txt_to_png(self):
        print('-----------------开始输出我们的图像可视化--------------')

        # 获取每个时间段
        for time in self.time_slot:
            truthfilepath =  os.path.join(self.config_dict['TruthFileDir'],  time.strftime('_%Y_%m_%d') + '.txt')
            if not os.path.exists(truthfilepath):
                print('Lighting data file  `{}` not exist! (这个文件不存在 {})'.format(truthfilepath, truthfilepath))
                continue
            output_file = os.path.join(self.config_dict['output'],'adtd' + time.strftime('_%Y_%m_%d_%H_%M') + '.png')
            grid = self.txt_to_grid(truthfilepath,time)

            drawImage(grid, output_file)

            print('{}输出成功，请您查看'.format(output_file))
        print('-----------------图像输出完毕--------------')

    # 将文件夹下所有的txt记录数据转换为npy数据格式
    def all_trueFile_txt_to_npy(self):
        print('-----------------开始输出npy--------------')

        # 获取每个时间段
        for time in self.time_slot:
            truthfilepath =  os.path.join(self.config_dict['TruthFileDir'],  time.strftime('_%Y_%m_%d') + '.txt')
            if not os.path.exists(truthfilepath):
                print('Lighting data file  `{}` not exist! (这个文件不存在 {})'.format(truthfilepath, truthfilepath))
                continue
            output_file = os.path.join(self.config_dict['output'],'adtd' + time.strftime('_%Y_%m_%d_%H_%M') + '.npy')
            grid = self.txt_to_grid(truthfilepath,time)
            np.save(output_file,grid)
            print('{}输出成功，请您查看,出现了{}个闪电点位'.format(output_file, np.sum(grid>0)))
        print('-----------------npy输出完毕--------------')

    # 将文件夹下所有的txt记录数据转换为nc数据格式
    def all_trueFile_txt_to_nc(self):
        print('-----------------开始输出nc--------------')
        # 获取每个时间段
        for time in self.time_slot:
            truthfilepath = os.path.join(self.config_dict['TruthFileDir'],
                                             time.strftime('_%Y_%m_%d') + '.txt')

            if not os.path.exists(truthfilepath):
                print('Lighting data file  `{}` not exist! (这个文件不存在 {})'.format(truthfilepath, truthfilepath))
                continue
            output_file = os.path.join(self.config_dict['output'],
                                           'adtd' + time.strftime('_%Y_%m_%d_%H_%M') + '.nc')
            grid = self.txt_to_grid(truthfilepath, time)

            self.create_nc(output_file, grid)


            print('{}输出成功，请您查看,出现了{}个闪电点位'.format(output_file, np.sum(grid>0)))

        print('-----------------nc输出完毕--------------')

        
    # 地闪数据转化为矩阵 将t1时间段内的闪电格点化
    def txt_to_grid(self, tFilePath, t1):
        # 当前时间段的矩阵 最多为1 他的形状和grid是一样的
        cur_grid = np.zeros(shape=[self.row, self.col], dtype=int)
        t2 = t1 + datetime.timedelta(hours=self.config_dict['timeGap'])
       # print(' t1 = {} t2 = {}'.format(t1, t2)) t1 = 2019-05-17 00:00:00 t2 = 2019-05-17 03:00:00

        with open(tFilePath, 'r', encoding='GBK') as tfile:
            for line in tfile:
                # 每条闪电数据
                lightning_data = {}
                linedata = line.split()
                temp_date = linedata[1]
                temp_time = linedata[2]
                temp_dt = datetime.datetime.strptime(temp_date + ' ' + temp_time[0:8], "%Y-%m-%d %H:%M:%S")

                lat = float(linedata[3].lstrip('纬度='))
                lon = float(linedata[4].lstrip('经度='))
                # print('temp_dt = {}'.format(temp_dt)) 2015-05-17 00:40:30
                #print('t1 <= temp_dt <= t2 is true ={},{},{},{}'.format(t1 <= temp_dt <= t2, temp_time,t1,t2))
                # 假如时间不在这个范围内 直接跳过
                if  t1 <= temp_dt <= t2 and self.check_in_area(lat, lon):
                    # 绘制到grid上 按照范围绘
                    row = int(self._cal_distance(lat, lon, lat, self.lon_min) / self.lat_gap)
                    col = int(self._cal_distance(lat, lon, self.lat_min, lon) / self.lon_gap)

                    if self.config_dict['equal_dis'] == 0:
                        row = int((lon - self.lon_min) / self.lat_gap)
                        col = int((lat - self.lat_min) / self.lon_gap)

                    # 按照范围绘制
                    cur_grid = self.draw_grid_all(cur_grid, row, col)



        Threshold = self.config_dict['Threshold']
        Threshold = int(Threshold)
        cur_grid[cur_grid<Threshold] = 0

        cur_grid[cur_grid!=0] = 1

        return cur_grid

    #将周边的范围绘制上地闪初猜
    def draw_grid_all(self, grid, x, y):
        row = grid.shape[0]
        col = grid.shape[1]

        edg = self.config_dict['edg']
        sr = max(x - edg,0)
        er = min(row - 1, x + edg)
        sc = max(y - edg, 0)
        ec = min(col - 1, y + edg)
        grid[sr:er,sc:ec] += 1

        return grid

    # 检测该点是否在范围呢
    def check_in_area(self, la,lo):
        if lo < self.lon_min or lo > self.lon_max or la < self.lat_min or la > self.lat_max:
            return False

        return True



    # 三个参数 分别是 输出文件夹,读取的txt文件,和time当前时间段
    def create_nc(self, output_path, cur_grid):
        f_w = nc.Dataset(output_path, 'w', format='NETCDF4')  # 创建一个格式为.nc的
        f_w.FileOrigins = 'equidistant'
        f_w.lon_begin = self.lon_min
        f_w.lon_end = self.lon_max
        f_w.lat_begin = self.lat_min
        f_w.lat_end = self.lat_max

        # --------------------上述是写入头文件-----------------------------

        # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
        f_w.createDimension('south_north', self.row)
        f_w.createDimension('west_east', self.col)

        f_w.createVariable('Flash_pre', np.float32, ('south_north', 'west_east'))
        f_w.variables['Flash_pre'].MemoryOrder = 'XY'
        f_w.variables['Flash_pre'].units = 'BJTime'
        f_w.variables['Flash_pre'].description = 'hourly grid prediction lightning'
        f_w.variables['Flash_pre'].coordinates = 'XLONG XLAT'
        # --------------------上述是创建变量信息-----------------------------
        # 获取该时间段下的矩阵数据
        f_w.variables['Flash_pre'] = cur_grid
        # 关闭文件
        f_w.close()



if __name__ == "__main__":
    config = config.read_config()
    t = TxtTrueFiletoGrid(config)
    t.all_trueFile_txt_to_wrf()