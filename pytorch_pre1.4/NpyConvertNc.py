import netCDF4 as nc
import numpy as np
from config import read_config
import datetime
import os


# path传入的是npy文件的路径 写入等经纬度
def createDistanceNc(config_dict, lon_min, lon_max, lat_min, lat_max, ncPath ='ResultDistance.nc'):
    f_w = nc.Dataset('Equal_distance.nc', 'w', format='NETCDF4')  # 创建一个格式为.nc的

    f_w.FileOrigins = 'equidistant'
    f_w.lon_begin = lon_min
    f_w.lon_end = lon_max
    f_w.lat_begin = lat_min
    f_w.lat_end = lat_max
    f_w.delta_lat = 0.03
    f_w.delta_dis ='4km'
    f_w.delta_time = '1 hour'

    # 确定基础变量的维度信息。相对与坐标系的各个轴(x,y,z)
    f_w.createDimension('south_north', config_dict['GridRowColNum'])
    f_w.createDimension('west_east', config_dict['GridRowColNum'])
    f_w.createDimension('Time', config_dict['ForecastHourNum'])
    # 创建变量。参数依次为：‘变量名称’，‘数据类型’，‘基础维度信息’
    f_w.createVariable('Flash_pre', np.float32, ('Time', 'south_north', 'west_east'))

    # 按小时来写入结果
    for hour_plus in range(config_dict['ForecastHourNum']):
        dt_d = datetime.datetime.strptime(config_dict['Datetime'], '%Y%m%d%H%M') + datetime.timedelta(hours=hour_plus)
        dis_npy_path = os.path.join(config_dict['ResultDistanceSavePath'], '{}_h{}.npy'.format(dt_d.strftime('%Y%m%d%H%M'), hour_plus))
        f_w.variables['Flash_pre'][hour_plus] = np.load(dis_npy_path)

    f_w.variables['Flash_pre'].MemoryOrder  = 'XY'
    f_w.variables['Flash_pre'].units = 'BJTime'

    f_w.variables['Flash_pre'].description = 'hourly grid prediction lightning'
    f_w.variables['Flash_pre'].coordinates = 'XLONG XLAT'

    # 获取开始时间
    start_time = datetime.datetime.strptime(config_dict['Datetime'], '%Y%m%d%H%M')
    # 获取结束时间
    end_time = datetime.datetime.strptime(config_dict['Datetime'], '%Y%m%d%H%M') + datetime.timedelta(hours=config_dict['ForecastHourNum'])


    f_w.variables['Flash_pre'].init_time = start_time.strftime('%Y%m%d') + '_' + start_time.strftime('%H%M%S')

    f_w.variables['Flash_pre'].valid_time = end_time.strftime('%Y%m%d') + '_' + end_time.strftime('%H%M%S')

    # 关闭文件
    f_w.close()




if __name__ == "__main__":

    config_dict = read_config()
    createDistanceNc(config_dict)