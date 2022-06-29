import numpy as np
from torch.utils.data import Dataset as py_Dataset
import datetime
from netCDF4 import Dataset
import os


variables3d = ['QICE', 'QGRAUP', 'QSNOW']
variables2d = ['W_max']
sumVariables2d = ['RAINNC']
# param_list = ['qice','qgraup','qsnow','w','rainnc']
param_list = ['QICE', 'QSNOW', 'QGRAUP', 'W_max', 'RAINNC']


def getTimePeriod(dt):
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


# 这是我自己该写的
# def getHoursGridFromSmallNC_npy(npy_father_filepath, delta_hour, config_dict):  # 20200619
#     variables3d = ['QICE', 'QGRAUP', 'QSNOW']
#     variables2d = ['W_max']
#     sumVariables2d = ['RAINNC']
#     param_list = ['QICE', 'QSNOW', 'QGRAUP', 'W_max', 'RAINNC']
#
#     m = config_dict['GridRowColNum']
#     n = config_dict['GridRowColNum']
#     grid_list = []
#     if config_dict['WRFChannelNum'] == 217:
#         varone_test = os.path.join(npy_father_filepath, 'V.npy')
#         grid = np.load(varone_test)
#         grid = grid[delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
#         grid = np.transpose(grid, (0, 2, 3, 1))  # (12, 159, 159, n)
#     else:
#         for s in param_list:
#             if s in variables3d:
#                 file_path = os.path.join(npy_father_filepath, s + '.npy')
#                 temp = np.load(file_path)[delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
#                 temp[temp < 0] = 0
#                 if config_dict['WRFChannelNum'] == 29:
#                     ave_3 = np.zeros((config_dict['ForecastHourNum'], m, n, 9))
#                     for i in range(9):
#                         ave_3[:, :, :, i] = np.mean(temp[:, 3 * i:3 * (i + 1), :, :], axis=1)  # (12, 159, 159, 9)
#                     grid_list.append(ave_3)
#                 else:
#                     temp = np.transpose(temp, (0, 2, 3, 1))  # (12, 159, 159, 27)
#                     grid_list.append(temp)
#             elif s in variables2d:
#                 if s == 'W_max':
#                     file_path = os.path.join(npy_father_filepath, 'W' + '.npy')
#                     tmp = np.load(file_path)[delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
#                     tmp = np.transpose(tmp, (0, 2, 3, 1))
#                     temp = np.max(tmp, axis=-1, keepdims=True)
#                 else:
#                     file_path = os.path.join(npy_father_filepath, s +'.npy')
#                     if not os.path.exists(file_path):
#                         print("这个文件没有={}".format(file_path))
#                     temp = np.load(file_path)[delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
#                 grid_list.append(temp)
#             elif s in sumVariables2d:
#                 file_path = os.path.join(npy_father_filepath, s + '.npy')
#                 if not os.path.exists(file_path):
#                     print("这个文件没有={}".format(file_path))
#                 temp = np.load(file_path)[delta_hour + 1:delta_hour + config_dict['ForecastHourNum'] + 1, 0:m, 0:n] - \
#                        np.load(file_path)[delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
#                 temp = temp[:, :, :, np.newaxis]
#                 grid_list.append(temp)
#         grid = np.concatenate(grid_list, axis=-1)
#
#     return grid


# 这个方法是魔改getHoursGridFromSmallNC方法，由于之前学长已经将nc处理为npy了
# 所以这里我们是读取npy文件 npy_father_filepath是一个文件夹，他就对应了过去的一个具体的nc文件
def getHoursGridFromSmallNC_npy(filepath, delta_hour, config_dict):  # 20200619
    # m = config_dict['GridRowColNum']
    # n = config_dict['GridRowColNum']
    grid_list = []
    param_list = ['QICE_ave3', 'QSNOW_ave3', 'QGRAUP_ave3', 'W_max', 'RAINNC']
    # delta_hour -= 6
    for s in param_list:
        npy_grid = np.load(os.path.join(filepath, '{}.npy'.format(s)))
        # npy_grid = npy_grid[delta_hour - config_dict['TruthHistoryHourNum']:delta_hour + config_dict['ForecastHourNum']]
        npy_grid = npy_grid[delta_hour:delta_hour + config_dict['ForecastHourNum']]
        if s == 'RAINNC':
            npy_grid = npy_grid[:, np.newaxis, :, :]
        elif s == 'W_max':
            npy_grid = np.max(npy_grid, axis=1, keepdims=True)
        npy_grid = np.transpose(npy_grid, (0, 2, 3, 1))  # (12, 159, 159, x)
        grid_list.append(npy_grid)

    grid = np.concatenate(grid_list, axis=-1)
    # grid.shape=(3, 159, 159, 29)
    return grid


def getHoursGridFromSmallNC(ncfilepath, delta_hour, config_dict):  # 20200619
    variables3d = ['QICE', 'QGRAUP', 'QSNOW']
    variables2d = ['W_max']
    sumVariables2d = ['RAINNC']
    param_list = ['QICE', 'QSNOW', 'QGRAUP', 'W_max', 'RAINNC']

    m = config_dict['GridRowColNum']
    n = config_dict['GridRowColNum']
    grid_list = []
    # if config_dict['WRFChannelNum'] == 217:
    #     with Dataset(ncfilepath) as nc:
    #         grid = nc.variables['varone'][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]
    #         grid = np.transpose(grid, (0, 2, 3, 1))  # (12, 159, 159, n)
    # else:

    with Dataset(ncfilepath) as nc:
            for s in param_list:
                if s in variables3d:
                    temp = nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], :, 0:m, 0:n]  # (12, 27, 159, 159)
                    temp[temp < 0] = 0
                    if config_dict['WRFChannelNum'] == 29:
                        ave_3 = np.zeros((config_dict['ForecastHourNum'], m, n, 9))
                        for i in range(9):
                            ave_3[:, :, :, i] = np.mean(temp[:, 3*i:3*(i+1), :, :], axis=1) # (12, 159, 159, 9)
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
                    temp = nc.variables[s][delta_hour + 1:delta_hour + config_dict['ForecastHourNum'] + 1, 0:m, 0:n] - \
                           nc.variables[s][delta_hour:delta_hour + config_dict['ForecastHourNum'], 0:m, 0:n]
                    temp = temp[:, :, :, np.newaxis]
                    grid_list.append(temp)
    grid = np.concatenate(grid_list, axis=-1)

    return grid

class DataGenerator(py_Dataset):
    def __init__(self, lists, config_dict):
        self.lists = lists
        self.config_dict = config_dict

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        X, y = self.__data_generation(self.lists[index])
        return X, y

    def __data_generation(self, datetime_peroid):
        m = self.config_dict['GridRowColNum']
        n = self.config_dict['GridRowColNum']
        wrf_batch = np.zeros(shape=[self.config_dict['ForecastHourNum'], m, n, self.config_dict['WRFChannelNum']], dtype=np.float32)
        label_batch = np.zeros(shape=[self.config_dict['ForecastHourNum'], m * n, 1], dtype=np.float32)
        history_batch = np.zeros(shape=[self.config_dict['TruthHistoryHourNum'], m, n, 1], dtype=np.float32)

        ddt = datetime.datetime.strptime(datetime_peroid, '%Y%m%d%H%M')
        # read WRF
        # UTC是世界时
        utc = ddt + datetime.timedelta(hours=-8)
        ft = utc + datetime.timedelta(hours=(-6))
        nchour, delta_hour = getTimePeriod(ft)
        delta_hour += 6
        # ncFileDir = NCDir + 'gfs.' + ft.strftime("%Y%m%d") + '/' + 'gfs.' + ft.strftime("%Y%m%d%M") + '/'
        # nc_grid = getHoursGridFromSingleParamNC(ncFileDir, delta_hour, param_list )

        ncFilepath = self.config_dict['WRFFileDir'] + ft.strftime("%Y-%m-%d") + '_' + nchour + '.wrfvar.nc'
        # '/home/wrfelec05/bjdata_test/2020-06-11_00.wrfvar.nc'

        npyFilepath = os.path.join(self.config_dict['WRFFileDir'], ft.strftime("%Y%m%d"), nchour)

        # 由于wrf(也就是nc格式)存了好几个矩阵，所以一个nc文件转成了好几个npy，这里一个wrf对应的
        # 是一个文件夹  '/data/wenjiahua/light_data/ADSNet_testdata/WRF_data/20200831'
        nc_grid = getHoursGridFromSmallNC_npy(npyFilepath, delta_hour, self.config_dict)

        wrf_batch[:, :, :, 0:self.config_dict['WRFChannelNum']] = nc_grid

        # read labels
        for hour_plus in range(self.config_dict['ForecastHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            tFilePath = self.config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth' + '.npy'
            truth_grid = np.load(tFilePath)
            truth_grid[truth_grid > 1] = 1
            truth_grid = truth_grid.reshape(m * n)
            label_batch[hour_plus, :, :] = truth_grid[:, np.newaxis]
        # read history observations
        for hour_plus in range(self.config_dict['TruthHistoryHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus - self.config_dict['TruthHistoryHourNum'])
            tFilePath = self.config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth.npy'
            truth_grid = np.load(tFilePath)
            truth_grid = truth_grid.reshape(m, n)
            history_batch[hour_plus, :, :, :] = truth_grid[:, :, np.newaxis]
        return [wrf_batch, history_batch, ft.strftime("%Y%m%d")], label_batch


if __name__ == "__main__":
    from config import read_config
    from torch.utils.data import DataLoader
    config_dict = read_config()
    # data index
    TrainSetFilePath = 'train_lite_new_12h.txt'
    ValSetFilePath = 'July.txt'
    train_list = []
    with open(TrainSetFilePath) as file:
        for line in file:
            train_list.append(line.rstrip('\n').rstrip('\r\n'))
    val_list = []
    with open(ValSetFilePath) as file:
        for line in file:
            val_list.append(line.rstrip('\n').rstrip('\r\n'))
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=8)
    for i, (X, y) in enumerate(train_loader):
        wrf, obs = X
        label = y
        wrf = wrf.to(config_dict['Device'])
        obs = obs.to(config_dict['Device'])
        label = label.to(config_dict['Device'])
        print(label.shape)
