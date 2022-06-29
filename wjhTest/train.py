# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from config import read_config
from layers.ADSNet_model import ADSNet_Model
from layers.LightNet_model import LightNet_Model
from layers.OnlyObsNet_model import OnlyObsNet_Model
from layers.OnlyWRFNet_model import OnlyWRFNet_Model
from generator import DataGenerator
from sklearn import datasets
from generator import getTimePeriod
import datetime

def selectModel(config_dict):
    if config_dict['NetName'] == 'ADSNet':
        model = ADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                  wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'LightNet':
        model = LightNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'OnlyObs':
        model = OnlyObsNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                               pre_frames=config_dict['ForecastHourNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'OnlyWRF':
        model = OnlyWRFNet_Model(wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model

def time_data_iscomplete(time_str, config_dict):

    time_str = time_str.rstrip('\n')
    time_str = time_str.rstrip('\r\n')
    if time_str == '':
        return False
    is_complete = True
    m = config_dict['GridRowColNum']
    n = config_dict['GridRowColNum']
    wrf_batch = np.zeros(shape=[config_dict['ForecastHourNum'], m, n, config_dict['WRFChannelNum']],
                         dtype=np.float32)
    label_batch = np.zeros(shape=[config_dict['ForecastHourNum'], m * n, 1], dtype=np.float32)
    history_batch = np.zeros(shape=[config_dict['TruthHistoryHourNum'], m, n, 1], dtype=np.float32)

    ddt = datetime.datetime.strptime(time_str, '%Y%m%d%H%M')
    # read WRF
    # UTC是世界时
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6
    npyFilepath = os.path.join(config_dict['WRFFileDir'], ft.strftime("%Y%m%d"), nchour)

    if not os.path.exists(npyFilepath):
        is_complete = False

    # read labels
    for hour_plus in range(config_dict['ForecastHourNum']):
        dt = ddt + datetime.timedelta(hours=hour_plus)
        tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth' + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False
        else :
            a = np.load(tFilePath)
            if np.sum(a) <= 10 :
                False
    # read history observations
    for hour_plus in range(config_dict['TruthHistoryHourNum']):
        dt = ddt + datetime.timedelta(hours=hour_plus - config_dict['TruthHistoryHourNum'])
        tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth.npy'
        if not os.path.exists(tFilePath):
            is_complete = False

    return is_complete


def DoTrain(config_dict):
    # data index wjh改 我觉得没必要用读取txt的方式 为了便于我的开发 我这里直接将扫描开始的位置作为数据开始的时间
    # TrainSetFilePath = 'TrainCase.txt'
    # ValSetFilePath = 'ValCase.txt'
    # train_list = []
    # with open(TrainSetFilePath) as file:
    #     for line in file:
    #         # 由于数据不全 所以需要校验数据的完整
    #         if time_data_iscomplete(line, config_dict):
    #             train_list.append(line.rstrip('\n').rstrip('\r\n'))
    # val_list = []
    # with open(ValSetFilePath) as file:
    #     for line in file:
    #         # 由于数据不全 所以需要校验数据的完整
    #         if time_data_iscomplete(line, config_dict):
    #             val_list.append(line.rstrip('\n').rstrip('\r\n'))


    st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H')
    # 训练集开始载入的时间
    train_time = datetime.datetime.strptime('2020070200', '%Y%m%d%H')

    train_list = []
    val_list = []
    print('加载从{}到{}之间的数据集，其中{}时间到{}时间作为测试集'.format(st,et,train_time,et))
    while st <= et:
        line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
        # 由于数据不全 所以需要校验数据的完整

        if time_data_iscomplete(line, config_dict):
            if st >= train_time:
                val_list.append(line.rstrip('\n').rstrip('\r\n'))
            train_list.append(line.rstrip('\n').rstrip('\r\n'))

        st += datetime.timedelta(hours=3)

    print('加载数据完毕，一共有{}训练集，val{}测试集'.format(len(train_list), len(val_list)))

    # data
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)

    # model
    model = selectModel(config_dict)

    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])

    for epoch in range(config_dict['EpochNum']):
        # train_calparams_epoch = Cal_params_epoch()
        for i, (X, y) in enumerate(train_loader):
            wrf, obs, npyFilepath = X
            label = y
            wrf = wrf.to(config_dict['Device'])
            obs = obs.to(config_dict['Device'])
            label = label.to(config_dict['Device'])



if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    #
    # config_dict = read_config()
    # #
    # # init_old_data(config_dict)
    #
    # # #train
    # DoTrain(config_dict)
    import sklearn
    config_dict = read_config()
    model = selectModel(config_dict)

    st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H')
    # 训练集开始载入的时间
    train_time = datetime.datetime.strptime('2018081200', '%Y%m%d%H')

    train_list = []
    val_list = []
    print('加载从{}到{}之间的数据集，其中{}时间到{}时间作为测试集'.format(st,et,train_time,et))
    while st <= et:
        line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
        # 由于数据不全 所以需要校验数据的完整
        if time_data_iscomplete(line, config_dict):
            if st >= train_time:
                val_list.append(line.rstrip('\n').rstrip('\r\n'))
            train_list.append(line.rstrip('\n').rstrip('\r\n'))

        st += datetime.timedelta(hours=3)

    print('加载数据完毕，一共有{}训练集，val{}测试集'.format(len(train_list), len(val_list)))

    data_train = DataGenerator(train_list, config_dict)
    data_test = DataGenerator(val_list, config_dict)

    # wrf.shape=(3, 159, 159, 29),obs.shape=(3, 159, 159, 1),label.shape=(3, 25281, 1)
    # 数组
    train_load = DataLoader(data_train)
    test_load = DataLoader(data_test)

    models = [('AdsNet'),model]



    # wrf.shape=torch.Size([4, 3, 159, 159, 29]),obs.shape=torch.Size([4, 3, 159, 159, 1]),y.shape=torch.Size([4, 3, 25281, 1])
    train_loader = DataLoader(dataset=data_train, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=data_test, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)

    for i, (X, y) in enumerate(train_loader):
        wrf, obs, npyFilepath = X
        label = y

