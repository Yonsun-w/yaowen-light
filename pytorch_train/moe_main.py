# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datetime
from config import read_config
from layers.ADSNet_model import ADSNet_Model as ads1
from layers.moe_ADSNet import ADSNet_Model as ads2
from layers.LightNet_model import LightNet_Model as l1
from layers.moe_LightNet import LightNet_Model as l2

from layers.OnlyObsNet_model import OnlyObsNet_Model
from layers.OnlyWRFNet_model import OnlyWRFNet_Model
from layers.MOE import MOE_Model
from moe_generator import DataGenerator
from MOE_scores import Cal_params_epoch, Model_eval
from generator import getTimePeriod
import datetime

def selectModel(config_dict):
    if config_dict['NetName'] == 'ADSNet':
        model = ads2(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                  wrf_channels=config_dict['WRFChannelNum'], row_col=config_dict['GridRowColNum']).to(config_dict['Device'])
    elif config_dict['NetName'] == 'LightNet':
        model = l2(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], row_col=config_dict['GridRowColNum']).to(config_dict['Device'])
    elif config_dict['NetName'] == 'OnlyObs':
        model = OnlyObsNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                               pre_frames=config_dict['ForecastHourNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'OnlyWRF':
        model = OnlyWRFNet_Model(wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict).to(config_dict['Device'])
    elif config_dict['NetName'] == 'MOE':
        model = MOE_Model(truth_history_hour_num=config_dict['TruthHistoryHourNum'],
                forecast_hour_num=config_dict['ForecastHourNum'],
                row_col=config_dict['GridRowColNum'], wrf_channels=config_dict['WRFChannelNum'],obs_channel=1).to(config_dict['Device'])
    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    return model

def time_data_iscomplete(time_str, WRFFileDir, ForecastHourNum,TruthFileDirGrid,TruthHistoryHourNum):
    time_str = time_str.rstrip('\n')
    time_str = time_str.rstrip('\r\n')
    if time_str == '':
        return False

    is_complete = True
    ddt = datetime.datetime.strptime(time_str, '%Y%m%d%H%M')
    # read WRF
    # UTC是世界时
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6
    npyFilepath = os.path.join(WRFFileDir, ft.strftime("%Y%m%d"), nchour)

    if not os.path.exists(npyFilepath):
        is_complete = False

    # read labels
    for hour_plus in range(ForecastHourNum):
        dt = ddt + datetime.timedelta(hours=hour_plus)
        tFilePath = TruthFileDirGrid + dt.strftime('%Y%m%d%H%M') + '_truth' + '.npy'
        if not os.path.exists(tFilePath):
            is_complete = False
        else:
            a = np.load(tFilePath)
            if np.sum(a) <= 10:
                False
    # read history observations
    for hour_plus in range(TruthHistoryHourNum):
        dt = ddt + datetime.timedelta(hours=hour_plus - TruthHistoryHourNum)
        tFilePath = TruthFileDirGrid + dt.strftime('%Y%m%d%H%M') + '_truth.npy'
        if not os.path.exists(tFilePath):
            is_complete = False


    ddt = datetime.datetime.strptime(time_str, '%Y%m%d%H%M') + datetime.timedelta(hours=-TruthHistoryHourNum)
    # read WRF
    # UTC是世界时
    utc = ddt + datetime.timedelta(hours=-8)
    ft = utc + datetime.timedelta(hours=(-6))
    nchour, delta_hour = getTimePeriod(ft)
    delta_hour += 6
    npyFilepath = os.path.join(WRFFileDir, ft.strftime("%Y%m%d"), nchour)
    if not os.path.exists(npyFilepath):
        is_complete = False

    return is_complete


def DoTrain(config_dict):



    st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H')
    # 验证集开始载入的时间
    test_time = datetime.datetime.strptime(config_dict['testTime'], '%Y%m%d%H')



    # # 直接从时间段自动加载
    # train_list = []
    # val_list = []
    # print('加载从{}到{}之间的数据集，其中{}时间到{}时间作为测试集'.format(st, et, test_time, et))
    # while st <= et:
    #     line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
    #     # 由于数据不全 所以需要校验数据的完整
    #     if time_data_iscomplete(line,  WRFFileDir=config_dict['WRFFileDir'],ForecastHourNum=config_dict['ForecastHourNum'],
    #                             TruthFileDirGrid=config_dict['TruthFileDirGrid'], TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
    #         if st >= test_time:
    #             val_list.append(line.rstrip('\n').rstrip('\r\n'))
    #         else:
    #             train_list.append(line.rstrip('\n').rstrip('\r\n'))
    #     st += datetime.timedelta(hours=3)

    # 直接从文件读
    train_list = []
    val_list = []
    train_path = 'TrainCase.txt'
    val_path = 'ValCase.txt'


    train_path = 'pre_train.txt'
    val_path = 'pre_val.txt'


    print('加载{}的为训练集，{}的为验证集'.format(train_path,val_path))
    with open(train_path) as file:
        for line in file:
            if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                    ForecastHourNum=config_dict['ForecastHourNum'],TruthFileDirGrid=config_dict['TruthFileDirGrid'], TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
                train_list.append(line.rstrip('\n').rstrip('\r\n'))

    with open(val_path) as file:
        for line in file:
            if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                    ForecastHourNum=config_dict['ForecastHourNum'],TruthFileDirGrid=config_dict['TruthFileDirGrid'], TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
                val_list.append(line.rstrip('\n').rstrip('\r\n'))



    print('加载数据完毕，一共有{}训练集，val{}测试集'.format(len(train_list), len(val_list)))

    # data
    train_data = DataGenerator(train_list, config_dict)
    train_loader = DataLoader(dataset=train_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)
    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=False, num_workers=0)

    # model
    model = selectModel(config_dict)


    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16))
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])
    # eval
    model_eval_valdata = Model_eval(config_dict)

    for epoch in range(config_dict['EpochNum']):
        # train_calparams_epoch = Cal_params_epoch()
        for i, (X, y) in enumerate(train_loader):
            wrf, obs, wrf_old = X

            # print('wrf.shape={},obs.shape={}'.format(wrf.shape, obs.shape))
            # wrf.shape=torch.Size([4, 未来3小时, 159, 159, 29]),obs.shape=torch.Size([4, 历史6小时, 159, 159, 1])

            label = y
            wrf = wrf.to(config_dict['Device'])

            obs = obs.to(config_dict['Device'])

            wrf_old = wrf_old.to(config_dict['Device'])



            label = label.to(config_dict['Device'])

            if config_dict['NetName'] == 'MOE':
                pre_frames = model(wrf, obs, wrf_old)

            else:
                pre_frames,h = model(wrf, obs)


            # backward
            optimizer.zero_grad()

            loss = criterion(torch.flatten(pre_frames), torch.flatten(label))

            loss.backward()

            # update weights
            optimizer.step()

            print('pre  有{}个, label  有{}个, obs 有{}个,wrf ={}'.format(torch.sum(pre_frames>0), torch.sum(label>0)
                  , torch.sum(obs > 0), torch.sum(wrf > 0)))

            print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
            # pod, far, ts, ets = train_calparams_epoch.cal_batch(label, pre_frames)
            # sumpod, sumfar, sumts, sumets = train_calparams_epoch.cal_batch_sum(label, pre_frames)
            # info = 'TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}\nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'\
            #     .format(epoch, i+1, len(train_loader), loss.item(), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
            # print(info)

        model_eval_valdata.eval(val_loader, model, epoch)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'



    config_dict = read_config()
    #
    # init_old_data(config_dict)

    print('-------------模型名称为={}--------------'.format(config_dict['NetName']))

    # #train
    DoTrain(config_dict)
    # moedl = selectModel(config_dict)


    # model = selectModel(config_dict)
    #
    # for name, param  in model.named_parameters():
    #     print(name)


