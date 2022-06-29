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
from generator import DataGenerator
from scores import Cal_params_epoch, Model_eval
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
                row_col=config_dict['GridRowColNum'], wrf_channels=config_dict['WRFChannelNum'],obs_channel = 1).to(config_dict['Device'])

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
        else:
            a = np.load(tFilePath)
            if np.sum(a) <= 100:
                is_complete = False

    # read history observations
    for hour_plus in range(config_dict['TruthHistoryHourNum']):
        dt = ddt + datetime.timedelta(hours=hour_plus - config_dict['TruthHistoryHourNum'])
        tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth.npy'
        if not os.path.exists(tFilePath):
            is_complete = False

    return is_complete






def DoTrain(config_dict):
    st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H')
    # 验证集开始载入的时间
    test_time = datetime.datetime.strptime(config_dict['testTime'], '%Y%m%d%H')

    train_list = []
    val_list = []
    print('加载从{}到{}之间的数据集，其中{}时间到{}时间作为测试集'.format(st, et, test_time, et))
    while st <= et:
        line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
        # 由于数据不全 所以需要校验数据的完整
        if time_data_iscomplete(line, config_dict):
            if st >= test_time:
                val_list.append(line.rstrip('\n').rstrip('\r\n'))
            else:
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
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)


    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16))
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])
    # eval
    model_eval_valdata = Model_eval(config_dict)

    for epoch in range(config_dict['EpochNum']):
        # train_calparams_epoch = Cal_params_epoch()
        for i, (X, y) in enumerate(train_loader):
            wrf, obs, npyFilepath = X

            # print('wrf.shape={},obs.shape={}'.format(wrf.shape, obs.shape))
            # wrf.shape=torch.Size([4, 未来3小时, 159, 159, 29]),obs.shape=torch.Size([4, 历史6小时, 159, 159, 1])
            label = y
            wrf = wrf.to(config_dict['Device'])

            obs = obs.to(config_dict['Device'])

            label = label.to(config_dict['Device'])

            pre_frames, h = model(wrf, obs)

            # backward
            optimizer.zero_grad()
            loss = criterion(torch.flatten(pre_frames), torch.flatten(label))
            loss.backward()

            # update weights
            optimizer.step()

            print('pre  有{}个, label  有{}个, obs 有{}个,wrf ={}, npyFilepath={}'.format(torch.sum(pre_frames>0), torch.sum(label>0)
                  , torch.sum(obs > 0), torch.sum(wrf > 0), npyFilepath))

            print('TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}'.format(epoch, i+1, len(train_loader), loss.item()))
            # pod, far, ts, ets = train_calparams_epoch.cal_batch(label, pre_frames)
            # sumpod, sumfar, sumts, sumets = train_calparams_epoch.cal_batch_sum(label, pre_frames)
            # info = 'TRAIN INFO: epoch:{} ({}/{}) loss:{:.5f}\nPOD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}\nsumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}\n'\
            #     .format(epoch, i+1, len(train_loader), loss.item(), pod, far, ts, ets, sumpod, sumfar, sumts, sumets)
            # print(info)

        model_eval_valdata.eval(val_loader, model, epoch)

# 生产原始训练数据,主要工作室生成各个时间段内的闪电npy文件 flag代表是否覆盖生成，默认为否
def init_old_data(config_dict,flag = False):

    # generator light grid
    if os.path.exists(config_dict['TruthFileDirGrid']) and len(glob.glob(config_dict['TruthFileDirGrid']+'*')) != 0 and not flag:
        print('Light grid data existed')
    else:
        from ConvertToGird import LightingToGird
        print('Converting light data to grid...')
        light_grid_generator = LightingToGird(config_dict)
        if not os.path.exists(config_dict['TruthFileDirGrid']):
            os.makedirs(config_dict['TruthFileDirGrid'])

        st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H%M')
        et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H%M')
        tt = st
        datetimelist = []
        while (tt < et):
            datetimelist.append(tt)
            tt += datetime.timedelta(hours=1)
        for dt in datetimelist:
            truthfilepath = os.path.join(config_dict['TruthFileDir'], dt.strftime('%Y'), dt.strftime('%Y_%m_%d') + '.txt')
            if not os.path.exists(truthfilepath):
                print('Lighting data file `{}` not exist!'.format(truthfilepath))
                continue
            grid = light_grid_generator.getPeroid1HourGridFromFile(truthfilepath, dt)
            dt_str = dt.strftime('%Y%m%d%H%M')
            truthgridfilename = dt_str + '_truth'
            truthgridfilepath = config_dict['TruthFileDirGrid'] + truthgridfilename
            if not os.path.exists(truthgridfilepath) or flag:
                np.save(truthgridfilepath + '.npy', grid)
                print('{}_truth.npy generated successfully'.format(dt_str))
            else:
                print('{}.npy已经存在并且模式为不覆盖'.format(truthgridfilepath))


    # Constructing set automatically
    if os.path.exists('TrainCase.txt') and os.path.exists('ValCase.txt'):
        print('set list existed')
    else:
        from ConstructSet import constructSet
        print('Constructing train and val set automatically...')
        constructSet(config_dict)

    if not os.path.isdir(config_dict['ModelFileDir']):
        os.makedirs(config_dict['ModelFileDir'])

    if not os.path.isdir(config_dict['RecordFileDir']):
        os.makedirs(config_dict['RecordFileDir'])

    print('初始化数据已经生产完毕,现在可以开始训练了')




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    config_dict = read_config()
    #
    # init_old_data(config_dict)

    # #train
    DoTrain(config_dict)
    # moedl = selectModel(config_dict)


    # model = selectModel(config_dict)
    #
    # for name, param  in model.named_parameters():
    #     print(name)


