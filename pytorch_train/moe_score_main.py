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
        model_file = torch.load('MOE/moe_ads_model_maxETS.pkl', map_location=torch.device('cuda'))

    elif config_dict['NetName'] == 'LightNet':
        model = l2(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1, wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], row_col=config_dict['GridRowColNum']).to(config_dict['Device'])

        model_file = torch.load('MOE/moe_light_model_maxETS.pkl', map_location=torch.device('cuda'))
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
        model_file = torch.load('MOE/moe_model_maxETS.pkl', map_location=torch.device('cuda'))

    else:
        print('`{}` not support'.format(config_dict['NetName']))
        assert False
    model.load_state_dict(model_file)
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


def DoScore(config_dict):



    st = datetime.datetime.strptime(config_dict['testTime'], '%Y%m%d%H')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H')

    val_list = []
    while st <= et:
        line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
        # 由于数据不全 所以需要校验数据的完整
        if time_data_iscomplete(line,  WRFFileDir=config_dict['WRFFileDir'],ForecastHourNum=config_dict['ForecastHourNum'],
                                TruthFileDirGrid=config_dict['TruthFileDirGrid'], TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):

            val_list.append(line.rstrip('\n').rstrip('\r\n'))
        st += datetime.timedelta(hours=3)

    print('加载数据完毕，val{}测试集'.format(len(val_list)))

    val_data = DataGenerator(val_list, config_dict)
    val_loader = DataLoader(dataset=val_data, batch_size=config_dict['Batchsize'], shuffle=True, num_workers=0)

    # model
    model = selectModel(config_dict)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


    # loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(16))
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['LearningRate'])
    # eval
    model_eval_valdata = Model_eval(config_dict, False)

    for epoch in range(config_dict['EpochNum']):
        model_eval_valdata.eval(val_loader, model, epoch)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    config_dict = read_config()
    #
    # init_old_data(config_dict)

    # #train
    DoScore(config_dict)
    # moedl = selectModel(config_dict)


    # model = selectModel(config_dict)
    #
    # for name, param  in model.named_parameters():
    #     print(name)


