# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datetime
from config import read_config
# from generator import DataGenerator, DataGeneratorOnlyObs, DataGeneratorOnlyWRF
from generator import Dataloader
from writefile import Writefile
from NpyConvertNc import createDistanceNc
# from newnc_generator import DataGenerator
from layers.ADSNet_model import ADSNet_Model
from layers.LightNet_model import LightNet_Model
from layers.OnlyObsNet_model import OnlyObsNet_Model
from layers.OnlyWRFNet_model import OnlyWRFNet_Model


def WriteINIFile(is_obsfile_exist, is_wrffile_exist, nc_filename, config_dict):
    '''
        [AI短时预报结果描述]
        闪电=正常
        模式产品=正常 
        模式产品文件名=2020-07-16_12.wrfvar.nc
        （或是
        闪电=不正常，无所用闪电数据文件
        模式产品=不正常，2020-07-16_12.wrfvar.nc不存在
    '''
    gap = 60
    with open(config_dict['IniFileDir'] + "%s_%s_%s.ini" % (config_dict['Datetime'], config_dict['ForecastHourNum'] * gap, gap), 'w') as file:
        file.write('[AI短时预报结果描述]\n')
        if is_obsfile_exist and is_wrffile_exist:
            file.write('闪电=%s\n' % '正常')
            file.write('模式产品=正常\n')
            file.write('模式产品文件名=%s\n' % nc_filename)
        elif is_wrffile_exist:
            file.write('闪电=%s\n' % '不正常，无所用闪电数据文件')
            file.write('模式产品=正常\n')
            file.write('模式产品文件名=%s\n' % nc_filename)
        elif is_obsfile_exist:
            file.write('闪电=%s\n' % '正常')
            file.write('模式产品=%s\n' % ('不正常，模式产品文件不存在'))
        else:
            file.write('闪电=%s\n' % '不正常，无所用闪电数据文件')
            file.write('模式产品=%s\n' % ('不正常，模式产品文件不存在'))


def selectModel(is_wrffile_exist, is_obsfile_exist, config_dict):
    # model
    if is_obsfile_exist and is_wrffile_exist:
        model_file = torch.load(config_dict['ModelFilePath'], map_location=torch.device(config_dict['Device']))
        ADSNet = ADSNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                             wrf_tra_frames=config_dict['ForecastHourNum'],
                             wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict)
        LightNet = LightNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                               wrf_tra_frames=config_dict['ForecastHourNum'],
                               wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict)
        if config_dict['NetName'] == 'ADSNet':
            ADSNet.load_state_dict(model_file)
            model = ADSNet.to(config_dict['Device'])
        elif config_dict['NetName'] == 'LightNet':
            LightNet.load_state_dict(model_file)
            model = LightNet.to(config_dict['Device'])
        else:
            model = None

    elif is_wrffile_exist:
        model = OnlyWRFNet_Model(wrf_tra_frames=config_dict['ForecastHourNum'],
                                 wrf_channels=config_dict['WRFChannelNum'], config_dict=config_dict)
        model.load_state_dict(torch.load(config_dict['WRFOnlyModelFilePath'], map_location=torch.device(config_dict['Device'])))
        model = model.to(config_dict['Device'])
    elif is_obsfile_exist:
        model = OnlyObsNet_Model(obs_tra_frames=config_dict['TruthHistoryHourNum'], obs_channels=1,
                                 pre_frames=config_dict['ForecastHourNum'], config_dict=config_dict)
        model.load_state_dict(torch.load(config_dict['LightOnlyModelFilePath'], map_location=torch.device(config_dict['Device'])))
        model = model.to(config_dict['Device'])
    else:
        print('Both WRF data and Light data are not found!')
        model = None
    return model

def DoPredict(config_dict):
    # 加载数据
    data = Dataloader(config_dict)
    # 查看wrf和obs文件是否完整
    is_wrffile_exist, is_obsfile_exist = data.getDataStates()
    # 选择模型
    model = selectModel(is_wrffile_exist, is_obsfile_exist, config_dict)
    if model == None:
        print('error: model file or data not found!')
        return

    wrf, obs = data.getData()


    pre_frames = model(wrf, obs)
    # 这里sigmoid一下
    pre_frames = torch.sigmoid(pre_frames)
    if config_dict['Threshold'] < 0:
        # probability output
        pre_frames = pre_frames * 100
    else:
        pre_frames[pre_frames > config_dict['Threshold']] = 1
        pre_frames[pre_frames < 1] = 0
    WriteINIFile(is_obsfile_exist, is_wrffile_exist, data.WRFFileName, config_dict)
    print('Successfully writing information to INI file!')
    output_file_writer = Writefile(config_dict)
    lon_min, lon_max, lat_min, lat_max = output_file_writer.getEdge()

    # 按小时来写入结果
    for hour_plus in range(config_dict['ForecastHourNum']):
        output_file_writer.writeResultFile(pre_frames[0, hour_plus, :, :, 0], hour_plus)

    # 写入等距离
    createDistanceNc(config_dict, lon_min, lon_max, lat_min, lat_max)

    #todo  nc文件time是否需要保留



if __name__ == "__main__":
    # Read in the configuration file and allocate GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
    config_dict = read_config()
    if not os.path.isdir(config_dict['IniFileDir']):
        os.makedirs(config_dict['IniFileDir'])
    if not os.path.isdir(config_dict['ResultSavePath']):
        os.makedirs(config_dict['ResultSavePath'])
    if not os.path.isdir(config_dict['ResultDistanceSavePath']):
        os.makedirs(config_dict['ResultDistanceSavePath'])

    DoPredict(config_dict)
