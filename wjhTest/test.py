import datetime

import numpy as np
import torch
import os
import config
import datetime
from generator import getTimePeriod




def time_data_iscomplete(time_str, WRFFileDir, TruthFileDir):
    time_str = time_str.rstrip('\n')
    time_str = time_str.rstrip('\r\n')
    dt = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    is_complete = True
    # 首先检查wrf
    wrf_path = os.path.join(WRFFileDir, dt.strftime("%Y%m%d"))
    wrf_data = ['00','06','12','18']
    for s in wrf_data :
        if not os.path.isdir(os.path.join(wrf_path,s)):
            is_complete = False
        else :

            print(os.path.join(wrf_path,s, 'V.npy'))
            a = np.load(os.path.join(wrf_path,s, 'V.npy') )
            print(a[a!=0].sum())
    #检测obs真实数据
    obs_path = os.path.join(TruthFileDir,dt.strftime("%Y"))
    txt_path = os.path.join(obs_path, dt.strftime("%Y_%m_%d") + '.txt')
    if not os.path.exists(txt_path):
        is_complete = False
    return is_complete


def getDataPath():
    filePath = '/data/wenjiahua/light_data/ADSNet_testdata/light_data_ori_18-20'
    wrf_path = '/data/wenjiahua/light_data/ADSNet_testdata/WRF_data'
    save_path = '/data/wenjiahua/light_data/ADSNet_testdata/val.txt'
    f = open(save_path, 'w')
    start = datetime.datetime.strptime('20180101', '%Y%m%d')
    end = start + datetime.timedelta(days=700)
    tt = start
    while tt <= end :
        tt_str = datetime.datetime.strftime(tt,'%Y%m%d%H%M')

        if time_data_iscomplete(tt_str,wrf_path,filePath):
            tt_str = tt_str + '\n'

           # f.write(tt_str)
        tt += datetime.timedelta(hours=1)
    f.close()
    print('end')
def merge_npy():
    output = '/data/wenjiahua/light_data/ADSNet_testdata/merge_light_grid'
    intput = '/data/wenjiahua/light_data/ADSNet_testdata/true_npy'
    st = datetime.datetime.strptime('201507020000', '%Y%m%d%H%M')
    et = datetime.datetime.strptime('202012310000', '%Y%m%d%H%M')
    while st <= et:
        npy_name = datetime.datetime.strftime(st, '%Y%m%d%H%M') + '.npy'
        npy_path = os.path.join(intput, npy_name)
        name = os.path.join(output, datetime.datetime.strftime(st, '%Y%m%d%H%M')) + '_truth.npy'
        if not os.path.exists(name) and os.path.exists(npy_path):
            print('{},save'.format(name))
            grid = np.load(npy_path)
            np.save(name, grid)
        st += datetime.timedelta(hours=1)
    print('ok')

def count_zero():
    path = '/data/wenjiahua/light_data/ADSNet_testdata/true_npy'
    st = datetime.datetime.strptime('201507020000', '%Y%m%d%H%M')
    et = datetime.datetime.strptime('202001310000', '%Y%m%d%H%M')
    num = 0
    while st <= et:
        npy_name = datetime.datetime.strftime(st, '%Y%m%d%H%M') + '.npy'
        npy_path = os.path.join(path, npy_name)
        name = os.path.join(path, datetime.datetime.strftime(st, '%Y%m%d%H%M')) + '.npy'
        if os.path.exists(name) :
            a = np.load(name)
            if (np.sum(a) > 10) :
                print(np.sum(a))
                num+=1

        st += datetime.timedelta(hours=1)
    print(num)


def time_data_iscomplete(time_str):

    time_str = time_str.rstrip('\n')
    time_str = time_str.rstrip('\r\n')
    if time_str == '':
        return False
    ddt = datetime.datetime.strptime(time_str, "%Y%m%d%H%M")
    is_complete = True
    # 首先检查wrf
    wrf_path = os.path.join('WRFFileDir=/data/wenjiahua/light_data/ADSNet_testdata/WRF_data', ddt.strftime("%Y%m%d"))
    wrf_data = ['00', '06', '12', '18']
    for s in wrf_data :
        if not os.path.isdir(os.path.join(wrf_path,s)):
            is_complete = False

    #检测obs真实数据
    obs_path = os.path.join('/data/wenjiahua/light_data/ADSNet_testdata/merge_light_grid', ddt.strftime("%Y"))
    txt_path = os.path.join(obs_path, ddt.strftime("%Y_%m_%d") + '.txt')
    if not os.path.exists(txt_path):
        is_complete = False

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
        # 假如标签缺失 返回False
        if not os.path.exists(tFilePath):
            is_complete = False
        # 假如数据量太小(发生闪电次数不到20次) 也返回false
        else:
            grid = np.load(tFilePath)
            if np.sum(grid) < 20:
                print('tFilePath={}，sum={}'.format(tFilePath, np.sum(grid)))
                is_complete = False

    # read history observations
    for hour_plus in range(config_dict['TruthHistoryHourNum']):
        dt = ddt + datetime.timedelta(hours=hour_plus - config_dict['TruthHistoryHourNum'])
        tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth.npy'
        if not os.path.exists(tFilePath):
            is_complete = False
    return is_complete



if __name__ == '__main__':

    path = '/data/wenjiahua/light_data/ADSNet_testdata/merge_light_grid/201608201200_truth.npy'

    a = np.load(path)

    print(a.shape)
