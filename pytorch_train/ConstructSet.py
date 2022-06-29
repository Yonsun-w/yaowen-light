# -*- coding: utf-8 -*-
import numpy as np
import os
import datetime

def getTimePeriod(dt):
    time = dt.strftime("%H:%M:%S")
    hour = int(time[0:2])
    if 0 <= hour < 6: nchour = '00'
    elif 6 <= hour < 12: nchour = '06'
    elif 12 <= hour < 18: nchour = '12'
    elif 18 <= hour <= 23: nchour = '18'
    else: print('error')
    delta_hour = hour - int(nchour)
    return nchour, delta_hour

def constructSet(config_dict):
    st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H%M')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H%M')
    tt = st
    datetimelist = []
    available_list = []
    while (tt < et):
        datetimelist.append(tt)
        tt += datetime.timedelta(hours=1)
    for ddt in datetimelist:
        # check historical lightning data  -----------------------------------------------------
        flag = 1  # available
        totalgrid = np.zeros(shape=(config_dict['GridRowColNum'] * config_dict['GridRowColNum'], ), dtype=np.int)
        for hour_plus in range(-config_dict['TruthHistoryHourNum'], config_dict['ForecastHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            dt_str = dt.strftime('%Y%m%d%H%M')
            tFilePath = config_dict['TruthFileDirGrid'] + dt_str + '_truth' + '.npy'
            if not os.path.exists(tFilePath):
                flag = 0  # unavailable
                break
            else:
                tgrid = np.load(tFilePath)
                totalgrid += tgrid
        #  remove the forecast case in which all 6/12 hours have no lightning ----
        # totalgrid[totalgrid>0] = 1
        # if np.sum(totalgrid) < int(mn*mn/100):
        #     flag = 0
        # ------------------------------------------------------------------------
        if flag == 0:
            print('Fail: lightning data (time:{})'.format(ddt))
            continue
        # --------------------------------------------------------------------------------------
        # check WRF data
        for hour_plus in range(config_dict['ForecastHourNum']):
            dt = ddt + datetime.timedelta(hours=hour_plus)
            utc = dt + datetime.timedelta(hours=-8)
            ft = utc + datetime.timedelta(hours=(-6))
            nchour, delta_hour = getTimePeriod(ft)
            delta_hour += 6
            # ncFileDir = NCDir + 'gfs.' + ft.strftime("%Y%m%d") + '/' + 'gfs.' + ft.strftime("%Y%m%d%M") + '/'
            ncFilepath = config_dict['WRFFileDir'] + ft.strftime("%Y-%m-%d") + '_' + nchour + '.wrfvar.nc'
            if not os.path.exists(ncFilepath):
                flag = 0
                break
        if flag == 0:
            print('Fail: WRF data (time:{})'.format(ddt))
            continue
        print('available case: time={}'.format(ddt))
        available_list.append(ddt.strftime('%Y%m%d%H%M'))


    np.random.shuffle(available_list)
    available_num = len(available_list)
    validate_ratio = 0.2
    train_list = available_list[:int(available_num*(1-validate_ratio))]
    val_list = [item for item in available_list if item not in train_list]
    if len(train_list) > 0:
        with open('TrainCase.txt', 'w') as file:
            for item in train_list:
                file.write(item + '\r\n')
    if len(val_list) > 0:
        with open('ValCase.txt', 'w') as file:
            for item in val_list:
                file.write(item + '\r\n')

if __name__ == "__main__":
    a = np.array([i for i in range(10)])
    b = np.random.shuffle(a)
    print(a, b)