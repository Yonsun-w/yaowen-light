import os
import numpy as np
import datetime
from config import read_config
from readFiles import EvalData
from scores import Cal_params_neighbor
import pandas as pd

def checkData(time, pre,obs, pre_duration, time_step):
    is_predata = True
    is_obsdata = True

    for i in range(0, pre_duration // time_step):
        pre_path = os.path.join(pre, '{}_h{}.dat'.format(time.strftime('%Y%m%d%H%M'), i))
        if not os.path.exists(pre_path):
            is_predata = False
        obs_path = os.path.join(obs, 'RealDF{}_60.DAT'.format(time.strftime('%Y%m%d%H%M')))
        if not os.path.exists(obs_path):
            is_obsdata = False
        time += datetime.timedelta(minutes=time_step)
        # print('pre_path = {}'.format(pre_path))
        # print('obs_path = {}'.format(obs_path))
        # print('pre={},obs={}'.format(os.path.exists(pre_path), os.path.exists(obs_path)))

    return is_predata and is_obsdata

# 检测地闪数据是否完整 npy
def checkGround(time, ground_path, pre_duration, time_step):
    is_ground = True
    for i in range(0, pre_duration // time_step):
        ground_path = os.path.join(ground_path, 'adtd{}.npy'.format(time.strftime('_%Y_%m_%d_%H_%M')))
        if not os.path.exists(ground_path):
            is_ground = False
        time += datetime.timedelta(minutes=time_step)
    return is_ground



def main(config_dict):
    eval_results = pd.DataFrame(columns=['Time', 'Time Period', 'Threshold', 'Neighborhood Range'] + config_dict['EvaluationMethod'])
    # 预报产品路径
    pre_duration = config_dict['PreDuration']
    time_step = config_dict['TimeStep']

    ground_path = config_dict['GroundFilePath']
    obs_path = config_dict['TrueFileGrid']
    pre_path = config_dict['preFile']

    st = datetime.datetime.strptime(config_dict['StartTime'], '%Y%m%d%H%M')
    et = datetime.datetime.strptime(config_dict['EndTime'], '%Y%m%d%H%M')
    all_available_time = []
    while st <= et:
        if checkData(st, pre_path,obs_path, pre_duration, time_step):
            if config_dict['NeedGround'] == 0 or (config_dict['NeedGround'] == 1 and checkGround(st,ground_path,pre_duration,time_step)):
                all_available_time.append(st.strftime('%Y%m%d%H%M'))
            else:
                print('{} time ,地闪数据不完整'.format(st.strftime('%Y%m%d%H%M')))
        else:
            print('{} data is not complete.'.format(st.strftime('%Y%m%d%H%M')))
        st += datetime.timedelta(hours=1)


    flag = True

    for time in all_available_time:
        for ptl in config_dict['PreTimeLimit']:
            for threshold in config_dict['Threshold']:
                p = EvalData(pre_path,obs_path,ground_path,config_dict['NeedGround'], config_dict['NeedObs'], time, ptl, time_step, threshold)
                for nbh in config_dict['NeighborhoodRange']:
                    # print('\nTime = {}  start&end = [{},{})  Threshold = {}  Neighborhood = {}'.
                    #       format(time, ptl[0], ptl[1], threshold, nbh))
                    cal = Cal_params_neighbor(neighbor_size=nbh)
                    scores = cal.cal_params_ones(p.obs_data, p.pre_data)
                    results_dict = {}
                    results_dict['Time'] = datetime.datetime.strptime(time, '%Y%m%d%H%M')
                    results_dict['Time Period'] = ptl
                    results_dict['Threshold'] = threshold
                    results_dict['Neighborhood Range'] = nbh
                    for name in config_dict['EvaluationMethod']:
                        # print(name, scores[name])
                        results_dict[name] = scores[name]
                    eval_results = eval_results.append(results_dict, ignore_index=True)
                print('time={},PreTimeLimit={},threshold={},的数据已经输入完毕'.format(time, ptl, threshold))
                if flag :
                    flag = False
                    eval_results.to_csv('result.csv', mode='a', header=True, index=False)
                else :
                    eval_results.to_csv('result.csv', mode='a', header=False, index=False)

    print('时间内所有数据已输入进result.csvs，请查看')



if __name__ == "__main__":
    config_dict = read_config()
    main(config_dict)



