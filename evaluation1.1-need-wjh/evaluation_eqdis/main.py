import os
import numpy as np
import pandas as pd
from config import read_config
from readFiles import EvalData
from scores import Cal_params_neighbor
# from pandas import datetime
import datetime


def checkData(time, true_file_grid, pre_file, pre_equal_distance, pre_duration, time_step):
    is_predata = True
    is_obsdata = True
    is_predis_data = True
    for i in range(0, pre_duration // time_step):
        pre_path = os.path.join(pre_file, '{}_h{}.dat'.format(time.strftime('%Y%m%d%H%M'), i))
        if not os.path.exists(pre_path):
            print("not exits pre_path = ", pre_path)
            is_predata = False
        predis_path = os.path.join(pre_equal_distance, '{}_h{}.npy'.format(time.strftime('%Y%m%d%H%M'), i))
        if not os.path.exists(predis_path):
            is_predis_data = False
        obs_path = os.path.join(true_file_grid, 'RealDF{}_60.DAT'.format(time.strftime('%Y%m%d%H%M')))
        if not os.path.exists(obs_path):
            is_obsdata = False
        time += datetime.timedelta(minutes=time_step)
    print(is_predata, is_predis_data, is_obsdata)
    return is_predata and is_obsdata and is_predis_data


def main(config_dict):

    eval_results = pd.DataFrame(columns=['Time', 'Time Period', 'Threshold', 'Neighborhood Range'] + config_dict['EvaluationMethod'])

    pre_duration = config_dict['PreDuration']

    time_step = config_dict['TimeStep']

    ##这三个参数用来替代path obs pre pre_dis
    true_file_grid = config_dict['TrueFileGrid']

    pre_file = config_dict['preFile']

    pre_equal_distance = config_dict['preEqualDistance']

    st = datetime.datetime.strptime(config_dict['StartTime'], '%Y%m%d%H%M')

    et = datetime.datetime.strptime(config_dict['EndTime'], '%Y%m%d%H%M')
    all_available_time = []
    while st <= et:
        # if checkData(st, path, pre_duration, time_step):
        if checkData(st, true_file_grid, pre_file, pre_equal_distance, pre_duration, time_step):
            all_available_time.append(st.strftime('%Y%m%d%H%M'))
        else:
            print('{} data is not complete.'.format(st.strftime('%Y%m%d%H%M')))
        st += datetime.timedelta(hours=1)

    eval_results.to_csv('result.csv', header=True, index=False)

    print("all_available_time is", all_available_time)
    for time in all_available_time:
        for ptl in config_dict['PreTimeLimit']:
            print("enter,and ptl = ", ptl)
            print("config_dict['Threshold']", config_dict['Threshold'])
            for threshold in config_dict['Threshold']:
                print("time = ",time," ptl = ",ptl ,"time_step",time_step,"threshold",threshold)
                p = EvalData(true_file_grid, pre_file, pre_equal_distance, time, ptl, time_step, threshold,)

                for nbh in config_dict['NeighborhoodRange']:
                    # print('\nTime = {}  start&end = [{},{})  Threshold = {}  Neighborhood = {}'.
                    #       format(time, ptl[0], ptl[1], threshold, nbh))
                    cal = Cal_params_neighbor(neighbor_size=nbh)
                    scores = cal.cal_params_ones(p.obs_data, p.pre_data_dis)
                    results_dict = {}
                    results_dict['Time'] = datetime.datetime.strptime(time, '%Y%m%d%H%M')
                    results_dict['Time Period'] = ptl
                    results_dict['Threshold'] = threshold
                    results_dict['Neighborhood Range'] = nbh
                    for name in config_dict['EvaluationMethod']:
                        # print(name, scores[name])
                        results_dict[name] = scores[name]
                    eval_results = eval_results.append(results_dict, ignore_index=True)
                    # wjh 改 将结果实时输出 MODE = 'a' 是追加的意思 按照每个邻居去输出
                eval_results.to_csv('result.csv', mode='a', header=False, index=False)
        print(time, '时间内所有数据已输入进result.csvs，请查看')
    # wjh 改 将结果实时输出
    # eval_results.to_csv('result.csv', index=False)


if __name__ == "__main__":
    config_dict = read_config()
    main(config_dict)