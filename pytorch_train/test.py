
import os
import numpy as np
import datetime
from config import read_config
if __name__ == '__main__':
    print('这个文件是用来获得训练和验证的集合的')
    config_dict = read_config()
    train_path = 'TrainCase.txt'
    num = 0
    count = 0
    with open(train_path) as file:
        for line in file:
            for hour_plus in range(config_dict['ForecastHourNum']):
                d_time = line.rstrip('\n').rstrip('\r\n')
                ddt = datetime.datetime.strptime(d_time, '%Y%m%d%H%M')
                dt = ddt + datetime.timedelta(hours=hour_plus)
                tFilePath = config_dict['TruthFileDirGrid'] + dt.strftime('%Y%m%d%H%M') + '_truth' + '.npy'
                if not os.path.exists(tFilePath) :
                    continue
                truth_grid = np.load(tFilePath)
                truth_grid[truth_grid > 1] = 1
                count +=1
                if np.sum(truth_grid) > 100 :
                    num+=1
                    print(d_time)

    print('一共有={}个，可用={}'.format(count,num))


