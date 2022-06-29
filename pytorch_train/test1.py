from moe_main import time_data_iscomplete
import datetime
from config import  read_config

if __name__ == '__main__':

    config_dict = read_config()

    # 直接从时间段自动加载
    train_list = []
    val_list = []



    st = datetime.datetime.strptime(config_dict['ScanStartTime'], '%Y%m%d%H')
    et = datetime.datetime.strptime(config_dict['ScanEndTime'], '%Y%m%d%H')
    # 验证集开始载入的时间
    test_time = datetime.datetime.strptime(config_dict['testTime'], '%Y%m%d%H')


    print('加载从{}到{}之间的数据集，其中{}时间到{}时间作为测试集'.format(st, et, test_time, et))
    print('ok')
    while st <= et:
        line = datetime.datetime.strftime(st, '%Y%m%d%H%M')
        # 由于数据不全 所以需要校验数据的完整
        if time_data_iscomplete(line, WRFFileDir=config_dict['WRFFileDir'],
                                ForecastHourNum=config_dict['ForecastHourNum'],
                                TruthFileDirGrid=config_dict['TruthFileDirGrid'],
                                TruthHistoryHourNum=config_dict['TruthHistoryHourNum']):
            if st >= test_time:
                print(line.rstrip('\n').rstrip('\r\n'))
                val_list.append(line.rstrip('\n').rstrip('\r\n'))
            else:
                train_list.append(line.rstrip('\n').rstrip('\r\n'))
                print(line.rstrip('\n').rstrip('\r\n'))
        st += datetime.timedelta(hours=1)







