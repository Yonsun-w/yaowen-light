
def read_config():
    ConfigFilePath = 'config_pre'
    config_info = {}
    with open(ConfigFilePath) as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')
            item = line.split('=')
            key = item[0]
            if key == 'ModelFilePath':
                config_info[key] = item[1]
            elif key == 'LightOnlyModelFilePath':
                config_info[key] = item[1]
            elif key == 'WRFOnlyModelFilePath':
                config_info[key] = item[1]
            elif key == 'LatlonFilePath':
                config_info[key] = item[1]
            elif key == 'NetName':
                config_info[key] = item[1]
            elif key == 'ResultSavePath':
                config_info[key] = item[1]
            elif key == 'ForecastHourNum':
                config_info[key] = int(item[1])
            elif key == 'TruthHistoryHourNum':
                config_info[key] = int(item[1])
            elif key == 'WRFFileDir':
                config_info[key] = item[1]
            elif key == 'TruthFileDirGrid':
                config_info[key] = item[1]
            elif key == 'VisResultFileDir':
                config_info[key] = item[1]
            elif key == 'IniFileDir':
                config_info[key] = item[1]
            elif key == 'GridRowColNum':
                config_info[key] = int(item[1])
            elif key == 'Datetime':
                config_info[key] = item[1]
            elif key == 'Threshold':
                config_info[key] = float(item[1])
            elif key == 'LatlonFilePath':
                config_info[key] = item[1]
            elif key == 'WRFChannelNum':
                config_info[key] = int(item[1])
            elif key == 'Device':
                config_info[key] = item[1]
            elif key == 'ResultDistanceSavePath':
                config_info[key] = item[1]
            else:
                print('no this item: {}'.format(key))
                assert False
    return config_info


if __name__ == "__main__":
    t = read_config()
    print(t)



