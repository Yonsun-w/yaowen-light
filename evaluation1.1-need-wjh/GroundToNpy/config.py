
def read_config():
    ConfigFilePath = 'config'
    config_info = {}
    with open(ConfigFilePath) as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')
            item = line.split('=')
            key = item[0]
            if key == 'startTime':
                config_info[key] = item[1].replace('_', '')
            elif key == 'endTime':
                config_info[key] = item[1].replace('_', '')
            elif key == 'lonBegin':
                config_info[key] = float(item[1])
            elif key == 'lonEnd':
                config_info[key] = float(item[1])
            elif key == 'latBegin':
                config_info[key] = float(item[1])
            elif key == 'latEnd':
                config_info[key] = float(item[1])
            elif key == 'timeGap':
                config_info[key] = float(item[1])
            elif key == 'TruthFileDir':
                config_info[key] = item[1]
            elif key == 'lon_lat_Gap':
                config_info['latGap'] = float(item[1])
                config_info['lonGap'] = float(item[1])
            elif key == 'latGap':
                config_info[key] = float(item[1])
            elif key == 'Threshold':
                config_info[key] = int(item[1])
            elif key == 'edg':
                config_info[key] = int(item[1])
            elif key == 'output':
                config_info[key] = item[1]
            elif key == 'npy':
                config_info[key] = int(item[1])
            elif key == 'nc':
                config_info[key] = int(item[1])
            elif key == 'saveImage':
                config_info[key] = int(item[1])
            elif key == 'equal_dis':
                config_info[key] = int(item[1])
            else:
                print('no this item: {}'.format(key))
                assert False
    return config_info


if __name__ == "__main__":
    t = read_config()
    print(t)



