
def read_config():
    ConfigFilePath = 'config'
    config_info = {}
    with open(ConfigFilePath) as file:
        for line in file:
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')
            item = line.split('=')
            key = item[0]
            if key == 'preFile':
                config_info[key] = item[1]
            elif key == 'TrueFileGrid':
                config_info[key] = item[1]
            elif key == 'preEqualDistance':
                config_info[key] = item[1]
            elif key == 'ObsFilePath':
                config_info[key] = item[1]
            elif key == 'LatlonFilePath':
                config_info[key] = item[1]
            elif key == 'StartTime':
                config_info[key] = item[1]
            elif key == 'EndTime':
                config_info[key] = item[1]
            elif key == 'Threshold':
                tmp = []
                for i in item[1].split(','):
                    tmp.append(float(i))
                config_info[key] = tmp
            elif key == 'TimeStep':
                config_info[key] = int(item[1])
            elif key == 'PreDuration':
                config_info[key] = int(item[1])
            elif key == 'PreTimeLimit':
                tmp = []
                for i in item[1].split(','):
                    i = i.split('/')
                    tmp.append((int(i[0]), int(i[1])))
                config_info[key] = tmp
            elif key == 'NeighborhoodRange':
                tmp = []
                for i in item[1].split(','):
                    tmp.append(int(i))
                config_info[key] = tmp
            elif key == 'EvaluationMethod':
                tmp = []
                for i in item[1].split(','):
                    tmp.append(i)
                config_info[key] = tmp
            elif key == 'lonBegin':
                config_info[key] = item[1]
            elif key == 'lonEnd':
                config_info[key] = item[1]
            elif key == 'latBegin':
                config_info[key] = item[1]
            elif key == 'latEnd':
                config_info[key] = item[1]
            elif key == 'groundFile':
                config_info[key] = item[1]
            else:
                print('no this item: {}'.format(key))
                assert False
    return config_info


if __name__ == "__main__":
    t = read_config()
    print(t)



