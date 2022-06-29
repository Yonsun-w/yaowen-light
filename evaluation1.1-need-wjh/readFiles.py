import numpy as np
import datetime
import os
from netCDF4 import Dataset
import struct
from scores import Cal_params_neighbor

class EvalData(object):
    def __init__(self, pre_path, obs_path, ground_path, need_ground, need_obs, time, pre_timelimit, time_step, threshold=0.5):
        self.threshold = threshold
        with open(os.path.join(pre_path, '{}_h0.dat'.format(time)), 'rb') as f:
            data = f.read()
            self.lon_left = struct.unpack('f', data[16:20])[0]
            self.lat_top = struct.unpack('f', data[20:24])[0]
            self.lon_center = struct.unpack('f', data[24:28])[0]
            self.lat_center = struct.unpack('f', data[28:32])[0]
            self.lon_step = struct.unpack('f', data[32:36])[0]
            self.lat_step = struct.unpack('f', data[36:40])[0]
            self.x = struct.unpack('i', data[40:44])[0]
            self.y = struct.unpack('i', data[44:48])[0]
        # print(self.lon_left, self.lat_top, self.lon_center, self.lat_center, self.lon_step, self.lat_step, self.x, self.y)

        time = datetime.datetime.strptime(time, '%Y%m%d%H%M')
        time += datetime.timedelta(minutes=pre_timelimit[0])
        self.pre_data = np.zeros([self.y, self.x])
        self.obs_data = np.zeros([self.y, self.x])
        for i in range(pre_timelimit[0] // time_step, pre_timelimit[1] // time_step):
            p_path = os.path.join(pre_path, '{}_h{}.dat'.format(time.strftime('%Y%m%d%H%M'), i))
            o_path = os.path.join(obs_path, 'RealDF{}_60.DAT'.format(time.strftime('%Y%m%d%H%M')))
            self.pre_data += self._loadPreData_timestep(p_path)

            # 算上真实数据
            if need_obs == 1:
                self.obs_data += self._loadObsData_timestep(o_path)
            # 算上地闪数据
            if need_ground == 1:
                g_path = os.path.join(ground_path, 'obs', 'RealDF{}_60.DAT'.format(time.strftime('%Y%m%d%H%M')))
                self.obs_data += np.load(g_path)

            time += datetime.timedelta(minutes=time_step)




        self.pre_data[self.pre_data > 1] = 1
        self.obs_data[self.obs_data > 1] = 1
        # show(self.obs_data)
        # show(self.pre_data)

    def _loadPreData_timestep(self, path):
        pre = []
        with open(path, 'rb') as f:
            data = f.read()
            head = 23 * 4
            # m = struct.unpack('i', data[40:44])[0]
            # n = struct.unpack('i', data[44:48])[0]
            for i in range(head, len(data)):
                tmp = struct.unpack('c', data[i:i + 1])
                tmp = ord(tmp[0])
                pre.append(tmp)
        pre = np.array(pre).reshape([self.y, self.x])
        pre = pre / 100
        pre[pre > self.threshold] = 1
        pre[pre < 1] = 0
        return pre

    def _loadObsData_timestep(self, path):
        obs = []
        with open(path, 'rb') as f:
            data = f.read()
            head = 23 * 4
            # m = struct.unpack('i', data[40:44])[0]
            # n = struct.unpack('i', data[44:48])[0]
            for i in range(head, len(data), 2):
                tmp = struct.unpack('h', data[i:i + 2])
                tmp = int(tmp[0])
                obs.append(tmp)
        obs = np.array(obs).reshape([self.y, self.x])
        return obs


def show(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # loadPreData()
    path = 'C:\\Users\\Lenovo\\Desktop\\lighting_project\\evaluation\\input_examples2'
    time = '202005201800'
    p = EvalData(path, time)
    for i in range(5):
        cal = Cal_params_neighbor(neighbor_size=i)
        t = cal.cal_params_ones(p.obs_data, p.pre_data)
        print(t['FSS'], t['ETS'], t['N1'], t['N2'], t['N3'], t['N4'], t['N1'] + t['N2'] + t['N3'] + t['N4'])
    # def loadPreData(path):
    #     pre = []
    #     with open(path, 'rb') as f:
    #         data = f.read()
    #         x = struct.unpack('i', data[40:44])[0]
    #         y = struct.unpack('i', data[44:48])[0]
    #         head = 23 * 4
    #         m = struct.unpack('i', data[40:44])[0]
    #         n = struct.unpack('i', data[44:48])[0]
    #         for i in range(head, len(data)):
    #             tmp = struct.unpack('c', data[i:i + 1])
    #             tmp = ord(tmp[0])
    #             pre.append(tmp)
    #     pre = np.array(pre).reshape([y, x])
    #     return pre
    # show(loadPreData('C:\\Users\\Lenovo\\Desktop\\202005-09短时预报产品和实测数据\\对比实测结果\\RealDF202005191300_60.DAT'))
