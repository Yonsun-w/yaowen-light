import numpy as np
import datetime
import os
#from netCDF4 import Dataset
import struct
import cv2 as cv

np.set_printoptions(threshold=np.inf)

from scores import Cal_params_neighbor


class EvalData(object):
    def __init__(self, true_file_grid, ground_file_path, pre_file, pre_equal_distance, time, pre_timelimit, time_step, threshold=0.5):
        self.threshold = threshold

        # #
        # with open(os.path.join(path, 'pre', '{}_h0.dat'.format(time)), 'rb') as f:
        #     data = f.read()
        #     self.lon_left = struct.unpack('f', data[16:20])[0]
        #     self.lat_top = struct.unpack('f', data[20:24])[0]
        #     self.lon_center = struct.unpack('f', data[24:28])[0]
        #     self.lat_center = struct.unpack('f', data[28:32])[0]
        #     self.lon_step = struct.unpack('f', data[32:36])[0]
        #     self.lat_step = struct.unpack('f', data[36:40])[0]
        #     self.x = struct.unpack('i', data[40:44])[0]
        #     self.y = struct.unpack('i', data[44:48])[0]
        # # print(self.lon_left, self.lat_top, self.lon_center, self.lat_center, self.lon_step, self.lat_step, self.x, self.y)


        time = datetime.datetime.strptime(time, '%Y%m%d%H%M')

        time += datetime.timedelta(minutes=pre_timelimit[0])
        self.pre_data = np.zeros([159, 159])

        self.obs_data = np.zeros([149, 201])

        self.pre_data_dis = np.zeros([159, 159])
        for i in range(pre_timelimit[0] // time_step, pre_timelimit[1] // time_step):

            pre_path = os.path.join(pre_file, '{}_h{}.dat'.format(time.strftime('%Y%m%d%H%M'), i))

            obs_path = os.path.join(true_file_grid, 'RealDF{}_60.DAT'.format(time.strftime('%Y%m%d%H%M')))

            pre_dis_path = os.path.join(pre_equal_distance, '{}_h{}.npy'.format(time.strftime('%Y%m%d%H%M'), i))

            # self.pre_data += self._loadPreData_timestep(pre_path) #todo 这里为什么注解掉了？

            self.obs_data += self._loadObsData_timestep(obs_path)

            self.pre_data_dis += self._loadPredis_Data_timestep(pre_dis_path)

            time += datetime.timedelta(minutes=time_step)

        self.pre_data[self.pre_data > 1] = 1

        self.obs_data[self.obs_data > 1] = 1

        self.obs_data = self._disResize(self.obs_data)

        self.pre_data_dis[self.pre_data_dis > 1] = 1

        ground_path = os.path(ground_file_path, 'adtd' + time.strftime('_%Y_%m_%d_%H_%M') + '.npy')

        if not os.path.exists(ground_path):
            print('{}时间内没有地闪数据引入观测'.format(ground_path))
        else:
            self.obs_data += np.load(ground_path)


        # show(self.pre_data)
        # show(self.obs_data)
        # show(self.pre_data_dis)

    # def _loadPreData_timestep(self, path):
    #     # WRF(jingweidu)\PRED(juli) nc,npy dengjuli ->159x159
    #     pre = []
    #     with open(path, 'rb') as f:
    #         data = f.read()
    #         head = 23 * 4
    #         # m = struct.unpack('i', data[40:44])[0]
    #         # n = struct.unpack('i', data[44:48])[0]
    #         for i in range(head, len(data)):
    #             tmp = struct.unpack('c', data[i:i + 1])
    #             tmp = ord(tmp[0])
    #             pre.append(tmp)
    #     pre = pre / 100
    #     pre[pre > self.threshold] = 1
    #     pre[pre < 1] = 0
    #     return pre

    def _loadPredis_Data_timestep(self, path):
        # PRED(juli) npy.shaper[159x159] dengjuli
        pre = np.load(path)
        pre = pre / 100
        pre[pre > self.threshold] = 1
        pre[pre < 1] = 0
        return pre

    def _loadObsData_timestep(self, path):
        # lighting.txt jingweidu->juli jingwei->159x159
        obs = []
        with open(path, 'rb') as f:
            data = f.read()
            head = 23 * 4
            # m = struct.unpack('i', data[40:44])[0]
            # n = struct.unpack('i', data[44:48])[0]
            for i in range(head, len(data), 2):
                tmp = struct.unpack('h', data[i:i + 2])
                tmp = int(tmp[0])
                # print(tmp)
                obs.append(tmp)
        obs = np.array(obs).reshape([149, 201])
        return obs

    def _disResize(self, img):
        dst = cv.resize(img, (159, 159))
        obs = np.array(dst)
        obs[obs >= 0.5] = 1
        obs[obs < 0.5] = 0
        return obs

    # def _loadObsData_timestep(self, path):
    #     # lighting.txt jingweidu->juli jingwei->159x159
    #     obs = np.load(path)
    #     obs = np.array(obs).reshape([159, 159])
    #     return obs


def show(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # loadPreData()
    path_obs = '/Users/yonsun/gitTest/light_train/pre-eval/evaluation_eqdis/input_examples/obs'
    path_pre = '/Users/yonsun/gitTest/light_train/pre-eval/evaluation_eqdis/input_examples/pre'

    path_dis = '/Users/yonsun/gitTest/light_train/pre-eval/evaluation_eqdis/input_examples/pre_dis'

    # path = '/home/pengqingjie/Pytorch/LightNet_AMS/evaluation_eqdis/input_examples'
    # time = '202005201800'
    time = '202005211000'
    # # # 这里后边的几个参数不传 无法运行 但是原始的也没传入呀
    ptl = (0, 360)
    time_step = 60
    threshold = 0.1
    p = EvalData(path_obs, path_pre, path_dis, time, ptl, time_step, threshold)
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
