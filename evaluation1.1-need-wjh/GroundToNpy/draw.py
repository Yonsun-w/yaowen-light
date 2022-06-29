import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os


# 传入的是 0 1 矩阵
def drawImage(arr, name):
    arr *= 255
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save(name)
    # b = np.load('./npy/adtd_2015_05_17_09_00.npy') 766428
    # b = np.load('./npy/adtd_2015_05_17_18_00.npy') 1552237


if __name__ == '__main__':
    name = 'adtd_2015_05_17_12_00'

    arr = np.load('./npy/{}.npy'.format(name))

    drawImage(arr,'/有病吧')
    print('有病')
    arr *= 255
    arr = arr.astype(np.uint8)
    im = Image.fromarray(arr)
    im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
    im.save('./npy/{}.png'.format(name))


    print('ok')
    # b = np.load('./npy/adtd_2015_05_17_09_00.npy') 766428
    # b = np.load('./npy/adtd_2015_05_17_18_00.npy') 1552237


