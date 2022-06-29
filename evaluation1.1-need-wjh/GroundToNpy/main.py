# -*- coding: utf-8 -*-
import FileWrite
import config
import datetime
import os
if __name__ == "__main__":
    config = config.read_config()



    dataTime = datetime.datetime.strptime(config['startTime'], "%Y%m%d%H%M%S")
    path = os.path.join(config['TruthFileDir'], 'adtd'+ dataTime.strftime('_%Y_%m_%d') + '.txt')

    if not os.path.exists(config['TruthFileDir']):
        assert print('您的输入路径{}错误'.format(config['TruthFileDir']))
    if not os.path.exists(config['output']):
        assert print('您的输出路径{}错误'.format(config['TruthFileDir']))

    file_write = FileWrite.TxtTrueFiletoGrid(config)

    if config['saveImage'] == 1:
        file_write.all_trueFile_txt_to_png()

    if config['npy'] == 1:
        file_write.all_trueFile_txt_to_npy()

    if config['nc'] == 1:
        file_write.all_trueFile_txt_to_nc()



