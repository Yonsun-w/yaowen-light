# -*- coding: utf-8 -*-
import FileWrite
import config


if __name__ == "__main__":
    config = config.read_config()

    file_write = FileWrite.TxtTrueFiletoGrid(config)

    if config['nc'] == 1 :
        file_write.all_trueFile_txt_to_wrf()

    if config['np'] == 1 :
        file_write.all_trueFile_txt_to_npy()

