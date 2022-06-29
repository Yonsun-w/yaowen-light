#-*- coding:utf-8 -*-
import os
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from typing import Any


def alter_configtime(time):
    ConfigFilePath = 'config_pre'
    config_info = {}
    with open(ConfigFilePath) as file1:
        message=''
        for line in file1:
            line = line.rstrip('\n')
            line = line.rstrip('\r\n')
            item = line.split('=')
            key = item[0]
            if key == 'Datetime':
                config_info[key] = item[1]
                line=line.replace(item[1],time)
            message = message + (line+'\r\n')
            
    with open(ConfigFilePath,'w') as file2:
        file2.write(message)

    return config_info

def job(text):
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))  # type: Any
    split_time = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    str = ('python main.py')
    config_time=alter_configtime(split_time)
    if os.system(str) == 0:
      print("calling success...")
    print('{} --- {}'.format(text, t))


scheduler = BlockingScheduler()
# 在每天22点，每隔 1分钟 运行一次 job 方法
# scheduler.add_job(job, 'cron', hour=16, minute='*/5', args=['Prediction Execute'])
# 在每周一至每周五，8-20点的每分钟(工作时间)执行任务
# scheduler.add_job(job, 'cron', day_of_week='0-6', minute='*/1',hour='8-23', args=['Prediction Execute'])
# 每天隔一个小时执行一次
scheduler.add_job(job, 'cron',hour='0-23', args=['Prediction Execute'])


# 在每天22和23点的25分，运行一次 job 方法
#scheduler.add_job(job, 'cron', hour='22-23', minute='25', args=['job2'])

scheduler.start()
