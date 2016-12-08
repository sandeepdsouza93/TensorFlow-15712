#script to log network bandwidth periodically

import subprocess
import time
import sys

def FileSizeTooLarge():
        "function to check if the network "
        subprocess.call('ls -alh /mnt/CPU_stats.txt > out.txt',shell=True)
        f = open('out.txt','r')
        lines = f.readlines()
        if len(lines) != 0:
                if 'G' in lines[0]:
                        return 1
        return 0

time.sleep(180) # allow tensorflow to start     

while(FileSizeTooLarge() == 0):
        time.sleep(300)  #log every 5 minutes 
        subprocess.call('sar -u 5 5 >> /mnt/CPU_stats.txt',shell=True)
