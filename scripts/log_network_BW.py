#script to log network bandwidth periodically

import subprocess
import time
import sys

def FileSizeTooLarge():
        "function to check if the network "
        subprocess.call('ls -alh /mnt/BW_stats.txt > out.txt',shell=True)
        f = open('out.txt','r')
        lines = f.readlines()
        if len(lines) != 0:
                if 'G' in lines[0]:
                        return 1
        return 0

time.sleep(180) # allow tensorflow to start     

while(FileSizeTooLarge() == 0):
        time.sleep(1800)  #log every half hour
        subprocess.call('vnstat > /mnt/temp_nw_stat.txt',shell=True)
        f = open('/mnt/temp_nw_stat.txt','r')
        lines = f.readlines()
        if len(lines) >= 3:
                temp = lines[len(lines) - 3].strip('\n')
                line = temp.split(' ')
                g = open('/mnt/BW_stats.txt','a')
                g.write(line[len(line) - 2] + ' ' + line[len(line) - 1] + '\n')
                g.close()
                f.close()
        else:
                f.close()
