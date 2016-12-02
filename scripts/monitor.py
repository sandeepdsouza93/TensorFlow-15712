import argparse
import sys
import subprocess
import os
import time

# CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--name",help="Specify basename for vms (dont use '_' )")
parser.add_argument("--num_workers", help="Specify number of workers",type=int)
parser.add_argument("--num_parameter_servers", help="Specify number of parameter servers",type=int)
parser.add_argument("--train_steps",help="Specify number of training steps",type=int)
args = parser.parse_args()


# setting up parameters
name = args.name
num_workers = args.num_workers
num_ps = args.num_parameter_servers
train_steps = args.train_steps

#create folder to store output for this experiment
subprocess.call('mkdir ' + name + '_data',shell=True)

progress_ctr = 0
stall_ctr = 0
checkpoint = 0
# code to monitor progress
temp_main = ''
while('Training elapsed' not in temp_main):
        time.sleep(30) #check periodically 
        subprocess.call('ssh root@' + name + '-' + str(num_ps) + ' "tail /mnt/train_output.log" > ' + name + '_data/progress.txt',shell=True)
        f = open(name + '_data/progress.txt','r')
        lines = f.readlines()
        temp_main = lines[len(lines) - 1].strip('\n')
        print(temp_main)
        if 'Training elapsed' in temp_main:
                break
        line = temp_main.split('\t')
        progress_ctr = int(line[2])
        progress_percent = float(progress_ctr) / float(train_steps)
        print('progress = ' + str(progress_percent*100))
        f.close()
        if progress_ctr - checkpoint > 1000:
                checkpoint = progress_ctr
                subprocess.call('ssh root@' + name + '-0 "vnstat" > ' + name + '_data/temp_nw_stat.txt',shell=True)
                f = open(name + '_data/temp_nw_stat.txt','r')
                lines = f.readlines()
                if len(lines) >= 3:
                        temp = lines[len(lines) - 3].strip('\n')
                        line = temp.split(' ')
                        g = open(name + '_data/bigdata_BW_stats.txt','a')
                        g.write('@' + str(checkpoint) + ' ' + line[len(line) - 2] + ' ' + line[len(line) - 1] + '\n')
                        g.close()
                        f.close()
                        #subprocess.call('ssh root@' + name + '-0 "rm /var/lib/vnstat/eth0"',shell=True)
                        #subprocess.call('ssh root@' + name + '-0 "vnstat -u -i eth0"',shell=True) 
                        #the previous commands resets vnstat's database
                else:
                        f.close()

# code to check if other workers are done
for worker in range(1,num_workers):
        temp_main = ''
        while('Training elapsed' not in temp_main):
                time.sleep(30) #check periodically 
                subprocess.call('ssh root@' + name + '-' + str(num_ps + worker) + ' "tail /mnt/train_output.log" > ' + name + '_data/progress.txt',shell=True)
                f = open(name + '_data/progress.txt','r')
                lines = f.readlines()
                temp_main = lines[len(lines) - 1].strip('\n')
                if 'Training elapsed' in temp_main:
                        break
                line = temp_main.split('\t')
                progress_ctr = int(line[2])
                progress_percent = float(progress_ctr) / float(train_steps)
                print('progress for worker ' + str(worker) + ' = ' + str(progress_percent*100))
                f.close()


# code to copy stuff to bigdata after completion
# a subdir for checkpoints
subprocess.call('mkdir ' + name + '_data/checkpoints',shell=True)
subprocess.call('scp -r root@' + name + '-' + str(num_ps) + ':/mnt/checkpoint* ' + name + '_data/checkpoints', shell=True)
subprocess.call('scp -r root@' + name + '-0:/mnt/BW_stats.txt ' + name + '_data/', shell=True)


# for each worker save the train.log and <name>_out.txt
for i in range(num_workers):
        subprocess.call('mkdir ' + name + '_data/workerinfo_' + str(i),shell=True)
        subprocess.call('scp root@' + name + '-' + str(num_ps + i) + ':/mnt/train_output.log ' + name + '_data/workerinfo_' + str(i), shell=True)
        subprocess.call('scp root@' + name + '-' + str(num_ps + i) + ':/mnt/*out.txt ' + name + '_data/workerinfo_' + str(i), shell=True)
        # compute the accuracy at different checkpoints
        subprocess.call('ssh root@' + name + '-' + str(num_ps + i) + ' "python /root/tensorflow/tensorflow/models/image/cifar10_new/cifar10_replica_eval.py --trained_steps 59999 > /mnt/accuracy_with_steps_worker' + str(i) + '.txt"',shell=True)
