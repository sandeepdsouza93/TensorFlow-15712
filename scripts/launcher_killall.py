import argparse
import sys
import subprocess
import os
import time

# CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", help="Specify number of workers",type=int)
parser.add_argument("--num_parameter_servers", help="Specify number of parameter servers",type=int)
args = parser.parse_args()

# setting up parameters
name = 'candidat' 
num_workers = args.num_workers
num_ps = args.num_parameter_servers
async = args.async
num_sync = args.num_sync
train_steps = args.train_steps

## # creating the vms for ps and workers
## create_cmd = 'tashi createMany --basename ' + name + ' --cores 8 --memory 4096 --disks tensorflow15712-new,ext3-900GB.qcow2 --count ' \
## + str(num_workers + num_ps) + ' --hints nicModel=e1000'
## #TEMP: trying a nicModel of virtio instead of e1000
## subprocess.call('ssh tashi "' + create_cmd + '"',shell=True)
## subprocess.call('ssh tashi "tashi getMyInstances" > status.txt',shell=True)

#sleep for a while
#time.sleep(45)
print('woke up from sleep')
## subprocess.call('ssh tashi "tashi getMyInstances" > status.txt',shell=True)
## f = open('status.txt','r')
## lines = f.readlines()
## temp = lines[len(lines) - 1].strip('\n')
## f.close()

## #check status of file in a loop
## while 'Running' not in temp:
##      subprocess.call('ssh tashi "tashi getMyInstances" > status.txt',shell=True)
##      f = open('status.txt','r')
##      lines = f.readlines()
##      temp = lines[len(lines) - 1].strip('\n')
##      time.sleep(5) #snooze
##      f.close()

# specifying the ip and port num of ps
ps_str = ''
for i in range(num_ps):
        if i == num_ps - 1:
                ps_str += (name + '-' + str(i) + ':2500')
        if i != num_ps - 1:
                ps_str += (name + '-' + str(i) + ':2500,')

# specifying the ip and port num of workers
workers_str = ''
for i in range(num_workers):
        if i == num_workers - 1:
                workers_str += (name + '-' + str(i + num_ps) + ':2500')
        if i != num_workers - 1:
                workers_str += (name + '-' + str(i + num_ps) + ':2500,')

# first launching the parameter servers
for i in range(num_ps):
        # tensorflow python args
        if async == 'False':
                comm_ps = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="ps" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=True'
        if async == 'True':
                comm_ps = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="ps" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=False'

        #copy the network monitoring tool
        #subprocess.call('scp log_network_BW.py root@' + name + '-0:/root/tensorflow',shell=True)

        # commands to launch on each vm
        comm0 = 'export PATH=\"$PATH:$HOME/bin\";'
        comm1 = 'mount /dev/sdb2 /mnt; apt-get install vnstat;'
        comm2 = 'cd tensorflow; git pull base master;'
        comm3 = 'nohup python ' + comm_ps + ' >> /mnt/' + name + '_out.txt 2>> /mnt/' + name + '_stderr.txt &'

        comm = 'killall -9 python;rm /mnt/BW_stats.txt; rm /mnt/CPU_stats.txt;'

        # finally launching the ps instances
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i) + ' "' + comm + '"',shell=True)
        print('ssh -o StrictHostKeyChecking=no root@' + name + '-' + str(i) + ' "' + comm + '"')
#       time.sleep(3) #give time to launch the command
#       subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + '-' + str(i) + ' "nohup python /root/tensorflow/log_network_BW.py > /dev/null 2>&1 &"',shell=True)
        subprocess.call('echo "launched a ps"',shell=True)

# next launching the workers
for i in range(num_workers):
        # tensorflow python args
        if async == 'False':
                comm_worker = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="worker" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=True'
        if async == 'True':
                comm_worker = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="worker" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=False'

        #commands to launch on each vm
        comm0 = 'export PATH=\"$PATH:$HOME/bin\";'
        comm1 = 'mount /dev/sdb2 /mnt;'
        comm2 = 'cd tensorflow; git pull base master;'
        comm3 = 'nohup python ' + comm_worker + ' >> /mnt/' + name + '_out.txt 2>> /mnt/' + name + '_stderr.txt &'

        comm0 = 'killall -9 python;rm /mnt/CPU_stats.txt;'
        comm1 = 'rm /mnt/train_output.log;'
        comm2 = 'rm -rf /mnt/checkpoint*'

        # finally launching the worker instances        
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i+num_ps) + ' "' + comm0 + comm1 + comm2 + '"',shell=True)
        print('ssh -o StrictHostKeyChecking=no root@' + name + str(i+num_ps) + ' "' + comm0 + comm1 + comm2 + '"')
        #time.sleep(3) #give time to launch the command
        subprocess.call('echo "launched a worker"',shell=True)                                                            
