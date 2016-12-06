import argparse
import sys
import subprocess
import os
import time

# CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", help="Specify number of workers",type=int)
parser.add_argument("--num_parameter_servers", help="Specify number of parameter servers",type=int)
parser.add_argument("--async",help="Specify if the training should be done synchronously or not")
parser.add_argument("--num_sync",help="If async=True, how many workers should be synchronously updated",type=int)
parser.add_argument("--train_steps",help="Specify number of training steps",type=int)
parser.add_argument("--approx_step",help="Step to start approximations",type=int, default=1000000000)
parser.add_argument("--approx_interval",help="Approximate every x steps",type=int,default=1)
parser.add_argument("--layers_to_train",help="Comma separated list of cifar10 model layers",type=str,default='softmax_linear,local4,local3,conv2,conv1')
#SANDEEP add flags that you will need for the approximation script
args = parser.parse_args()

# setting up parameters
name = 'candidat'
num_workers = args.num_workers
num_ps = args.num_parameter_servers
async = args.async
num_sync = args.num_sync
train_steps = args.train_steps
approx_step = args.approx_step
approx_interval = args.approx_interval
layers_to_train = args.layers_to_train

# specifying the ip and port num of ps
ps_str = ''
for i in range(num_ps):
        if i == num_ps - 1:
                ps_str += (name  + str(i) + ':2500')
        if i != num_ps - 1:
                ps_str += (name  + str(i) + ':2500,')

# specifying the ip and port num of workers
workers_str = ''
for i in range(num_workers):
        if i == num_workers - 1:
                workers_str += (name + str(i + num_ps) + ':2500')
        if i != num_workers - 1:
                workers_str += (name + str(i + num_ps) + ':2500,')

# first launching the parameter servers
for i in range(num_ps):
        # tensorflow python args
        if async == 'False':
                comm_ps = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="ps" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=True'
	# SANDEEP: add flags above based on your approximation script
        if async == 'True':
                comm_ps = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="ps" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=False' + \
	' --approx_step=' + str(approx_step) + ' --approx_interval=' + str(approx_interval) + \
	' --layers_to_train=' + str(layers_to_train)
	# SANDEEP: add flags above based on your approximation script

        #copy the network monitoring tool
        subprocess.call('scp log_network_BW.py root@' + name + '0:/root/tensorflow',shell=True)
        subprocess.call('scp log_cpu_util.py root@' + name + '0:/root/tensorflow',shell=True)

        # commands to launch on each vm
        comm0 = 'export PATH=\"$PATH:$HOME/bin\";'
        comm1 = 'mount /dev/sdb2 /mnt; apt-get install vnstat;apt-get install sysstat;'
        comm1 = 'mount /dev/sdb2 /mnt;'
        comm2 = 'cd tensorflow; git pull base master;'
        comm3 = 'nohup python ' + comm_ps + ' >> /mnt/' + name + '_out.txt 2>> /mnt/' + name + '_stderr.txt &'

        # finally launching the ps instances
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i) + ' "' + comm0 +  comm1 + comm2 + comm3 + '"',shell=True)
        print('ssh -o StrictHostKeyChecking=no root@' + name + str(i) + ' "' + comm0 + comm1 + comm2 + comm3 + '"')
        time.sleep(3) #give time to launch the command
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i) + ' "nohup python /root/tensorflow/log_network_BW.py > /dev/null 2>&1 &"',shell=True)
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i) + ' "nohup python /root/tensorflow/log_cpu_util.py > /dev/null 2>&1 &"',shell=True)
        subprocess.call('echo "launched a ps"',shell=True)

# next launching the workers
for i in range(num_workers):
        # tensorflow python args
        if async == 'False':
                comm_worker = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="worker" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=True'
	# SANDEEP: add flags above based on your approximation script
        if async == 'True':
                comm_worker = './tensorflow/models/image/cifar10_new/cifar10_replica.py --ps_hosts=' \
        + ps_str + ' --worker_hosts=' + workers_str + ' --job_name="worker" --task_index=' + str(i) + \
        ' --num_gpus=0 --train_steps=' + str(train_steps) + ' --sync_replicas=False' + \
	' --approx_step=' + str(approx_step) + ' --approx_interval=' + str(approx_interval) + \
	' --layers_to_train=' + str(layers_to_train)

        subprocess.call('scp log_cpu_util.py root@' + name + str(i+num_ps) + ':/root/tensorflow',shell=True)
	# SANDEEP: add flags above based on your approximation script

        #commands to launch on each vm
        comm0 = 'export PATH=\"$PATH:$HOME/bin\";'
        comm1 = 'mount /dev/sdb2 /mnt; apt-get install sysstat;'
        comm1 = 'mount /dev/sdb2 /mnt;'
        comm2 = 'cd tensorflow; git pull base master;'
        comm3 = 'nohup python ' + comm_worker + ' >> /mnt/' + name + '_out.txt 2>> /mnt/' + name + '_stderr.txt &'

        # finally launching the worker instances        
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i+num_ps) + ' "' + comm0 + comm1 + comm2 + comm3 + '"',shell=True)
        print('ssh -o StrictHostKeyChecking=no root@' + name + str(i+num_ps) + ' "' + comm0 + comm1 + comm2 + comm3 +'"')
        subprocess.call('ssh -o StrictHostKeyChecking=no root@' + name + str(i+num_ps) + ' "nohup python /root/tensorflow/log_cpu_util.py > /dev/null 2>&1 &"',shell=True)
        time.sleep(3) #give time to launch the command
        subprocess.call('echo "launched a worker"',shell=True)


### machine hints r1r4u38-*u32 and *u23

