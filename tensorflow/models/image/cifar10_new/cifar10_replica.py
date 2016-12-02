# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Distributed CIFAR-10 training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10


flags = tf.app.flags
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 0,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", True,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_integer("checkpoint_restore", 0,
                     "Checkpoint Step to Restore from")
flags.DEFINE_string("checkpoint_dir", '/mnt/checkpoint',"job name: worker or ps")


FLAGS = flags.FLAGS


IMAGE_PIXELS = 28


def main(unused_argv):
  cifar10.maybe_download_and_extract()
  if FLAGS.download_only:
    sys.exit(0)
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)
  num_ps = len(ps_spec)

  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    if FLAGS.num_gpus < num_workers:
      raise ValueError("number of gpus is less than number of workers")
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu 
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU
  with tf.device(
      tf.train.replica_device_setter(
          ps_tasks=num_ps,
          worker_device=worker_device,
          ps_device="/job:ps/cpu:0",
          cluster=cluster)):
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    #train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries();
    # Variables that affect learning rate.
    num_batches_per_epoch = 50000 / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * 350)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(0.1,
                                    global_step,
                                    decay_steps,
                                    0.1,
                                    staircase=True)

    opt = tf.train.GradientDescentOptimizer(lr)
    
    if FLAGS.sync_replicas:
      if FLAGS.replicas_to_aggregate is None:
        replicas_to_aggregate = num_workers
      else:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

      opt = tf.train.SyncReplicasOptimizerV2(
          opt,
          replicas_to_aggregate=replicas_to_aggregate,
          total_num_replicas=num_workers,
          name="cifar10_sync_replicas")

    train_step = opt.minimize(loss, global_step=global_step)

    if FLAGS.sync_replicas:
      local_init_op = opt.local_step_init_op
      if is_chief:
        local_init_op = opt.chief_init_op

      ready_for_local_init_op = opt.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      chief_queue_runner = opt.get_chief_queue_runner()
      sync_init_op = opt.get_init_tokens_op()

    init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp(dir="/mnt",suffix="data", prefix="cifar10_train")

    if FLAGS.sync_replicas:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          local_init_op=local_init_op,
	  saver=None,
          summary_op=summary_op,
          save_summaries_secs=120, 
          save_model_secs=600,
          checkpoint_basename='model.ckpt', 
          ready_for_local_init_op=ready_for_local_init_op,
          recovery_wait_secs=1,
          global_step=global_step)
    else:
      sv = tf.train.Supervisor(
          is_chief=is_chief,
          logdir=train_dir,
          init_op=init_op,
          saver=None,
          summary_op=summary_op,
          save_summaries_secs=120,  
          save_model_secs=600,
          checkpoint_basename='model.ckpt',
          recovery_wait_secs=1,
          global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    if FLAGS.existing_servers:
      server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
      print("Using existing server at: %s" % server_grpc_url)

      sess = sv.prepare_or_wait_for_session(server_grpc_url,
                                            config=sess_config)
    else:
      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if FLAGS.sync_replicas and is_chief:
      # Chief worker will start the chief queue runner and call the init op.
      sess.run(sync_init_op)
      sv.start_queue_runners(sess, [chief_queue_runner])

    # Restore from Checkpoint
    if FLAGS.checkpoint_restore > 0:
      checkpoint_directory = FLAGS.checkpoint_dir + str(FLAGS.checkpoint_restore)
      ckpt = tf.train.get_checkpoint_state(checkpoint_directory)
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      else:
        print('No checkpoint file found')
        return


    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    num_examples_per_step = 128
    f = open('/mnt/train_output.log', 'w')
    #f.write("Training begins @ " + str(time_begin) +"\n")
    f.write("Duration\tWorker\tLocalStep\tGlobalStep\tLoss\tExamplesPerSec\tTrainingTime\n")
    f.close()
    last = time_begin
    while True:
      start_time = time.time()
      _, step, loss_value = sess.run([train_step, global_step, loss])
      duration = time.time() - start_time
      local_step += 1
      if local_step % 10 == 0:
        now = time.time()
        examples_per_sec = 10*num_examples_per_step/(now-last)
        print("%f: Worker %d: step %d (global step: %d of %d) loss = %.2f examples_per_sec = %.2f \n" % (now - last, FLAGS.task_index, local_step, step, FLAGS.train_steps, loss_value, examples_per_sec))
        f = open('/mnt/train_output.log', 'a')
        f.write(str(now-last) + "\t" + str(FLAGS.task_index) + "\t" + str(local_step) + "\t" + str(step) + "\t" + str(loss_value) + "\t"+str(examples_per_sec)+"\t"+str(now-time_begin) +"\n")
        f.close()
        last = now
      
      if step >= FLAGS.train_steps:
        break

      if sv.should_stop():
	print('Stopped due to abort')
	break
      # Save the model checkpoint periodically.
      #if is_chief and (step % 1000 == 0 or (step + 1) == FLAGS.train_steps):
      if (step % 1000 == 0 or (step + 1) == FLAGS.train_steps):
        print('Taking a Checkpoint @ Global Step '+str(step))
        checkpoint_dir = "/mnt/checkpoint"+str(step) 
        if tf.gfile.Exists(checkpoint_dir):
    	   tf.gfile.DeleteRecursively(checkpoint_dir)
        tf.gfile.MakeDirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir,"model.ckpt")
        saver.save(sess, checkpoint_path, global_step=step)

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    f = open('/mnt/train_output.log', 'a')
    #f.write("Training ends @ " + str(time_end) +"\n")
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
    f.write("Training elapsed time: " + str(training_time) +" s\n")
    f.close()
    # Validation feed
    # val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    # val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    # print("After %d training step(s), validation cross entropy = %g" %
    #       (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()
