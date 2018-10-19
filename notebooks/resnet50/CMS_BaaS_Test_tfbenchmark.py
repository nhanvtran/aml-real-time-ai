import os
import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import importer

#import amlrealtimeai
from collections import namedtuple
import numpy as np
from amlrealtimeai import resnet50
import amlrealtimeai.resnet50.utils
from amlrealtimeai.resnet50.model import LocalQuantizedResNet50
#from amlrealtimeai.pipeline import ServiceDefinition, TensorflowStage, BrainWaveStage
#from amlrealtimeai import DeploymentClient
#from amlrealtimeai import PredictionClient
from tensorflow.python.framework import graph_util
from tensorflow.python.util import nest
import requests

import time

params = namedtuple(
                    'params',
                   [
                     'num_inter_threads',
                     'num_intra_threads',
                     'device',
                     'gpu_memory_frac_for_testing',
                     'allow_growth',
                     'use_unified_memory',
                     'xla',
                     'rewriter_config',
                     'enable_optimizations',
                     'variable_update',
                     'force_gpu_compatible'
                   ])

def create_config_proto(params):
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  if params.num_intra_threads is None:
    if params.device == 'gpu':
      config.intra_op_parallelism_threads = 1
  else:
    config.intra_op_parallelism_threads = params.num_intra_threads
  config.inter_op_parallelism_threads = params.num_inter_threads
  config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
  config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
  if params.allow_growth is not None:
    config.gpu_options.allow_growth = params.allow_growth
  if params.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = (
        params.gpu_memory_frac_for_testing)
  if params.use_unified_memory:
    config.gpu_options.experimental.use_unified_memory = True
  if params.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
    config.graph_options.rewrite_options.pin_to_host_optimization = (rewriter_config_pb2.RewriterConfig.OFF)
  if params.rewriter_config:
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    text_format.Merge(params.rewriter_config, rewriter_config)
    config.graph_options.rewrite_options.CopyFrom(rewriter_config)
  elif not params.enable_optimizations:
    off = rewriter_config_pb2.RewriterConfig.OFF
    config.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.layout_optimizer = off
    rewrite_options.constant_folding = off
    rewrite_options.shape_optimization = off
    rewrite_options.remapping = off
    rewrite_options.arithmetic_optimization = off
    rewrite_options.dependency_optimization = off
    rewrite_options.loop_optimization = off
    rewrite_options.function_optimization = off
    rewrite_options.debug_stripper = off
    rewrite_options.disable_model_pruning = True
    rewrite_options.scoped_allocator_optimization = off
    rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.NO_MEM_OPT)
    rewrite_options.pin_to_host_optimization = off
  elif params.variable_update == 'collective_all_reduce':
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')
  return config

def unfreezable_local_variables(graph):
  return graph.get_collection(
    tf.GraphKeys.LOCAL_VARIABLES,
    scope='.*' + 'gpu_cached_inputs')

def load_graph(frozen_graph_filename, input_name, dummy_input):
    f = tf.gfile.GFile(frozen_graph_filename, "rb")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in  = tf.import_graph_def(graph_def)
    graph = tf.Graph().as_default()
    tf.import_graph_def(graph_def, name="prefix",input_map={input_name : dummy_input})
    return graph, graph_def

def setup_queue(image, batch_size, threads):
    tensor_list = [image]
    dtypes = [tf.float32]
    shapes = [image.shape]
    min_after_dequeue = 10000
    capacity = min_after_dequeue + (threads + 1) * batch_size
    q = tf.FIFOQueue(capacity=capacity, dtypes=dtypes, shapes=shapes)
    enqueue_op = q.enqueue(tensor_list)
    tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * threads))
    image_batch = q.dequeue_many(batch_size)
    return image_batch


def main():
    params_info=params(
        num_intra_threads=1,
        num_inter_threads=6,
        device='gpu',
        gpu_memory_frac_for_testing=0,
        allow_growth=None,
        use_unified_memory=False,
        xla=False,
        rewriter_config=None,
        enable_optimizations=True,
        variable_update='parameter_server',
        force_gpu_compatible=False
    )
    useGPU = True
    batch_size=100
    num_threads = 16
    if useGPU:
        device_name = "/device:GPU:0"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        device_name = "/cpu:0"
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_path = '/tmp/pharris/'
    
    dummy_input = tf.random_normal([224,224,3], mean=0, stddev=1)
    #dummy_input = tf.random_normal([1,1,2048], mean=0, stddev=1)          
    dummy_label = tf.random_uniform([batch_size,1], minval=0, maxval=1, dtype=tf.int32)          
    image_batch = setup_queue(dummy_input,batch_size,num_threads)

    graph, graph_def = load_graph(model_path+'/resnet50.pb','InputImage',image_batch)
    graph=tf.get_default_graph()
    y                  = graph.get_tensor_by_name('prefix/resnet_v1_50/pool5:0')
    graph, graph_def = load_graph(model_path+'/resnet50_classifier.pb','Input', y)
    graph=tf.get_default_graph()
    out = graph.get_tensor_by_name('prefix_1/resnet_v1_50/logits/Softmax:0')
    
    predictions = tf.argmax(out, 1)
    accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, dummy_label)
    metrics_op = tf.group(accuracy_update)

    with tf.device(device_name):
        init_run_options=[]
        with tf.name_scope('local_variable_initialization'):
          local_var_init_op = tf.variables_initializer(unfreezable_local_variables(tf.get_default_graph()))
          variables_to_keep = {
            local_variable: tf.GraphKeys.LOCAL_VARIABLES
            for local_variable in unfreezable_local_variables(graph)
          }
        variable_initializers = [variable.initializer.name for variable in variables_to_keep]
        output_node_names = (
          #flattened_op_names +
          variable_initializers + [variable.value().op.name for variable in variables_to_keep])
        graphdef = graph.as_graph_def(add_shapes=True)
        with graph.as_default():
          with tf.Session(config=create_config_proto(params_info)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            graphdef = graph_util.convert_variables_to_constants(sess,graphdef,output_node_names,variable_names_blacklist=[variable.op.name for variable in variables_to_keep])
          importer.import_graph_def(graph_def=graphdef, name='')
          for variable in variables_to_keep:
            updated_variable = tf.Variable.from_proto(variable.to_proto())
            tf.add_to_collection(variables_to_keep[variable], updated_variable)
        ready_for_local_init_op = tf.report_uninitialized_variables(tf.global_variables())
        local_var_init_op = tf.local_variables_initializer()
        #sv = tf.train.Supervisor(is_chief=True,logdir=None,ready_for_local_init_op=None,local_init_op=local_var_init_op,saver=None,summary_op=None,save_model_secs=0,summary_writer=None,local_init_run_options=init_run_options)
        sv = tf.train.Supervisor(is_chief=True,logdir=None,ready_for_local_init_op=ready_for_local_init_op,local_init_op=local_var_init_op,saver=None,summary_op=None,save_model_secs=0,summary_writer=None,local_init_run_options=init_run_options)
        target=''
        with sv.managed_session(master=target,config=create_config_proto(params_info),start_standard_services=False) as sess:
            coord = tf.train.Coordinator()
            with sess.as_default():
              threads = tf.train.start_queue_runners(coord=coord)
              for j in range(10):
                time0 = time.time() 
                for i in range(10):
                    #result = sess.run(image_batch)
                    result = sess.run(metrics_op)
                time1 = time.time()
                print("Time to infer Resnet50 classifier on %s per image"%device_name, (time1-time0)/batch_size/10)
                print("Raw Time to infer Resnet50 classifier on %s "%device_name, (time1-time0))
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    main()
