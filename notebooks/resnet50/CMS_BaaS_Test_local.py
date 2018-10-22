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
    useGPU = False
    batch_size=100
    num_threads = 16
    if useGPU:
        device_name = "/device:GPU:0"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        device_name = "/cpu:0"
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_path = '/tmp/pharris/'

    dummy_input = tf.random_normal([224,224,3], mean=0, stddev=1) #Note I am using a truly random image 
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
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for j in range(10):
                time0 = time.time() 
                for i in range(10):
                    #result = sess.run(image_batch) # Check the queueing time
                    result = sess.run(metrics_op)
                time1 = time.time()
                print("Time to infer Resnet50 classifier on %s per image"%device_name, (time1-time0)/batch_size/10)
                print("Raw Time to infer Resnet50 classifier on %s "%device_name, (time1-time0))
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    main()
