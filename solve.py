caffe_root = '../caffe-dev/'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import numpy as np
import argparse

def set_properties(prototxt, properties):
    print properties
    basename = os.path.basename(prototxt)
    with open(prototxt, 'r') as fr:
        lines = fr.readlines()
    for i, line in enumerate(lines):
        for key in properties:
            if line.strip().startswith(key + ':'):
                index = line.index(':')
                if type(properties[key]) == str:
                    lines[i] = line[:index+1] + ''' "''' + properties[key] + '''"\n'''
                else:
                    lines[i] = line[:index+1] + ' ' + str(properties[key]) + '\n'
    new_prototxt = '.' + prototxt
    with open(new_prototxt, 'w') as fw:
        for line in lines:
            fw.write(line)
    return new_prototxt

verb_net_proto_file = 'motion/deploy.prototxt'
verb_weights = 'model/Motion_iter_4000.caffemodel'

obj_net_proto_file = 'object/deploy.prototxt'
obj_weights = 'model/Object_iter_4000.caffemodel'

train_prototxt = 'train.prototxt'
solver_prototxt = 'solver.prototxt'

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver(solver_prototxt)
verb_net = caffe.Net(verb_net_proto_file, verb_weights, caffe.TEST)
obj_net = caffe.Net(obj_net_proto_file, obj_weights, caffe.TEST)

#['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8_gtea']
#['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7_gtea', 'fc8_gtea']
#['verb_conv1', 'verb_conv2', 'verb_conv3', 'verb_conv4', 'verb_conv5', 'verb_fc6', 'verb_fc7', 'verb_fc8', 'object_conv1', 'object_conv2', 'object_conv3', 'object_conv4', 'object_conv5', 'object_fc6', 'object_fc7', 'object_fc8', 'fuse_fc8', 'fuse_fc9']

print verb_net.params.keys()
print obj_net.params.keys()
print solver.net.params.keys()

verb_copy_pairs = {
            'verb_conv1' : 'conv1',
            'verb_conv2' : 'conv2',
            'verb_conv3' : 'conv3',
            'verb_conv4' : 'conv4',
            'verb_conv5' : 'conv5',
            'verb_fc6' : 'fc6',
            'verb_fc7' : 'fc7',
            'verb_fc8' : 'fc8_gtea'}

obj_copy_pairs = {
            'object_conv1' : 'conv1',
            'object_conv2' : 'conv2',
            'object_conv3' : 'conv3',
            'object_conv4' : 'conv4',
            'object_conv5' : 'conv5',
            'object_fc6' : 'fc6',
            'object_fc7' : 'fc7_gtea',
            'object_fc8' : 'fc8_gtea'}

for dst in verb_copy_pairs:
    src = verb_copy_pairs[dst]
    print 'copy {}, {} -> {}, {}'.format(src, \
        verb_net.params[src][0].data.shape, \
        dst, \
        solver.net.params[dst][0].data.shape)
    solver.net.params[dst][0].data[...] = verb_net.params[src][0].data
    solver.net.params[dst][1].data[...] = verb_net.params[src][1].data

for dst in obj_copy_pairs:
    src = obj_copy_pairs[dst]
    print 'copy {}, {} -> {}, {}'.format(src, \
        obj_net.params[src][0].data.shape, \
        dst, \
        solver.net.params[dst][0].data.shape)
    solver.net.params[dst][0].data[...] = obj_net.params[src][0].data
    solver.net.params[dst][1].data[...] = obj_net.params[src][1].data

#solver.net.copy_from(base_weights)
#solver.restore('PLUSM_model/VERB_OBJ_PLUSM_AHMAD_iter_6000.solverstate')
solver.step(4000)
