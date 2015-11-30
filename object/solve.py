import sys,os
caffe_root = os.path.abspath('../../caffe-dev/')
print caffe_root
sys.path.insert(0, os.path.join(caffe_root, 'python'))    #insert 0
import caffe
import argparse
import numpy as np

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

base_weights = "VGG_CNN_M.caffemodel"
solver_prototxt = 'solver.prototxt'

# init
caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver(solver_prototxt)
solver.net.copy_from(base_weights)
solver.step(4000)
