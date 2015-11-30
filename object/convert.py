import sys
import cv2
sys.path.append("../../caffe-dev/python/")

import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
data = open("imagenet_mean.binaryproto", 'rb').read()
blob.ParseFromString(data)
arr = np.array(caffe.io.blobproto_to_array(blob))
out = arr[0]
np.save('imagenet_mean.npy', out)
