import sys, os
import cv2
sys.path.append("../../caffe-dev/python/")
import caffe
import numpy as np

batch_size = 200

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net("deploy.prototxt", "../model/Object_iter_4000.caffemodel", caffe.TEST)
net.blobs['data'].reshape(batch_size, 3, 224, 224)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load('imagenet_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

res_fd = open('obj_res.txt','w')

image_batch = np.zeros((batch_size, 3, 224, 224))
GT_labels = np.zeros(batch_size, dtype=np.uint8)
image_name = []
count = 0
correct1 = 0
correct5 = 0
with open('test_object_list.txt') as fd:
	sum_lines = len(fd.readlines())

with open('test_object_list.txt') as fd:
	for line in fd:
		tokens = line.strip().split(' ')
		im = caffe.io.load_image(tokens[0])

		#crop in the middle
		im = im[16:-16,16:-16,:]
		
		#save information
		image_batch[count % batch_size, :, :, :] = transformer.preprocess('data', im)
		GT_labels[count % batch_size] = int(tokens[1]) 
		image_name.append(tokens[0])
		
		count += 1
		if count % batch_size == 0:
			net.blobs['data'].data[...] = image_batch
			out = net.forward()
			ranks = out['prob'].argsort()[:, -1:-6:-1]
			for im_nm, label, rank in zip(image_name, GT_labels, ranks):
				res_fd.write(im_nm+' '+str(label)+' '+' '.join(str(ele) for ele in rank)+'\n')
				if label in rank:
					correct5 += 1
				if label == rank[0]:
					correct1 += 1
				print os.path.basename(im_nm) + " in " + "{:.3%}".format((0.0 + count)/sum_lines)
				print " Predicted class is "+' '.join(str(ele) for ele in rank)+" and ground truth is "+str(label)
				print " first one accuracy: " + str((0.0 + correct1)/count) + " first five accuracy: " + str((0.0 + correct5)/count)
			image_batch.fill(0)
			image_name = []
			GT_labels.fill(0)

remain = count % batch_size;
if remain != 0:
	net.blobs['data'].data[...] = image_batch
	out = net.forward()
	ranks = out['prob'].argsort()[:remain, -1:-6:-1]
	for im_nm, label, rank in zip(image_name, GT_labels[:remain], ranks):
				res_fd.write(im_nm+' '+str(label)+' '+' '.join(str(ele) for ele in rank)+'\n')
				if label in rank:
					correct5 += 1
				if label == rank[0]:
					correct1 += 1
				print os.path.basename(im_nm) + " in " + "{:.3%}".format((0.0 + count)/sum_lines)
				print " Predicted class is "+' '.join(str(ele) for ele in rank)+" and ground truth is "+str(label)
				print " first one accuracy: " + str((0.0 + correct1)/count) + " first five accuracy: " + str((0.0 + correct5)/count)

res_fd.write("first one accuracy: " + str((0.0 + correct1)/count))
res_fd.write("first five accuracy: " + str((0.0 + correct5)/count))

res_fd.close()
