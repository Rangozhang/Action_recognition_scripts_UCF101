import sys, os
import cv2
sys.path.append("../caffe-dev/python/")
import caffe
import numpy as np
from scipy import stats

rand_time = 20 # choose 20 stacks of frames randomly

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net("deploy.prototxt", "model/Action_iter_4000.caffemodel", caffe.TEST)
net.blobs['data'].reshape(rand_time, 23, 224, 224)

transformer = caffe.io.Transformer({'data': (rand_time, 20, 224, 224), 'rgb_data': (rand_time, 3, 224, 224)})
transformer.set_transpose('data',(2, 0, 1))
transformer.set_transpose('rgb_data',(2, 0, 1))
transformer.set_mean('data', np.tile(128, 20))
transformer.set_mean('rgb_data', np.load('object/imagenet_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_raw_scale('rgb_data', 255)
transformer.set_channel_swap('rgb_data', (2, 1, 0))

res_fd = open('act_res.txt','w')

image_batch = np.zeros((rand_time, 23, 224, 224))
image_name = []
count = 0
correct1 = 0
correct5 = 0
with open('test_action_list.txt') as fd:
	sum_lines = len(fd.readlines())

with open('test_action_list.txt') as fd:
	for line in fd:
		tokens = line.strip().split(' ')
		
		frames = [fr for fr in os.listdir(os.path.join(tokens[0], 'x')) if fr.endswith(".jpg")]
		frames_num = len(frames)
		randInd = np.random.randint(frames_num-10, size=rand_time)
		for j in range(rand_time):
			clip_buff = np.zeros((20, 224, 224))
			for i, frame in enumerate(frames[randInd[j]:randInd[j]+10]):
				im_x = caffe.io.load_image(os.path.join(tokens[0], 'x', frame))
				im_y = caffe.io.load_image(os.path.join(tokens[0], 'y', frame))
				#crop in the middle
				clip_buff[2*i, :, :] = im_x[16:-16,16:-16,0]
				clip_buff[2*i+1, :, :] = im_y[16:-16,16:-16,0]

			clip_buff = clip_buff.transpose((1, 2, 0))
			image_batch[j , 0:20, :, :] = transformer.preprocess('data', clip_buff)
			im_rgb = caffe.io.load_image(os.path.join(tokens[1], frame))
			temp = transformer.preprocess('rgb_data', im_rgb)
			image_batch[j , 20:,  :, :] = temp

		label = int(tokens[2]) 
		image_name = tokens[0]
		
		count += 1
		net.blobs['data'].data[...] = image_batch
		out = net.forward()
		ranks = out['prob'].argsort()[:, -1:-6:-1]
		final_rank = np.zeros(5)
		
		# find mode for each position
		# ranks = ranks.transpose((1, 0))
		final_rank = stats.mode(ranks)[0][0].astype(int)
		
		if label in final_rank:
			correct5 += 1
		if label == final_rank[0]:
			correct1 += 1
		print os.path.basename(image_name) + " in " + "{:.3%}".format((0.0 + count)/sum_lines)
		print " Predicted class is "+' '.join(str(ele) for ele in final_rank)+" and ground truth is "+str(label)
		print " first one accuracy: " + str((0.0 + correct1)/count) + " first five accuracy: " + str((0.0 + correct5)/count)
		res_fd.write(os.path.basename(image_name) + " " + str(label) + " " + " ".join(str(ele) for ele in final_rank) + '\n')
		image_batch.fill(0)
		image_name = []
'''
remain = count % rand_time;
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
'''
res_fd.write("first one accuracy: " + str((0.0 + correct1)/count) + '\n')
res_fd.write("first five accuracy: " + str((0.0 + correct5)/count) + '\n')

res_fd.close()

def replace_path(path, frm, to):
    pre, match, post = path.rpartition(frm)
    print pre
    print match
    print post
    return ''.join(((pre, to, post) if match else (pre, match, post)))
