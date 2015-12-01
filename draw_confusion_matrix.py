import os, sys
#sys.path.insert(0, '../')
import numpy as np
import matplotlib.pyplot as plt
from confusion_matrix import get_confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename', help = 'result file with label and first 5 predicted class')
args = parser.parse_args()

if args.filename:
	file_name = args.filename
else:
	file_name = 'object/obj_res_real.txt'

class_num = 101
conf_mat = get_confusion_matrix(file_name, class_num)


sum_per_row = np.sum(conf_mat, axis=1)
conf_mat = (0.0 + conf_mat) / sum_per_row[:, np.newaxis]
#print " ".join(str(ele) for ele in np.sum(conf_mat, axis=1))

plt.matshow(conf_mat)
#plt.show()
plt.xticks(np.arange(0,101,5))
plt.yticks(np.arange(0,101,5))

plt.savefig(os.path.basename(file_name).strip().split('.')[0] + ".png")

ind_ranking = conf_mat.argsort()[:, -1:-6:-1]
conf_mat.sort()
value_ranking = conf_mat[:,-1:-6:-1]

res_fd = open(os.path.basename(file_name).strip().split('.')[0] + "_res"+ ".txt", 'w')

for i, (ind, value) in enumerate(zip(ind_ranking, value_ranking)):
	res_fd.write(str(i) + " :\t " +  " ".join("{:3d}".format(ele) for ele in ind) + " :\t " + " ".join("{:.2f}".format(ele) for ele in value) + '\n')
