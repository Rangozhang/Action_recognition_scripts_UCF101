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

print conf_mat
print " ".join(str(ele) for ele in np.sum(conf_mat, axis=1))

plt.matshow(conf_mat)
plt.show()
