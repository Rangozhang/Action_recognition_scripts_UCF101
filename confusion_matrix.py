import os, sys
import numpy as np

def get_confusion_matrix(filename, num_class):
	conf_mat = np.zeros((num_class, num_class), dtype=np.uint16)

	with open(filename) as fd:
		for line in fd:
			tokens = line.strip().split();
			if tokens[1].isdigit():
				conf_mat[int(tokens[1])][int(tokens[2])] += 1
	return conf_mat

if __name__ == '__main__':
	conf_mat = get_confusion_matrix('object/obj_res_real.txt', 101)
