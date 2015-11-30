import os, sys

def training_data_preprocessing(filename):
	mt_fd = open("motion_list.txt", 'w')
	ob_fd = open("object_list.txt", 'w')
	ac_fd = open("action_list.txt", 'w')
	
	with open(filename) as f:
		for line in f:
			line_splited = line.strip().split(' ')
			if int(line_splited[1]) == 101:
				line_splited[1] = "0"
			abs_line_splited = os.path.join(os.path.abspath("../flow"), line_splited[0].split('.')[0])
			abs_obj_line_splited = os.path.join(os.path.abspath("../images"), line_splited[0].split('.')[0])
			#print abs_line_splited, abs_obj_line_splited
			mt_fd.write(abs_line_splited + ' ' + line_splited[1] + '\n')
			ac_fd.write(abs_line_splited + ' ' + abs_obj_line_splited + ' ' + line_splited[1] + ' ' + line_splited[1] + ' ' + line_splited[1] + '\n')
			for each_video in sorted([d for d in os.listdir(abs_obj_line_splited)]):
				ob_fd.write(os.path.join(os.path.abspath("../images"), line_splited[0].split('.')[0], each_video) + ' ' + line_splited[1] + '\n')

def test_data_preprocessing(filename):
	mt_fd = open("test_motion_list.txt", 'w')
	ob_fd = open("test_object_list.txt", 'w')
	ac_fd = open("test_action_list.txt", 'w')
	
	action_dic = dict()
	with open("classInd.txt") as f:
		for line in f.readlines():
			tokens = line.strip().split(' ')	
			action_dic[tokens[1]] = tokens[0]

	with open(filename) as f:
		for line in f:
			action_name = line.strip().split('/')[0]
			line_splited = line.strip().split(' ')

			abs_line_splited = os.path.join(os.path.abspath("../flow"), line_splited[0].split('.')[0])
			abs_obj_line_splited = os.path.join(os.path.abspath("../images"), line_splited[0].split('.')[0])
			mt_fd.write(abs_line_splited + ' ' + action_dic[action_name] + '\n')
			ac_fd.write(abs_line_splited + ' ' + abs_obj_line_splited + ' ' + action_dic[action_name] + ' ' + action_dic[action_name] + ' ' + action_dic[action_name] + '\n')
			for each_video in sorted([d for d in os.listdir(abs_obj_line_splited)]):
				ob_fd.write(os.path.join(os.path.abspath("../images"), line_splited[0].split('.')[0], each_video) + ' ' + action_dic[action_name] + '\n')

if __name__ == "__main__":
	#training_data_preprocessing("trainlist01.txt")	
	test_data_preprocessing("testlist01.txt")
