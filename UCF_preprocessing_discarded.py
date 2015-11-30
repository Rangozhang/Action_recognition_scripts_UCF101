import os, sys

if __name__ == "__main__":

	data_dir = "../data/images"
	fd_id = open("action_ids.txt", 'w')
	fd_mt = open("motion_data.txt", 'w')
	fd_ob = open("object_data.txt", 'w')
	fd_ac = open("action_data.txt", 'w')
	action_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
	label = 0;
	for action in action_list:
		fd_id.write(action + ' ' + str(label) + '\n')
		action_dir = os.path.join(os.path.abspath(data_dir), action)
		subdir_actions = [os.path.join(action_dir, d) for d in os.listdir(action_dir) if os.path.isdir(os.path.join(action_dir, d))]
		frames = []
		for subdir_action in subdir_actions:
			tmp_frms = sorted([os.path.join(subdir_action, d) for d in os.listdir(subdir_action) if d.endswith('.jpg')])	
			frames = frames + tmp_frms
			fd_mt.write(subdir_action + ' ' + str(label) + '\n')
			fd_ac.write(subdir_action + ' ' + subdir_action + ' ' + str(label) + ' ' + str(label) + ' ' + str(label) + '\n')
		for frame in frames:
			fd_ob.write(frame + ' ' + str(label) + '\n')
		label += 1
			
	fd_id.close()
	fd_mt.close()
	fd_ob.close()
	fd_ac.close()	
