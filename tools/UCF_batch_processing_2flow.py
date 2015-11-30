import os
import sys
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source_dir')
	parser.add_argument('-t', '--target_dir')
	args = parser.parse_args()

	if args.source_dir is None or args.target_dir is None:
		parser.print_help()
		sys.exit(0)
	
	if not os.path.exists(args.target_dir):
		os.mkdir(args.target_dir)
	
	video_dirs = [d for d in os.listdir(args.source_dir) if os.path.isdir(os.path.join(args.source_dir,d))]
	for video_dir in video_dirs:
		tgt_path = os.path.join(args.target_dir, video_dir)
		if not os.path.exists(tgt_path):
			os.mkdir(tgt_path)
		src_path = os.path.join(args.source_dir, video_dir)
		os.system('python image_to_flow.py '+'calc_flow_folder '+src_path+' '+tgt_path)
