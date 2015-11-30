import sys, os
import argparse
import cv2

if __name__ == "__main__":
	parser = argparse.ArgumentParser();
	parser.add_argument('src_dir')	
	args = parser.parse_args()

	if args.src_dir is None:
		parser.print_help()
		sys.exit(0)

	im_dirs = [os.path.join(args.src_dir, d) for d in os.listdir(args.src_dir) if os.path.isdir(os.path.join(args.src_dir, d))]
	#print len(im_dirs)
	for im_dir in im_dirs:
		im_sub_dirs = [d for d in os.listdir(im_dir) if os.path.isdir(os.path.join(im_dir, d))] 
		#print im_sub_dirs
		for im_sub_dir in im_sub_dirs:
			im_files = [os.path.join(im_dir, im_sub_dir, f) for f in os.listdir(os.path.join(im_dir, im_sub_dir)) if f.endswith('.jpg')]
			for im_file in im_files:
				print 'processing ', im_file
				im = cv2.imread(im_file)	 
				cv2.imwrite(im_file, cv2.resize(im, (256, 256)))
