import os
import sys
import random
from PIL import Image

def write_box(bid, b):
	bid.write('\t<object>\n')
	bid.write('\t\t<name>face</name>\n')
	bid.write('\t\t<bndbox>\n')
	bid.write('\t\t\t<xmin>%d</xmin>\n' % b['xmin'])
	bid.write('\t\t\t<ymin>%d</ymin>\n' % b['ymin'])
	bid.write('\t\t\t<xmax>%d</xmax>\n' % b['xmax'])
	bid.write('\t\t\t<ymax>%d</ymax>\n' % b['ymax'])
	bid.write('\t\t</bndbox>\n')
	bid.write('\t</object>\n')
	return

def walk_vgg_directory(root):
	images = {}
	lut = []

	for dirpath, dnames, fnames in os.walk(root):
		# read all the boxes
		for f in fnames:

			fullpath = os.path.join(root, dirpath, f)
			classid, ext = os.path.splitext(f)

			if ext == '.txt' and classid != 'licence':
				print 'parsing ' + f
				fid = open(fullpath, 'r')
				for line in fid:
					cols = line.split(' ')
					if len(cols) > 5:
						img_id = cols[0]
						img_url = cols[2]
						box_left = float(cols[3])
						box_top = float(cols[4])
						box_right = float(cols[5])
						box_bottom = float(cols[6])
						img_filepath = os.path.join('img', classid, img_id + '.jpg')

						# generate the box
						box = {'xmin':box_left, 'ymin':box_top, 'xmax':box_right, 'ymax':box_bottom}

						if img_url not in images:
							images[img_url] = {'boxes':[], 'path':img_filepath}
							lut.append(img_url)

						images[img_url]['boxes'].append(box)

				fid.close()

	# randomly select 90% of the dataset for training and leave reset for test
	endsize = len(lut) * 0.1
	fid = open('trainval.txt', 'w')
	if not os.path.exists('labels'):
		os.mkdir('labels')

	random.shuffle(lut)

	while len(lut) > endsize:
		url = lut[0]
		imgpath = images[url]['path']	
		imgname = os.path.splitext(os.path.basename(imgpath))[0]
		classname = os.path.basename(os.path.split(imgpath)[0])
		anno_path = os.path.join('labels', classname, imgname + '_annotation.xml')
		
		fid.write(imgpath + ' ' + os.path.join('annos', anno_path) + '\n')

		if not os.path.exists(os.path.join('labels', classname)):
			os.makedirs(os.path.join('labels', classname))

		# write all boxes to file
		bid = open(anno_path, 'w')
		bid.write('<annotation>\n')
		for b in images[url]['boxes']:
			write_box(bid, b)
		bid.write('</annotation>\n')
		lut.remove(url)
		bid.close()
		
		if len(lut) % 1000 == 0:
			print len(lut)
			
	fid.close()

	print 'Created trainval.txt creating test.txt'

	# write the remaining images to test file
	fid = open('test.txt', 'w')
	if not os.path.exists('test_labels'):
		os.mkdir('test_labels')

	for url in lut:
		imgpath = images[url]['path']	
		imgname = os.path.splitext(os.path.basename(imgpath))[0]
		classname = os.path.basename(os.path.split(imgpath)[0])
		anno_path = os.path.join('test_labels', classname, imgname + '_annotation.xml')
			
		fid.write(imgpath + ' ' + os.path.join('annos', anno_path) + '\n')
		
		if not os.path.exists(os.path.join('test_labels', classname)):
			os.makedirs(os.path.join('test_labels', classname))

		bid = open(anno_path, 'w')
		bid.write('<annotation>\n')
		for b in images[url]['boxes']:
			write_box(bid, b)
		bid.write('</annotation>\n')
		bid.close()
	fid.close()
	
def convert_voc_file(input_filepath, images_root, savefolder):
	# load all the lines from file and build up the map
	fin = open(input_filepath, 'r')

	images = {}
	
	for line in fin:
		cols = line.split(' ')
		img_id = cols[0]
		box_left = float(cols[1])
		box_top = float(cols[2])
		box_right = float(cols[3])
		box_bottom = float(cols[4])
		img_width = float(cols[5])
		img_height = float(cols[6])
		
		# generate the box
		box = {'xmin':box_left, 'ymin':box_top, 'xmax':box_right, 'ymax':box_bottom}

		# convert the box to yolo format
		if img_id not in images:
			images[img_id] = {'boxes':[]}

		images[img_id]['boxes'].append(box)
		

	fin.close()
	
	# save the image map out
	if not os.path.exists(savefolder):
        	os.makedirs(savefolder)

	fout = open(os.path.join(savefolder, 'test.txt'), 'w')
	if not os.path.exists(os.path.join(savefolder, 'test_labels')):
		os.makedirs(os.path.join(savefolder, 'test_labels'))
	
	for id in images:
		anno_path = os.path.join('test_labels', id + '_annotation.xml')
		
		imgpath = os.path.join("images", "JPEGImages", id)
		
		fout.write(imgpath + ' ' + os.path.join(anno_path) + '\n')
		
		bid = open(os.path.join(savefolder, anno_path), 'w')
		bid.write('<annotation>\n')
		for b in images[id]['boxes']:
			write_box(bid, b)
		bid.write('</annotation>\n')
		bid.close()
	fout.close()
	
def walk_ffld_directory(root):

	for dirpath, dnames, fnames in os.walk(root):
		# read all the boxes
		for f in fnames:
			fullpath = os.path.join(root, dirpath, f)
			classid, ext = os.path.splitext(f)

			if ext == '.txt' and classid != 'licence':
				print 'parsing ' + f
				fid = open(fullpath, 'r')
				for line in fid:
					cols = line.split(' ')
					if len(cols) == 5:
						img_path = os.path.basename(cols[0])
						left = cols[1]
						top = cols[2]
						width = cols[3]
						height = cols[4]
					
					# generate the box
					box = {'xmin':left, 'ymin':top, 'xmax':left+width, 'ymax':top+height}

					if img_path not in images:
						images[img_path] = {'boxes':[], 'path':img_filepath}
						lut.append(img_path)

					images[img_path]['boxes'].append(box)
			elif ext.lower() == '.jpg' or ext.lower() == '.jpeg':
				if img_path not in images:
					images[img_path] = {'boxes':[], 'path':img_filepath}
					lut.append(img_path)

	# randomly select 90% of the dataset for training and leave rest for test
	endsize = len(lut) * 0.1
	fid = open('trainval.txt', 'w')
	if not os.path.exists('labels'):
		os.mkdir('labels')

	random.shuffle(lut)

	while len(lut) > endsize:
		url = lut[0]
		imgpath = images[url]['path']	
		imgname = os.path.splitext(os.path.basename(imgpath))[0]
		anno_path = os.path.join('labels', imgname + '_annotation.xml')
		
		fid.write(imgpath + ' ' + os.path.join('annos', anno_path) + '\n')

		# write all boxes to file
		bid = open(anno_path, 'w')
		bid.write('<annotation>\n')
		for b in images[url]['boxes']:
			write_box(bid, b)
		bid.write('</annotation>\n')
		lut.remove(url)
		bid.close()
		
		if len(lut) % 1000 == 0:
			print len(lut)
			
	fid.close()

	print 'Created trainval.txt creating test.txt'

	# write the remaining images to test file
	fid = open('test.txt', 'w')
	if not os.path.exists('test_labels'):
		os.mkdir('test_labels')

	for url in lut:
		imgpath = images[url]['path']	
		imgname = os.path.splitext(os.path.basename(imgpath))[0]
		anno_path = os.path.join('test_labels', imgname + '_annotation.xml')
			
		fid.write(imgpath + ' ' + os.path.join('annos', anno_path) + '\n')
		
		bid = open(anno_path, 'w')
		bid.write('<annotation>\n')
		for b in images[url]['boxes']:
			write_box(bid, b)
		bid.write('</annotation>\n')
		bid.close()
	fid.close()
	
					
if __name__ == "__main__":
	input = sys.argv[1]
	encoding = sys.argv[2]
	images_root = sys.argv[3]
	savepath = sys.argv[4]

	print 'crawling ' + input

	if input.endswith('.txt'):
		if encoding == 'pascal-faces':
			convert_voc_file(input, images_root, savepath)
	else:
		if encoding == 'vgg':
			walk_vgg_directory(input)
		else:
			walk_ffld_directory(input)
			
	print 'Created test.txt'
