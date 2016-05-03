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

root = sys.argv[1]

images = {}
lut = []

print 'crawling ' + root

for dirpath, dnames, fnames in os.walk(root):
	# read all the boxes
	for f in fnames:

		fullpath = os.path.join(root, dnames, f)
		classid, ext = os.path.splitext(f)

		if ext == '.txt':
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
					img_filepath = os.path.join(root, 'imgs', classid, img_id + '.jpg')

					# generate the box
					box = {'xmin':box_left, 'ymin':box_top, 'xmax':box_right, 'ymax':box_bitton}

					# convert the box to yolo format
					if img_url not in images:
						images[img_url] = {'boxes':[], 'path':img_filepath}
						lut.append(img_url)

					images[img_url]['boxes'].append(box)

			fid.close()

# randomly select 70% of the dataset for training and leave 30% for test
endsize = len(lut) * 0.3
fid = open('trainval.txt', 'w')
if not os.path.exists('labels'):
	os.mkdir('labels')

while len(lut) > endsize:
	url = random.choice(lut)
	imgpath = images[url]['path']	
	imgname = os.path.splitext(os.path.basename(imgpath))[0]
	classname = os.path.basename(os.path.split(imgpath)[0])
	anno_path = os.path.join('labels', classname, imgname + '_annotation.xml')
	
	fid.write(imgpath + ',' + anno_path + '\n')

	if not os.path.exists(os.path.join('labels', classname)):
		os.mkdir(os.path.join('labels', classname))

	# write all boxes to file
	bid = open(anno_path, 'w')
	bid.write('<annotation>'\n)
	for b in images[url]['boxes']:
		write_box(bid, b)
	bid.write('</annotation>\n')
	lut.remove(url)
	bid.close()
fid.close()

# write the remaining images to test file
fid = open('test.txt', 'w')
if not os.path.exists('test_labels'):
	os.mkdir('test_labels')

for url in lut:
	imgpath = images[url]['path']	
	imgname = os.path.splitext(os.path.basename(imgpath))[0]
	classname = os.path.basename(os.path.split(imgpath)[0])
	anno_path = os.path.join('test_labels', classname, imgname + '_annotation.xml')
		
	fid.write(imgpath + ',' + anno_path + '\n')
	
	if not os.path.exists(os.path.join('test_labels', classname)):
		os.mkdir(os.path.join('test_labels', classname))

	bid = open(anno_path, 'w')
	bid.write('<annotation>'\n)
	for b in images[url]['boxes']:
		write_box(bid, b)
	bid.write('</annotation>\n')
	bid.close()
fid.close()
