import sys
import os
import csv

output_folder = sys.argv[2]
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder) 

csvfile = open(sys.argv[1], 'rb')
csvrows = csv.reader(csvfile, delimiter=',')
next(csvrows, None) # skip header

boxes = []
last_filename = ''

imglist_file = open(os.path.join(output_folder, 'image_list.txt'), 'w')

for row in csvrows:
    filename = row[0]
    if last_filename == '':
        last_filename = filename

    if filename != last_filename:
        # create the annotation file
        print last_filename
        annofilename = os.path.join(output_folder, last_filename + '_annotation.xml')
        try:
            os.stat(os.path.dirname(annofilename))
        except:
            os.mkdir(os.path.dirname(annofilename))

        annofile = open(annofilename, 'w')
        annofile.write('<annotation>\n')
        annofile.write('\t<object>\n')
        annofile.write('\t\t<name>face</name>\n')
        for box in boxes:
            annofile.write('\t\t\t<bndbox>\n')
            annofile.write('\t\t\t\t<xmin>%d</xmin>\n' % int(round(box['left'])))
            annofile.write('\t\t\t\t<xmax>%d</xmax>\n' % int(round(box['left'] + box['width'])))
            annofile.write('\t\t\t\t<ymin>%d</ymin>\n' % int(round(box['top'])))
            annofile.write('\t\t\t\t<ymax>%d</ymax>\n' % int(round(box['top'] + box['height'])))
            annofile.write('\t\t\t</bndbox>\n')
        annofile.write('\t</object>\n')
        annofile.write('</annotation>\n')
        annofile.close()
        boxes = []
        last_filename = filename

        # add the line to the images file
        imglist_file.write('img/%s.jpg annos/%s_annotation.xml\n' % (last_filename, last_filename))

    box = {}
    box['left'] = float(row[1])
    box['top'] = float(row[2])
    box['width'] = float(row[3])
    box['height'] = float(row[4])
    boxes.append(box)

imglist_file.close()

    

