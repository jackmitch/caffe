import sys
import os
import csv
from PIL import Image

output_file = open(sys.argv[3], 'w')
output_csv = csv.writer(output_file, delimiter=',')

input_file = open(sys.argv[1], 'rb')
input_csv = csv.reader(input_file, delimiter=',')

root_dir = sys.argv[2]

header = input_csv.next()
header.append('image_width')
header.append('image_height')
output_csv.writerow(header);

for row in input_csv:
    filename = os.path.join(root_dir, row[0] + '.jpg')
    im = Image.open(filename)
    row.append(im.size[0])
    row.append(im.size[1])
    output_csv.writerow(row)

output_file.close()
input_file.close()



