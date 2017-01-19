import os
import sys
import cv2
import random

def cropFace(input_file, box, output_side_length = 256):
    '''Takes an image name, resize it and crop the center square
    '''
    img = cv2.imread(input_file)
    
    if box:
        img = img[box['ymin']:box['ymax'], box['xmin']:box['xmax']]
        
    height, width, depth = img.shape
    new_height = output_side_length
    new_width = output_side_length
    if height > width:
        new_height = output_side_length * height / width
    else:
        new_width = output_side_length * width / height
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img
    
    
root = sys.argv[1]
outputFolder = sys.argv[2]
cid = 0
if not os.path.exists(os.path.join(outputFolder, 'data')):
    os.makedirs(os.path.join(outputFolder, 'data'))
    
classFile = open(os.path.join(outputFolder, 'data', 'class_list.txt'), 'w')
trainFile = open(os.path.join(outputFolder, 'data', 'train.txt'), 'w')
valFile = open(os.path.join(outputFolder, 'data', 'val.txt'), 'w')

for dirpath, dnames, fnames in os.walk(root):
    for f in fnames:
        fullpath = os.path.join(root, dirpath, f)
        className, ext = os.path.splitext(f)
        
        if ext == '.txt' and className != 'licence':
            print 'parsing ' + f
            fid = open(fullpath, 'r')
            images = {}
            lut = []
            
            for line in fid:
                cols = line.split(' ')
                if len(cols) > 5:
                    img_id = cols[0]
                    img_url = cols[2]
                    box_left = float(cols[3])
                    box_top = float(cols[4])
                    box_right = float(cols[5])
                    box_bottom = float(cols[6])
                    curation = float(cols[9])
                    
                    if curation > 0:
                        # add to the image list
                        img_filepath = os.path.join(root, 'img', className, img_id + '.jpg')
                        
                        # generate the box
                        box = {'xmin':box_left, 'ymin':box_top, 'xmax':box_right, 'ymax':box_bottom}

                        if img_url not in images:
                            images[img_url] = {'box':box, 'path':img_filepath, 'img_id':img_id}
                            lut.append(img_url)
            
            fid.close()
            
            # randomly shuffle to create train and val sets
            random.shuffle(lut)
            endsize = len(lut) * 0.8
    
            for i in xrange(0, len(lut)):
                # crop out the box from each each and put into the right folder	
                try:
                    crop = cropFace(images[lut[i]]['path'], images[lut[i]]['box'])
                except Exception as e:
                    print(e)  
                    continue

                if i < endsize:
                    part = os.path.join('n%08d'%cid, 'n%08d_%s.jpg'%(cid, images[lut[i]]['img_id']))
                    croppath = os.path.join(outputFolder, 'train', part)
                    trainFile.write('%s %d\n'%(part, cid))
                else:
                    part = 'n%08d_%s.jpg'%(cid, images[lut[i]]['img_id'])
                    croppath = os.path.join(outputFolder, 'val', part) 
                    valFile.write('%s %d\n'%(part, cid))
                
                cv2.imwrite(croppath, crop)
            
            classFile.write('n%08d\n'%cid)
            cid += 1
                
                
classFile.close()
trainFile.close()
valFile.close()
