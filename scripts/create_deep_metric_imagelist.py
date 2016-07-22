import os
import sys
import itertools
import random

# Usage:
#   Inputs: Directory to crawl 
#           Class list file containing class_name class_id
#           maximum size of the final training set 
#           batchsize that is going to be used during training
#           the name of the output file 
#   Outputs: text file with rows containing imagepath label.
# The output file can be used by convert_imageset tool to create a lmdb.

def generate_neg_samples(num_negs, img_map, ignore_cid):
    negs = set()
    img_ids = img_map.keys()
    while len(negs) < num_negs:
        idxs = random.sample(xrange(len(img_ids)), num_negs)
        for idx in idxs:
            if img_map[img_ids[idx]][1] != ignore_cid:
                negs.add(img_ids[idx])         
    return list(negs);
    
root_folder = sys.argv[1]
class_list = open(sys.argv[2],'r').readlines()
max_training_set_size = int(sys.argv[3])
batch_size = int(sys.argv[4])
outfp = open(sys.argv[5],'w')

class_count = 0
cid_map = dict()
for line in class_list:
        parts = line.split()
        cid = class_count
        if len(parts) == 2:
            cid = parts[1]
        else:
            cid = class_count
        cid_map[parts[0].strip()] = cid
        class_count = class_count + 1

img_map = dict()
cid_img_map = dict()
for cname in cid_map:
    cid_img_map[cid_map[cname]] = []
    
dir_list = os.listdir(root_folder)
img_id = 0
for dirname in dir_list:
        cid = cid_map[dirname]
        files  = os.listdir(root_folder + '/' + dirname)
        for filename in files:
            imgpath = '''%s/%s/%s'''%(root_folder, dirname, filename)
            img_map[img_id] = (imgpath, cid)
            cid_img_map[cid].append(img_id) 
            img_id = img_id + 1

# calculate the total number of positive pairs 
num_pos_pairs = dict()
total_pos_pairs = 0;
for cid in cid_img_map:
    num_pos_pairs[cid] = len(list(itertools.combinations(cid_img_map[cid], 2))) + len(cid_img_map[cid])
    total_pos_pairs = total_pos_pairs + num_pos_pairs[cid]

subsample = min(1.0, float(max_training_set_size) / (total_pos_pairs * 2))

# select training pairs 
all_pos_pairs = []
all_neg_ids = []
for cid in cid_img_map:
    pos_pairs = list(itertools.combinations(cid_img_map[cid], 2))
    print str(cid) + ' ' + str(len(cid_img_map[cid])) + ' ' + str(len(pos_pairs))
    neg_ids = generate_neg_samples(len(pos_pairs), img_map, cid)

    # subsample the pairs to hit the target
    if subsample < 1:
        idxs = random.sample(xrange(len(pos_pairs)), int(round(subsample*len(pos_pairs))))
        pos_pairs = [pos_pairs[i] for i in idxs]
        idxs = random.sample(xrange(len(neg_ids)), int(round(subsample*len(neg_ids))))
        neg_ids = [neg_ids[i] for i in idxs]
    
    all_pos_pairs.extend(pos_pairs)
    all_neg_ids.extend(neg_ids)
   
# randomly shuffle the pairs
print 'Randomly shuffling data'
random.shuffle(all_pos_pairs)
print 'Shuffled pos pairs'
random.shuffle(all_neg_ids)
print 'Shuffled neg pairs'

# write batches out to file 
pn = 0
nn = 0
print 'Number of pos pairs ' + str(len(all_pos_pairs))
print 'Number of neg ids ' + str(len(all_neg_ids))

while pn < len(all_pos_pairs):
    batch = set()
    while len(batch) < batch_size:
        if pn >= len(all_pos_pairs) or nn >= len(all_neg_ids):
            print 'run out of pairs'
            pn = len(all_pos_pairs)
            break
        
        if img_map[all_pos_pairs[pn][0]][1] != img_map[all_pos_pairs[pn][1]][1]:
            print 'OUCH something is total bust'

        batch.add(all_pos_pairs[pn][0])
        batch.add(all_pos_pairs[pn][1])
        batch.add(all_neg_ids[nn])
        pn = pn + 1
        nn = nn + 1
    batch = list(batch)
    print str(len(batch)) + ' ' + str(pn) 
    if len(batch) >= batch_size:
        for n in xrange(len(batch)):
            fLine = '''%s %d\n'''%(img_map[batch[n]][0],img_map[batch[n]][1])
            outfp.write(fLine)

outfp.close()
