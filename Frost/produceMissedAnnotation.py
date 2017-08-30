from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import random
import time
if 1:
    annotation_file = "/home/xum/Desktop/current/data/coco/annotations/~instances_train2014.json"
    saved_file = "/home/xum/Desktop/current/data/coco/annotations/instances_train2014.json"
else:
    annotation_file = "/home/xum/Desktop/current/data/coco/annotations/~instances_minival2014.json"
    saved_file = "/home/xum/Desktop/current/data/coco/annotations/instances_minival2014.json"

cls = [3,44,84]
missing_rate =0.00

# load
print('loading annotations into memory...')
tic = time.time()
dataset = json.load(open(annotation_file, 'r'))
assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
print('Done (t={:0.2f}s)'.format(time.time()- tic))

# check length
import pprint
pprint.pprint(len(dataset))

'''
info v
images
 82783
 dataset["images"][0]:
{u'coco_url': u'http://mscoco.org/images/57870',
 u'date_captured': u'2013-11-14 16:28:13',
 u'file_name': u'COCO_train2014_000000057870.jpg',
 u'flickr_url': u'http://farm4.staticflickr.com/3153/2970773875_164f0c0b83_z.jpg',
 u'height': 480,
 u'id': 57870,
 u'license': 5,
 u'width': 640}

licenses v
annotations
 604907
 {u'area': 54652.9556,
 u'bbox': [116.95, 305.86, 285.3, 266.03],
 u'category_id': 58,
 u'id': 86,
 u'image_id': 480023,
 u'iscrowd': 0,
 u'segmentation': [[312.29,
   ...
   571.89]]}

categories
 {u'id': 1, u'name': u'person', u'supercategory': u'person'},
 ...
 {u'id': 90, u'name': u'toothbrush', u'supercategory': u'indoor'}]

'''

# stat
import numpy as np

cats_stat = np.zeros((100,))
for anno in dataset["annotations"]:
    cat = anno["category_id"]
    cats_stat[cat] = cats_stat[cat]+1


# draw
import matplotlib.pyplot as plt
N = len(cats_stat)
x = range(N)
width = 1/1.5
plt.bar(x, cats_stat, width, color="blue")
fig = plt.gcf()

# new annotations_3cls
annotations_3cls = []
for anno in dataset["annotations"]:
    if anno["category_id"]==cls[0] or anno["category_id"]==cls[1] or anno["category_id"]==cls[2]:
        annotations_3cls.append(anno)
all_anna_num = len(annotations_3cls) #=75247
missed_num = int(all_anna_num * missing_rate)

# new images_3cls
images_3cls = []
count = 0.0
for pict in dataset["images"]:
    id = pict["id"]
    count = count + 1
    if count%100==0:
        print(count,len(dataset["images"]))

    for anno in annotations_3cls:
        if anno["image_id"] == id:
            # print("add class number ",anno["id"])
            images_3cls.append(pict)
            break

# new categories
new_cat = []
for cat in dataset["categories"]:
    if cat["id"]==cls[0] or cat["id"]==cls[1] or cat["id"]==cls[2]:
        new_cat.append(cat)

# delete missing label
missed_index = []
for k in range(0,missed_num):
    missed_index = random.randint(0,len(annotations_3cls))
    discard = annotations_3cls.pop(missed_index)
remaind_anna_num = len(annotations_3cls)

# merge
dataset_3cls = {
    "info":dataset["info"],
    "images":images_3cls,
    "licenses":dataset["licenses"],
    "annotations":annotations_3cls,
    "categories":new_cat
}

# save
filePath = saved_file
with open(filePath, 'w') as fid:
        json.dump(dataset_3cls, fid)

print("missing label file saved to " + saved_file)
print("original annotation number: " , all_anna_num)
print("missing rate: " , missing_rate)
print("remaind annotation number: " , remaind_anna_num)
print("Categories are: ", new_cat)

