from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from datasets.imdb import imdb
# import datasets.ds_utils as ds_utils
# from model.config import cfg
# import os.path as osp
# import sys
# import os
import numpy as np
# import scipy.sparse
# import scipy.io as sio
# import pickle
import json
# import uuid
# # COCO API
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from pycocotools import mask as COCOmask

def ReadBoxFile(txtPath):
    annTxt =  np.loadtxt(txtPath,delimiter=',')
    boxes = annTxt[:,0:-1]
    classes = annTxt[:,-1]
    print(boxes.shape,classes.shape)
    return boxes,classes

def WriteCOCOFile(classes, boxes, filePath):
    # info
    info = {
       'contributor': "Frost M. Xu",
       'date_created':"2017-04-27",
       'description':"This is stable 1.0 version of the 2017 KAUST VCC IVUL COCO-style dataset.",
       'url':"https://ivul.kaust.edu.sa",
       'version':"1.0",
       'year':2017,
    }
    # licenses
    licenses = "not yet"
    # images
    images = []
    for cls_ind, cls in enumerate(classes):
        name = 'COCO_train2017_'+str(cls_ind+1).zfill(12)+'.jpg'
        inNum = cls_ind+1
        # {"file_name": name, "height":1080, "width": 1920, "id": inNum}
        image = {
            'date_captured':'2017-04-28',
            'file_name' : name,
            'height' : 1080,
            'id' : inNum,
            'license' : 0  ,
            'url' : 'None',
            'width' : 1920,
        }
        images.append(image)
    # categories
    category1 = {'id':1, 'name':'Plectropomus'}
    category2 = {'id':2, 'name':'Lethrinus'}
    category3 = {'id':3, 'name':'Lutjanus'}
    categories = [category1,category2,category3]
    # annotations
    annotations = []
    for cls_ind, cls in enumerate(classes):
        # bbox = [boxxy for boxxy in boxes[cls_ind,:]]
        bbox = [boxes[cls_ind, 1],boxes[cls_ind, 0],boxes[cls_ind, 3],boxes[cls_ind, 2]]
        inNum = cls_ind+1
        annotation = {
            'bbox':bbox,
            'category_id' : cls,
            'image_id' : inNum,
            'id' : inNum,
            'area':bbox[2]*bbox[3],
            'iscrowd': 0
        }
        annotations.append(annotation)


    results = {
        'annotations':annotations,
        'categories':categories,
        'images':images,
        'info':info,
        'licenses':licenses,
        'type':'instances'
    }
    with open(filePath, 'w') as fid:
        json.dump(results, fid)
    print('Writing results json to {}'.format(filePath))
    return results

boxes,classes = ReadBoxFile("/home/xum/Documents/Git/tf-faster-rcnn-BAI/Frost/GT of boxes/boxes.txt")
fakeDataset = WriteCOCOFile(classes, boxes, "/home/xum/Documents/Git/tf-faster-rcnn-BAI/Frost/test.json")
dataset = json.load(open("/home/xum/Documents/Git/tf-faster-rcnn-BAI/data/coco/annotations/instances_train2017.json", 'r'))
pass
print('dataset')
