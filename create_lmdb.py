import os
import glob
import random
import numpy as np
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227
IMG_PATH='../input/data/'
train_lmdb = '../input/train_lmdb'
validation_lmdb = '../input/validation_lmdb'
train_data=open('train.csv')
validation_data=open('validation.csv')


def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    # img[:, :] = cv2.equalizeHist(img[:, :])
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=img.tostring()        )

def parse_line(line):
    words=line.split(',')
    label=words[1].strip()
    img_path=IMG_PATH+words[0].strip()
    return img_path,int(label)

print 'Creating train_lmdb'
os.system('rm -rf  ' + train_lmdb)
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    in_idx=1
    for line in train_data.readlines():
        img_path,label=parse_line(line)
        img_name=img_path.split('/')[3]
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
        in_idx=in_idx+1
in_db.close()


print '\nCreating validation_lmdb'
os.system('rm -rf  ' + validation_lmdb)
in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    in_idx=1
    for line in validation_data.readlines():
        img_path,label=parse_line(line)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR )
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
        in_idx=in_idx+1
in_db.close()

print '\nFinished processing all images'
