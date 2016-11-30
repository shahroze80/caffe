rm -rf ../input/train_lmdb
GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset \
    --resize_height=227 --resize_width=227 --gray \
    ../input/ims/ \
    ../input/train.txt \
    ../input/train_lmdb

rm -rf ../input/validation_lmdb
GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset \
    --resize_height=227 --resize_width=227 --gray \
    ../input/ims/ \
    ../input/validation.txt \
    ../input/validation_lmdb

rm -rf ../input/test_lmdb
GLOG_logtostderr=1 $CAFFE_ROOT/build/tools/convert_imageset \
    --resize_height=227 --resize_width=227 --gray \
    ../input/ims/ \
    ../input/test.txt \
    ../input/test_lmdb


# Flags from tools/convert_imageset.cpp:
# -backend (The backend {lmdb, leveldb} for storing the result) type: string
#   default: "lmdb"
# -check_size (When this option is on, check that all the datum have the same
#   size) type: bool default: false
# -encode_type (Optional: What type should we encode the image as
#   ('png','jpg',...).) type: string default: ""
# -encoded (When this option is on, the encoded image will be save in datum)
#   type: bool default: false
# -gray (When this option is on, treat images as grayscale ones) type: bool
#   default: false
# -resize_height (Height images are resized to) type: int32 default: 0
# -resize_width (Width images are resized to) type: int32 default: 0
# -shuffle (Randomly shuffle the order of images and their labels) type: bool
#   default: false
