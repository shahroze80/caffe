CAFFE_ROOT=~/caffe
LMDB=../input/train_lmdb
OUTPUT=../input/mean.binaryproto
$CAFFE_ROOT/build/tools/compute_image_mean $LMDB $OUTPUT