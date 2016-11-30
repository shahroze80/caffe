import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

# free parameters
TEST_DB_PATH = '../input/test_lmdb'
ITERATIONS_START=1000
ITERATIONS_END=36000
SNAPSHOT=1000   # constant
SNAPSHOT_PREFIX='caffe_model_1'
DEPLOY_FILE = '../caffe_models/caffe_model_2/deploy.prototxt'
MODEL_LOCATION = '../caffe_models/caffe_model_2/iterations/'

PREDICTIONS_FILE=open('../reports/current_predictions.csv','w')
MEAN_FILE='../input/mean.binaryproto'
###############

# def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
#     # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
#     img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
#     return img

def parse_line(line):
    words=line.split(',')
    label=words[1].strip()
    img_path=IMG_PATH+words[0].strip()
    img_name=words[0].strip()
    return img_name,img_path,int(label)


# mean_blob = caffe_pb2.BlobProto()
# with open(mean_file) as f:
#     mean_blob.ParseFromString(f.read())

# mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
#     (mean_blob.channels, mean_blob.height, mean_blob.width))
# net = caffe.Net(DEPLOY_FILE,
#                 MODEL_LOCATION,
#                 caffe.TEST)


# transformer.set_mean('data', mean_array)     
   
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# x=[(k, v.data.shape) for k, v in solver.net.blobs.items()]
# test_file.write(x)

model_files=[]
for i in range(ITERATIONS_START,ITERATIONS_END+1):
    if (float(i)%SNAPSHOT==0):
        model_file= SNAPSHOT_PREFIX+'_iter_'+str(i)+'.caffemodel'
        model_files.append(model_file)

# lines=test_data.readlines()
PREDICTIONS_FILE.write('filename,predicted,actual\n')
# making prediction using each model
for model in model_files:
    PREDICTIONS_FILE.write("USING MODEL "+model+'\n')
    net = caffe.Net(DEPLOY_FILE, MODEL_LOCATION+model, caffe.TEST)
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_transpose('data', (2,0,1)) 
    # transformer.set_mean('data', mean_array) 
    lmdb_env = lmdb.open(TEST_DB_PATH)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor() 
    print "USING MODEL "+model  
    for key, value in lmdb_cursor:
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        true_label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        # image = image.transpose(1,2,0)
        # image = image.astype(np.float32)
        # key is image name and value is the true label
        # img_name,img_path,true_label=parse_line(line)
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # net.blobs['data'].reshape(,1,227,227) 
        # y=image.T
        net.blobs['data'].data[...] = image
        out = net.forward()
        prob = out['prob']
        predicted_label=prob.argmax()
        print key
        print prob
        print predicted_label
        print true_label
        print '-------'
        PREDICTIONS_FILE.write(key+','+str(predicted_label)+','+str(true_label)+','+str(prob)+'\n' )
