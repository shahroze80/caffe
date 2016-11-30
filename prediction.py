import sys
# sys.path.append("/home/omair/caffe-master/python")
import caffe
import numpy as np
import Image
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
import numpy as np
import lmdb
from caffe.proto import caffe_pb2
import io

# free parameters
TEST_DB_PATH = '../input/test_lmdb'
ITERATIONS_START=1000
ITERATIONS_END=56000
SNAPSHOT=1000   # constant
SNAPSHOT_PREFIX='caffe_model_1'
DEPLOT_FILE = '../caffe_models/caffe_model_1/deploy.prototxt'
PRETRAINED_MODEL_LOCATION = '../caffe_models/caffe_model_1/iterations/'

PREDICTIONS_FILE=open('../reports/current_predictions.csv','w')
MEAN_FILE='../input/mean.binaryproto'
###############

caffe.set_mode_gpu()
PREDICTIONS_FILE.write('img_path,predicted,actual,prob\n')

# loading mean file
mean_blob = caffe_pb2.BlobProto()
with open(MEAN_FILE) as f:
    mean_blob.ParseFromString(f.read())
mu= np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

# generating all model names
model_files=[]
for i in range(ITERATIONS_START,ITERATIONS_END+1):
    if (float(i)%SNAPSHOT==0):
        model_file= SNAPSHOT_PREFIX+'_iter_'+str(i)+'.caffemodel'
        model_files.append(model_file)

# # prediction code
i=1
for pretrained in model_files:
    PREDICTIONS_FILE.write("USING MODEL "+pretrained+'\n')
    # net1 = caffe.Classifier(DEPLOT_FILE, 
    #                         PRETRAINED_MODEL_LOCATION+pretrained,
    #                         mean=mu,
    #                         raw_scale=255,
    #                         input_scale=1.0
    #                         # image_dims=(1,227,227)
    #                         )
    net2=caffe.Net(DEPLOT_FILE, 
                    PRETRAINED_MODEL_LOCATION+pretrained,
                    caffe.TEST)

    lmdb_env = lmdb.open(TEST_DB_PATH)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor() 
    print "USING MODEL "+pretrained  
    for key, value in lmdb_cursor:
        # key is image name and value is the true label
        print i
        i=i+1
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        label = int(datum.label)
        image = caffe.io.datum_to_array(datum)
        image = image.transpose(1,2,0)
        image2 = image.astype(np.float32)
        # prediction = net1.predict([image2], oversample=False)
        # x=np.zeros((1,1,227,227), dtype=np.float32)
        x=np.ones((1,1,227,227), dtype=np.float32)

        
        prediction=net2.forward(data=x)

        # predicted_label=np.argmax(prediction)
        # prob= prediction[0,predicted_label]
        # PREDICTIONS_FILE.write(key+','+str(predicted_label)+','+str(label)+','+str(prob)+'\n' )


# making Summary report
# PREDICTIONS_FILE=open('predictions.csv','r')
# results_file=open('report_interim.csv','w')
# lines=PREDICTIONS_FILE.readlines()
# lines.pop(0)
# lines.pop(0)
# results_file.write('Iteration,True Positive,True Negative,False Positive,False Negative\n')
# tp=0
# tn=0
# fp=0
# fn=0
# ITERATIONS_START=60000
# write=False
# for line in lines:
#     words=line.split(',')
#     if 'USING' not in words[0]:
#         actual=words[2].rstrip()
#         predicted=words[1].rstrip()
#         if(actual=="0" and predicted=="0"):
#             tn=tn+1
#         elif (actual=="1" and predicted=="1"):
#             tp=tp+1
#         elif (actual=="0" and predicted=="1"):
#             fp=fp+1
#         elif (actual=="1" and predicted=="0"):
#             fn=fn+1
#     else:
#         results_file.write(str(ITERATIONS_START)+','+str(tp)+','+str(tn)+','+str(fp)+','+str(fn)+'\n')
#         ITERATIONS_START=ITERATIONS_START+SNAPSHOT
#         tp=0
#         tn=0
#         fp=0
#         fn=0
# results_file.write(str(ITERATIONS_START)+','+str(tp)+','+str(tn)+','+str(fp)+','+str(fn)+'\n')



