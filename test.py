datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(value)
true_label = int(datum.label)
image = caffe.io.datum_to_array(datum)
# image = image.transpose(1,2,0)
image = image.astype(np.float32)
# key is image name and value is the true label
# img_name,img_path,true_label=parse_line(line)
# img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
# net.blobs['data'].reshape(,1,227,227) 
# y=image.T
net.blobs['data'].data[...] = image
out = net.forward()
pred_probas = out['prob']
prediction=pred_probas.argmax()
print key
print pred_probas
print prediction
print true_label
print '-------'