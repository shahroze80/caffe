mkdir -f ../caffe_models/bvlc_googlenet/iterations/
~/caffe/build/tools/caffe train \
--solver /media/cvlab/50D08EC3D08EAEB2/shahroze/chest-xray/caffe_models/bvlc_googlenet/quick_solver.prototxt 2>&1 | \
tee /media/cvlab/50D08EC3D08EAEB2/shahroze/chest-xray/caffe_models/bvlc_googlenet/model_1_train.log
