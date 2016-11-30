with open('../input/annotation2.txt') as f:
	files=f.readlines()
import random


train=open('../input/train.txt','w')
test=open('../input/test.txt','w')
validation=open('../input/validation.txt','w')

n=0
for line in files:
	r=random.random()
	if (r <=0.1):
		test.write(line)
	elif (r>0.1 and r<0.2):
		validation.write(line)
	else:
		train.write(line)