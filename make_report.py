predictions_file=open('../reports/current_predictions.csv')
results_file=open('../reports/current_report.csv','w')

lines=predictions_file.readlines()
lines.pop(0)
lines.pop(0)
results_file.write('Iteration,True Positive,True Negative,False Positive,False Negative,Accuracy\n')
tp=0
tn=0
fp=0
fn=0
iteration=5000
# write=False
# accuracy=0
for line in lines:
	words=line.split(',')

	if 'USING' not in words[0]:
		# print words
		actual=words[2].rstrip()
		predicted=words[1].rstrip()
		# print words
		if(actual=="0" and predicted=="0"):
			tn=tn+1
		elif (actual=="1" and predicted=="1"):
			tp=tp+1
		elif (actual=="0" and predicted=="1"):
			fp=fp+1
		elif (actual=="1" and predicted=="0"):
			fn=fn+1
	else:
		accuracy=(tp*1.0+tn)/(tp+tn+fp+fn)
		results_file.write(str(iteration)+','+str(tp)+','+str(tn)+','+str(fp)+','+str(fn)+','+str(accuracy)+'\n')
		iteration=iteration+1000
		tp=0
		tn=0
		fp=0
		fn=0
accuracy=(tp*1.0+tn)/(tp+tn+fp+fn)
results_file.write(str(iteration)+','+str(tp)+','+str(tn)+','+str(fp)+','+str(fn)+','+str(accuracy)+'\n')
print "DONE"


