from sklearn.metrics import f1_score,precision_score,recall_score,matthews_corrcoef

probability = open('modelsave/hlung_epoch200.txt')
lines = probability.readlines()
a = []
for i in lines:
    temp = i.strip('\n')
    temp_list = temp.split(',')
    t = []
    for j in range(1,len(temp_list)-1):
        t.append(float(temp_list[j]))
    a.append(t)

labela = []
for i in a:
    labela.append(i.index(max(i))+1)
probability.close()

labelb = open('finaldata/hlung/test_labels.csv')
linesb = labelb.readlines()
a = linesb[0].strip('\n').split(',')
b= []
for i in range(1,len(a)):
    b.append(int(a[i]))

count = 0
for i in range(len(b)):
    if (b[i]==labela[i]):
        count +=1
print('accuracy:',count/len(b))
f1 = f1_score(y_true=b,y_pred=labela,average='macro')
precision = precision_score(y_true=b,y_pred=labela,average='macro')
recall = recall_score(y_true=b,y_pred=labela,average='macro')
mcc = matthews_corrcoef(y_true=b,y_pred=labela,sample_weight=None)
print('f1:',f1)
print('precision:',precision)
print('recall:',recall)
print('mcc:',mcc)