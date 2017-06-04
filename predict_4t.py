# coding=utf-8
import os, sys, time
#os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu"    
import numpy as np
import scipy.io
import  h5py
import codecs
time.clock()
np.random.seed(1337) 


from sklearn.externals import joblib
from sklearn import cross_validation
import keras
from keras.models import Model, Sequential, model_from_yaml
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from keras.utils import np_utils
from loader_4t import read_conll_file,prepare_data,load_data



tag_to_id = {"O": 1, "LOC": 2, 
                   "PER": 3, "ORG": 4 }    

# tag_to_id = {"O": 1, "B-LOC": 2, "I-LOC": 3,
#                  "B-PER": 4, "I-PER": 5, "B-ORG": 6, "I-ORG": 7 }  
id_to_tag = {i:t for t,i in tag_to_id.items()}

word_to_id, id_to_tag, train_data,dev_data,test_data= load_data(tag_to_id)
id_to_word = {ii:jj for jj,ii in word_to_id.items()}
print (len(id_to_word))

X_test = []
Y_test = []
for sentence in test_data:
    X_test.append(sentence['words'])
    Y_test.append(sentence['tags'])
teY = []
for sent in Y_test:
    sent2 = []
    for wtag in sent:
        xx = np.zeros(4)
        xx[wtag-1] = 1
        sent2.append(xx)

    teY.append(sent2)
            
X_test, Y_test  = np.array(X_test, dtype='int32'), np.array(teY, dtype='int32')
print(X_test.shape)

# print('word_num:', word_num, X_test.max())
#X = X[:1000]

modelfile = './model/model_4t.h5'

with open('./model_4t.yaml') as fin: model = model_from_yaml(fin.read())
if os.path.exists(modelfile): model.load_weights(modelfile)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Z = model.predict(X_test , batch_size=256)
print(Z.shape)

# tZ = np_utils.probas_to_classes(Z.reshape((-1,1)))
amount = 0
score = 0
entityNum = 0
findentityNum = 0
flagY = []
flagZ = []
with open('pred_results/NERresult0516_4t01.txt' , 'w', encoding='utf-8') as fout:
    for x,y, z in zip(X_test,Y_test, Z):
        for xx, yy, zz in zip(x,y,z ):
            tagN = 0
            for i in range(4):
                if yy[i] == 1:
                    tagN = i+1
                    break
            tagY = id_to_tag[tagN]
            flagY.append(tagY)
            maxpro = 0
            otagN = 0
            for j in range(4):
                if zz[j] > maxpro:
                    maxpro = zz[j]
                    otagN = j+1
            tagZ = id_to_tag[otagN]
            flagZ.append(tagZ)
            word = id_to_word[xx]
            if word!='<PAD>':
                amount+=1
                if tagY == tagZ:
                    score = score+1
                fout.writelines([str(word)+'  '+str(tagY) +'   '+str(tagZ)+ '\n'])

print (len(flagZ))
print (len(flagY))
correctEntityNER = 0
for e in range(len(flagY)):
    if ((flagY[e] == 'ORG') or (flagY[e] == 'LOC') or (flagY[e] == 'PER')):
        entityNum +=1
        j = e+1
        isTure = False
        if flagZ[e] == flagY[e]:
        	isTure = True
        for l in range(50):
            if flagY[j+l] != flagY[i]:
                e = j+l
                break
            else:
            	if flagZ[j+l]!=flagY[j+l]:
            		isTure = False
        if isTure:
        	correctEntityNER += 1
        	
for i in range(len(flagZ)):
    if ((flagZ[i] == 'ORG') or (flagZ[i] == 'LOC') or (flagZ[i] == 'PER')):
        findentityNum +=1
        j = i+1
        for l in range(50):
            if flagY[j+l] != flagY[i]:
                e = j+l
                break


Precision =  correctEntityNER/findentityNum   
Recall = correctEntityNER/entityNum
print ('Num of Character :　'+ str(amount))
print ('Num of Character NER :　'+ str(score))
print ('Character Precison　'+ str(score/amount))
print ('Num of entity : ' + str(entityNum))
print ('Num of entity NER :　'+str(findentityNum))
print ('Num of entity correct : '+str(correctEntityNER))
print ('NER Precision :　'+str(Precision))
print ('NER Recall : '+str(Recall))
print ('NER F1 : ' + str(2*Precision*Recall/(Precision+Recall)))