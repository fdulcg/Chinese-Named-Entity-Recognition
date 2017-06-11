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
from loader import read_conll_file,prepare_data,load_data



# tag_to_id = {"O": 1, "LOC": 2, 
#                    "PER": 3, "ORG": 4 }    

tag_to_id = {"O": 1, "B-LOC": 2, "I-LOC": 3,
                 "B-PER": 4, "I-PER": 5, "B-ORG": 6, "I-ORG": 7 }  
id_to_tag = {i:t for t,i in tag_to_id.items()}

word_to_id, id_to_tag, train_data , dev_data , test_data= load_data(tag_to_id)
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
        xx = np.zeros(7)
        xx[wtag-1] = 1
        sent2.append(xx)
    teY.append(sent2)
            
X_test, Y_test  = np.array(X_test, dtype='int32'), np.array(teY, dtype='int32') 
print(X_test.shape)
# print('word_num:', word_num, X_test.max())
#X = X[:1000]

modelfile = './model/model.h5'

with open('model/model.yaml') as fin: model = model_from_yaml(fin.read())
if os.path.exists(modelfile): model.load_weights(modelfile)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Z = model.predict(X_test, batch_size=256)
print(Z.shape)

# tZ = np_utils.probas_to_classes(Z.reshape((-1,1)))
amount = 0
score = 0
entityNum = 0
findentityNum = 0
flagY = []
flagZ = []
with open('pred_results/NERresult0516_blstm.txt' , 'w', encoding='utf-8') as fout:
    for x,y, z in zip(X_test,Y_test, Z):
        for xx, yy, zz in zip(x,y,z ):
            tagN = 0
            for i in range(7):
                if yy[i] == 1:
                    tagN = i+1
                    break
            tagY = id_to_tag[tagN]
            flagY.append(tagY)
            maxpro = 0
            otagN = 0
            for j in range(7):
                if zz[j] > maxpro:
                    maxpro = zz[j]
                    otagN = j+1
            tagZ = id_to_tag[otagN]
            flagZ.append(tagZ)
            word = id_to_word[xx]
            if word!='<PAD>':
                amount = amount+1
                if ((tagY == 'B-ORG') or (tagY == 'B-LOC') or (tagY == 'B-PER')):
                    entityNum = entityNum + 1 
                if tagY == tagZ:
                    score = score+1
                    if ((tagY == 'B-ORG') or (tagY == 'B-LOC') or (tagY == 'B-PER')):
                        findentityNum = findentityNum + 1 
            # #wd = ee.id2wd[wid]
            # wd = tt
            # if zz > 0.5: wd = wd + '@%.3f' % zz
            # es.append(wd)
            # if wid == 0: break
                fout.writelines([str(word)+'  '+str(tagY) +'   '+str(tagZ)+ '\n'])

print (len(flagZ))
print (len(flagY))
correctEntityNER = 0
for i in range(len(flagZ)):
    if ((flagZ[i] == 'B-ORG') or (flagZ[i] == 'B-LOC') or (flagZ[i] == 'B-PER')):
        isflag = False
        if flagY[i] == flagZ[i]:
            isflag = True
            j = i+1
            for l in range(50):
                if ((flagZ[j+l]=='O') or ('B' in flagZ[j+l])):
                    break
                else:
                    if flagZ[j+l]!=flagY[j+l] :
                        isflag = False
                        break
            if isflag:
                correctEntityNER+=1
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