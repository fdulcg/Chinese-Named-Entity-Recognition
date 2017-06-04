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
from loader import read_conll_file,doc_to_sentence



tag_to_id = {"O": 1, "LOC": 2,
                 "PER": 3, "ORG": 4}
id_to_tag = {i:t for t,i in tag_to_id.items()}

word_to_id = {}
with codecs.open('./wordlist.txt','r','utf-8') as wd:
    for each in wd:
        mapbt = str(each).split('    ')
        if len(mapbt)>1:
            word = str(mapbt[0]).strip()
            wordid = int(str(mapbt[1]))
            word_to_id[word] = wordid
id_to_word = {j:i for i, j in word_to_id.items()}
print ('************  id_to_word length     ********')
print (len(id_to_word))

X_test = []
with codecs.open('./testNER.txt','r','utf-8') as fout:
    for doc in fout:
        if '\r' in doc:
            doc = doc.replace('\r','')
        if '\n' in doc:
            doc = doc.replace('\n','')
        doc = doc_to_sentence(doc,50)
        for each in doc:
            sentence = []
            for word in each:
                sentence.append(word_to_id[word] if word in word_to_id
                             else word_to_id["<UNK>"])
            sentence += [word_to_id["<PAD>"]] * (50-len(each))
            X_test.append(sentence)
            
X_test = np.array(X_test, dtype='int32')
print(X_test.shape)

modelfile = './model/model_cnn.h5'

with open('./model_cnn.yaml') as fin: model = model_from_yaml(fin.read())
if os.path.exists(modelfile): model.load_weights(modelfile)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Z = model.predict(X_test, batch_size=256)
print(Z.shape)

# tZ = np_utils.probas_to_classes(Z.reshape((-1,1)))

with open('pred_results/NERresult_cnn.txt' , 'w', encoding='utf-8') as fout:
    for x, z in zip(X_test, Z):
        for xx, zz in zip(x,z ):
            maxpro = 0
            otagN = 0
            for j in range(4):
                if zz[j] > maxpro:
                    maxpro = zz[j]
                    otagN = j+1
            tagZ = id_to_tag[otagN]
            word = id_to_word[xx]
            if (str(word)!='<UNK>' and str(word)!='<PAD>'):
                fout.writelines([str(word)+'   '+str(tagZ)+ '\n'])
