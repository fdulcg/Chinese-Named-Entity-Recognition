# coding=utf-8
import os, sys, time
#os.environ['THEANO_FLAGS'] = "mode=FAST_RUN,device=cpu"    
import numpy as np
import scipy.io
import  h5py
import codecs
import jieba
import jieba.posseg as pos
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
from loader import prepare_data,doc_to_sentence



tag_to_id = {"O": 1, "B-LOC": 2, "I-LOC": 3,
                 "B-PER": 4, "I-PER": 5, "B-ORG": 6, "I-ORG": 7 }
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
BIO_test = []
with codecs.open('test_demo/test_20170511.txt','r','utf-8') as fout:
    for doc in fout:
        if '\r' in doc:
            doc = doc.replace('\r','')
        if '\n' in doc:
            doc = doc.replace('\n','')
        doc = doc_to_sentence(doc,50)
        for each in doc:
            sentence = []
            origin = ''
            for word in each:
                origin+=word
                sentence.append(word_to_id[word] if word in word_to_id
                             else word_to_id["<UNK>"])
            sentence += [word_to_id["<PAD>"]] * (50-len(each))
            X_test.append(sentence)
            # Word segment using jieba , and part-of-speech , preparing BIO_test
            BIO = []
            sent = pos.cut(origin)
            for word in sent:
                if ((len(word.word)>1) and (word.flag == 'nr' or word.flag == 'ns' or word.flag == 'nt' or word.flag == 'nz')):
                    wordlist = []
                    for character in str(word.word):
                        wordlist.append(2)  # 2 indicate 'I'
                    wordlist[0] = 1         # 1 indicate 'B'
                    BIO = BIO+wordlist      
                else:
                    for character in str(word.word):
                        BIO.append(0)  # 0 indicate 'O'
            BIO += [0]* (50-len(BIO))
            BIO_test.append(BIO)


X_test = np.array(X_test, dtype='int32') 
BIO_test = np.array(BIO_test , dtype = 'int32')
print ('############### test and BIO length #################')
print(X_test.shape)
print (BIO_test.shape)
# print('word_num:', word_num, X_test.max())
#X = X[:1000]

modelfile = './model/model_bio.h5'

with open('./model_bio.yaml') as fin: model = model_from_yaml(fin.read())
if os.path.exists(modelfile): model.load_weights(modelfile)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Z = model.predict([X_test,BIO_test], batch_size=256)
print(Z.shape)

# tZ = np_utils.probas_to_classes(Z.reshape((-1,1)))

with open('test_demo/result/result_20170511.txt' , 'w', encoding='utf-8') as fout:
    for x, z in zip(X_test, Z):
        for xx, zz in zip(x,z ):
            maxpro = 0
            otagN = 0
            for j in range(7):
                if zz[j] > maxpro:
                    maxpro = zz[j]
                    otagN = j+1
            tagZ = id_to_tag[otagN]
            word = id_to_word[xx]
            if (str(word)!='<UNK>' and str(word)!='<PAD>'):
                fout.writelines([str(word)+'   '+str(tagZ)+ '\n'])
