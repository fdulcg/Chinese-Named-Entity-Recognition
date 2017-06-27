# coding=utf-8
import os, sys, time
import numpy as np
from sklearn.externals import joblib
from sklearn import cross_validation
np.random.seed(1337) 

import keras
from keras.models import  model_from_json,Sequential
from keras.models import Model as Md
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import *
from keras.utils import np_utils
from utils import get_logger, load_word2vec, calculate_accuracy
import pickle
from loader import load_data


def prepareData(sentences):
    word = []
    tag = []
    for each in sentences:
        for i in each:  
            try:
                word.append(i[0])
                tag.append(i[1])
            except Exception as e:
                pass

    return word,tag


def  create_model(word_to_id,train_data,dev_data,test_data,embedding_layer=None):
    
    id_to_word = {jj:ii for ii,jj in word_to_id.items()}
    word_num = len(id_to_word)

    X = []
    Y = []
    for sentence in train_data:
        # app = False
        # for wordtag in sentence['tags']:
        #     if wordtag!=1:
        #         app = True
        #         break
        # if app:
        X.append(sentence['words'])
        Y.append(sentence['tags'])
    newY = []
    for sent in Y:
        sent2 = []
        for wtag in sent:
            xx = np.zeros(7)
            xx[wtag-1] = 1
            sent2.append(xx)
        newY.append(sent2)
            
    X, Y = np.array(X, dtype='int32'), np.array(newY, dtype='int32')
    # X, X_test, Y, Y_test = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=123)
    print(X.shape[0])
    print(Y.shape[0])
    seqlen = 50 
    # Y = Y.reshape((-1, seqlen, 1))

    
    seq_input = Input(shape=(seqlen,),dtype='int32')
    if False:
        embedded = embedding_layer(seq_input)
        print('Loading pretrained embedding')
    else:
        print ('Default embedding')
        embedded = Embedding(len(id_to_word)+1, 50 , input_length=seqlen,dropout=0.2)(seq_input)
    
    ## Bi-directional LSTM Layer 
    forwards = LSTM(128,return_sequences=True )(embedded)
    backwards = LSTM(128,return_sequences=True,go_backwards = True)(embedded)
    merged = merge([forwards,backwards],mode='concat',concat_axis=-1)
    after_dp = Dropout(0.2)(merged)
     
    ## Convoluntion1D Layer
    half_window_size=2
    paddinglayer=ZeroPadding1D(padding=half_window_size)(embedded)
    conv=Conv1D(nb_filter=50,filter_length=(2*half_window_size+1),border_mode='valid')(paddinglayer)
    conv_d = Dropout(0.1)(conv)
    dense_conv = TimeDistributed(Dense(50))(conv_d)
    ## Concat Bi-LSTM Layer and Convolution Layer 
    rnn_cnn_merge=merge([after_dp,dense_conv], mode='concat', concat_axis=2)

    #output Layer and CRFLayer 
    outputs = TimeDistributed(Dense(7,activation='softmax'))(rnn_cnn_merge)

    model = Md(input = seq_input, output=outputs)
    with open('model/model_cnn.json', 'w') as fout: fout.write(model.to_json())
    modelfile = './model/model_cnn.h5'


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    with open('model/model_cnn.yaml', 'w') as fout: fout.write(model.to_yaml())


	## Number of Epoch should be decided by actual experiment environment
    if X.shape[0] > 5000: nb_epoch = 200
    if X.shape[0] > 10000: nb_epoch = 150
    if X.shape[0] > 40000: nb_epoch = 10
    if X.shape[0] > 70000: nb_epoch = 7
    if X.shape[0] > 100000: nb_epoch = 4
    model.fit(X, Y, batch_size=256, callbacks=[ModelCheckpoint(modelfile,save_best_only=True)], 
             validation_split=0.1,nb_epoch=nb_epoch)
    
    print ("###########     dev        ##############")

    X_dev = []
    Y_dev = []
    for sentence in dev_data:
        X_dev.append(sentence['words'])
        Y_dev.append(sentence['tags'])
    tempY = []
    for sent in Y_dev:
        sent2 = []
        for wtag in sent:
            xx = np.zeros(7)
            xx[wtag-1] = 1
            sent2.append(xx)
        tempY.append(sent2)
            
    X_dev, Y_dev = np.array(X_dev, dtype='int32'), np.array(tempY, dtype='int32')
    
    print(X_dev.shape[0])
    print(Y_dev.shape[0]) 

    loss_and_metrics = model.evaluate(X_dev,Y_dev,batch_size = 128)
    print (loss_and_metrics)

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
            
    X_test, Y_test = np.array(X_test, dtype='int32'), np.array(teY, dtype='int32')
    
    print ("###########     test        ##############")
    print(X_test.shape[0])
    print(Y_test.shape[0]) 
    Z = model.predict(X_test, batch_size=256,verbose=1)
    print(Z.shape)
 
    tZ = np_utils.probas_to_classes(Z.reshape((-1,1)))
    tY = np_utils.probas_to_classes(Y_test.reshape((-1,1)))

    TP = (tZ + tY == 2).sum()
    PP = (tZ == 1).sum()
    RR = (tY == 1).sum()
    prec, reca = TP / PP, TP / RR
    F1 = 2 * prec * reca / (prec + reca)
    ret = 'P=%d/%d %.5f\tR=%d/%d %.5f\tF1=%.5f' % ( TP, PP, prec, TP, RR, reca, F1)
    print(ret)
    show = True
    if show:
        amount = 0
        entityNum = 0
        findentityNum = 0
        score = 0
        with open('./results/ner_cnn.txt' , 'w', encoding='utf-8') as fout:
            for x, y, z in zip(X_test, Y_test, Z):
                for xx, yy, zz in zip(x,y,z ):
                    tagN = 0
                    for i in range(7):
                        if yy[i] == 1:
                            tagN = i+1
                            break
                    tagY = id_to_tag[tagN]
                    maxpro = 0
                    otagN = 0
                    for j in range(7):
                        if zz[j] > maxpro:
                            maxpro = zz[j]
                            otagN = j+1
                    tagZ = id_to_tag[otagN]
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
    print ('Num of Character :　'+ str(amount))
    print ('Num of Character NER :　'+ str(score))
    print ('Precison　'+ str(score/amount))
    print ('Num of entity : ' + str(entityNum))
    print ('Num of entity NER :　'+str(findentityNum))
    print ('NER accuracy :　'+str(findentityNum/entityNum))


if __name__ == '__main__':
    tag_to_id = {"O": 1, "B-LOC": 2, "I-LOC": 3,
                 "B-PER": 4, "I-PER": 5, "B-ORG": 6, "I-ORG": 7 }
    # tag_to_id = {"O": 1, "LOC": 2, 
    #                "PER": 3, "ORG": 4 }        
    id_to_word, id_to_tag, train_data , dev_data , test_data= load_data(tag_to_id)
    embedding_layer = get_embedding(id_to_word)
    create_model(id_to_word,train_data,dev_data,test_data,embedding_layer)

    