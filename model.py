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
    if not embedding_layer:
        embedded = embedding_layer(seq_input)
        print('Loading pretrained embedding')
    else:
        print ('Default embedding')
        embedded = Embedding(len(id_to_word)+1, 50 , input_length=seqlen,mask_zero = True,dropout=0.2)(seq_input)
    
    forwards = LSTM(128,return_sequences=True )(embedded)
    backwards = LSTM(128,return_sequences=True,go_backwards = True)(embedded)
    merged = merge([forwards,backwards],mode='concat',concat_axis=-1)
    after_dp = Dropout(0.2)(merged)
    
    outputs = TimeDistributed(Dense(7,activation='softmax'))(after_dp)
    #output = TimeDistributed(Dense(1,activation='sigmod'))
    model = Md(input = seq_input, output=outputs)
    with open('model/model.json', 'w') as fout: fout.write(model.to_json())
    modelfile = './model/model.h5'


    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    with open('model/model.yaml', 'w') as fout: fout.write(model.to_yaml())

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
        with open('./results/ner.txt' , 'w', encoding='utf-8') as fout:
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

def get_embedding(id_to_word):
    EMBEDDING_DIM = 100
    MAX_SEQLEG = 100
    embeddings_index = {}
    with open('./embedding/wiki_word2vec.pkl', "rb") as f:
        word_vec = pickle.load(f)
    embedding_matrix = np.zeros((len(id_to_word)+1 , EMBEDDING_DIM))
    for i , word in id_to_word.items():
        embedding_vector = word_vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print ('#################')
    print (len(embedding_matrix))
    print ('#################')
    
    embedding_layer = Embedding(len(id_to_word)+1, EMBEDDING_DIM, 
                               input_length=MAX_SEQLEG, 
                               weights=[embedding_matrix],
                               mask_zero = True,
                               trainable=False)
    return embedding_layer
    #print ('Found %s word vectors')% len(embeddings_index)

def testNer(X,Y,show=False):
    Z = model.predict(X, batch_size=256,verbose=1)
    print(Z.shape)

    tZ = np_utils.probas_to_classes(Z.reshape((-1,1)))
    tY = np_utils.probas_to_classes(Y.reshape((-1,1)))

    TP = (tZ + tY == 2).sum()
    PP = (tZ == 1).sum()
    RR = (tY == 1).sum()
    prec, reca = TP / PP, TP / RR
    F1 = 2 * prec * reca / (prec + reca)
    ret = 'P=%d/%d %.5f\tR=%d/%d %.5f\tF1=%.5f' % ( TP, PP, prec, TP, RR, reca, F1)
    print(ret)
    if show:
        with open('results/ner.txt' , 'w', encoding='utf-8') as fout:
            for x, y, z in zip(X, tY.reshape((-1, seqlen)), tZ.reshape((-1, seqlen))):
                es = []
                for wid, yy, zz in zip(x, y, z):
                    wd = id_to_word[wid]
                    if yy > 0: wd = '#' + wd + '#'
                    if zz > 0: wd = '@' + wd + '@'
                    es.append(wd)
                    if wid == 0: break
                fout.write(' '.join(es) + '\n')
    return ret
class Model(object):
    def __init__(self, name, word_to_id, id_to_tag, parameters):

        self.logger = get_logger(name)
        self.params = parameters
        self.num_words = len(word_to_id)
        self.learning_rate = self.params.lr
        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = tf.contrib.layers.xavier_initializer
        self.tags = [tag for i, tag in id_to_tag.items()]
        self.tag_num = len(self.tags)
        
        
        # add placeholders for the model
        # Keras x_train = x_train.astype('float32')
        self.inputs = tf.placeholder(dtype=tf.int32,
                                     shape=[None, self.params.word_max_len],
                                     name="Inputs")
        self.labels = tf.placeholder(dtype=tf.int32,
                                     shape=[None, self.params.word_max_len],
                                     name="Labels")
        self.lengths = tf.placeholder(dtype=tf.int32,
                                      shape=[None],
                                      name="Lengths")
        if self.params.feature_dim:
            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.params.word_max_len,
                                                  self.params.feature_dim],
                                           name="Features")
        self.dropout = tf.placeholder(dtype=tf.float32,
                                      name="Dropout")
        # model.add(Dropout(0.35))
        #embedded = Embedding(word_num, 50, input_length=seqlen, dropout=0.2)(seq_input)
        #forwards = LSTM(128, return_sequences=True)(merged_input) 
        #backwards = LSTM(128, return_sequences=True, go_backwards=True)(merged_input)
        #merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
        #after_dp = Dropout(0.2)(merged)      ZZ  
        # get embedding of input sequence
        embedding = self.get_embedding(self.inputs, word_to_id)
        # apply dropout on embedding
        
        rnn_inputs = tf.nn.dropout(embedding, self.dropout)
        # concat extra features with embedding
        if self.params.feature_dim:
            rnn_inputs = tf.concat(2, [rnn_inputs, self.features])
        # extract features
        rnn_features = self.bilstm_layer(rnn_inputs)
        # projection layer
        self.scores = self.project_layer(rnn_features, self.tag_num)
        # calculate loss of crf layer
        self.trans, self.loss = self.loss_layer(self.scores, self.tag_num)
        # optimizer of the model
        self.opt = tf.train.AdamOptimizer(self.learning_rate)
        # apply grad clip to avoid gradient explosion
        grads_vars = self.opt.compute_gradients(self.loss)
        capped_grads_vars = [(tf.clip_by_value(g, -self.params.clip, self.params.clip), v)
                             for g, v in grads_vars]  # gradient capping
        self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

  

    def bilstm_layer(self, inputs):
        # bidirectional lstm layer for feature extration
        with tf.variable_scope("BiLSTM"):
            forwards = LSTM(150)()
            fw_cell = rnn_cell.LSTMCell(self.params.word_hidden_dim,
                                        use_peepholes=True,
                                        initializer=self.initializer())
            bw_cell = rnn_cell.LSTMCell(self.params.word_hidden_dim,
                                        use_peepholes=True,
                                        initializer=self.initializer())
            length64 = tf.cast(self.lengths, tf.int64)
            forward_output, _ = tf.nn.dynamic_rnn(
                fw_cell,
                inputs,
                dtype=tf.float32,
                sequence_length=self.lengths,
                scope="fw"
            )
            backward_output, _ = tf.nn.dynamic_rnn(
                bw_cell,
                tf.reverse_sequence(inputs, length64, seq_dim=1),
                dtype=tf.float32,
                sequence_length=self.lengths,
                scope="bw"
            )
            backward_output = tf.reverse_sequence(backward_output, length64, seq_dim=1)
            # concat forward and backward outputs into a 2*hiddenSize vector
            outputs = tf.concat(2, [forward_output, backward_output])
            #merged = merge([forwards,backwards],mode='concat',concat_axis=-1)
            S = tf.reshape(outputs, [-1, self.params.word_hidden_dim * 2])
            return lstm_features

    def project_layer(self, lstm_features, tag_num):
        # projection layer
        with tf.variable_scope("Project",
                               initializer=self.initializer()):
            w1 = tf.get_variable(
                'W1',
                [self.params.word_hidden_dim * 2, tag_num],
                regularizer=tf.contrib.layers.l2_regularizer(0.001))
            b1 = tf.get_variable(
                'b1', [tag_num])
            scores = tf.batch_matmul(lstm_features, w1) + b1
            scores = tf.reshape(scores, [-1, self.params.word_max_len, tag_num])
            return scores

    def loss_layer(self, scores, tag_num):
        # crf layer
        with tf.variable_scope("CRF"):
            trans = tf.get_variable('trans',
                                    shape=[tag_num, tag_num],
                                    initializer=self.initializer())
            log_likelihood, _ = crf_log_likelihood(scores,
                                                   self.labels,
                                                   self.lengths,
                                                   trans)
            loss = tf.reduce_mean(-1.0 * log_likelihood)
            return trans, loss

    def create_feed_dict(self, is_train, **kwargs):
        feed_dict = {
            self.inputs: kwargs["words"],
            self.lengths: kwargs["len"],
            self.features: kwargs["features"],
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.labels] = kwargs["tags"]
            feed_dict[self.dropout] = self.params.dropout
        return feed_dict

    def run_step(self, sess, is_train, batch):
        feed_dict = self.create_feed_dict(is_train, **batch)
        if is_train:
            loss, _ = sess.run(
                [self.loss, self.train_op],
                feed_dict)
            return loss
        else:
            scores = sess.run(self.scores, feed_dict)
            return scores

    @staticmethod
    def decode(scores, lengths, matrix):
        # inference final labels usa viterbi Algorithm
        paths = []
        for score, length in zip(scores, lengths):
            score = score[:length]
            path, _ = viterbi_decode(score, matrix)
            paths.append(path)
        return paths

    def valid(self, sess, data):
        trans = self.trans.eval()
        total_correct = 0
        total_labels = 0
        for batch in data.iter_batch():
            lengths = batch["len"]
            tags = batch["tags"]
            scores = self.run_step(sess, None, batch)
            batch_paths = self.decode(scores, lengths, trans)
            batch_correct, batch_total = calculate_accuracy(tags, batch_paths, lengths)
            total_correct += batch_correct
            total_labels += batch_total
        return total_correct / total_labels

    def predict(self, sess, data):
        results = []
        trans = self.trans.eval()
        for batch in data.iter_batch():
            tags = batch["tags"]
            lengths = batch["len"]
            str_lines = batch["str_lines"]
            end_of_doc = batch["end_of_doc"]
            scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(batch)):
                result = []
                for char, gold, pred in zip(str_lines[i][:lengths[i]],
                                            tags[i][:lengths[i]],
                                            batch_paths[i][:lengths[i]]):
                    result.append(" ".join([char, self.tags[int(gold)], self.tags[int(pred)]]))
                results.append([result, end_of_doc[i]])
        return results

if __name__ == '__main__':
    tag_to_id = {"O": 1, "B-LOC": 2, "I-LOC": 3,
                 "B-PER": 4, "I-PER": 5, "B-ORG": 6, "I-ORG": 7 }
    # tag_to_id = {"O": 1, "LOC": 2, 
    #                "PER": 3, "ORG": 4 }        
    id_to_word, id_to_tag, train_data , dev_data , test_data= load_data(tag_to_id)
    # train_manager = BatchManager(train_data, len(id_to_tag), 100, 128)
    # dev_manager = BatchManager(dev_data, len(id_to_tag), 100, 128)
    # test_manager = BatchManager(test_data, len(id_to_tag), 100, 128)
    embedding_layer = get_embedding(id_to_word)
    create_model(id_to_word,train_data,dev_data,test_data,embedding_layer)
    # ret = testNer(X_test,Y_test)

    