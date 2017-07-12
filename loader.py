import codecs
#coding = utf-8
import sys
print (sys.getdefaultencoding())  
import codecs
import jieba
import numpy as np

def read_file(path):
    """
    This function will Load sentences from your data file.
    A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []  
    for line in codecs.open(path, "rb", "utf8"):
        line = line.rstrip()
        #try:
            #print (line)
        #except Exception as e:
            #pass
        if not line:
            if len(sentence) > 0:
                if "DOCSTART" not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if '\t' in line:
                word = line.split('\t')
            else:
                word = line.split(" ")
            if word:
                if len(word)<=1:
                    print ('wrong+ ' + str(word))               
                #assert len(word) > 1, word
                sentence.append(word)
    if len(sentence) > 0:
        if "DOCSTART" not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def doc_to_sentence(doc, max_len):
    """
    This function will cut doc to sentences with ！|。|；,
    If sentence is longer than max_len, it will be cut with ，|、|／
    The function return a list of integers with length of each sentence
    """
    pattern1 = "。！|？；"
    pattern2 = "，、／。；"
    pre_index = -1
    pattern_index = -1
    sentences = []
    sentence = []
    for i, line in enumerate(doc):
        sentence.append(line)
        if i - pre_index > max_len-1:
            if pattern_index > pre_index:
                sentences.append(sentence[:pattern_index-pre_index])
                sentence = sentence[pattern_index-pre_index:]
                pre_index = pattern_index
            else:
                pre_index = i-1
                sentences.append(sentence)
                sentence = []
        else:
            if line[0] in pattern2:
                pattern_index = i
                if line[0] in pattern1:
                    sentences.append(sentence)
                    sentence = []
                    pre_index = i
    if sentence:
        sentences.append(sentence)
    return sentences


def word_mapping(data, min_freq):
    """
    create word dictionary
    min_freq set Minimal appearance of each Chinese word
    word to id is the mapping of word and number of times the word appeared
    """
    vocab = dict() 
    word_to_id = dict()
    word_id = 0
    for doc in data:
        for line in doc:
            word = line[0]
            if word not in word_to_id:
                if word in vocab:
                    vocab[word] += 1
                    if vocab[word] >= min_freq:
                        word_to_id[word] = word_id
                        word_id += 1
                else:
                    vocab[word] = 1
    word_to_id["<UNK>"] = word_id
    word_to_id["<PAD>"] = word_id+1
    
    with codecs.open('./wordlist.txt','a') as wf:
        for each in word_to_id.keys():
            try:
                wf.writelines(each+'    '+ str(word_to_id[each])+ '\n')
            except Exception as e:
                pass
           
    return word_to_id, {v: k for k, v in word_to_id.items()}


def prepare_data(data, word_to_id, tag_to_id, max_words):
    """
    Grasp as much as information though sometimes useless temporarily.
    In the data form that a model could accept
    """
    processed_data = []
    for doc in data:
        doc = doc_to_sentence(doc, max_words)
        len_doc = len(doc)

        for i, sentence in enumerate(doc):

            len_sen = len(sentence)
            str_words = []
            words = []
            tags = []
            for line in sentence:
                word = line[0].lower()  #get a Chinese Character  '科\tO'
                #try:
                    #print(i)
                    #print (word)
                #except Exception as e:
                    #pass 
                str_words.append(word)
                words.append(word_to_id[word] if word in word_to_id
                             else word_to_id["<UNK>"])
                tags.append(tag_to_id[line[-1]])
            words += [word_to_id["<PAD>"]] * (max_words-len_sen)
            tags += [tag_to_id["O"]] * (max_words-len_sen)
            features = np.zeros([max_words, 4],dtype=np.float32)
            index = 0
            # BIES tags
            for word in jieba.cut("".join(str_words)):
                len_word = len(word)
                if len_word == 1:
                    index += 1
                else:
                    index += 1
                    for i_ in range(len_word-2):
                        index += 1
                    index += 1
            processed_data.append({"str_line": str_words,
                                   "words": words,
                                   "tags": tags,
                                   "len": len_sen,
                                   "end_of_doc": i == len_doc-1})
    return processed_data


'''
	Entry Function of this loader.py file
	Usage: Read dataset file, create word_to_id dictionary , create train_data,test_data and dev_data
	Return: word_to_id Dict,id_to_tag Dict,train_data,dev_data,test_data 
'''
def load_data(tag_to_id,train,dev,test,word_freq,lengthofsen):

    train_file = read_file(train)
    dev_file = read_file(dev)
    # test_file = read_conll_file('./data/SIGHAN.NER.test')
    test_file = read_file(test)
    word_to_id, id_to_word = word_mapping(train_file+dev_file, word_freq)

    train_data = prepare_data(train_file, word_to_id, tag_to_id,lengthofsen)
    dev_data = prepare_data(dev_file, word_to_id, tag_to_id, lengthofsen)
    test_data = prepare_data(test_file, word_to_id, tag_to_id, lengthofsen)
    # train_data += dev_data
    print (len(word_to_id))
    print ('$$$$$$$$$$$$$$$$  word_to_id size   $$$$$$$$$$$$$$$$')

    return word_to_id, {v: k for k, v in tag_to_id.items()}, train_data, dev_data, test_data
    # return word_to_id,{v: k for k, v in tag_to_id.items()},train_data
