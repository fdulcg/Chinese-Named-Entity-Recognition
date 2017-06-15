# Chinese-Named-Entity-Recognition-
Chinese Named Entity Recognition via BLSTM-CNN and BIO Input.  Using Keras and Theano backend . 


### Requirements

*keras
*Theano
*Python3.5


### Chinese Named Entity Recognition using Bi-directional LSTMs and Convolutional Nerual Network using Keras. Added BIO sequence as the other input layer of our model.


### File description
./data : Labeled Chinese text data
./embedding : pre-trained embedding
./model : Well-trained model Repository
./pred_results : Predict results using our well-trained model on test data
./test_demo : Randomly selected Chinese texts and their ner result.
./loader.py  loader_4t.py : Data preparation in 7 categories(B-LOC,I-LOC,B-ORG,I-ORG,B-PER,I-PER,O) and 4 categories(PER,LOC,ORG,O)
./model.py model_4t.py : Chinese NER via BLSTM model in 7 categories(B-LOC,I-LOC,B-ORG,I-ORG,B-PER,I-PER,O) and 4 categories(PER,LOC,ORG,O)
./model_cnn.py model_cnn_4t.py : Chinese NER via BLSTM-CNN model in 7 categories(B-LOC,I-LOC,B-ORG,I-ORG,B-PER,I-PER,O) and 4 categories(PER,LOC,ORG,O)
./model_cnn_bio.py : Chinese NER via BLSTM-CNN-BIO input layer model in 7 categories(B-LOC,I-LOC,B-ORG,I-ORG,B-PER,I-PER,O) and 4 categories(PER,LOC,ORG,O)
./predict_bio predict_cnn predict_4t : Predict our model
./wordlist.txt : Chinese word dictionary :{ Chinese Character : Num}