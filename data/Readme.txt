#######
SIGHAN Named Entity Recognition Data：
SIGHAN.NER.train
SIGHAN.NER.dev
SIGHAN.NER.test

Data from Github Repository https://github.com/zjy-ucas/ChineseNER
#######

*******
weiboNER.conll.train

Chinese weibo data as informal Chinese data，only used to evaluate our model's behaviors in informal Chinese text.
Data from Github Repository https://github.com/hltcoe/golden-horse
*******

Thanks for the work below.




######
weibo_4t.test : Adjust weibo data into 4 categories: ORG, PER, LOC and O for my experiment.
modified.NER.train
modified.NER.dev
modified.NER.test : Adjust SIGHAN data into 4 categories: ORG, PER, LOC and O for my experiment.
*******
loc2org.train.train : Added some Chinese cities' name and organizaion names.Combined a city name and a organization name to create a new 
organizaion name : 上海 LOC  大学 ORG ——》 上海大学  ORG

Aimming at solving the kind of error“ 上海（LOC）大学（ORG）” which frequently come out in case study.