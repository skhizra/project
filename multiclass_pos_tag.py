#text features with POS tags 


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn import metrics
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn import svm


# read the csv to the dataframe for emojis dataset
data = pd.DataFrame.from_csv('emoji_tweets.csv', encoding='utf-8')


# read the csv to the dataframe for dataset without emojis
#data = pd.DataFrame.from_csv('without_emoji_tweets.csv', encoding='utf-8')


#calculates the term frequencies of the pos tags

def pos_tagging(x):
    pos_tag = { 'CC': 0, 'CD': 0, 'DT':0, 'DT':0, 'EX':0, 'FW':0, 'IN':0, 'JJ':0, 'JJR':0, 'JJS':0,
'LS':0, 'MD':0, 'NN':0, 'NNS':0, 'NNP':0, 'NNPS':0, 'PDT':0, 'POS':0, 'PRP':0, 'PRP$':0,
'RB':0, 'RBR':0, 'RBS':0, 'RP':0, 'TO':0, 'UH':0, 'VB':0, 'VBD':0, 'VBG':0, 'VBN':0, 
'VBP':0, 'VBZ':0, 'WDT':0, 'WP':0, 'WP$':0, 'WRB':0}	

    t = nltk.pos_tag(nltk.word_tokenize(x))
    tag_list = [p[1] for p in t]
    for k in tag_list:
        for i in pos_tag:
            if k == i:
                pos_tag[i] = pos_tag[i] + 1      
                       
    for j in pos_tag:
        pos_tag[j] = pos_tag[j]/len(tag_list)

    return(pos_tag)

    


# calls the function 
data.text = data.text.apply(lambda x: pos_tagging(x))


#cross validation 
cv = cross_validation.KFold(data.shape[0], n_folds=10)


for traincv, testcv in cv:
    
    vec = DictVectorizer()
    pos_vectorized = vec.fit_transform(data.text[traincv])
    
    
    #classifiers
    clf = OneVsRestClassifier(LogisticRegression())
    #clf = OneVsRestClassifier(svm.LinearSVC())

    # train the classifier
    clf = clf.fit(pos_vectorized, data.emotion[traincv])
    
    
    pos_vectorized1 = vec.transform(data.text[testcv])
    
    #predict emotion for the test data 
    predicted = clf.predict(pos_vectorized1)
    
    
    print("Accuracy =")
    
    print(np.mean(predicted == data.emotion[testcv]))
    
    print(metrics.confusion_matrix(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad']))
    
    
    target_names = ['joy', 'anger', 'fear', 'sad']
    
    print(metrics.classification_report(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad'], target_names=target_names))
    
