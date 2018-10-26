
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import nltk
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import cross_validation





# read the csv to the dataframe for emojis dataset

data = pd.DataFrame.from_csv('emoji_tweets.csv', encoding='utf-8')


# read the csv to the dataframe for dataset without emojis

#data = pd.DataFrame.from_csv('without_emoji_tweets.csv', encoding='utf-8')




#reading positive and negative words from text file

positive_words = []
negative_words = []
     
with open('positive.txt', 'r') as f:
    positive_words.append(f.read().split('\n'))
    
with open('negative.txt', 'r') as f:
    negative_words.append(f.read().split('\n'))



liwc_list= []

#calculates the term frequency of the positive and negative words
def liwc_freq(x):
    negative_counter = 0

    positive_counter = 0

    t = nltk.word_tokenize(x)
    for word in t:
        if word in positive_words[0]:
            positive_counter = positive_counter + 1
        if word in negative_words[0]:
            negative_counter = negative_counter + 1
    word_dict = {}
    word_dict['pos_word'] = positive_counter/len(t)
    word_dict['neg_word'] = negative_counter/len(t)
    return word_dict


data.text = data.text.astype(str)

    

data.text = data.text.apply(lambda x: liwc_freq(x))


#corss validation
cv = cross_validation.KFold(data.shape[0], n_folds=10)



for traincv, testcv in cv:
    
    
    vec = DictVectorizer()
    pos_vectorized = vec.fit_transform(data.text[traincv])
    
    
    #cassifiers
    clf = OneVsRestClassifier(LogisticRegression())
    #clf = OneVsRestClassifier(svm.LinearSVC())
    
    
    clf = clf.fit(pos_vectorized, data.emotion[traincv])
    pos_vectorized1 = vec.transform(data.text[testcv])
    
    predicted = clf.predict(pos_vectorized1)
    
    
    print("Accuracy =")
    print(np.mean(predicted == data.emotion[testcv]))
    
    print(metrics.confusion_matrix(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad']))
    
    
    target_names = ['joy', 'anger', 'fear', 'sad']
    
    print(metrics.classification_report(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad'], target_names=target_names))



