

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import cross_validation




# read the csv to the dataframe for emojis dataset
#data = pd.DataFrame.from_csv('emoji_tweets.csv', encoding='utf-8')

# read the csv to the dataframe for dataset without emojis
data = pd.DataFrame.from_csv('without_emoji_tweets.csv', encoding='utf-8')


#cross validation
cv = cross_validation.KFold(data.shape[0], n_folds=10)



for traincv, testcv in cv:
    
    
    count_vect = CountVectorizer(ngram_range=(2, 2))
    X_train_counts = count_vect.fit_transform(data.text[traincv])
    
        
    
    X_new_counts = count_vect.transform(data.text[testcv])
    
    
    #clf = OneVsRestClassifier(LogisticRegression())
    clf = OneVsRestClassifier(svm.LinearSVC())
    
    
    predicted = clf.fit(X_train_counts, data.emotion[traincv]).predict(X_new_counts)
    

    
    print("Accuracy = ")
    
    print(np.mean(predicted == data.emotion[testcv]))
    
    print(metrics.confusion_matrix(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad']))
    
    
    target_names = ['joy', 'anger', 'fear', 'sad']
    
    print(metrics.classification_report(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad'], target_names=target_names))


