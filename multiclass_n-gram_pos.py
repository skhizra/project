
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import nltk
import string
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack
from sklearn import svm
from sklearn import cross_validation



# read the csv to the dataframe for emojis dataset
#data = pd.DataFrame.from_csv('emoji_tweets.csv', encoding='utf-8')
#data1 = pd.DataFrame.from_csv('emoji_tweets.csv', encoding='utf-8')


# read the csv to the dataframe for dataset without emojis
data = pd.DataFrame.from_csv('without_emoji_tweets.csv', encoding='utf-8')
data1 = pd.DataFrame.from_csv('without_emoji_tweets.csv', encoding='utf-8')

# creates a pos tag list for the tweets 

def pos_tagging(x):
    
        # dictionary of pos tags
    pos_tag = { 'CC': 0, 'CD': 0, 'DT':0, 'DT':0, 'EX':0, 'FW':0, 'IN':0, 'JJ':0, 'JJR':0, 'JJS':0,
'LS':0, 'MD':0, 'NN':0, 'NNS':0, 'NNP':0, 'NNPS':0, 'PDT':0, 'POS':0, 'PRP':0, 'PRP$':0,
'RB':0, 'RBR':0, 'RBS':0, 'RP':0, 'TO':0, 'UH':0, 'VB':0, 'VBD':0, 'VBG':0, 'VBN':0, 
'VBP':0, 'VBZ':0, 'WDT':0, 'WP':0, 'WP$':0, 'WRB':0}	

    # tokenizes the words in tweets and pos tags of the words
    t = nltk.pos_tag(nltk.word_tokenize(x))
    
        # creates a list of all the words tokenized
    tag_list = [p[1] for p in t]
    
     # for each word in the list
    for k in tag_list:
        
                # for each word in the pos tag dictionary
        for i in pos_tag:
            
                        # it matches the pos tags of the tag list and the pos tag dictionary
            if k == i:
                
                                # appends the values by 1
                pos_tag[i] = pos_tag[i] + 1      
                       
       # calcuates the term frequency               
    for j in pos_tag:
        pos_tag[j] = pos_tag[j]/len(tag_list)

    # return the dictionary of the pos tags 
    return(pos_tag)


# converts data to string type
data.text = data.text.astype(str)


# call the function pos_tagging for the tweets
data.text = data.text.apply(lambda x: pos_tagging(x))



# cross validation
cv = cross_validation.KFold(data.shape[0], n_folds=10)

# for each fold predicting the emotion 
for traincv, testcv in cv:
    
        # creating an object of DictVectorizer
    vec = DictVectorizer()
        # calling the transform function on training data
    pos_vectorized = vec.fit_transform(data.text[traincv])
    
    
        # creating an object of CountVectorizer for bigrams
    count_vect = CountVectorizer(ngram_range=(2, 2))
    
        # calling the transform function on training data
    X_train_counts = count_vect.fit_transform(data1.text[traincv])
    
    
              # adding all the features
    new_vector = hstack((pos_vectorized, X_train_counts))
    
    
    #classifiers
    #clf = OneVsRestClassifier(LogisticRegression())
    clf = OneVsRestClassifier(svm.LinearSVC())
    
        # transfroming the test data 
    pos_vectorized1 = vec.transform(data.text[testcv])
    
    
    X_new_counts = count_vect.transform(data1.text[testcv])
    
    
    
    new_vector_test = hstack((pos_vectorized1, X_new_counts))
    
    
        # predicting the data
    predicted = clf.fit(new_vector, data.emotion[traincv]).predict(new_vector_test)
    
        # calculating the accuracy
    print("Accuracy = ")
    print(np.mean(predicted == data.emotion[testcv]))
    
        # prints confusion matrix
    print(metrics.confusion_matrix(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad']))
    
    target_names = ['joy', 'anger', 'fear', 'sad']
    
        # calculate and prints precision, recall and f1 score
    print(metrics.classification_report(data.emotion[testcv], predicted, labels=['joy', 'anger', 'fear', 'sad'], target_names=target_names))


