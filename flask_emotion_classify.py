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
from flask import Flask, render_template, request, redirect, url_for
import emoji
import re

app = Flask(__name__)

app.config.update(SEND_FILE_MAX_AGE_DEFAULT=10)
@app.route("/")
def home():
    return render_template('index.html')


@app.route("/emotion_analysis", methods = ['POST', 'GET'])
def emotion_analysis():
    test_data = ""
    if request.method == 'POST':
        test_data = request.form["text"]
        
    test_data1 = [test_data]

    data = pd.DataFrame.from_csv('freaking_without_emojis.csv', encoding='utf-8')
   
    training_data = data[:3000]
    
    count_vect = CountVectorizer(ngram_range=(1, 1))
    X_train_counts = count_vect.fit_transform(training_data.text)
    
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    
    X_new_counts = count_vect.transform(test_data1)
    
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    #clf = OneVsRestClassifier(LogisticRegression())
    clf = OneVsRestClassifier(svm.LinearSVC())
    
    predicted = clf.fit(X_train_tfidf, training_data['emotion']).predict(X_new_tfidf)
            
    return render_template('emotion.html', predicted = predicted, test_data = test_data)



@app.route("/emotion_analysis_emojis", methods = ['POST', 'GET'])
def emotion_analysis_emojis():
    test_data = ""
    if request.method == 'POST':
        test_data = request.form["text1"]
        
    test_data1 = [test_data]

    

    data = pd.DataFrame.from_csv('emoji_tweets.csv', encoding='utf-8')
   
    training_data = data[:5000]
    
    count_vect = CountVectorizer(ngram_range=(1, 1))
    X_train_counts = count_vect.fit_transform(training_data.text)
    
    
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    
    X_new_counts = count_vect.transform(test_data1)
    
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    #clf = OneVsRestClassifier(LogisticRegression())
    clf = OneVsRestClassifier(svm.LinearSVC())
    
    predicted = clf.fit(X_train_tfidf, training_data['emotion']).predict(X_new_tfidf)
            
    return render_template('emotion.html', predicted = predicted, test_data = test_data)






if __name__ == "__main__":
    app.run()
