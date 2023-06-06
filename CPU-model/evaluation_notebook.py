#!/usr/bin/env python
# coding: utf-8

# ## Evaluation Notebook

# !python -m spacy download en_core_web_sm
#pip install spacy

import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin
import re
import os
from spacy import displacy
from sklearn import metrics
#from sklearn.metrics import plot_confusion_matrix

from spacy.util import filter_spans

FOODKEEPER_PATH = "datasets/FoodKeeper-Data.xls"
TRAINING_DATA_PATH = "datasets/data.csv"
MODEL_PATH = "output/model-last"
TEST_DATA_PATH = "datasets/test_data.csv"
REAL_TWITTER_DATA_PATH = "datasets/data.csv"
#STARTING_KEYWORD_COUNT = 10
#TRAINING_LOOP_ITERATIONS = 3
#REQUIRED_KEYWORDS = 3

pd.options.mode.chained_assignment = None

food_data = pd.read_excel(FOODKEEPER_PATH, sheet_name = "Product")
all_data = pd.read_csv(TRAINING_DATA_PATH,index_col = False, header = None)
#live_tweets = pd.read_csv(REAL_TWITTER_DATA_PATH, header = None)  
    
#Gathers all the keywords from the FoodKeeper database
def foodKeeperInfo():
    keywords = []
    for word in food_data['Name']: # food_data['Keywords']:
        word = word.replace(" or ", " ") # TO DO: break up phrases with commas like "Pastries, danish" into separate words
        word = re.sub('[/,]', ' ', word)
        word = word.lstrip()
        word = word.rstrip()
        if word.lower() not in keywords: 
            keywords.append(word.lower())

    #print("Total foodkeeper food names: " + str(len(keywords)))        
    #for element in sorted(keywords):
        #print(element)
    return keywords


def preProcess(tweet):
    #Converts a tweet to lowercase, replaces anyusername w/ <USERNAME> and URLS with <URL>
    tweet = tweet.lower()
    tweet = re.sub('@[a-zA-z0-9]*', '', tweet)              # <USERNAME>
    tweet = re.sub('http[a-zA-z0-9./:]*', '', tweet)       # <URL>
    tweet = re.sub('[.,-]*', '', tweet)
    tweet = re.sub('&amp;', 'and', tweet)
    return tweet


#---------------------------------------------

# foodKeeperKeywords = foodKeeperInfo()
# print(foodKeeperKeywords)

# for i in range(len(all_data[0])):
#     all_data[0][i] = preProcess(all_data[0][i])

# TO BE COMPLETED: 
# select tweets which contain one of the foodKeeperKeywords
# then select 250 random tweets of these; this is the test_data
# save test_data in a "new_test_tweets.csv" file
#quit()

#---------------------------------------------

    
def ent_recognize(text):
    doc = nlp(text)
    displacy.render(doc,style = "ent")
    
def predict(tweet):
    doc = nlp(str(tweet))
    if doc.ents:
        displacy.render(doc,style = "ent")

def returnPrediction(tweet):
    nlp = spacy.load(MODEL_PATH)
    doc = nlp(str(tweet))
    if doc.ents:
        return 1
    else:
        return 0
    
def get_predictions():
    predictions = []
    for tweet in test_data['tweet'].tolist():
        predictions.append(returnPrediction(tweet))
    return predictions
    
def eval_model():
    nlp = spacy.load(MODEL_PATH)
    predictions = get_predictions()
    print(metrics.confusion_matrix(y,predictions, labels = [1,0]))
    print(metrics.classification_report(y,predictions, labels = [1,0]))
    
def show_tp():
    counter = 0
    tweets = test_data['tweet'].tolist()
    predictions = get_predictions()
    for i in range(len(y)):
        if predictions[i] == 1 and y[i] == 1:
            print("True positives:", tweets[i], "\n")
            counter += 1
    print(counter)
    
def show_tn():
    counter = 0
    predictions = get_predictions()
    tweets = test_data['tweet'].tolist()
    for i in range(len(y)):
        if predictions[i] == 0 and y[i] == 0:
            print("True Negative:", tweets[i], "\n")
            counter += 1
    print(counter)
    
def show_fn():
    predictions = get_predictions()
    tweets = test_data['tweet']
    counter = 0
    for i in range(len(y)):
        if predictions[i] == 0 and y[i] == 1:
            print("False Negative:", tweets[i], "\n")
            counter += 1
    print(counter)
    
def show_fp():
    predictions = get_predictions()
    tweets = test_data['tweet'].tolist()
    for i in range(len(y)):
        if predictions[i] == 1 and y[i] == 0:
            print("False Positive:")
            doc = nlp(str(tweets[i]))
            if doc.ents:
                displacy.render(doc,style = "ent")

# Evaluation: how well does the existing spacy model identify foods
# Read "new_test_tweets.csv",  

test_data = pd.read_csv(TEST_DATA_PATH)
#test_data = pd.read_csv("datasets/new_test_tweets.csv")

#y = test_data['food'].tolist()
nlp = spacy.load(MODEL_PATH)
#nlp = spacy.blank("en")
print(nlp.pipe_names)
ent_recognize("My rice cakes is tasty")
ent_recognize("My chicken cakes is tasty")
quit()

# # Checking for overlapping words
   
#     print(data)
# keywords = foodKeeperInfo()
# testdata = convertToTrainingFormat("My rice cakes is tasty ", keywords)

# # for word in keywords:
# #     for word2 in wordsInFoodkeeper:
# #         if word in word2 and word != word2:
# #             print(word,word2)



# ## Use the function below to check individual sentences
# nlp = spacy.load("en_core_web_sm")
# ent_recognize("my friend is chicken because he is scared")
# print(live_tweets)
# testTweets = live_tweets[5]
# for tweet in testTweets[:500]:
#     if nlp(preProcess(tweet)).ents:
#         ent_recognize(preProcess(tweet))


# ## Use the function below to check model performance on the entire test set
eval_model()

# ## Use the functions below to see TP, TN, FP, FN respectively
# show_tp()
# show_tn()
# show_fp()
#show_fn()

# foodkeeper = foodKeeperInfo()
# print(foodkeeper)
# sortedKeywords =  sorted(keywordRanker, key=keywordRanker.get, reverse=True)

# for i in range(15): #sortedKeywords
#     keywords.append(sortedKeywords[i])
# print(keywords)
# keywords.append("chicken")
# #keywords.append("cream cheese")

# abc = convertToTrainingFormat("I like to eat cream and cheese with chicken test", keywords)
# print(abc)

# # See what keywords are found by the created model
# keywordsFound = []
# nlp = spacy.load(MODEL_PATH)
# print(nlp.pipeline)
# for tweet in live_tweets[5]: #test_data['tweet']:
#     modeledTweet = nlp(preProcess(tweet))
#     for token in modeledTweet.doc.ents:
#         if str(token) in keywordsFound: continue
#         keywordsFound.append(str(token))


# # If the model finds food keywords print the Tweet
# To help visualize which keywords are being found this loop iterates through the Tweets testing the model to see what keywords it finds. If it finds a keyword in the Tweet it will print the Tweet and highlight the keyword

# for tweet in test_data['tweet'][:500]:
#     ents = nlp(preProcess(tweet))
#     #if ents.doc.ents:
#     ent_recognize(preProcess(tweet))


