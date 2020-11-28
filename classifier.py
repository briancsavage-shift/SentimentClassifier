import numpy as np
import pandas as pd
import os
import string
import nltk
import spacy
import sklearn
import matplotlib
import matplotlib.pyplot as plt

from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from tqdm import tqdm

def loadCSVs():
    data_dir = 'data_reviews'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df  = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))
    if x_train_df.shape[0] == 0:
        print('X Training Set is Empty')
    if y_train_df.shape[0] == 0:
        print('Y Training Set is Empty')
    if x_test_df.shape[0] == 0:
        print('X Testing Set is Empty')
    return x_train_df, y_train_df, x_test_df

def preprocessReviews(reviews):
    negatives = ['no', 'not', 'never', 'neither', 'none', 'nobody', 'cant', 'nor', 'wont', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'almost', 'kinda', 'doesnt', 'isnt', 'isnt', 'cannt', 'wasnt', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 'dont', '1', '2', '3', '4', '5', 'stars', 'star', 'again', 'all']
    nlp = spacy.load('en_core_web_sm')
    shortened = list()
    for j in range(len(reviews)):
        reviews[j] = reviews[j].replace("'",'')
        parsed = nlp(reviews[j]) 
        words = list()
        for i in range(len(parsed)):
            if parsed[i].pos_ in ['VERB', 'ADV', 'ADJ'] or parsed[i].lemma_ in negatives:
                if i > 0 and parsed[i-1].lemma_ in negatives:
                    words.append(parsed[i-1].lemma_ + parsed[i].lemma_)
                elif parsed[i].lemma_ not in negatives:
                    words.append(parsed[i].lemma_)
            elif parsed[i].pos_ in ['NOUN'] and parsed[i].lemma_ not in ['.'] and parsed[i].lemma_ not in negatives:
                words.append(parsed[i].lemma_)
            elif parsed[i].pos_ in ['PROPN']:
                words.append(parsed[i].text)        
        add = ''
        if len(words) >= 3:
            merged = " ".join(words)
            add = merged
        else:
            larger = [token.lemma_ for token in parsed if token.pos_ not in ['NUM', 'SYM', 'PRON', 'PROPN', 'AUX']]
            if len(larger) >= 3:
                merged = " ".join(larger)
                add = merged
            else:
                add = reviews[j].lower()
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for punc in punctuations: 
            if punc in add: 
                add = add.replace(punc, "")
        shortened.append(add)
    return shortened

def TreePipeline(n_estimators=200, max_depth=32, min_samples_split=2, min_samples_leaf=1):
    return sklearn.pipeline.Pipeline(steps=[
        ('vectorizer', TfidfVectorizer(lowercase=True, 
                                       stop_words='english', 
                                       ngram_range=(1, 3),
                                       analyzer='word',
                                       max_df=0.95, 
                                       min_df=1,
                                       norm='max')),
        ('transformer', TfidfTransformer(norm='l2', 
                                         use_idf=True, 
                                         smooth_idf=True)),
        ('classifier', RandomForestClassifier(n_estimators=n_estimators, 
                                              criterion='gini', 
                                              max_depth=max_depth, 
                                              min_samples_split=min_samples_split, 
                                              min_samples_leaf=min_samples_leaf, 
                                              min_weight_fraction_leaf=0.0, 
                                              max_features='auto', 
                                              max_leaf_nodes=None, 
                                              min_impurity_decrease=0.0, 
                                              min_impurity_split=None, 
                                              bootstrap=True, 
                                              oob_score=False, 
                                              n_jobs=2, 
                                              verbose=0, 
                                              ccp_alpha=0.0))
    ])

def BoostedPipeline(min_df=1, max_df=0.95, activation='tanh', solver='lbfgs', batch_size=10, max_iter=500, alpha=0.001, hidden_layer_sizes=(16,16)):
    return sklearn.pipeline.Pipeline(steps=[
        ('vectorizer', TfidfVectorizer(lowercase=True, 
                                       stop_words=None, 
                                       ngram_range=(1, 3), 
                                       analyzer='word',
                                       max_df=max_df, 
                                       min_df=min_df,
                                       norm='max')),
        ('transformer', TfidfTransformer(norm='l2', 
                                         use_idf=True, 
                                         smooth_idf=True)),
        ('classifier', MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                     activation=activation, 
                                     solver=solver, 
                                     alpha=alpha, 
                                     batch_size=batch_size, 
                                     learning_rate='adaptive', 
                                     max_iter=max_iter, 
                                     early_stopping=False, 
                                     validation_fraction=0.10))
    ])
    
def getErrors(processed, tr_sentiment, model=BoostedPipeline()):
    model.fit(processed, tr_sentiment)
    sentiments = ['Positive' if sentiment == 1 else 'Negative' for sentiment in tr_sentiment]
    yhat = model.predict(processed)
    predictions = ['Positive' if i == 1 else 'Negative' for i in yhat]
    falsepositives = [True if pred == 'Positive' and sent == 'Negative' else False for pred, sent in zip(predictions, sentiments)]
    falsenegatives = [True if pred == 'Negative' and sent == 'Positive' else False for pred, sent in zip(predictions, sentiments)]
    fpidx = [processed[i] for i in range(len(falsepositives)) if falsepositives[i] == True]
    orig_fp = [tr_reviews[i] for i in range(len(falsepositives)) if falsepositives[i] == True]
    fnidx = [processed[i] for i in range(len(falsenegatives)) if falsenegatives[i] == True]
    orig_fn = [tr_reviews[i] for i in range(len(falsenegatives)) if falsenegatives[i] == True]
    df = pd.DataFrame(list(zip(sentiments, predictions, processed, tr_reviews)), columns=['Real Sentiment', 'Prediction', 'Shortened', 'Original'])
    fp = pd.DataFrame(list(zip(fpidx, orig_fp)), columns=['False Positives Shortened', 'False Positive Original'])
    np = pd.DataFrame(list(zip(fnidx, orig_fn)), columns=['False Negatives Shortened', 'False Negative Original'])
    return df, fp, np

def tester(reviews, tr_sentiment, min_df=False, max_df=False, alpha=False, max_iter=False, batch_size=False, hidden_layer_size=False):
    folds = 5
    if min_df is True:
        print(' -- New Parameter -- ')
        xvar = 'min_df'
        ba_min_df = list()
        for var in [1, 2, 5, 10, 20, 50]:
            score = cross_validate(BoostedPipeline(min_df=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds)
            ba_min_df.append(float(sum(score['test_score']) / len(score['test_score'])))
            print('Balanced Accuracy Test Set Score: ' + str(ba_min_df[-1]) + '   with option ' + xvar + '=' + str(var))
    if max_df is True:
        print(' -- New Parameter -- ')
        xvar = 'max_df'
        ba_max_df = list()
        for var in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
            score = cross_validate(BoostedPipeline(max_df=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds)
            ba_max_df.append(float(sum(score['test_score']) / len(score['test_score'])))
            print('Balanced Accuracy Test Set Score: ' + str(ba_max_df[-1]) + '   with option ' + xvar + '=' + str(var))
    if alpha is True:
        print(' -- New Parameter -- ')
        xvar = 'alpha'
        ba_alpha = list()
        for var in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
            score = cross_validate(BoostedPipeline(alpha=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds)
            ba_alpha.append(float(sum(score['test_score']) / len(score['test_score'])))
            print('Balanced Accuracy Test Set Score: ' + str(ba_alpha[-1]) + '   with option ' + xvar + '=' + str(var))
    if max_iter is True:
        print(' -- New Parameter -- ')
        xvar = 'max_iter'
        ba_max_iter = list()
        for var in [10, 15, 20, 25, 50, 100, 200, 500, 1000]:
            score = cross_validate(BoostedPipeline(max_iter=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds)
            ba_max_iter.append(float(sum(score['test_score']) / len(score['test_score'])))
            print('Balanced Accuracy Test Set Score: ' + str(ba_max_iter[-1]) + '   with option ' + xvar + '=' + str(var))
    if batch_size is True:
        print(' -- New Parameter -- ')
        xvar = 'batch_size'
        ba_batch_size = list()
        for var in [1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000]:
            score = cross_validate(BoostedPipeline(batch_size=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds)
            ba_batch_size.append(float(sum(score['test_score']) / len(score['test_score'])))
            print('Balanced Accuracy Test Set Score: ' + str(ba_batch_size[-1]) + '   with option ' + xvar + '=' + str(var))    
    if hidden_layer_size is True:
        print(' -- New Parameter -- ')
        xvar = 'hidden_layer_size'
        ba_hidden_layer_size = list()
        for var in [(4,8), (8,8), (8,16), (8, 32) , (32, 32)]:
            score = cross_validate(BoostedPipeline(hidden_layer_sizes=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds)
            ba_hidden_layer_size.append(float(sum(score['test_score']) / len(score['test_score'])))
            print('Balanced Accuracy Test Set Score: ' + str(ba_hidden_layer_size[-1]) + ' with option ' + xvar + '=' + str(var))
    print(' Done. ')

def grapher(model, reviews, sentiment):

    fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2, 2, figsize=(15,15))
    fig.suptitle('Hyperparameter Selection for Classifier')

    folds = 5
    ba_alpha = list()
    ba_alpha_train = list()
    variables = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    for var in tqdm(variables):
        score = cross_validate(BoostedPipeline(alpha=var), reviews, sentiment, scoring='balanced_accuracy', cv=folds, return_train_score=True)
        ba_alpha.append(float(sum(score['test_score']) / len(score['test_score'])))
        ba_alpha_train.append(float(sum(score['train_score']) / len(score['train_score'])))

    ax11.scatter(variables,ba_alpha, c='r')
    ax11.plot(variables, ba_alpha, label='Testing Accuracy', c='r')
    ax11.scatter(variables,ba_alpha_train, c='b')
    ax11.plot(variables, ba_alpha_train, label='Training Accuracy', c='b')
    ax11.errorbar

    ax11.legend(loc="lower left")
    ax11.set_xscale('log')
    ax11.set_title("Alpha to MLPClassifier")
    ax11.set_xlabel('Alpha (logarithmic)')
    ax11.set_ylabel('Balanced Accuracy Score')


    ba_max_iter = list()
    ba_max_iter_train = list()
    iterations = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for var in tqdm(iterations):
            score = cross_validate(BoostedPipeline(max_iter=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds, return_train_score=True)
            ba_max_iter.append(float(sum(score['test_score']) / len(score['test_score'])))
            ba_max_iter_train.append(float(sum(score['train_score']) / len(score['train_score'])))

    ax12.scatter(iterations,ba_max_iter, c='r')
    ax12.plot(iterations, ba_max_iter, label='Testing Accuracy', c='r')
    ax12.scatter(iterations,ba_max_iter_train, c='b')
    ax12.plot(iterations, ba_max_iter_train, label='Training Accuracy', c='b')
    ax12.legend(loc="lower left")
    ax12.set_xscale('log')
    ax12.set_title("Max Iterations to MLPClassifier")
    ax12.set_xlabel('Max Iterations (logarithmic)')
    ax12.set_ylabel('Balanced Accuracy Score')

    ba_min_df = list()
    ba_min_df_train = list()
    mins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for var in tqdm(mins):
        score = cross_validate(BoostedPipeline(batch_size=var), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds, return_train_score=True)
        ba_min_df.append(float(sum(score['test_score']) / len(score['test_score'])))
        ba_min_df_train.append(float(sum(score['train_score']) / len(score['train_score'])))

    ax21.scatter(mins,ba_min_df, c='r')
    ax21.plot(mins, ba_min_df, label='Testing Accuracy', c='r')
    ax21.scatter(mins, ba_min_df_train, c='b')
    ax21.plot(mins, ba_min_df_train, label='Training Accuracy', c='b')
    ax21.legend(loc="lower left")
    ax21.set_title("Batch Size for MLPClassifier")
    ax21.set_xlabel('Size of batch to stochastic optimizer')
    ax21.set_ylabel('Balanced Accuracy Score')

    ba_max_df = list()
    ba_max_df_train = list()
    maxs = [1, 2, 4, 8, 16, 32, 64, 128]
    for var in tqdm(maxs):
        score = cross_validate(BoostedPipeline(hidden_layer_sizes=(16,var)), reviews, tr_sentiment, scoring='balanced_accuracy', cv=folds, return_train_score=True)
        ba_max_df.append(float(sum(score['test_score']) / len(score['test_score'])))
        ba_max_df_train.append(float(sum(score['train_score']) / len(score['train_score'])))

    ax22.scatter(maxs,ba_max_df, c='r')
    ax22.plot(maxs, ba_max_df, label='Testing Accuracy', c='r')
    ax22.scatter(maxs,ba_max_df_train, c='b')
    ax22.plot(maxs, ba_max_df_train, label='Training Accuracy', c='b')
    ax22.legend(loc="lower left")
    ax12.set_xscale('log')
    ax22.set_title("Hidden Layer Size of MLPClassifier")
    ax22.set_xlabel('Number of Neurons in single Hidden Layer (logarithmic)')
    ax22.set_ylabel('Balanced Accuracy Score')


    fig.subplots_adjust(left=0.1, hspace=0.3, wspace=0.3)
    fig.savefig('PipelineOptimization.pdf')

if __name__ == '__main__':
    x_tr_df, y_tr_df, x_te_df = loadCSVs()
    tr_reviews = preprocessReviews(x_tr_df['text'].values.tolist())
    tr_sentiment = y_tr_df['is_positive_sentiment'].values.tolist()
    tester(tr_reviews, tr_sentiment)
    grapher(BoostedPipeline(), tr_reviews, tr_sentiment)

