import numpy as np   
import pandas as pd  
import string
#!pip install -q emoji
import emoji
import itertools
import time
import pickle
  
# Import dataset 
dataset = pd.read_csv('Labelled_Dataset.csv')  

#dataset = dataset.reindex(np.random.permutation(dataset.index))

pd.set_option('display.max_columns',None)      #To display all the columns while printing in place of ...
pd.set_option('display.max_rows',None)
pd.options.mode.chained_assignment = None

# library to clean data 
import re  
  
# Natural Language Tool Kit 
import nltk  
  
# to remove stopwordsk
nltk.download('stopwords')
from nltk.corpus import stopwords 
print(stopwords)
  
# for Stemming propose  
from nltk.stem.porter import PorterStemmer 

x = dataset.iloc[:,1].values
y = dataset.iloc[:,0].values 

SMILEY = {':-)' : 'happy', ':)':'happy', ';-)' : 'winking happy', ';)' : 'winking happy', '^^':'happy', '(-:':'happy', ':-(':'sad', ':(':'sad', ':-*' : 'kiss', ':-/' : 'skeptical', ':-\\' : 'undecided', ':\'-(':'cry', ":\'(" :'cry'}
n = x.shape
n = n[0]    #n = number of rows in dataset
print(n)

#------------------------------Data Pre-Processing-------------------------------------------------------------------
# 14641 (reviews) rows to clean 
for i in range(0, n):  
    review = x[i]
    review = review.lower()
    review = re.sub('http://\S+ | https://\S+', '', review)   #remove url
    words = review.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]   #replace smileys with corresponding word 
    review = " ".join(reformed)
    review = emoji.demojize(review)     #replace emojis with values
    review = review.replace(":"," ")
    review = re.sub('rt','',review)
    review = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",review).split())   #Remove words that have @
    #review = ' '.join(re.sub("(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",review).split())   #Remove words that have #
    review = re.sub(r'/\[].*?','',review)
    #We must use [] to give multiple characters for resubstituition
    #[%s] = The string in
    #re.escape(string.punctuation) = all the punctuations with \ before them (not necessary in Python3)
    review = re.sub('[%s]' % re.escape(string.punctuation),'',review)
    review = re.sub('\w*\d\w*','',review)
    review = re.sub('http','',review)
    review = re.sub('https','',review)
    review = re.sub('co','',review)
    review = re.sub('com','',review)
    review = re.sub('\n','',review)
    review = re.sub(r'"<>/\;','',review)
    
    review = ''.join(''.join(s)[:2] for _, s in itertools.groupby(review))
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    x[i] = review

#-----------------Count Vectorizer----------------------------------------------

# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
  
# To extract max 1500 feature. 
# "max_features" is attribute to 
# experiment with to get better results 
cv = CountVectorizer()  
  
# X contains corpus (dependent variable) 
X = cv.fit_transform(x).toarray()

#----------------------Splitting Data-------------------------------------------

from sklearn.model_selection import train_test_split 
  
# experiment with "test_size" 
# to get better results 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state = 32)

#-----------------------Naive Bayes-------------------------------------------

from sklearn.naive_bayes import MultinomialNB

MNB = MultinomialNB(alpha=1, fit_prior=True,)

MNB.fit(X_train, y_train)

print('-----------MNB Test--------------')
# Predicting the Test set results 
y_pred = MNB.predict(X_test)
  
print(MNB.score(X_test,y_test)) 

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred) 
  
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=[0, 2, 4]))

#-------------------Random Forest-----------------------------------------------

# Fitting Random Forest Classification 
# to the Training set 
from sklearn.ensemble import RandomForestClassifier 
  
# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results  
RF = RandomForestClassifier( n_estimators=200, max_features='sqrt', min_samples_leaf=10,
                            criterion = 'entropy')
                              
RF.fit(X_train, y_train)  

print('-----------RF Test--------------')
# Predicting the Test set results 
y_pred = RF.predict(X_test)
  
print(RF.score(X_test,y_test)) 

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred) 
  
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=[0, 2, 4]))

#------Parameter Tuning for Random Forest---------------------------------------
'''
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Create the parameter grid
model_params = {
    'n_estimators': [100,200],
    'max_features': ['sqrt', 'auto', 0.2],
    'min_samples_leaf' : [10,50, 100]
}

from sklearn.ensemble import RandomForestRegressor

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = GridSearchCV(estimator=rf, param_grid=model_params, cv= 5)
print()
# Fit the random search model
rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
'''

#------------------Support Vector Machine---------------------------------------

from sklearn.svm import LinearSVC

SVM = LinearSVC()

SVM.fit(X_train, y_train)

print('-----------SVM Test--------------')
# Predicting the Test set results 
y_pred = SVM.predict(X_test)
  
print(SVM.score(X_test,y_test)) 

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred) 
  
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=[0, 2, 4]))

#--------------Parameter Tuning for SVM-------------------------------------------
'''
# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train) 

print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_)
'''

#---------------------------Voting----------------------------------------------

from sklearn.ensemble import VotingClassifier
#Voting
voting_clf = VotingClassifier(
	estimators = [('MNB', MNB), ('RF', RF),('SVM',SVM)],
	voting = 'soft')
voting_clf.fit(X_train, y_train)

print('-------Voting Classifier Test---------')
# Predicting the Test set results 

y_pred = voting_clf.predict(X_test)
print(voting_clf.score(X_test,y_test))

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred) 
  
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=[0, 2, 4]))

#---------------------------Pickle-----------------------------------------------

filename = 'Pickled_model.sav'
pickle.dump(model, open(voting_clf, 'wb'))



