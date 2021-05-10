# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:10:48 2020

@author: nickpaine
"""
##################################################################################################
#PACKAGE AND DATA LOAD#


# Load libraries
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix #for confusion matrix
import seaborn as sns #heat map for confusion matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import io
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz #to draw Decision Tree
from sklearn.externals.six import StringIO #to draw Decision Tree
from IPython.display import Image #to draw Decision Tree
from sklearn import tree
import pydotplus #to draw Decision Tree



df = pd.read_csv('rftJulia.csv')


X_train = pd.get_dummies(df[['a','b','c','d']])   #convert training data to dummy variables
#y_train = pd.get_dummies(df[['T']])
y_train = df['T']

dftest = pd.read_csv('rfctestJulia.csv')


X_test = pd.get_dummies(dftest[['a','b','c','d']])   #convert training data to dummy variables
#y_test = pd.get_dummies(dftest[['T']])
y_test = dftest['T']


##################################################################################################
#MODEL#

# Split dataset into training set and test set
#uncomment the below line to split the data- for purposes of the example, we have uploaded the test data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test



# Create Decision Tree classifer object
#dtc= DecisionTreeClassifier()

rfc = RandomForestClassifier()


# Train Decision Tree Classifer
#dtc = dtc.fit(X_train,y_train)

rfc = rfc.fit(X_train, y_train)

#y_pred = dtc.predict(X_test)

y_pred = rfc.predict(X_test)

#print('Testing Set Evaluation F1-Score=>',f1_score(y_test,y_pred))

# n_estimators = [100, 300, 500, 800, 1200]
# max_depth = [5, 8, 15, 25, 30]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)

# gridF = GridSearchCV(rfc, hyperF, cv = 3, verbose = 1, 
#                       n_jobs = -1)
# bestF = gridF.fit(X_train, y_train)



# Create a dictionary of hyperparameters to search
grid = {'n_estimators': [3,4], 'max_depth': [3,4,5,10], 'max_features': [2,3], 'random_state': [42]} #depth 4 gives 73% while depth 3 gives 87%
test_scores = []

# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfc.set_params(**g)  # ** is "unpacking" the dictionary
    rfc.fit(X_train, y_train)
    test_scores.append(rfc.score(X_test,y_test))

from pprint import pprint
pprint(list(ParameterGrid(grid)))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])


rfc = RandomForestClassifier(n_estimators=4, max_depth=3, max_features=3, random_state=42)
rfc = rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

##################################################################################################
#RESULTS#

len(rfc.estimators_)

plt.figure(figsize=(20,20))
_ = tree.plot_tree(rfc.estimators_[0], feature_names=X_train.columns, filled=True)

rfc.estimators_[0].tree_.max_depth

tree = rfc.estimators_[1]

#draw out the Decision Tree
dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = list(X_train.columns.values),class_names=['qualify','notqualify'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('RF.png')
Image(graph.create_png())

#plot one specific tree
fn=list(X_train.columns.values)
cn=['qualify','notqualify']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rfc.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True);
fig.savefig('rf_individualtree.png')



#plot many trees
# This may not the best way to view each estimator as it is small
fn=list(X_train.columns.values)
cn=['qualify','notqualify']
fig, axes = plt.subplots(nrows = 1,ncols = 4,figsize = (10,2), dpi=800)
for index in range(0, 4):
    tree.plot_tree(rfc.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);
fig.savefig('rf_5trees.png')


##################################################################################################
#ACCURACY#


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #which means 8 out of 9 times were correctly predicted

#Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred)) 
cf_matrix=confusion_matrix(y_test, y_pred)

#See error
y_pred1=pd.Series(y_pred) #convert the prediction to a series rather than array
pd.concat([y_test, y_pred1], axis=1) #we can see index 4 was incorrect

#Make heatmap of confusion matrix 
group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')


parameterunchanged = y_pred
parameterchanged = y_pred

dftestans = dftest
dftestans['pred1'] = parameterunchanged
dftestans['pred2'] = parameterchanged

dftestans.to_csv('resultsrandomforest.csv')


##################
#FEATURE IMPORTANCES
#################

# Get feature importances from our random forest model
importances = rfc.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels 
labels = np.array(X_train.columns)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
plt.savefig('feat_impt.png')

# Rotate tick labels to vertical
# plt.xticks(rotation=90)
# plt.show()

#testing what happens with a change