# Given the following confusion matrix, evaluate (by hand) the model's performance:
'''
|               | pred dog   | pred cat   |
|:------------  |-----------:|-----------:|
| actual dog    |         46 |         7  |
| actual cat    |         13 |         34 |
'''

pos_case = 'dog'
TP = 46
TN = 34
FP = 7
FN = 13

# In the context of this problem, what is a false positive?

# ANSWER: predicted dog and is cat

# In the context of this problem, what is a false negative?

# ANSWER: predicted cat and is dog

# How would you describe this model?
accuracy = (TP+TN)/(TP+TN+FP+FN)
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('The description of this model is:')
print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(round(precision,2)))
print('Recall: {}'.format(round(recall,2)))



# You are working as a datascientist working for Codeup Cody Creator (C3 for short), a rubber-duck manufacturing plant.
# Unfortunately, some of the rubber ducks that are produced will have defects. 
# Your team has built several models that try to predict those defects, and the data from their predictions
# can be found here: https://ds.codeup.com/data/c3.csv
import pandas as pd
c3_df = pd.read_csv('c3.csv')
c3_df.actual.value_counts()
# Use the predictions dataset and pandas to help answer the following questions:

#q1.
# An internal team wants to investigate the cause of the manufacturing defects. 
# They tell you that they want to identify as many of the ducks that have a defect as possible. 
# Which evaluation metric would be appropriate here? Which model would be the best fit for this use case?

# RECALL:
# defect actual set for model eval
defects = c3_df[c3_df.actual == 'Defect']
defects

# Recall m1
m1_recall = (defects.actual == defects.model1).mean()
print("Model 1")
print('Model recall: {:.2%}'.format(m1_recall))

# Recall m2
m2_recall = (defects.actual == defects.model2).mean()
print("Model 2")
print('Model recall: {:.2%}'.format(m2_recall))

# Recall m3
m3_recall = (defects.actual == defects.model3).mean()
print("Model 3")
print('Model recall: {:.2%}'.format(m3_recall))

# model3 is most viable with target of ID defects at 81.25% success rate 

# q2.
# Recently several stories in the local news have come out highlighting 
# customers who received a rubber duck with a defect, and portraying C3 
# in a bad light. The PR team has decided to launch a program that gives 
# customers with a defective duck a vacation to Hawaii. 
# They need you to predict which ducks will have defects, but tell you 
# the really don't want to accidentally give out a vacation package when 
# the duck really doesn't have a defect. Which evaluation metric would be 
# appropriate here? Which model would be the best fit for this use case?

# select pos. case of defects per model (precision)
#model1 
m1_defects = c3_df[c3_df.model1 == 'Defect']
m1_defects
model1_precision = (m1_defects.actual == m1_defects.model1).mean()
print('Model 1')
print('Model precision: {:.2%}'.format(model1_precision))
#model2
m2_defects = c3_df[c3_df.model2 == 'Defect']
model2_precision = (m2_defects.actual == m2_defects.model2).mean()
print('Model 2')
print('Model precision: {:.2%}'.format(model2_precision))
#model3
m3_defects = c3_df[c3_df.model3 == 'Defect']
model3_precision = (m3_defects.actual == m3_defects.model3).mean()
print('Model 3')
print('Model precision: {:.2%}'.format(model3_precision))

# from the above, model1 is the most viable in precision of ID defects

# q3.
# You are working as a data scientist for Gives You Paws ™, a subscription 
# based service that shows you cute pictures of dogs or cats 
# (or both for an additional fee).
# At Gives You Paws, anyone can upload pictures of their cats or dogs. 
# The photos are then put through a two step process. 
# First an automated algorithm tags pictures as either a cat or a dog (Phase I). 
# Next, the photos that have been initially identified are put through another 
# round of review, possibly with some human oversight, before being presented 
# to the users (Phase II).
# PHASE I: recall
# PHASE II: precision

paws_df = pd.read_csv("https://ds.codeup.com/data/gives_you_paws.csv")
paws_df

paws_df.actual.value_counts() # [3254 Dogs, 1746 Cats]

# Given this dataset, use pandas to create a baseline model 
# (i.e. a model that just predicts the most common class) and answer the 
# following questions:

#q3a.
# In terms of accuracy, how do the various models compare to the baseline 
# model? Are any of the models better than the baseline?
paws_df["baseline"] = paws_df.actual.value_counts().idxmax() # baseline for most: dogs
paws_df.head()

# create list of column names
versions = list(paws_df.columns)
versions = versions[1:] # removing actual column

# fx for creating dict of model accuracy per model
accuracy = lambda versions, paws_df: {version: (paws_df.actual == paws_df[version]).mean() for version in versions}
accuracy_out = accuracy(versions, paws_df)
accuracy_out

# turn into df
accuracy_df = pd.DataFrame(accuracy_out.items(), columns = ['model', 'accuracy'] )

# in terms of accuracy, model1 outperforms baseline and peer models



# q4.
# q4a.
# Suppose you are working on a team that solely deals 
# with dog pictures. Which of these models would you 
# recommend?
# Phase I: Automated algorithm tags pictures as either a cat or a dog
# For phaseI, we should choose a model with highest Recall

# RECALL
dogs = paws_df[paws_df.actual == 'dog']
#m1
(dogs.actual == dogs.model1).mean()
#m2
(dogs.actual == dogs.model2).mean()
#m3
(dogs.actual == dogs.model3).mean()
#m4
(dogs.actual == dogs.model4).mean()

# create table for above values
model_list = list(paws_df.columns)
model_list = model_list[1:]
recall = lambda model_list, paws_df: {model: (paws_df.actual == paws_df[model]).mean() for model in model_list}
recall_table = recall(model_list, dogs)
recall_table
# recc. model4 with 95% ID of dogs


# Phase II: Photos that have been initially identified are put 
# through another round of review
# Precision is the appropriate metric since we are trying to 
# minimize false positives

# PRECISION
pos_pred1 = paws_df[paws_df.model1 == 'dog']
pos_pred2 = paws_df[paws_df.model2 == 'dog']
pos_pred3 = paws_df[paws_df.model3 == 'dog']
pos_pred4 = paws_df[paws_df.model4 == 'dog']

#m1
(pos_pred1.actual == pos_pred1.model1).mean()
#m2
(pos_pred2.actual == pos_pred2.model2).mean()
#m3
(pos_pred3.actual == pos_pred3.model3).mean()
#m4
(pos_pred4.actual == pos_pred4.model4).mean()
# model2 has highest precision value at 89.3%


#q4b.
# Suppose you are working on a team that solely deals with 
# cat pictures. Which of these models would you recommend?

# RECALL
cats = paws_df[paws_df.actual == 'cat']
#m1
(cats.actual == cats.model1).mean()
#m2
(cats.actual == cats.model2).mean()
#m3
(cats.actual == cats.model3).mean()
#m4
(cats.actual == cats.model4).mean()

recall_table2 = recall(model_list, cats)
recall_table2

# recc. model2 with 89% ID of cats

# PRECISION
pos_pred5 = paws_df[paws_df.model1 == 'cat']
pos_pred6 = paws_df[paws_df.model2 == 'cat']
pos_pred7 = paws_df[paws_df.model3 == 'cat']
pos_pred8 = paws_df[paws_df.model4 == 'cat']

#m1
(pos_pred5.actual == pos_pred5.model1).mean()
#m2
(pos_pred6.actual == pos_pred6.model2).mean()
#m3
(pos_pred7.actual == pos_pred7.model3).mean()
#m4
(pos_pred8.actual == pos_pred8.model4).mean()

# recc. model4 with 81% precision



#q5
# Follow the links below to read the documentation about 
# each function, then apply those functions to the data 
# from the previous problem.

#    sklearn.metrics.accuracy_score
from sklearn.metrics import accuracy_score

#sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)

y_true = paws_df['actual']
y_pred = paws_df['model1']
accuracy_score(y_true, y_pred)

y_pred2 = paws_df['model2']
accuracy_score(y_true, y_pred2)

y_pred3 = paws_df['model3']
accuracy_score(y_true, y_pred3)

y_pred4 = paws_df['model4']
accuracy_score(y_true, y_pred4)

#    sklearn.metrics.precision_score
from sklearn.metrics import precision_score
#sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')

#m1
y_true = paws_df['actual']
y_pred5 = paws_df['model1']
precision_score(y_true, y_pred5, pos_label='cat')
#m2
y_pred6 = paws_df['model2']
precision_score(y_true, y_pred6, pos_label='cat')
#m3
y_pred7 = paws_df['model3']
precision_score(y_true, y_pred7, pos_label='cat')
#m4
y_pred8 = paws_df['model4']
precision_score(y_true, y_pred8, pos_label='cat')


#    sklearn.metrics.recall_score
from sklearn.metrics import recall_score
#sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
#m1
y_true = paws_df['actual']
y_pred9 = paws_df['model1']
recall_score(y_true, y_pred9, pos_label = 'cat')
#m2
y_pred10 = paws_df['model2']
recall_score(y_true, y_pred10, pos_label = 'cat')
#m3
y_pred11 = paws_df['model3']
recall_score(y_true, y_pred11, pos_label = 'cat')
#m4
y_pred12 = paws_df['model4']
recall_score(y_true, y_pred12, pos_label = 'cat')


#    sklearn.metrics.classification_report
from sklearn.metrics import classification_report
#sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
#m1
y_true = paws_df['actual']
y_pred13 = paws_df['model1']
classification_report(y_true, y_pred13,labels = ['cat', 'dog'],output_dict=True).T


print('m1')
print(pd.DataFrame(classification_report(y_true, paws_df.model1,
                                   labels = ['cat', 'dog'],
                                   output_dict=True)).T)
print('\nm2')
print(pd.DataFrame(classification_report(y_true, paws_df.model2,
                                   labels = ['cat', 'dog'],
                                   output_dict=True)).T)
print('\nm3')
print(pd.DataFrame(classification_report(y_true, paws_df.model3,
                                   labels = ['cat', 'dog'],
                                   output_dict=True)).T)
print('\nm4')
print(pd.DataFrame(classification_report(y_true, paws_df.model4,
                                   labels = ['cat', 'dog'],
                                   output_dict=True)).T)
