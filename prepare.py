# The end product of this exercise should be the specified functions in a python script named prepare.py.
# Do these in your classification_exercises.ipynb first, then transfer to the prepare.py file. 


import pandas as pd
from acquire import *



# Iris Function
def prep_iris():
    iris = get_iris_data()
    iris = iris.drop(columns=['species_id','measurement_id','Unnamed: 0'])
    iris = iris.rename(columns= {'species_name':'species'})
    dummy_iris = pd.get_dummies(iris.species, drop_first=True)
    iris = pd.concat([iris, dummy_iris], axis=1)
    return iris


# Titanic Function
def prep_titanic():
    titanic = get_titanic_data()
    titanic = titanic.drop(columns=['embarked', 'class', 'age', 'deck','Unnamed: 0'])
    tit_dummy = pd.get_dummies(data=titanic[['sex','embark_town']],drop_first=True)
    titanic = pd.concat([titanic, tit_dummy],axis=1)
    return titanic


# Telco Function
def prep_telco():
    telco = get_telco_data()
    telco = telco.drop(columns=['internet_service_type_id', 'contract_type_id', 'payment_type_id','Unnamed: 0'])

    telco['gender_encoded'] = telco.gender.map({'Female': 1, 'Male': 0})
    telco['partner_encoded'] = telco.partner.map({'Yes': 1, 'No': 0})
    telco['dependents_encoded'] = telco.dependents.map({'Yes': 1, 'No': 0})
    telco['phone_service_encoded'] = telco.phone_service.map({'Yes': 1, 'No': 0})
    telco['paperless_billing_encoded'] = telco.paperless_billing.map({'Yes': 1, 'No': 0})
    telco['churn_encoded'] = telco.churn.map({'Yes': 1, 'No': 0})
    
    dummy_df = pd.get_dummies(telco[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    telco = pd.concat( [telco, dummy_df], axis=1 )
    
    return telco


# Data Split

# a good data split is 60/20/20 with 60 being train
import pandas as pd
from sklearn.model_selection import train_test_split


# Train-Test-Validate Function
def my_train_test_split(df, target=None):
    if target:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
        train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    else:
        train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
        train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])
    return train, validate, test




