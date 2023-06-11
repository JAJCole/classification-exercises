import os
import pandas as pd
from env import get_db_url
import env

def get_titanic_data():
    titanic_csv = 'titanic_csv.csv'
    url = get_db_url('titanic_db')
    if os.path.isfile(titanic_csv):
        return pd.read_csv(titanic_csv)
    else:
        df_titanic = pd.read_sql('SELECT * FROM passengers', url)
        df_titanic.to_csv(titanic_csv)
        return df_titanic



def get_iris_data():
    iris_csv = 'iris_csv.csv'
    url = get_db_url('iris_db')
    if os.path.isfile(iris_csv):
        return pd.read_csv(iris_csv)
    else:
        df_iris = pd.read_sql('SELECT * FROM species join measurements using(species_id)', url)
        df_iris.to_csv(iris_csv)
        return df_iris


# be sure to join contract_types, internet_service_types, payment_types tables with the customers table,


def get_telco_data():
    telco_csv = 'telco_churn.csv'
    url = get_db_url('telco_churn')
    if os.path.isfile(telco_csv):
        return pd.read_csv(telco_csv)
    else:
        SQL = '''
    select * from customers
    join contract_types using (contract_type_id)
    join internet_service_types using (internet_service_type_id)
    join payment_types using (payment_type_id)
    '''
        df_telco = pd.read_sql(SQL, url)
        df_telco.to_csv(telco_csv)
        return df_telco


