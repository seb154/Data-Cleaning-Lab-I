# Imports
import pandas as pd
import numpy as np 
#make sure to install sklearn in your terminal first!
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_csv(url):
    return pd.read_csv(url)

def set_categoricals(df, cat_cols):
    #Set selected columns as categorical dtypes
    df[cat_cols] = df[cat_cols].astype('category') 
    return df

def normalize_numerical(df, num_cols):
    #Normalize selected numerical columns using MinMaxScaler
    df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return df

def one_hot_encode(df, cat_cols):
    #One Hot Encode selected categorical columns
    df_encoded = pd.get_dummies(df, columns = cat_cols) 
    return df_encoded

def create_target_variable(df, target_col, cutoff_quantile):
    #Create binary target on quantile of original target column
    cutoff = df[target_col].quantile(cutoff_quantile)
    df[target_col + '_f'] = (df[target_col] >= cutoff).astype(int)
    return df

def calculate_prevalence(df, target_col_f):
    #Calculate prevalence 
    prevalence = df[target_col_f].value_counts()[1] / len(df[target_col_f])
    return prevalence

def train_test_split_data(df, target_col_f, train_size=0.7):
    X = df.drop(columns=[target_col_f])
    y = df[target_col_f]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    return X_train, X_test, y_train, y_test