import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

df_raw = pd.read_csv("penguins.csv")

def split_data( species):
    subset = df_raw[df_raw['Species'] == species].copy()
    train_sub, test_sub = train_test_split(subset,train_size=30,test_size=20,random_state=42)
    return train_sub, test_sub

def split_the_species( train_size=30, test_size=20, random_state=42):
    train1, test1 = split_data( "Adelie")
    train2, test2 = split_data( "Chinstrap")
    train3, test3 = split_data( "Gentoo")

    # Merge both species sets
    train_df = pd.concat([train1, train2, train3], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = pd.concat([test1, test2, test3], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, test_df

def fit_preprocessor(train_df):
    df = train_df.copy()

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    medians = df[numeric_cols].median()
    df[numeric_cols] = df[numeric_cols].fillna(medians)

    encoder_dict = {}
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder_dict[col] = le

    scaler = MinMaxScaler()
    exclude = ['Species']
    scale_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in exclude]
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    df['Species'] = LabelEncoder().fit_transform(df['Species'])
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return {'medians': medians,'encoders': encoder_dict,'scaler': scaler,'columns': df.columns}, df

def transform_preprocessor(df, fitted):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(fitted['medians'])

    for col, le in fitted['encoders'].items():
        df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    exclude = ['Species']
    scale_cols = [c for c in df.select_dtypes(include=['float64', 'int64']).columns if c not in exclude]
    df[scale_cols] = fitted['scaler'].transform(df[scale_cols])

    df = df.reindex(columns=fitted['columns'], fill_value=0)
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df
