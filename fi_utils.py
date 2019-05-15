# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
from sklearn import preprocessing

def load_dataset():
    # Path of the file to read
    fifa_filepath = "datasets/fifadata.csv"
    # Read the file into a variable iris_data
    data = pd.read_csv(fifa_filepath)
    # Print the first 5 rows of the data
    data.head()
    
    
    df2 = data.loc[:, 'Crossing':'Release Clause']
    df1 = data[['Age', 'Overall', 'Value', 'Wage', 'Preferred Foot', 'Skill Moves', 'Position', 'Height', 'Weight']]
    df = pd.concat([df1, df2], axis=1)
    
    df = df.dropna()
    
    def value_to_int(df_value):
        try:
            value = float(df_value[1:-1])
            suffix = df_value[-1:]
    
            if suffix == 'M':
                value = value * 1000000
            elif suffix == 'K':
                value = value * 1000
        except ValueError:
            value = 0
        return value
      
    df['Value_float'] = df['Value'].apply(value_to_int)
    df['Wage_float'] = df['Wage'].apply(value_to_int)
    df['Release_Clause_float'] = df['Release Clause'].apply(lambda m: value_to_int(m))
    
    def weight_to_int(df_weight):
        value = df_weight[:-3]
        return value
      
    df['Weight_int'] = df['Weight'].apply(weight_to_int)
    df['Weight_int'] = df['Weight_int'].apply(lambda x: int(x))
    
    def height_to_int(df_height):
        try:
            feet = int(df_height[0])
            dlm = df_height[-2]
    
            if dlm == "'":
                height = round((feet * 12 + int(df_height[-1])) * 2.54, 0)
            elif dlm != "'":
                height = round((feet * 12 + int(df_height[-2:])) * 2.54, 0)
        except ValueError:
            height = 0
        return height
    
    df['Height_int'] = df['Height'].apply(height_to_int)
    
    
    df = df.drop(['Value', 'Wage', 'Release Clause', 'Weight', 'Height'], axis=1)
    
    le_foot = preprocessing.LabelEncoder()
    df["Preferred Foot"] = le_foot.fit_transform(df["Preferred Foot"].values)
    
    
    for i in ['ST', 'CF', 'LF', 'LS', 'LW', 'RF', 'RS', 'RW']:
      df.loc[df.Position == i , 'Pos'] = 'Strikers' 
    
    for i in ['CAM', 'CDM', 'LCM', 'CM', 'LAM', 'LDM', 'LM', 'RAM', 'RCM', 'RDM', 'RM']:
      df.loc[df.Position == i , 'Pos'] = 'Midfielder' 
    
    for i in ['CB', 'LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB','GK']:
      df.loc[df.Position == i , 'Pos'] = 'Defender' 
    
    le_class = preprocessing.LabelEncoder()
    
    df['Pos'] = le_class.fit_transform(df['Pos'])
    
    y = df["Pos"]
    
    df.drop(columns=["Position","Pos"],inplace=True)
    
    return df, y