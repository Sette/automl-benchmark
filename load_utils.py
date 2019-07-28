# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

from autokeras.image.image_supervised import load_image_dataset
import pandas as pd
from sklearn import preprocessing

def load_invasive_species():
    data_dir = "datasets/invasive-species"
    x_train, y_train = load_image_dataset(csv_file_path=data_dir+"/train_labels_real.csv",
                                          images_path=data_dir+"/train")
    print(x_train.shape)
    print(y_train.shape)

    
    x_test = load_image_dataset(csv_file_path=data_dir+"/sample_submission_real.csv",
                                        images_path=data_dir+"/test")
    print(x_test[0].shape)
    
    return x_train, y_train,x_test[0]

def load_dog_breed():
    data_dir = "datasets/dog-breed"
    x_train, y_train = load_image_dataset(csv_file_path=data_dir+"/labels_real.csv",
                                          images_path=data_dir+"/train")
    print(x_train.shape)
    print(y_train.shape)

    
    x_test = load_image_dataset(csv_file_path=data_dir+"/sample_submission_real.csv",
                                        images_path=data_dir+"/test")
    print(x_test[0].shape)
    
    return x_train, y_train,x_test[0]

def load_aerial_cactus():
    data_dir = "datasets/aerial-cactus"
    x_train, y_train = load_image_dataset(csv_file_path=data_dir+"/train.csv",
                                          images_path=data_dir+"/train")
    print(x_train.shape)
    print(y_train.shape)

    
    x_test = load_image_dataset(csv_file_path=data_dir+"/sample_submission.csv",
                                        images_path=data_dir+"/test")
    print(x_test[0].shape)
    
    return x_train, y_train,x_test[0]
    


def load_house_prices():
    overfit_filepath_train = "datasets/house-prices/train.csv"
    overfit_filepath_test = "datasets/house-prices/test.csv"
    data_train = pd.read_csv(overfit_filepath_train)
    data_test = pd.read_csv(overfit_filepath_test)
    
    y_train = pd.to_numeric(data_train['SalePrice'])

    data_train.drop(columns=['SalePrice','Id'],inplace=True)
    id_test = data_test.Id
    data_test.drop(columns=['Id'],inplace=True)

    id_name = 'Id'
    target_name = 'SalePrice'
    
    return data_train,y_train,data_test,id_test,id_name,target_name



def load_santander_customer():
    santander_filepath_train = "datasets/santander-customer/train.csv"
    santander_filepath_test = "datasets/santander-customer/test.csv"

    data_train = pd.read_csv(santander_filepath_train)
    data_test = pd.read_csv(santander_filepath_test)
    
    y_train = pd.to_numeric(data_train['target'])

    data_train.drop(columns=['target','ID_code'],inplace=True)
    id_test = data_test.ID_code
    data_test.drop(columns=['ID_code'],inplace=True)
    
    return data_train,y_train,data_test,id_test

def load_microsoft_malware():
    microsoft_filepath_train = "datasets/microsoft-malware/train.csv"
    microsoft_filepath_test = "datasets/microsoft-malware/test.csv"

    data_train = pd.read_csv(microsoft_filepath_train)
    data_test = pd.read_csv(microsoft_filepath_test)
    
    y_train = pd.to_numeric(data_train['HasDetections'])

    data_train.drop(columns=['HasDetections','MachineIdentifier'],inplace=True)
    id_test = data_test.MachineIdentifier
    data_test.drop(columns=['MachineIdentifier'],inplace=True)
    
    
    return data_train,y_train,data_test,id_test


def load_porto_seguro():
    porto_filepath_train = "datasets/porto-seguro/train.csv"
    porto_filepath_test = "datasets/porto-seguro/test.csv"

    data_train = pd.read_csv(porto_filepath_train)
    data_test = pd.read_csv(porto_filepath_test)
    
    y_train = pd.to_numeric(data_train['target'])
    data_train.drop(columns=['target','id'],inplace=True)
    id_test = data_test.id
    data_test.drop(columns=['id'],inplace=True)
    
    return data_train,y_train,data_test,id_test

def load_dont_overfit():
    overfit_filepath_train = "datasets/dont-overfit/train.csv"
    overfit_filepath_test = "datasets/dont-overfit/test.csv"
    data_train = pd.read_csv(overfit_filepath_train)
    data_test = pd.read_csv(overfit_filepath_test)
    
    y_train = pd.to_numeric(data_train['target'])

    data_train.drop(columns=['target','id'],inplace=True)
    id_test = data_test.id
    data_test.drop(columns=['id'],inplace=True)
    
    return data_train,y_train,data_test,id_test

def load_taxi_fare():
    overfit_filepath_train = "datasets/taxi-fare/train.csv"
    overfit_filepath_test = "datasets/taxi-fare/test.csv"
    data_train = pd.read_csv(overfit_filepath_train)
    data_test = pd.read_csv(overfit_filepath_test)
    
    y_train = pd.to_numeric(data_train['fare_amount'])

    data_train.drop(columns=['fare_amount','key','pickup_datetime'],inplace=True)
    id_test = data_test.key
    data_test.drop(columns=['key','pickup_datetime'],inplace=True)

    id_name = 'key'
    target_name = 'fare_amount'
    
    return data_train,y_train,data_test,id_test,id_name,target_name

def load_trip_duration():
    overfit_filepath_train = "datasets/trip-duration/train.csv"
    overfit_filepath_test = "datasets/trip-duration/test.csv"
    data_train = pd.read_csv(overfit_filepath_train)
    data_test = pd.read_csv(overfit_filepath_test)
    print(data_train.columns)
    y_train = data_train['trip_duration']

    data_train.drop(columns=['id','trip_duration'],inplace=True)
    id_test = data_test.id
    data_test.drop(columns=['id'],inplace=True)

    id_name = 'id'
    target_name = 'trip_duration'
    
    return data_train,y_train,data_test,id_test,id_name,target_name

def load_santander_value():
    overfit_filepath_train = "datasets/santander-value/train.csv"
    overfit_filepath_test = "datasets/santander-value/test.csv"
    data_train = pd.read_csv(overfit_filepath_train)
    data_test = pd.read_csv(overfit_filepath_test)
    
    y_train = pd.to_numeric(data_train['target'])

    data_train.drop(columns=['target','ID'],inplace=True)
    id_test = data_test.ID
    data_test.drop(columns=['ID'],inplace=True)

    id_name = 'ID'
    target_name = 'target'
    
    return data_train,y_train,data_test,id_test,id_name,target_name


def load_fifa():
    # Path of the file to read
    fifa_filepath = "datasets/fifa/fifadata.csv"
    # Read the file into a variable iris_data
    data = pd.read_csv(fifa_filepath)

    
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
    
    target_names = df["Pos"].unique()

    le_class = preprocessing.LabelEncoder()
    
    df['Pos'] = le_class.fit_transform(df['Pos'])
    
    y = df["Pos"]
    
    df.drop(columns=["Position","Pos"],inplace=True)
    
    return df, y,target_names
