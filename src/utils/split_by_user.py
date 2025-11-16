import polars as pl
import pandas as pd
import numpy
import random

def split_by_user(data_frame: pd.DataFrame, num_user_test = 4, num_user_val = 4):

    list_users  = list(data_frame['Subject-id'].unique())
    num_user    = len(data_frame['Subject-id'].unique())

    if num_user < num_user_test + num_user_val:
        print("Validar número de usuarios")
        return None

    list_users_filter = random.sample(list_users, num_user_test + num_user_val)

    train_users = [train_user for train_user in list_users if train_user not in list_users_filter]
    test_users = random.sample(list_users_filter, num_user_test)
    val_users = [val_user for val_user in list_users_filter if val_user not in test_users]

    train_data = data_frame[data_frame['Subject-id'].isin(train_users)]
    test_data = data_frame[data_frame['Subject-id'].isin(test_users)]
    val_data = data_frame[data_frame['Subject-id'].isin(val_users)]

    return train_data, test_data, val_data
