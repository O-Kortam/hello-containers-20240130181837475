import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time


def read_data(connection):
    df = pd.DataFrame(list(connection.chatbotDataBase.Unit.find()))[
        ['_id', 'floor', 'status_desc',
         'ru_view_desc', 'usage_type', 'garden', 'bathroom', 'numberofrooms', 'balcony',
         'area', 'baseprice', 'delivery']]
    return df


def process_split_data(df):
    df.floor.replace({'GF': 0, '': 'No Floor'}, inplace=True)
    df.balcony.fillna(0, inplace=True)
    df.usage_type.replace({'1': 'Appartment', '2': 'Villa'}, inplace=True)
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    df['delivery'] = df['delivery'].str[:1]
    df = df.drop('usage_type', axis=1).join(pd.get_dummies(df['usage_type']))
    df = df.drop('ru_view_desc', axis=1).join(pd.get_dummies(df['ru_view_desc']))
    return df.loc[(df.Appartment == 1) & ((df.status_desc == 'Available') | (df.status_desc == 'Available Ready'))], \
           df.loc[(df.Villa == 1) & ((df.status_desc == 'Available') | (df.status_desc == 'Available Ready'))]


def process_appartments(df_app):
    df_app.drop(columns=['Villa', 'Appartment'], inplace=True)
    df_app.drop(columns='status_desc', inplace=True)
    df_app.floor.replace({'GF': 0}, inplace=True)
    df_app.set_index('_id', inplace=True)
    df_app = df_app.apply(pd.to_numeric)
    df_app.iloc[:, :] = preprocessing.StandardScaler().fit_transform(df_app)
    return df_app


def process_villas(df_villa):
    df_villa.drop(columns=['Villa', 'Appartment', 'floor'], inplace=True)
    df_villa.drop(columns='status_desc', inplace=True)
    df_villa.set_index('_id', inplace=True)
    df_villa = df_villa.apply(pd.to_numeric)
    df_villa.iloc[:, :] = preprocessing.StandardScaler().fit_transform(df_villa)
    return df_villa


def similarity_mats(app, villa):
    app_similarity = pd.DataFrame(cosine_similarity(app, app),
                                  columns=app.index,
                                  index=app.index)
    villa_similarity = pd.DataFrame(cosine_similarity(villa, villa),
                                    columns=villa.index,
                                    index=villa.index)
    return app_similarity, villa_similarity


class ContentBasedSim:
    def __init__(self, connection):
        self.connection = connection
        self.unit_sim_arr = []
    #  Update DB

    def update_record(self, key, arr):
        myquery = {"_id": key}
        newvalues = {"$set": {"recommended": arr}}
        self.connection.chatbotDataBase['Unit_Similarities'].update_one(myquery, newvalues)
        return 'done'

    def update_db(self):
        df = read_data(self.connection)
        app_data, villa_data = process_split_data(df)
        apartments = process_appartments(app_data)
        villas = process_villas(villa_data)
        app_similarity, villa_similarity = similarity_mats(apartments, villas)
        app_similarity.apply(
            lambda row: self.update_record(row.name, df.loc[df._id.isin(row.drop(labels=[row.name]).sort_values(ascending=False).head(
                20).index.values.tolist())][
            ['_id', 'bathroom', 'numberofrooms', 'area', 'delivery', 'baseprice', 'usage_type']].to_dict('records')), axis=1)
        villa_similarity.apply(
            lambda row: self.update_record(row.name, df.loc[df._id.isin(row.drop(labels=[row.name]).sort_values(ascending=False).head(
                20).index.values.tolist())][
            ['_id', 'bathroom', 'numberofrooms', 'area', 'delivery', 'baseprice', 'usage_type']].to_dict('records')), axis=1)
        return {}

    # Insert in DB

    def add_to_dict(self, key, arr):
        self.unit_sim_arr.append({"_id": key, "recommended": arr})
        return 'done'

    def insert_in_db(self):
        df = read_data(self.connection)
        app_data, villa_data = process_split_data(df)
        apartments = process_appartments(app_data)
        villas = process_villas(villa_data)
        app_similarity, villa_similarity = similarity_mats(apartments, villas)
        self.unit_sim_arr = []
        app_similarity.apply(
            lambda row: self.add_to_dict(row.name, df.loc[df._id.isin(row.drop(labels=[row.name]).sort_values(ascending=False).head(
                20).index.values.tolist())][
            ['_id', 'bathroom', 'numberofrooms', 'area', 'delivery', 'baseprice', 'usage_type']].to_dict('records')), axis=1)

        villa_similarity.apply(
            lambda row: self.add_to_dict(row.name, df.loc[df._id.isin(row.drop(labels=[row.name]).sort_values(ascending=False).head(
                20).index.values.tolist())][
            ['_id', 'bathroom', 'numberofrooms', 'area', 'delivery', 'baseprice', 'usage_type']].to_dict('records')), axis=1)
        self.connection.chatbotDataBase['Unit_Similarities'].insert_many(self.unit_sim_arr)
        return {}
