from concurrent.futures import process
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from connect import DBConnector
from sklearn import preprocessing
import numpy as np
import pandas as pd
import yaml
from clusteval import clusteval


def connect_to_db():
    return DBConnector().connection  # establish connection


class Clustered_Units:
    def __init__(self):
        self.connection = connect_to_db()

        with open(r'./columns.yaml') as file:
            self.all_columns = yaml.full_load(file)

        # Read Data
        self.original_df = self.read_data()
        # print(self.original_df.head())
        # Get nearest Neigbors dataframe
        # This dataframe holds the nearest 10 units for each unit
        self.nearest_neighbors_df = self.get_neighbors_metrices()

    def get_neighbors_metrices(self):
        # Proprocess data to get a dataframe with only the columns that are needed for getting the nearst neighbors
        processed_df = self.preprocess_data()
        # print(processed_df.head())
        # Sklearn Nearest Neighbors model
        nbrs = NearestNeighbors(n_neighbors=10)
        # Fit model with the processed data
        knbrs = nbrs.fit(processed_df)
        # Get 10 nearest neighbors for each unit
        dist, indices = knbrs.kneighbors(processed_df)
        # Return unit_id column in the dataframe
        processed_df.reset_index(inplace=True)
        # Convert indices 2-D array into dataframe
        indices_df = pd.DataFrame(indices)
        # Convert indices of the neares neighbors into unit_ids
        nearest_neighbors_df = indices_df.applymap(
            lambda x: processed_df.iloc[[x]]["unit_id"].values[0])

        # Set index of the nearest neighbors dataframe to be the unit_id column
        nearest_neighbors_df.set_index(processed_df['unit_id'], inplace=True)

        return nearest_neighbors_df

    def read_data(self):
        while True:
            try:
                sql = "SELECT * FROM eshtri.unit_search_engine where stat_id = 1;"
                df = pd.read_sql(sql, self.connection)
                self.connection.close()
                return df

            except Exception as e:
                print('Error', e)
                self.connection = connect_to_db()

    def preprocess_data(self):
        processed = self.original_df[self.all_columns["clustering_columns"]
        ].loc[self.original_df.lang_id == 1]
        processed.set_index('unit_id', inplace=True)
        processed = processed.apply(pd.to_numeric)
        processed.iloc[:, :] = preprocessing.MinMaxScaler(
        ).fit_transform(processed)
        processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        processed.dropna(inplace=True)
        return processed

    def get_recommendations(self, unit_id, lang):
        # Get Recommendations for the selected unit
        recommendations_unit_ids = self.nearest_neighbors_df.loc[unit_id]

        # Filter recommendations to remove the actual unit if it is presented in the recommendations
        recommendations_unit_ids = list(filter(
            lambda id: unit_id != id, recommendations_unit_ids))

        # Get recommended units from the original dataframe
        recommended_units = self.original_df.loc[(self.original_df.unit_id.isin(recommendations_unit_ids)) &
                                                 (self.original_df.lang_id == lang)]

        return recommended_units