import atexit
from apscheduler.schedulers.background import BackgroundScheduler
from connect import DBConnector
import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing


def connect_to_db():
    return DBConnector().connection  # establish connection

class Clustered_Units:
    def __init__(self):
        # Read Data
        print("Starting reading from database")
        self.og_units, self.og_clicks = self.read_data()
        # self.save_data_csv()
        print("data has been read successfully")

        # # Load data from CSV
        # self.og_units, self.og_clicks = self.load_data_csv()

        # Preprocessing
        self.units, self.clicks = self.data_preprocessing()

        # Get similarity matrix
        self.SimMatrix = self.get_similarity_dataframe()
        print(self.SimMatrix.head())

        # Scheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(func=self.update_df, trigger="interval", seconds=1800)
        scheduler.start()
        atexit.register(lambda: scheduler.shutdown())

    def save_data_csv(self):
        self.og_units.to_csv("temp_units.csv")
        self.og_clicks.to_csv("temp_clicks.csv")

    def load_data_csv(self):
        og_units = pd.read_csv("temp_units.csv", index_col=False)
        og_clicks = pd.read_csv("temp_clicks.csv", index_col=False)
        return og_units, og_clicks

    def update_df(self):
        print('updating dataframe')
        print(datetime.datetime.now())
        self.og_units, self.og_clicks = self.read_data()
        self.units, self.clicks = self.data_preprocessing()
        print("DONE!")
        print('updating similarity matrix')
        self.SimMatrix = self.get_similarity_dataframe()
        print("DONE !")
        print(self.SimMatrix.head())

    def read_data(self):
        """
        Reads clicks and units data from the database returning them
        Returns the dataframe of  clicks and units from database
        """
        while True:
            try:
                self.connection = connect_to_db()

                # Reading units
                sql = "SELECT * FROM eshtri.unit_search_sorting where price > 100000;"
                print("Reading Units from unit_search_sorting table...")
                og_units = pd.read_sql(sql, self.connection)
                print("successful!")

                # # there are some problems with exporting the data so we remove those corrupted units
                # columns = ["unit_id", "reg_id", "cat_id", "usg_id", "bathroom", "room", "area", "price", "balcony",
                #            "unit_search_status", "comp_id"]
                # og_units.drop(og_units[og_units.cat_id == "Q"].index, axis=0, inplace=True)
                # og_units.dropna(inplace=True)

                # Reading clicks
                sql2 = "SELECT * FROM eshtri.history;"
                print("Reading Clicks from history table...")
                og_clicks = pd.read_sql(sql2, self.connection)
                print("successful!")

                return og_units, og_clicks

            except Exception as e:
                print('Error', e)
                self.connection = connect_to_db()
            finally:
                self.connection.close()

    def data_preprocessing(self):
        """
        og_units/og_clicks: Original dataframes of clicks and units without any processing
        returns Both data frames after some preprocessig "described using comments"
        """
        print("Preprocessing data...")
        columns = ["unit_id", "reg_id", "cat_id", "usg_id", "bathroom", "room", "area", "price", "balcony",
                   "unit_search_status", "comp_id"]
        units = self.og_units[columns].copy()
        clicks = self.og_clicks[["usr_id", "unt_id"]].copy()
        ###### UNITS
        # Leave only available items (unit_search_status == 1)
        units = units[units.unit_search_status == 1]
        units.drop(columns=["unit_search_status"], inplace=True)

        # drop duplicates as units duplicated because of the two languages
        units.drop_duplicates(inplace=True, keep='first')
        units.reset_index(drop=True, inplace=True)

        # change datatype for later use
        units[["unit_id", "balcony", "room", "reg_id", "cat_id", "usg_id", "bathroom"]] = units[
            ["unit_id", "balcony", "room", "reg_id", "cat_id", "usg_id", "bathroom"]].astype(int)
        units[["area", "price"]] = units[["area", "price"]].astype(float)

        ###### CLICKS
        # Rename columns
        clicks.columns = ["user_id", "unit_id"]

        # remove duplicates (a user click on the same item multiple times is the same as clicking one time)
        clicks.drop_duplicates(inplace=True, keep='first')
        clicks.sort_values(by="user_id", inplace=True)

        print("Done!")
        return units, clicks

    def feature_scaling(self, dum):
        """
        dum: Dataframe with one hot encoded categorical variables
        returns: Scaled numeric features between 0 and 1
        """
        processed = dum.apply(pd.to_numeric)
        processed.iloc[:, :] = preprocessing.MinMaxScaler().fit_transform(processed)
        processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        return processed

    def model_preprocessing(self, df):
        """
        df: units dataframe
        Returns the dataframe after one hot encoding for categorical values and feature scaling for numeric values
        """
        # Get hot encoding for categorical variables in units
        units_dum = pd.get_dummies(df, columns=["reg_id", "cat_id", "usg_id"] , drop_first=True)
        units_dum.drop(columns=["comp_id"], inplace=True)

        # Setting index for later
        units_dum.set_index('unit_id', inplace=True)

        # Some preprocessing and scaling for features for the knn
        processed = self.feature_scaling(units_dum)

        return processed

    def getNN(self, processed):
        """
        processed: units dataframe after full processing ready to enter the KNN model
        Returns 2 arrays:
        dist: array of distances between each unit and all other units in the dataframe
        indices: array of indices of those distances
        """
        nbrs = NearestNeighbors(n_neighbors=100)
        # Fit model with the processed data
        print("KNN process....")
        knbrs = nbrs.fit(processed)
        # Get all nearest neighbors for each unit
        dist, indices = knbrs.kneighbors(processed, processed.shape[0], return_distance=True)
        print("Done")
        return dist, indices

    def get_similarity_dataframe(self):
        """
        data: input datafame of the units
        Returns Similarity matrix in form of dataframe of all units with each other
        """
        # Preprocessing Data
        processed = self.model_preprocessing(self.units)

        # Get nearest neighbours
        X, Y = self.getNN(processed)

        z = []
        print("Creating similarity matrix....")
        for i in range(len(X)):
            z.append([x for _, x in sorted(zip(Y[i], X[i]))])
        t = pd.DataFrame(z)
        print("Done!")
        return t

    # Based on User history
    def get_user_recommended_units(self, usrid, lang, k=10):
        """
        usrid: CurrentUserID
        k: number of required recommendations, default = 10
        Returns the dataframe of top recommended "unique compound" units for user clicks

        """

        # get current user clicked units
        currUnits = list(self.clicks[self.clicks.user_id == usrid].unit_id)

        # Get index of clicked units
        currIndx = self.units[self.units.unit_id.isin(currUnits)].index

        # subset from similarity dataframe
        curr_df = self.SimMatrix.loc[currIndx].copy()

        # Get mean of similarity between clicked units and all units
        curr_df.loc["mean"] = list(curr_df.mean(axis=0))

        # getting index of top (n) similarity {smallest distance} of the mean values
        topRidx = curr_df.loc[["mean"]].apply(pd.Series.nsmallest, axis=1, n=300).columns
        # print(topRidx)

        # get recommended units sorted by highet similarity
        sorted_recommended_units = self.units[self.units.index.isin(topRidx)].reindex(topRidx)

        # drop items clicked by user to recommended unclicked items
        sorted_recommended_units.drop(sorted_recommended_units[sorted_recommended_units.unit_id.isin(currUnits)].index,
                                      axis=0, inplace=True)

        ## get only unique compound units (This a required filter and can be changed if needed)
        sorted_recommended_units.drop_duplicates(subset=["comp_id"], inplace=True, keep='first')

        IDs = list(sorted_recommended_units.unit_id)
        t = self.og_units[self.og_units.unit_id.isin(IDs)]
        t = t[t.lang_id == lang]
        t.set_index('unit_id', drop=True, inplace=True)
        x = t.reindex(IDs)
        x.reset_index(inplace=True)
        x.dropna(subset=["lang_id"], inplace=True)
        return x.iloc[:k]

    # Find similar Items
    def get_similar_units(self, untID, lang, k=10):
        """
        untID: current unit id
        k: number of required recommendations, default = 10
        Returns the dataframe of top similar "unique compound" units  for clicked unit

        """
        # Get index of clicked units
        currIndx = self.units[self.units.unit_id == untID].index

        # subset from similarity dataframe
        curr_df = self.SimMatrix.loc[currIndx].copy()

        # getting index of top (n) similarity {smallest distance} from the item
        topRidx = curr_df.apply(pd.Series.nsmallest, axis=1, n=300).columns

        # get recommended units sorted by highest similarity
        sorted_recommended_units = self.units[self.units.index.isin(topRidx)].reindex(topRidx)

        ## get only unique compound units (This a required filter and can be changed if needed)
        sorted_recommended_units.drop_duplicates(subset=["comp_id"], inplace=True, keep='first')

        IDs = list(sorted_recommended_units.unit_id)
        t = self.og_units[self.og_units.unit_id.isin(IDs)]
        t = t[t.lang_id == lang]
        t.set_index('unit_id', drop=True, inplace=True)
        x = t.reindex(IDs)
        x.reset_index(inplace=True)
        x.dropna(subset=["lang_id"], inplace=True)

        return x.iloc[:k]

    def get_popular_units(self, lang,k=10):
        """
        k: number of required recommendations, default = 10
        Returns the dataframe of trending items "unique compound" units  for clicked unit

        """
        counted_list = self.og_clicks.unt_id.value_counts().index
        temp = self.units.set_index('unit_id')
        z = temp[temp.index.isin(counted_list)].reindex(counted_list)
        z.drop_duplicates(subset=["comp_id"], inplace=True, keep='first')
        z.dropna(inplace=True)
        IDs = list(z.index)
        t = self.og_units[self.og_units.unit_id.isin(IDs)]
        t = t[t.lang_id == lang]
        t.set_index('unit_id', drop=True, inplace=True)
        x = t.reindex(IDs)
        x.reset_index(inplace=True)
        x.dropna(subset=["lang_id"], inplace=True)
        return x.iloc[:k]

    def get_user_clicks(self, Usrid):
        currUnits = list(self.clicks[self.clicks.user_id == Usrid].unit_id)
        print("Current user clicked units")
        return self.units[self.units.unit_id.isin(currUnits)]

    def get_unit(self, untID):
        return self.units[self.units.unit_id == untID]
