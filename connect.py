import pymongo
from pymongo import MongoClient


class MongoConnector:
    def __init__(self):
        self.connection = MongoClient(
            'mongodb://ibm_cloud_c05b2233_1e07_49a1_a407_6f485d72f620:8d94edb1a13161e8b2aa158f1ad78611f2120e35d68921d5372e22c80b139358@0146c0f7-b4d2-49cb-b1e1-616a80c40926-0.bv72mkuf0ul4s4tm7p1g.databases.appdomain.cloud:32595,0146c0f7-b4d2-49cb-b1e1-616a80c40926-1.bv72mkuf0ul4s4tm7p1g.databases.appdomain.cloud:32595,0146c0f7-b4d2-49cb-b1e1-616a80c40926-2.bv72mkuf0ul4s4tm7p1g.databases.appdomain.cloud:32595/ibmclouddb?authSource=admin&replicaSet=replset',
            tls=True,
            tlsCAFile='./mongo.crt')
