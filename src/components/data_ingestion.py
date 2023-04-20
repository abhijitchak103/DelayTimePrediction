import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import getDistancefromLatLonginKm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Started')
        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'data.csv'))
            logging.info('Read csv file as pandas.DataFrame')
            
            df.drop(columns='ID', axis=1, inplace=True)
            logging.info('Dropped ID column')

            df['Delivery_distance'] = df.apply(
            lambda x: getDistancefromLatLonginKm(x['Restaurant_latitude'], 
                                                 x['Restaurant_longitude'], 
                                                 x['Delivery_location_latitude'], 
                                                 x['Delivery_location_longitude']), axis=1)
            logging.info("Created Column 'Delivery_distance'")
            
        except Exception as e:
            logging.info('Error occured in DataIngestion.initiate_data_ingestion')
            raise CustomException(e, sys)