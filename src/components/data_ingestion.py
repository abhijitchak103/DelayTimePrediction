import os, sys
from src.logger import logging
from src.exception import CustomException
from src.utils import preprocess_df

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

            logging.info('Preprocessing Data started')
            df = preprocess_df(df=df)
            logging.info('Successfully preprocessed DataFrame')

            
            


        except Exception as e:
            logging.info('Error occured in DataIngestion.initiate_data_ingestion')
            raise CustomException(e, sys)