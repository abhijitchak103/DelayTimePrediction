import os, sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, get_dummies_df

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        try:
            logging.info('Data Transfromation Started')

            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
                                 'multiple_deliveries', 'Festival', 'Delivery_distance', 'Time_to_pick', 
                                 'Weather_conditions_Cloudy', 'Weather_conditions_Fog', 'Weather_conditions_Sandstorms',
                                 'Weather_conditions_Stormy', 'Weather_conditions_Sunny', 'Weather_conditions_Windy', 
                                 'Road_traffic_density_High', 'Road_traffic_density_Jam', 'Road_traffic_density_Low', 
                                 'Road_traffic_density_Medium', 'Type_of_order_Buffet', 'Type_of_order_Drinks', 
                                 'Type_of_order_Meal', 'Type_of_order_Snack', 'Type_of_vehicle_electric_scooter', 
                                 'Type_of_vehicle_motorcycle', 'Type_of_vehicle_scooter', 'City_AGR', 'City_ALH', 
                                 'City_AURG', 'City_BANG', 'City_BHP', 'City_CHEN', 'City_COIMB', 'City_DEH', 'City_GOA', 
                                 'City_HYD', 'City_INDO', 'City_JAP', 'City_KNP', 'City_KOC', 'City_KOL', 'City_LUDH', 
                                 'City_MUM', 'City_MYS', 'City_PUNE', 'City_RANCHI', 'City_SUR', 'City_VAD', 
                                 'Time_of_Day_Ordered_Evening', 'Time_of_Day_Ordered_Morning', 'Time_of_Day_Ordered_Night', 
                                 'Month_Apr', 'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jan', 'Month_Jul', 'Month_Jun', 
                                 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep']

            logging.info('Defining Numerical Pipeline')
            numerical_pipeline = Pipeline(
                steps = [
                ('Imputer', SimpleImputer(strategy='median')),
                ('Scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Defining Preprocessor')
            preprocessor = ColumnTransformer(
                [
                ('Numerical_Pipeline', numerical_pipeline, numerical_columns)
                ]
            )

            logging.info('Pipeline Created')

            return preprocessor
        
        except Exception as e:
            logging.info('Error Occured in get_data_transformation_obj')
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info('Reading Training and test Data')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read Train and test data')

            logging.info(f"Training dataset head: \n{train_df.head().to_string()}")
            logging.info(f"Test dataset head: \n{test_df.head().to_string()}")

            train_df = get_dummies_df(train_df)
            logging.info('Created Dummies for Categorical Columns in Training Data')

            test_df = get_dummies_df(test_df)
            logging.info('Created Dummies for Categorical Columns in Test Data')

            logging.info('Obtaining Preprocessor Object')
            preprocessor_obj = self.get_data_transformation_obj()

            target_column = 'Time_taken (min)'
            
            input_feature_train_df = train_df.drop(columns = target_column, axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = train_df.drop(columns = target_column, axis = 1)
            target_feature_test_df = train_df[target_column]
            

            logging.info('Transforming using preprocessor object')
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info('Train and test data transformed')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Preprocessor pickle file saved')

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Error occured in initiate_data_transformation')
            raise CustomException(e, sys)