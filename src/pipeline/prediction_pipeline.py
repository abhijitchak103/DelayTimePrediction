import os, sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        pass


    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)

            return pred
        except Exception as e:
            logging.info('Error occured in PredictionPipeline.predict')
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 Delivery_person_Age: int, 
                 Delivery_person_Ratings: float, 
                 Vehicle_condition: int,
                 multiple_deliveries: int, 
                 Festival: int, 
                 Delivery_distance: float, 
                 Time_to_pick: float, 
                 Weather_conditions: str,
                 Road_traffic_density: str,
                 Type_of_order: str, 
                 Type_of_vehicle: str,
                 City: str, 
                 Time_of_Day_Ordered: str,
                 Month: str):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings 
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries 
        self.Festival = Festival
        self.Delivery_distance = Delivery_distance 
        self.Time_to_pick = Time_to_pick
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_order = Type_of_order
        self.Type_of_vehicle = Type_of_vehicle
        self.City = City
        self.Time_of_Day_Ordered = Time_of_Day_Ordered
        self.Month = Month

    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age' : [self.Delivery_person_Age],
                'Delivery_person_Ratings' : [self.Delivery_person_Ratings],
                'Vehicle_condition' : [self.Vehicle_condition],
                'multiple_deliveries' : [self.multiple_deliveries],
                'Festival' : [self.Festival],
                'Delivery_distance' : [self.Delivery_distance],
                'Time_to_pick' : [self.Time_to_pick],
                'Weather_conditions' : [self.Weather_conditions],
                'Road_traffic_density' : [self.Road_traffic_density],
                'Type_of_order' : [self.Type_of_order],
                'Type_of_vehicle' : [self.Type_of_vehicle],
                'City' : [self.City],
                'Time_of_Day_Ordered' : [self.Time_of_Day_Ordered],
                'Month' : [self.Month]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Error occured in PredictionPipeline.get_data_as_dataframe')
            raise CustomException(e, sys)