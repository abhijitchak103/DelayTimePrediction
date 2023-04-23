import math
import os, sys, pickle
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def deg2rad(deg):
    """
    Returns the radian value of angle.
    ---------------------------------

    Arguments:
    ----------
    deg: Latitude or Longitude value in degrees
    ---------------------------------
    
    Returns:
    --------
    Latitude or Longitude in radians
    """
    try:
        return deg*math.pi/180
    except Exception as e:
        logging.info('Error occured in utils.deg2rad')
        raise CustomException(e, sys)


def get_distance_from_latlong_in_km(lat1, lon1, lat2, lon2):
    """
    Returns the Distance in Kms upto 2 decimal places between 2 points with Latitude and Longitude.
    --------------------------------------

    Arguments:
    ----------
    lat1: Latitude value of first point in radians
    lon1: Longitude value of first point in radians
    lat2: Latitude value of second point in radians
    lon2: Longitude value of second point in radians
    --------------------------------------
    
    Returns:
    --------
    Distance in kms
    """
    try:
        earth_radius = 6371 # Radius of earth in kms
        diff_lat = deg2rad(lat2-lat1)
        diff_lon = deg2rad(lon2-lon1)
        
        a = math.sin(diff_lat/2)**2 + math.cos(deg2rad(lat1))*math.cos(deg2rad(lat2))*math.sin(diff_lon/2)**2
        c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = earth_radius*c
        
        return round(d, 2)
    except Exception as e:
        logging.info('Error occured in utils.getDistancefromLatLonginKm')
        raise CustomException(e, sys)
    

def preprocess_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input pd.DataFrame columns and returns the updated pd.DataFrame
    -----------------------------------------

    Arguments:
    ----------
    df: Pandas DataFrame

    -----------------------------------------

    Returns:
    --------
    df: Pandas DataFrame
    """
    try:
        order_date = list(df["Order_Date"])

        order_date = [x.replace("-", "/") for x in order_date]

        df["Order_Date"] = order_date
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True)
        logging.info('Changed Order_Date to datetime object')

        time_ordered = list(df["Time_Orderd"])
        time_picked = list(df["Time_Order_picked"])

        df['Time_Orderd_num'] = transform_time_columns(time_ordered)
        df['Time_Order_picked_num'] = transform_time_columns(time_picked)
        logging.info('Created temporary numerical columns from Time_orderd and Time_Order_picked')

        df = df.mask(df == '')
        df.dropna(axis=0, inplace=True)
        logging.info('Dropped null rows in new columns')


        df['Time_to_pick'] = (df['Time_Order_picked_num'] - df['Time_Orderd_num'])*60
        df['Time_to_pick'] = np.around(df['Time_to_pick'].astype(np.double),3)
        logging.info('Created new column Time_to_pick')

        df.drop(columns = ['Time_Orderd_num', 'Time_Order_picked_num', 'Time_Order_picked'], axis = 1, inplace = True)
        logging.info('Dropped temporary columns and Time_Order_picked column')

        df['Month'] = df["Order_Date"].dt.month
        df.drop(columns = "Order_Date", axis = 1, inplace = True)
        logging.info('Created new Month feature from Order_Date and dropped Order_Date')

        return df
    except Exception as e:
        logging.info('Error occured in utils.preprocessDataColumns')
        raise CustomException(e, sys)


def transform_time_columns(list_to_clean: list) -> list:
    """
    Returns a modified list
    -----------------------
    
    Arguments:
    ----------
    list_to_clean: list
    -----------------------

    Returns:
    --------
    list
    """
    try:

        # Cleaning data not having ':' because of irrelevant information 
        for i, x in enumerate(list_to_clean):
            if ':' not in str(x):
                list_to_clean[i] = ''
        
        # Cleaning data for typos
        for i, x in enumerate(list_to_clean):
            if x.count(':') > 1:
                list_to_clean[i]= x[:5]
        
        # Getting the time in hours for calculation 
        for i, x in enumerate(list_to_clean):
            if x != '':
                hour = float(x[:2])
                mins = float(x[3:])

                list_to_clean[i] = round(hour + mins/60, 2)
                
        return list_to_clean
    except Exception as e:
        logging.info('Error Occured in utils.transform time columns')
        raise CustomException(e, sys)
    

def get_part_of_day(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Creates a new column Time_of_Day_Ordered which holds the part of day of order
    
    Arguments:
    ----------
    df: pandas DataFrame
    column: column name to extract information from

    Returns:
    --------
    df: pandas DataFrame
    """
    try:    
        time_of_day = list(df[column])

        for i, x in enumerate(time_of_day):
            time_of_day[i] = float(x[:2]) + float(x[3:])/60
            
        for i, x in enumerate(time_of_day):
            if 0 < x <= 6:
                time_of_day[i] = 'Early Morning'
            elif 6 < x <= 12:
                time_of_day[i] = 'Morning'
            elif 12 < x <= 18:
                time_of_day[i] = 'Evening'
            elif 18 < x <= 24:
                time_of_day[i] = 'Night'
                
        df['Time_of_Day_Ordered'] = time_of_day
        df.drop(columns = column, axis = 1, inplace = True)
        return df
    except Exception as e:
        logging.info('Error occured in utils.get_part_of_day')
        raise CustomException(e, sys)
    

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a modified DataFrame
    
    Arguments:
    ----------
    df: pd.DataFrame

    Returns:
    --------
    df: pd.DataFrame
    """
    try:
        df.drop(columns='ID', axis=1, inplace=True)
        logging.info('Dropped ID column')

        df['Delivery_distance'] = df.apply(
        lambda x: get_distance_from_latlong_in_km(x['Restaurant_latitude'], 
                                                x['Restaurant_longitude'], 
                                                x['Delivery_location_latitude'], 
                                                x['Delivery_location_longitude']), axis=1)
        logging.info("Created Column 'Delivery_distance'")
        
        columns_to_drop = ['Restaurant_latitude', 'Restaurant_longitude', 
                            'Delivery_location_latitude', 'Delivery_location_longitude']
        df.drop(columns = columns_to_drop, axis=1, inplace=True)
        logging.info('Dropped Latitude and Longitude columns successfully')

        q = df['Delivery_distance'].quantile(0.99)
        df = df[df['Delivery_distance'] < q ]
        logging.info("Removed outliers from 'Delivery_distance'")

        logging.info('Preprocessing df to clean Datetime columns')
        df = preprocess_date_columns(df)
        logging.info('Preprocessing df completed')


        df = get_part_of_day(df=df, column='Time_Orderd')
        logging.info('Created Time_of_day_Ordered feature and dropped Time_Orderd')

        cities = list(df['Delivery_person_ID'])
        cities = [x[:x.find('RES')] for x in cities]

        df['City'] = cities
        df.drop(columns = 'Delivery_person_ID', axis = 1, inplace = True)
        logging.info('Created City feature for City of Order. Dropped Delivery_person_ID')

        return df
    except Exception as e:
        logging.info('Error Occcured in utils.preprocess_df')
        raise CustomException(e, sys)
    

def get_dummies_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        festival = {'No': 0, 'Yes': 1}
        months = {1: 'Jan',
                2: 'Feb',
                3: 'Mar',
                4: 'Apr',
                5: 'May',
                6: 'Jun',
                7: 'Jul',
                8: 'Aug',
                9: 'Sep',
                10: 'Oct',
                11: 'Nov',
                12: 'Dec'}

        df.replace({"Festival": festival}, inplace=True)
        df.replace({"Month": months}, inplace=True)

        cat_columns = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order', 
                    'Type_of_vehicle', 'City', 'Time_of_Day_Ordered', 'Month']

        df = pd.get_dummies(df, columns=cat_columns, dtype=float)

        return df
    except Exception as e:
        logging.info('Error occured in utils.reorganize_df')
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as path:
            pickle.dump(obj, path)

    except Exception as e:
        logging.info('Error occured in save_object')
        raise CustomException(e, sys)
    

def evaluate_model(X_train, X_test, y_train, y_test, models):
    try:
        report = {}
        for key, value in models.items():
            model = value

            logging.info(f'Training data with {value} model')
            model.fit(X_train, y_train)
            logging.info('Data trained')

            logging.info(f'Predicting with {value} model')
            y_pred = model.predict(X_test)
            logging.info('Prediction Complete')

            logging.info('Evaluating r2 scores for test data')
            test_model_score = r2_score(y_true=y_test, y_pred=y_pred)
            logging.info(f'Obtained R2 score for {value} model')

            report[key] = test_model_score
        
        return report
    except Exception as e:
        logging.info('Error occured in evaluate_model stage')
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info('Error occured in utils.load_object')
        raise CustomException(e, sys)