import math
import os, sys
from logger import logging
from exception import CustomException

import pandas as pd
import numpy as np


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


def getDistancefromLatLonginKm(lat1, lon1, lat2, lon2):
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