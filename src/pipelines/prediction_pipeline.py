import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,df):
        try:
            
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(df)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Item_Weight:float,
                 Item_Visibility:float,
                 Item_MRP:float,
                 Outlet_Establishment_Year:int,
                 Item_Fat_Content:str,
                 Item_Type:str,
                 Outlet_Identifier:str,
                 Outlet_Size:str,
                 Outlet_Location_Type:str,
                 Outlet_Type:str):
        
        self.Item_Weight=Item_Weight
        self.Item_Visibility=Item_Visibility
        self.Item_MRP=Item_MRP
        self.Outlet_Establishment_Year=Outlet_Establishment_Year
        self.Item_Fat_Content=Item_Fat_Content
        self.Item_Type=Item_Type
        self.Outlet_Identifier = Outlet_Identifier
        self.Outlet_Size = Outlet_Size
        self.Outlet_Location_Type = Outlet_Location_Type
        self.Outlet_Type = Outlet_Type

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Item_Weight':[self.Item_Weight],
                'Item_Visibility':[self.Item_Visibility],
                'Item_MRP':[self.Item_MRP],
                'Outlet_Establishment_Year':[self.Outlet_Establishment_Year],
                'Item_Fat_Content':[self.Item_Fat_Content],
                'Item_Type':[self.Item_Type],
                'Outlet_Identifier':[self.Outlet_Identifier],
                'Outlet_Size':[self.Outlet_Size],
                'Outlet_Location_Type':[self.Outlet_Location_Type],
                'Outlet_Type':[self.Outlet_Type]

            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)