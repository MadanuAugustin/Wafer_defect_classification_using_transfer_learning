
import os
import sys
from pymongo import MongoClient
import gridfs
from src.waferDetection.entity.config_entity import DataIngestionConfig
from src.waferDetection import CustomException, logger


class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config

    
    def download_file(self) -> str:
        try:
            # Connect to MongoDB
            client = MongoClient('mongodb://localhost:27017/')
            db = client['wafer-detection']
            fs = gridfs.GridFS(db)
            images_collection = db['wafer-data']

            logger.info(f'----------Connected to Monogdb server-----------------')

            # Directory to save retrieved images
            retrieve_directory = self.config.local_data_path

            logger.info(f'------------Feteching the data from database--------------')

            for image_document in images_collection.find():
                image_id = image_document['gridfs_id']
                filename = image_document['filename']
                folder_path = image_document['folder_path']
                output_data = fs.get(image_id).read()
                output_folder_path = os.path.join(retrieve_directory, folder_path)
                os.makedirs(output_folder_path, exist_ok=True)
                output_path = os.path.join(output_folder_path, filename)
                with open(output_path, 'wb') as f:
                    f.write(output_data)
                    print(f"Retrieved and saved {filename} to {output_folder_path}")

            
            logger.info(f'------------Successfully completed feteching data from the database---------------------')


        except Exception as e:
            raise(CustomException(e, sys))
        

