
import os
import sys
import math
import tensorflow
from src.waferDetection.entity.config_entity import TrainingConfig
from src.waferDetection import logger, CustomException
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



class Training:

    def __init__(self, config : TrainingConfig):

        self.config = config


    def training_model(self):
        try:

            logger.info(f'------------Entered training_model function-----------------------')

            logger.info(f'----------------creating train and test generators----------------')

            test_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.20)

            train_datagen = ImageDataGenerator(
                rotation_range = 40,
                horizontal_flip = True,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                shear_range = 0.2,
                zoom_range = 0.2,
                rescale = 1./255,
                validation_split = 0.20
            )


            train_generator = train_datagen.flow_from_directory(
                directory = self.config.training_data_path,
                subset = 'training',
                shuffle = True,
                target_size = self.config.params_image_size[:-1],
                batch_size = self.config.params_batch_size,
                interpolation = 'bilinear',
                class_mode='categorical'
            )

            logger.info(f'------------train_generator found {train_generator.samples} samples---------------')

            valid_generator = test_datagen.flow_from_directory(
                directory = self.config.training_data_path,
                subset = 'validation',
                shuffle = False,
                target_size = self.config.params_image_size[:-1],
                batch_size = self.config.params_batch_size,
                interpolation = 'bilinear',
                class_mode='categorical'
            )

            logger.info(f'------------test_generator found {valid_generator.samples} samples---------------')

            logger.info(f'-------------completed creating train and test generators---------------------')

            logger.info(f'------------loading the model----------------')

            model = load_model(self.config.updated_base_model_path)

            logger.info(f'---------------model successfully loaded------------------')

            logger.info(f'----------creating Modelcheckpoint and earlystopping---------------')

            model_checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                self.config.trained_model_path,
                monitor = 'val_accuracy',
                save_best_only = True,
                save_weights_only = False,
                mode = 'max',
                verbose = 1
            )

            early_stopping = EarlyStopping(
                monitor = 'val_accuracy',
                patience = 5,
                mode = 'max',
                verbose = 1
            )

            logger.info(f'--------------successfully created Modelcheckpoint and earlystopping--------------------')

            logger.info(f'------------Training of the model started-----------------------')

            model.fit(
                train_generator,
                steps_per_epoch = math.ceil(train_generator.samples // train_generator.batch_size),
                epochs = self.config.params_epochs,
                validation_data = valid_generator,
                validation_steps = math.ceil(valid_generator.samples // valid_generator.batch_size),
                callbacks = [model_checkpoint_callback, early_stopping]
            )


            logger.info(f'-----------Model Training completed and model saved successfully------------------------')



        except Exception as e:
            raise CustomException(e, sys)