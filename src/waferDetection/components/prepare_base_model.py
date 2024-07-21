

import os
import tensorflow
from src.waferDetection.entity.config_entity import PrepareBaseModelConfig
from src.waferDetection import logger, CustomException
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class PrepareBaseModel:
    def __init__(self, config : PrepareBaseModelConfig):

        self.config = config


    def model_construction(self):

        logger.info(f'-------------started model_construction function--------------------')

        logger.info(f'--------------creating base_model-------------------')

        base_model = VGG16(weights=self.config.params_weight, include_top=self.config.params_include_top, input_shape=self.config.params_image_size)     

        base_model.save(self.config.base_model_path, include_optimizer = False)

        logger.info(f'-------------successfully created base_model and saved-----------------')

        logger.info(f'-----------customizing the base_model-------------------------')

        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(256, activation = 'relu')(x)
        predictions = Dense(9, activation = 'softmax')(x)

        model = Model(inputs = base_model.input, outputs = predictions)

        model.compile(optimizer = tensorflow.keras.optimizers.Adam(learning_rate = self.config.params_learning_rate),
                       loss = 'categorical_crossentropy',
                         metrics = ['accuracy'])
        
        model.summary()

        model.save(self.config.updated_base_model_path)

        logger.info(f'-------------------successfully customized the base model and saved-------------------------')