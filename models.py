import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Add, GlobalAveragePooling2D, Dense, Dropout, Conv1D
import random
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import rasterio 
import timeit
from pandas import DataFrame

class InceptionTime(Model):
    def __init__(self, input_shape=(24, 4, 1)):
        super(InceptionTime, self).__init__()
        self.input_shape = input_shape
        self.input_layer = layers.Input(shape=input_shape)
        self.conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape, padding='same')
        self.pool1 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D(pool_size=(2,2))
        self.conv3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D(pool_size=(2,2), padding='same')
        self.flatten_cnn = layers.Flatten()
        
        self.tcn1 = layers.Conv2D(64, (7,1), dilation_rate=2, activation='relu', padding='same')
        self.tcn2 = layers.Conv2D(64, (7,1), dilation_rate=4, activation='relu', padding='same')
        self.tcn3 = layers.Conv2D(64, (7,1), dilation_rate=8, activation='relu', padding='same')

        self.conv1x1 = layers.Conv2D(64, (1,1), activation='relu', padding='same')
        self.conv3x3_d2 = layers.Conv2D(64, (3,3), dilation_rate=2, activation='relu', padding='same')
        self.conv5x5_d4 = layers.Conv2D(64, (5,5), dilation_rate=4, activation='relu', padding='same')
        self.concatenated = layers.Concatenate()
        
        self.residual = layers.Add()
        self.flatten_tcn = layers.Flatten()
        self.output_layer = layers.Dense(8, activation='softmax')
    
    def call(self, inputs):
        conv1_out = self.conv1(inputs)
        pool1_out = self.pool1(conv1_out)
        conv2_out = self.conv2(pool1_out)
        pool2_out = self.pool2(conv2_out)
        conv3_out = self.conv3(pool2_out)
        pool3_out = self.pool3(conv3_out)
        flatten_cnn_out = self.flatten_cnn(pool3_out)
        
        tcn1_out = self.tcn1(inputs)
        tcn2_out = self.tcn2(tcn1_out)
        tcn3_out = self.tcn3(tcn2_out)
        
        conv1x1_out = self.conv1x1(tcn3_out)
        conv3x3_d2_out = self.conv3x3_d2(tcn3_out)
        conv5x5_d4_out = self.conv5x5_d4(tcn3_out)
        concatenated_out = self.concatenated([conv1x1_out, conv3x3_d2_out, conv5x5_d4_out])
        
        residual_out = self.residual([tcn2_out, tcn3_out])
        flatten_tcn_out = self.flatten_tcn(residual_out)
        
        concatenated_output = self.concatenated([flatten_cnn_out, flatten_tcn_out])
        output = self.output_layer(concatenated_output)
        
        return output

class MultiRocket:
    def __init__(self):
        pass
    
    def evaluate(clf, rocket, x_train, y_train, x_valid, y_valid):
        clf.set_params(**rocket)
        clf.fit(x_train, y_train)
        accuracy = accuracy_score(y_valid, clf.predict(x_valid))
        return accuracy

    def update_rocket(clf, rocket, x_train, y_train, x_valid, y_valid):
        pass

    def multirocket(clf, num_rockets, num_iterations, x_train, y_train, x_valid, y_valid):
        param_grid = {"n_estimators": range(10, 101),
                      "max_depth": range(1, 11)}
        rockets = []
        for i in range(num_rockets):
            rocket = {"n_estimators": random.randint(10, 100), 
                      "max_depth": random.randint(1, 10)}
            rockets.append(rocket)
        for i in range(num_iterations):
            for j in range(num_rockets):
                random_search = RandomizedSearchCV(clf, param_grid, n_iter=1, cv=5, scoring=self.evaluate, refit=True)
                random_search.fit(x_train, y_train)
                rockets[j] = random_search.best_params_
        return max(rockets, key=lambda x: self.evaluate(clf, x, x_train, y_train, x_valid, y_valid))

class ResNet50:
    def __init__(self, input_shape, nb_classes):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
    
    def residual_block(X_start, filters, name, reduce=False, res_conv2d=False):
        nb_filters_1, nb_filters_2, nb_filters_3 = filters
        strides_1 = [2,2] if reduce else [1,1]
        X = Conv2D(filters=nb_filters_1, kernel_size=[1,1], strides=strides_1, padding='same', name=name)(X_start)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=nb_filters_2, kernel_size=[3,3], strides=[1,1], padding='same')(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=nb_filters_3, kernel_size=[1,1], strides=[1,1], padding='same')(X)
        X = BatchNormalization()(X)
        if res_conv2d:
            X_res = Conv2D(filters=nb_filters_3, kernel_size=[1,1], strides=strides_1, padding='same')(X_start)
            X_res = BatchNormalization()(X_res)
        else:
            X_res = X_start
        X = Add()([X, X_res])
        X = Activation('relu')(X)
        return X

    def resnet50(input_shape, nb_classes):
        X_input = layers.Input(shape=input_shape)
        X = Conv2D(filters=64, kernel_size=[7,7], strides=[2,2], padding='same', name='conv1')(X_input)
        X = BatchNormalization(name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D([3,3], strides=[2,2])(X)

        X = residual_block(X, filters=[64, 64, 256], name='conv2_a', reduce=False, res_conv2d=True)
        X = residual_block(X, filters=[64, 64, 256], name='conv2_b')
        X = residual_block(X, filters=[64, 64, 256], name='conv2_c')

        X = residual_block(X, filters=[128, 128, 512], name='conv3_a', reduce=True, res_conv2d=True)
        X = residual_block(X, filters=[128, 128, 512], name='conv3_b')
        X = residual_block(X, filters=[128, 128, 512], name='conv3_c')
        X = residual_block(X, filters=[128, 128, 512], name='conv3_d')

        X = residual_block(X, filters=[256, 256, 1024], name='conv4_a', reduce=True, res_conv2d=True)
        X = residual_block(X, filters=[256, 256, 1024], name='conv4_b')
        X = residual_block(X, filters=[256, 256, 1024], name='conv4_c')
        X = residual_block(X, filters=[256, 256, 1024], name='conv4_d')
        X = residual_block(X, filters=[256, 256, 1024], name='conv4_e')
        X = residual_block(X, filters=[256, 256, 1024], name='conv4_f')

        X = residual_block(X, filters=[512, 512, 2048], name='conv5_a', reduce=True, res_conv2d=True)
        X = residual_block(X, filters=[512, 512, 2048], name='conv5_b')
        X = residual_block(X, filters=[512, 512, 2048], name='conv5_c')

        X = GlobalAveragePooling2D(name='avg_pool')(X)
        X = Flatten()(X)
        X = Dense(units=nb_classes, activation='softmax')(X)

        model = tf.keras.models.Model(inputs=X_input, outputs=X)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                      loss="sparse_categorical_crossentropy", metrics=['accuracy'])

        return model

class RandomForest:
    def __init__(self, tuned_parameters={'n_estimators': [10], 'max_depth': [20]}):
        self.tuned_parameters = tuned_parameters
    
    def train(self, X, y, train_y, valid_X, valid_y):
        mytestfold = [-1] * len(y) + [0] * len(valid_y)
        ps = PredefinedSplit(test_fold=mytestfold)
        clf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=self.tuned_parameters, cv=ps, n_jobs=-1, verbose=2)
        clf.fit(X, y)
        return clf
    
class TempCNN(tf.keras.Model):
    '''
    TempCNN encoder from (Pelletier et al, 2019) 
    https://www.mdpi.com/2072-4292/11/5/523
    '''
    def __init__(self, n_filters=64, drop=0.5):
        super(TempCNN, self).__init__(name='TempCNN')
        self.conv1 = Conv1D(filters=n_filters, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(l=1E-6))
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.drop_layer1 = Dropout(rate=drop)
        self.conv2 = Conv1D(filters=n_filters, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(l=1E-6))
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.drop_layer2 = Dropout(rate=drop)
        self.conv3 = Conv1D(filters=n_filters, kernel_size=1, kernel_regularizer=tf.keras.regularizers.l2(l=1E-6))
        self.bn3 = BatchNormalization()
        self.act3 = Activation('relu')
        self.drop_layer3 = Dropout(rate=drop)
        self.flatten = Flatten()
    
    def call(self, inputs, is_training):
        x = self.drop_layer1(self.act1(self.bn1(self.conv1(inputs))), is_training)
        x = self.drop_layer2(self.act2(self.bn2(self.conv2(x))), is_training)
        x = self.drop_layer3(self.act3(self.bn3(self.conv3(x))), is_training)
        return self.flatten(x)

