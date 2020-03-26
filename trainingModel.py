# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:37:35 2020

@author: Affan
"""

# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger

#Creating the common Logging object


class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')
    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            self.log_writer.log(self.file_object, 'Starting of Training')
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()
            print(data.head())


            """doing the data preprocessing as dicussed in EDA"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            #data=preprocessor.remove_columns(data,['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.
            data = preprocessor.binning(data) 
            #removing unwanted columns as discussed in the EDA part in ipynb file
            data = preprocessor.dropUnnecessaryColumns(data,['Ageband'])
            #print(data.isnull().sum())
            data =preprocessor.combiningfornewfeature(data)
            data = preprocessor.dropUnnecessaryColumns(data,['Parch', 'Sibsp', 'FamilySize','Pid'])
            
            
            data = preprocessor.convertCategoricalfeatureIntonumeric(data)

            
            data = preprocessor.binningfare(data)
            data = preprocessor.dropUnnecessaryColumns(data,['FareBand'])
            print(data.head())
            #print(data.isnull().sum())
            
            # check if missing values are present in the dataset
            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            
            # if missing values are there, replace them appropriately.
            if(is_null_present):
                data=preprocessor.impute_missing_values(data) # missing value imputation
            
            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='Survived')
            print(Y)
            #We donot need to encode any value as we have opted Binning in this case. 
            #All data is fine and ready to scaling/Modeling.
            
            
            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            #for i in list_of_clusters:
            
            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=36)
            x_train_scaled = preprocessor.standardScalingData(x_train)
            x_test_scaled = preprocessor.standardScalingData(x_test)
            
            model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization
            #getting the best model.
            best_model_name,best_model,prediction,acc=model_finder.get_best_model(x_train_scaled,y_train,x_test_scaled,y_test)
            #saving the best model to the directory.
            print("Predictions:")
            print(prediction)
            print("Accuracy:")
            print(acc)
            
            file_op = file_methods.File_Operation(self.file_object,self.log_writer)
            self.log_writer.log(self.file_object, 'Going to create directory')
            
            #save_model=file_op.save_model(best_model,best_model_name+str(i))
            save_model=file_op.save_model(best_model,best_model_name)
            

            

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception