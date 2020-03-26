# -*- coding: utf-8 -*-
from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Logger


class dataTransformPredict:

     def __init__(self):
          self.goodDataPath = "Prediction_Raw_Files_Validated/Good_Raw"
          self.logger = App_Logger()
     
     def EDA(self):
        try:
            log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                data = pandas.read_csv(self.goodDataPath + "/" + file)
                median = data['Age'].median()
                data['Age'].fillna(median, inplace=True)
                self.logger.log(log_file, " %s: Age colum transformed!!" % file)
                data = data.drop(['Ticket', 'Cabin'], axis=1)
                self.logger.log(log_file, " %s: UnWanted Colums Dropped!!" % file)
                combine = [data]
                for dataset in combine:
                    dataset["Title"] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
                
                for dataset in combine:
                    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
                    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
                    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
                    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs') 
                    
                titlemapping = {'Mr':1, 'Miss':2,'Mrs':3, 'Master':4,'Rare':5}
                for row in combine:
                    row["Title"] = row["Title"].map(titlemapping)
                    row['Title'] = row['Title'].fillna(0)
                data = data.drop(['Name'],axis = 1)
                self.logger.log(log_file, " %s: Name Column Transformed" % file)
                #titlemapping = {'male':0, 'female':1}
                #for row in combine:
                    #row["Sex"] = row["Sex"].map(titlemapping).astype(int)
                    #row['Sex'] = row['Sex'].fillna(0)
                data['Embarked'] = data["Embarked"].apply(lambda x: "'" + str(x) + "'")
                data['Sex'] = data["Sex"].apply(lambda x: "'" + str(x) + "'")
                data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                self.logger.log(log_file, " %s: EDA successful!!" % file)
        
        except Exception as e:
            log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
            self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
            log_file.close()
            raise e
        log_file.close()


     def addQuotesToStringValuesInColumn(self):

          try:
               log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
               onlyfiles = [f for f in listdir(self.goodDataPath)]
               for file in onlyfiles:
                    data = pandas.read_csv(self.goodDataPath + "/" + file)
                    
                    data['Name'] = data["Name"].apply(lambda x: "'" + str(x) + "'")
                    data['Sex'] = data["Sex"].apply(lambda x: "'" + str(x) + "'")
                    data['Cabin'] = data["Cabin"].apply(lambda x: "'" + str(x) + "'")
                    data['Embarked'] = data["Embarked"].apply(lambda x: "'" + str(x) + "'")

                    
                    data.to_csv(self.goodDataPath + "/" + file, index=None, header=True)
                    self.logger.log(log_file, " %s: Quotes added successfully!!" % file)

          except Exception as e:
               log_file = open("Prediction_Logs/dataTransformLog.txt", 'a+')
               self.logger.log(log_file, "Data Transformation failed because:: %s" % e)
               #log_file.write("Current Date :: %s" %date +"\t" +"Current time:: %s" % current_time + "\t \t" + "Data Transformation failed because:: %s" % e + "\n")
               log_file.close()
               raise e
          log_file.close()
          
        
    
   