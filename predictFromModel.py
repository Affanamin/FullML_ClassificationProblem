import pandas
#from sklearn.preprocessing import StandardScaler
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation



class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()
            self.log_writer.log(self.file_object,'Let me chk data')
            print(data.head())
            #self.log_writer.log(self.file_object,data.head())
            #code change
            # wafer_names=data['Wafer']
            # data=data.drop(labels=['Wafer'],axis=1)

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            #data=preprocessor.remove_columns(data,['Wafer']) # remove the unnamed column as it doesn't contribute to prediction.
            data = preprocessor.binning(data) 
            #removing unwanted columns as discussed in the EDA part in ipynb file
            data = preprocessor.dropUnnecessaryColumns(data,['Ageband'])
            #print(data.isnull().sum())
            data =preprocessor.combiningfornewfeature(data)
            data = preprocessor.dropUnnecessaryColumns(data,['Parch', 'Sibsp', 'FamilySize'])
            
            
            data = preprocessor.convertCategoricalfeatureIntonumeric(data)

            
            data = preprocessor.binningfare(data)
            data = preprocessor.dropUnnecessaryColumns(data,['FareBand','PassengerId'])
            print(data.head())
            #print(data.isnull().sum())
            
            # check if missing values are present in the dataset
            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            
            # if missing values are there, replace them appropriately.
            if(is_null_present):
                data=preprocessor.impute_missing_values(data) # missing value imputation
            
            self.log_writer.log(self.file_object,'--Fati-03.5--')
            
            data_scaled = pandas.DataFrame(preprocessor.standardScalingData(data),columns=data.columns)
            
            #data_scaled = pandas.DataFrame(data,columns=data.columns)
            self.log_writer.log(self.file_object,'--Fati-04--,It worked :)')
            
            ##----Predictions left only, will work after lunch IA
            
            data=data.to_numpy()
            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)
            RfClassifier=file_loader.load_model('RandomForestClassifier')
            self.log_writer.log(self.file_object,'--Fati-05--')
            ##Code changed
            #pred_data = data.drop(['Wafer'],axis=1)
            classifier=RfClassifier.predict(data_scaled)#drops the first column for cluster prediction
            self.log_writer.log(self.file_object,'--Fati-06--')
            #data_scaled['clusters']=clusters
            #self.log_writer.log(self.file_object,'--Fati-07--')
            #clusters=data_scaled['clusters'].unique()
            self.log_writer.log(self.file_object,'--Fati-08--')
            print(classifier)
            #result=[] # initialize blank list for storing predicitons
            #with open('EncoderPickle/enc.pickle', 'rb') as file: #let's load the encoder pickle file to decode the values
             #   encoder = pickle.load(file)

            #for i in clusters:
             #   cluster_data= data_scaled[data_scaled['clusters']==i]
              #  cluster_data = cluster_data.drop(['clusters'],axis=1)
               # model_name = file_loader.find_correct_model_file(i)
                #model = file_loader.load_model(model_name)
                 #   result.append(val)
            result = pandas.DataFrame(classifier,columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path
        #return "worked"





