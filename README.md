# FullML_ClassificationProblem
Machine Learning model training with hyper parameter tuning in flask. 

**Problem Statement**
To build a classification model to predict the survivals in titanic incidents based on the given different factors in the training data..

**Architecture**

![1](https://user-images.githubusercontent.com/36659805/77675115-59cfba80-6fae-11ea-9190-ec56391e4708.PNG)

**Data Description**

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

**Data Dictionary**

Variable		Definition			Key
survival		Survival			0 = No, 1 = Yes
pclass		Ticket class		1 = 1st, 2 = 2nd, 3 = 3rd
sex			Sex	
Age			Age in years	
sibsp			# of siblings / spouses aboard the Titanic	
parch			# of parents / children aboard the Titanic	
ticket			Ticket number	
fare			Passenger fare	
cabin			Cabin number	
embarked		Port of Embarkation	C = Cherbourg, Q = 									Queenstown, S = 									Southampton

**Variable Notes**

pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children travelled only with a nanny, therefore parch=0 for them.

Apart from training files, we also require a "schema" file from the client, which contains all the relevant information about the training files such as:

Name of the files, Length of Date value in FileName, Length of Time value in FileName, Number of Columns, Name of the Columns, and their datatype.





