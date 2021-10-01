######################### problem1 ########################################
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn import preprocessing
#load the dataset
fedex_data = pd.read_csv("E:/ARTIFICIAL ASSIGNMENT/Deep Learning Practical Issues/fedex.csv")

#EDA
#checking for NA values and null values
fedex_data.isna().sum()
fedex_data.dropna(axis = 0, inplace =True)

#identify duplicated records in the data
duplicate = fedex_data.duplicated()
sum(duplicate)
fedex_data = fedex_data.drop_duplicates()

#checking unique value for each columns
fedex_data.nunique()

EDA  = {"column":fedex_data.columns,
        "mean":fedex_data.mean(),
        "median":fedex_data.median(),
        "mode":fedex_data.mode(),
        "standard deviation":fedex_data.std(),
        "kurtosis":fedex_data.kurt(),
        "skewness":fedex_data.skew(),
        "variance":fedex_data.var()}
EDA

#variance for each column
fedex_data.var() 

#graphical representation
#histogram and scatter plot
sns.pairplot(fedex_data, hue='Delivery_Status')

fedex_data.columns

#Drop the unwanted columns
fedex_data.drop(['Year','Month','DayofMonth','DayOfWeek','Carrier_Name'],axis = 1, inplace = True)

fedex_data.dtypes
 
#Label encoding
label_encoder = preprocessing.LabelEncoder()
fedex_data['Source'] = label_encoder.fit_transform(fedex_data['Source'])
fedex_data['Destination'] = label_encoder.fit_transform(fedex_data['Destination'])

#normalisation using z for all the continuous data
def norm_func(i):
    x = (i-i.mean()/i.std())
    return(x)

df = norm_func(fedex_data.iloc[:,:9])

#final dataframe
final_fedex_data = pd.concat([fedex_data.iloc[:,[9]],df],axis = 1)

#train test splitting
np.random.seed(10)

final_fedex_data_train,final_fedex_data_test = train_test_split(final_fedex_data, test_size = 0.2,random_state = 457) #20% test data

x_train = final_fedex_data_train.iloc[:,1:].values.astype("float32")
y_train = final_fedex_data_train.iloc[:,0].values.astype("float32")
x_test = final_fedex_data_test.iloc[:,1:].values.astype("float32")
y_test = final_fedex_data_test.iloc[:,0].values.astype("float32")

#model building
model = MLPRegressor(hidden_layer_sizes=(10,10,),activation='identity', max_iter=20 , solver = 'lbfgs')
model.fit(x_train,y_train)

#Evaluate the model on test data using mean absolute square error
mae1 = metrics.mean_absolute_error(y_test,model.predict(x_test))
print ("error on test data", mae1) 

# Evaluating the model on train data 
mae2 = metrics.mean_absolute_error(y_train, model.predict(x_train))
print("error on train data: ",mae2)

############################################ END #################################################