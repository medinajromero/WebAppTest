#Description: This program allows powerlifting athletes to predict their future performance in the sport when wearing gear or taking it off.

#Import the libraries
import pandas as pd
from sklearn.metrics import accuracy_score	
from sklearn.model_selection import train_test_split
from PIL import Image
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from gsheetsdb import connect


#Create a title and subtitle
st.write("""
# Powerlifting Raw vs Equipped Performance Prediction
Predicts powerlifting athletes performance when wearing gear or viceversa!
""")

#Open and display image
#maybe it wont work and I have to type the full path
image = Image.open('imageBack.png')
st.image(image, caption='Raw vs Non-Raw', use_column_width=True)

#Get the data
	#locally:
	#df = pd.read_csv('squat.csv')
#gsheets


"""
conn = connect()

@st.cache(ttl=600) #600 segundos aka 1 min
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows

rows = run_query(f'SELECT * FROM "{gsheet_url}"')
#rows = rows.fetchall()
#print(rows)
df = pd.DataFrame(rows)

#trasnformamos el tipo de datos de object a int
columnas = df.head()
for col in columnas:
	df[col] = df[col].astype(str).astype(int)

print(df.info())
"""

@st.cache(ttl=600) #600 segundos aka 1 min
def run_query(query):
	rows = conn.execute(query, headers=1)
	rows = rows.fetchall()
	df = pd.DataFrame(rows)

	columnas = df.head()
	for col in columnas:
		df[col] = df[col].astype(str).astype(int)

	return df


#rows = rows.fetchall()
#print(rows)


#trasnformamos el tipo de datos de object a int
gsheet_url = st.secrets["public_gsheets_url"]
conn = connect()
df = run_query(f'SELECT * FROM "{gsheet_url}"')
print(df.info())





"""
df = pd.DataFrame(rows.fetchall())
df.columns = rows.keys()
print(df.head())
"""
"""
conn = connect()
rows = conn.execute(f'SELECT * FROM "{gsheet_url}"')
df = pd.DataFrame(rows)
"""

#Set a subheader
st.subheader('Data Information:')
#Show data as a table
	#st.dataframe(df)
#Show statistics on the data
st.write(df.describe())
#Show the data as chart
	#chart = st.bar_chart(df)

X = df.drop("Best3SquatKg_y", axis=1)
#target
y = df[['Best3SquatKg_y']]

#Split the data into independent X and dependent y variables
#features
#X = df.iloc[:, 0:7].values

#target
#las column
#y = df.iloc[:, -1].values

#split dataset into training and test
#X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.25, random_state=1234)

PredictorScaler=StandardScaler()
TargetVarScaler=StandardScaler()
 
# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)
TargetVarScalerFit=TargetVarScaler.fit(y)
 
# Generating the standardized values of X and y
X=PredictorScalerFit.transform(X)
y=TargetVarScalerFit.transform(y)
 
# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#***Get the feature input from the user!
def get_user_input():
	# nombre, initial value, final value, default value
	#Sex_x = st.sidebar.slider('Sex', 0,1, 0)
	Sex_x = 0 #default
	GenderButton = st.sidebar.radio(
     "What's your gender?",
     ('Female', 'Male')
    )
	if GenderButton == 'Female':
	    Sex_x = 0
	else:
	    Sex_x = 1

	Age_x = st.sidebar.number_input('Current Age', 16, 80, 20)
	BodyweightKg_x = st.sidebar.number_input('Current BodyWeight', 30, 300, 70)
	Best3SquatKg_x = st.sidebar.number_input('Current Best Squat', 20, 700, 100)
	Age_y = st.sidebar.number_input('Future Age', 16, 80, 20)
	BodyweightKg_y = st.sidebar.number_input('Future BodyWeight', 30, 300, 70)
	DiffDays = st.sidebar.number_input('Days until next competition', 0,3650, 0)

	#Store a dictionary into a variable
	user_data = {
		'Sex_x':Sex_x,
		'Age_x':Age_x,
		'BodyweightKg_x':BodyweightKg_x,
		'Best3SquatKg_x':Best3SquatKg_x,
		'Age_y':Age_y,
		'BodyweightKg_y':BodyweightKg_y,
		'DiffDays':DiffDays
	}

	#Transform the data into a df
	features = pd.DataFrame(user_data,index =[0])

	return features

#Store the users input into a variable
user_input = get_user_input()

#Set a subheader and display the users input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
#... aqui cargamos el modelo que ya tenemos entrenado!
model = load_model('annModelV2.0')

"""
#Processing the data
Predictions=model.predict(X_test)
 
# Scaling the predicted Price data back to original price scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
 
# Scaling the y_test Price data back to original price scale
y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
 
# Scaling the test data back to original scale
Test_Data=PredictorScalerFit.inverse_transform(X_test)

TargetVariable=['Best3SquatKg_y']
Predictors=['Sex_x', 'Age_x', 'BodyweightKg_x', 'Best3SquatKg_x', 'Age_y', 'BodyweightKg_y', 'DiffDays']
 
TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
TestingData['Best3SquatKg_y']=y_test_orig
TestingData['PredictedBestSquatKg']=Predictions


#Show the model metrics
st.subheader('Model Test Accuracy Score:')
APE=100*(abs(TestingData['Best3SquatKg_y']-TestingData['PredictedBestSquatKg'])/TestingData['Best3SquatKg_y'])
st.write(str(APE)+'%')
"""

#Store the models predictions in a variable
X=PredictorScalerFit.transform(user_input)
Predictions=model.predict(X)
 
# Scaling the predicted output data back to original scale
Predictions=TargetVarScalerFit.inverse_transform(Predictions)
valor = Predictions[0]
print('\n\n\n',valor[0])

#Set a subheader and display the classification
st.subheader('Classification: ')
st.write(valor)

print("end")