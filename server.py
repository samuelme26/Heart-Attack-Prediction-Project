from flask import Flask, request, jsonify, render_template, redirect

app = Flask(__name__)

import pandas as pd
import numpy as np

from collections import Counter
import math
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate

import pickle


df = pd.read_csv('D:\\IntrainTech\\heart_attack_prediction_dataset.csv')


df = df.drop('Patient ID', axis=1)


df['BP_systolic'] = df['Blood_Pressure'].apply(lambda x: x.split("/")[0])
df['BP_diastolic'] = df['Blood_Pressure'].apply(lambda x: x.split("/")[1])

df = df[['Age', 'Sex', 'Cholesterol', 
#          'Blood Pressure',
          'BP_systolic', 'BP_diastolic',
         'Heart_Rate', 'Diabetes',
       'Family_History', 'Smoking', 'Obesity', 'Alcohol_Consumption',
       'Exercise_Hours_Per_Week', 'Diet', 'Previous_Heart_Problems',
       'Medication_Use', 'Stress_Level', 'Sedentary_Hours_Per_Day', 'Income',
       'BMI', 'Triglycerides', 'Physical_Activity_Days_Per_Week',
       'Sleep_Hours_Per_Day', 'Country', 'Continent', 'Hemisphere',
       'Heart_Attack_Risk']]


df['BP_systolic'] = pd.to_numeric(df['BP_systolic'])
df['BP_diastolic'] = pd.to_numeric(df['BP_diastolic'])


df2 = df

df2 = df2[['Age', 'Sex', 'Cholesterol','BP_systolic', 'BP_diastolic',
         'Heart_Rate', 'Diabetes','Family_History', 'Smoking', 'Obesity',
         'Alcohol_Consumption','Exercise_Hours_Per_Week', 'Diet',
         'Previous_Heart_Problems', 'Medication_Use', 'Stress_Level', 
         'Sedentary_Hours_Per_Day', 'Income','BMI', 'Triglycerides', 
         'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day', 
#         'Country', 'Continent', 'Hemisphere',
       'Heart_Attack_Risk']]

df3 = df2.select_dtypes(include=['object'])
le = LabelEncoder()
label_encoder = {}
for column in df3:
    label_encoder[column] = le
    df3[column] = label_encoder[column].fit_transform(df2[column])

df4 = pd.read_csv('D:\\IntrainTech\\heart_attack_prediction_dataset.csv')

df4 = df4[['Sex', 'Diet']]

result = pd.concat([df4, df3], axis=1)

df2 = df2.drop(['Sex', 'Diet'], axis=1)

df2 = pd.concat([df2, df3], axis=1)

df2 = df2.drop('Income', axis=1)



X = df2[['Age', 'Cholesterol', 'BP_systolic', 'BP_diastolic', 'Heart_Rate',
       'Diabetes', 'Family_History', 'Smoking', 'Obesity',
       'Alcohol_Consumption', 'Exercise_Hours_Per_Week',
       'Previous_Heart_Problems', 'Medication_Use',
        'BMI', 'Triglycerides',
       'Sleep_Hours_Per_Day',
       'Sex', 'Diet']]
y = df2[['Heart_Attack_Risk']]

smote = SMOTE(random_state = 50)
X_resample, y_resample = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resample)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,  y_resample, test_size=0.2, random_state=42)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)


model = RandomForestClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

def dicti_vals(dicti):
    x = list(dicti.values())
    x = np.array([x])
    return x

def determine_lifestyle_changes(predict_type, new_person):
    lifestyle_changes = []
    if predict_type > 0:
        if 'Smoking' in new_person and new_person['Smoking'] == 1:
            lifestyle_changes.append('quit smoking')
        if 'BMI' in new_person and new_person['BMI'] < 18.5:
            lifestyle_changes.append('gain weight')
        elif 'BMI' in new_person and new_person['BMI'] > 25:
            lifestyle_changes.append('lose weight')
        if 'Exercise_Hours_Per_Week' in new_person and new_person['Exercise_Hours_Per_Week'] < 1.25:
            lifestyle_changes.append('do more exercise')
        if 'Diet' in new_person and (new_person['Diet'] == 0 or new_person['Diet']==2):
            lifestyle_changes.append('eat healthy food')
        if 'Alcohol_Consumption' in new_person and new_person['Alcohol_Consumption'] == 1:
            lifestyle_changes.append('try reducing alcohol')
        # print("Heart attack risk:", predict_type)
        # for i in lifestyle_changes:
        #     print(f"Please {i},")
        # print("This can reduce your heart rate risk.")
        return {
            'Heart_attack_risk': predict_type[0],
            'Lifestyle_changes': lifestyle_changes
        }
        
    if predict_type > 0.6:
        print("You should consult a doctor immediately.")
        print("Heart attack risk:", predict_type)
# dict1 = pd.DataFrame([new_person])
# x = dicti_vals(new_person)



data = {'Heart_attack_risk': 1}

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/result', methods=['GET', 'POST'])
def resultPage():
    return render_template('result_template.html', data=data)


@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('Age'))
    sex = int(request.form.get('sex'))
    BP_systolic = int(request.form.get('BP_systolic'))
    BP_diastolic = int(request.form.get('BP_diastolic'))
    Cholesterol = float(request.form.get('Cholesterol'))
    Triglycerides = float(request.form.get('Triglycerides'))
    Heart_Rate = int(request.form.get('Heart_Rate'))
    Diabetes = int(request.form.get('Diabetes'))
    Family_History = int(request.form.get('Family_History'))
    Smoking = int(request.form.get('Smoking'))
    Obesity = int(request.form.get('Obesity'))
    Alcohol_Consumption = int(request.form.get('Alcohol_Consumption'))
    Medication_Use = int(request.form.get('Medication_Use'))
    Diet = int(request.form.get('Diet'))
    Sleep_Hours_Per_Day = int(request.form.get('Sleep_Hours_Per_Day'))
    Previous_Heart_Problems =  int(request.form.get('Previous_Heart_Problems'))
    BMI = float(request.form.get('BMI'))
    Exercise_Hours_Per_Week = int(request.form.get('Exercise_Hours_Per_Week'))

    new_person1 = {
        'Age': age,
        'Sex':sex,
        'BP_systolic': BP_systolic,
        'BP_diastolic': BP_diastolic,
        'Cholesterol': Cholesterol,
        'Triglycerides':Triglycerides,
        'Heart_Rate':Heart_Rate,
        'Diabetes':Diabetes,
        'Family_History':Family_History,
        'Smoking':Smoking,
        'Obesity':Obesity,
        'Alcohol_Consumption':Alcohol_Consumption,
        'Medication_Use':Medication_Use,
        'Diet':Diet,
        'Sleep_Hours_Per_Day':Sleep_Hours_Per_Day,
        'Previous_Heart_Problems':Previous_Heart_Problems,
        'BMI':BMI,
        'Exercise_Hours_Per_Week':Exercise_Hours_Per_Week
    }
    new_person = new_person1

    dict1 = pd.DataFrame([new_person])
    x = dicti_vals(new_person)

    model.predict(x)
    predict_type = model.predict_proba(x)[:, 1]
    result = determine_lifestyle_changes(predict_type, new_person)
    global data
    data = result
    print(result)


    # return "GOOD"
    # return render_template('result_template.html', result=result)
    # return redirect('/result')
    # return render_template('result_template.html')
    return jsonify(result)

if __name__ == '__main__':
    app.run()
