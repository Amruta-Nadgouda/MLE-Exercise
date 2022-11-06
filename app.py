#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template
import pandas as pd
import pickle
import math

app = Flask(__name__)

ohe_file = open("onehotencoder.pkl", 'rb')
ohe = pickle.load(ohe_file)

poly_file = open("polynomialFT.pkl", 'rb')
poly = pickle.load(poly_file)

algo_file = open("linregmodel.pkl", 'rb')
model = pickle.load(algo_file)

def bmi_rules(age, bmi, gender):

    if (age >= 18 and age <= 39) and (bmi < 17.49 or bmi > 38.5):
        bmi_reason = "Age is between 18 to 39 and BMI is either less than 17.49 or greater than 38.5"
        quote = 750
    elif (age >= 40 and age <= 59) and (bmi < 18.49 or bmi > 38.5):
        bmi_reason = "Age is between 40 to 59 and BMI is either less than 18.49 or greater then 38.5"
        quote = 1000
    elif (age >= 60) and (bmi < 18.49 or bmi > 45.5):
        bmi_reason = "Age is greater than 60 and BMI is either less than 18.49 or greater than 45.5"
        quote = 2000
    else:
        bmi_reason = "BMI is in right range"
        quote = 500
    
    if gender.lower() == 'female':
        quote = 0.9 * quote

    return bmi_reason, int(quote)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    ht = int(request.form.get('height'))
    wt = int(request.form.get('weight'))
    gender = request.form.get('gender')

    df = pd.DataFrame(columns = ['age', 'height', 'weight', 'gender'])
    df = df.append({'age' : age, 'height' : ht, 'weight' : wt, 'gender' : gender}, ignore_index = True)
    
    df['gender'] = pd.DataFrame(ohe.transform(df[['gender']]).toarray())
    
    df = poly.transform(df)
    
    prediction = model.predict(df)
    
    reason, quote = bmi_rules(age, prediction[0], gender)
    output = round(prediction[0], 2)

    return render_template('index.html',
                           predicted_bmi=f'Predicted BMI: {output}',
                           bmi_reason=f'Reason: {reason}',
                           ins_quote=f'Insurance Quote: {quote} USD'
                          )

if __name__=="__main__":
    app.run(debug=True, use_reloader=False)


# In[ ]:





# In[ ]:




