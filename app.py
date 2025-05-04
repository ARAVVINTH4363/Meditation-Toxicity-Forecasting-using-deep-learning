from flask import Flask, render_template, redirect, request
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE




app = Flask(__name__)

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='Toxicity'
)

mycur = mydb.cursor()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber= request.form['phonenumber']
        age  = request.form['age']
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password,`phone_number`,age) VALUES (%s, %s, %s, %s,%s)'
                val = (name, email, password, phonenumber,age)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
               msg = 'user logged successfully'
               return redirect("/viewdata")
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')



# Route to view the data
@app.route('/viewdata')
def viewdata():
    global x_train, x_test, y_train, y_test, df
    # Load the dataset
    dataset_path = 'prescriber-info.csv'  # Make sure this path is correct to the uploaded file
    df = pd.read_csv(dataset_path)
    df["Credentials"].fillna(df["Credentials"].mode()[0], inplace=True)
    df["Gender"].replace({'M': 1, 'F': 0}, inplace=True)
    df["Gender"] = df["Gender"].astype(int)
    ###################################
    le = LabelEncoder()
    df["State"] = le.fit_transform(df["State"])
    df["Credentials"] = le.fit_transform(df["Credentials"])
    df["Specialty"] = le.fit_transform(df["Specialty"])

    x = df.drop("Opioid.Prescriber", axis=1)
    y = df["Opioid.Prescriber"]
    sm = SMOTE()
    x, y = sm.fit_resample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1, stratify=y)

    x_train = x_train[['ACETAMINOPHEN.CODEINE', 'ALLOPURINOL', 'AZITHROMYCIN',
       'CYCLOBENZAPRINE.HCL', 'GABAPENTIN', 'HYDROCHLOROTHIAZIDE',
       'HYDROCODONE.ACETAMINOPHEN', 'LEVOTHYROXINE.SODIUM', 'LISINOPRIL',
       'LISINOPRIL.HYDROCHLOROTHIAZIDE', 'LYRICA', 'MELOXICAM',
       'METFORMIN.HCL', 'SIMVASTATIN', 'TRAMADOL.HCL']]
    
    x_test = x_test[['ACETAMINOPHEN.CODEINE', 'ALLOPURINOL', 'AZITHROMYCIN',
       'CYCLOBENZAPRINE.HCL', 'GABAPENTIN', 'HYDROCHLOROTHIAZIDE',
       'HYDROCODONE.ACETAMINOPHEN', 'LEVOTHYROXINE.SODIUM', 'LISINOPRIL',
       'LISINOPRIL.HYDROCHLOROTHIAZIDE', 'LYRICA', 'MELOXICAM',
       'METFORMIN.HCL', 'SIMVASTATIN', 'TRAMADOL.HCL']]
    
    dummy = df.head(1000)

    # Convert the dataframe to HTML table
    data_table = dummy.to_html(classes='table table-striped table-bordered', index=False)

    # Render the HTML page with the table
    return render_template('viewdata.html', table=data_table)

@app.route('/algo', methods=['GET', 'POST'])
def algo():

    # Placeholder variables for the results
    accuracy = None
    classification_rep = None
    selected_algo = None  

    if request.method == 'POST':
        selected_algo = request.form.get('algorithm')  

        # Random Forest
        if selected_algo == 'Random Forest':
            model = RandomForestClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        # XGBoost
        elif selected_algo == 'XGBoost':
            model = XGBClassifier()
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)


        # XGBoost
        elif selected_algo == 'Voting Classifier':
            # Initialize individual models
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            lr = LogisticRegression(max_iter=1000, random_state=42)

            # If you want to use soft voting (probabilistic)
            model = VotingClassifier(estimators=[ ('rf', rf),  ('gb', gb), ('lr', lr) ], voting='soft')  # Use 'soft' for averaging predicted probabilities
            # Train the ensemble model with soft voting
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = round(accuracy_score(y_test, y_pred), 2)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

        
    return render_template('algo.html', accuracy=accuracy, classification_report=classification_rep, selected_algo=selected_algo)


# Route for prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Get input data from form
        user_input = [ float(request.form['ACETAMINOPHEN.CODEINE']),
            float(request.form['ALLOPURINOL']),
            float(request.form['AZITHROMYCIN']),
            float(request.form['CYCLOBENZAPRINE.HCL']),
            float(request.form['GABAPENTIN']),
            float(request.form['HYDROCHLOROTHIAZIDE']),
            float(request.form['HYDROCODONE.ACETAMINOPHEN']),
            float(request.form['LEVOTHYROXINE.SODIUM']),
            float(request.form['LISINOPRIL']),
            float(request.form['LISINOPRIL.HYDROCHLOROTHIAZIDE']),
            float(request.form['LYRICA']),
            float(request.form['MELOXICAM']),
            float(request.form['METFORMIN.HCL']),
            float(request.form['SIMVASTATIN']),
            float(request.form['TRAMADOL.HCL']) ]
        
        # Initialize individual models
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)

        # If you want to use soft voting (probabilistic)
        VTC = VotingClassifier(estimators=[ ('rf', rf), ('gb', gb), ('lr', lr) ], voting='soft')  # Use 'soft' for averaging predicted probabilities

        # Train the ensemble model with soft voting
        VTC.fit(x_train, y_train)

        ## Prediction
        result = VTC.predict([user_input])
        result = int(result)
        if result == 0:
            msg = ("The prescriber did not exceed 10 opioid prescriptions in the year, suggesting minimal risk of opioid toxicity.")
        else:
            msg = ("The prescriber exceeded 10 opioid prescriptions in the year, indicating a higher risk of opioid toxicity and potential overprescribing concerns.")

        # Render the result to the prediction page
        return render_template('prediction.html', prediction=result, suggestion=msg)
    
    return render_template('prediction.html')



if __name__ == '__main__':
    app.run(debug=True)
