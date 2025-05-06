from flask import Flask, render_template, request, redirect, url_for, session, jsonify 
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Initialize Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = '2753b167d0de6d94420b18422a417389'

# Initialize SQLAlchemy and Bcrypt for database interaction and password hashing
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# Load datasets
testing_df = pd.read_csv('Testing.csv')
training_df = pd.read_csv('Training.csv')
description_df = pd.read_csv('symptom_Description.csv', header=None, names=['Condition', 'Description'])
drug_df = pd.read_csv('drugs.csv')
doctors_df = pd.read_csv('doctors.csv')

# Create dictionaries for disease descriptions and precautions
disease_descriptions = dict(zip(description_df['Condition'], description_df['Description']))
disease_drugs = {
    row['Drugs']: [drug for drug in [row['Drug1'], row['Drug2'], row['Drug3']] if pd.notna(drug)]
    for _, row in drug_df.iterrows()
}

    
# Function to preprocess data for the model
def preprocess_data(df):
    X = df.drop(columns=['prognosis'])
    y = df['prognosis']
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y, le

# Preprocess training data and initialize the classifier
X_train, y_train, label_encoder = preprocess_data(training_df)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Function to predict disease based on symptoms
def predict_disease(symptoms):
    symptom_data = {col: 0 for col in testing_df.columns[:-1]}
    for symptom in symptoms:
        if symptom in symptom_data:
            symptom_data[symptom] = 1
    symptom_df = pd.DataFrame([symptom_data])
    prediction = clf.predict(symptom_df)
    predicted_disease = label_encoder.inverse_transform(prediction)[0]
    return predicted_disease

# User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(50), nullable=False)
    medical_conditions = db.Column(db.Text, nullable=True)

# Function to create database tables
def create_tables():
    with app.app_context():
        db.create_all() # it creates the database

# Route for the home page
@app.route('/')
def home():
    return redirect(url_for('login'))

# Route for user signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        medical_conditions = request.form['medical_conditions']
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(
            username=username,
            email=email,
            password=hashed_password,
            age=int(age),
            gender=gender,
            medical_conditions=medical_conditions
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            return redirect(url_for('main'))
        else:
            return render_template('login.html', error='Invalid email or password')
    return render_template('login.html')

# Route for the main user page
@app.route('/main')
def main():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)
    health_information = {
        'username': user.username,
        'email': user.email,
        'profile': {
            'age': user.age,
            'gender': user.gender,
            'conditions': user.medical_conditions,
        }
    }
    return render_template('main.html', user=user, health_info=health_information)

#
@app.route('/update-profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)

    if user:
        # Retrieve form data
        username = request.form.get('name')
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        medical_conditions = request.form.get('conditions')

        # Ensure required fields are not None
        if not username or not email or not age or not gender:
            return render_template('main.html', user=user, health_info={
                'username': user.username,
                'email': user.email,
                'profile': {
                    'age': user.age,
                    'gender': user.gender,
                    'conditions': user.medical_conditions,
                }
            }, error="All fields are required!")

        # Update user data
        user.username = username
        user.email = email
        user.age = int(age)
        user.gender = gender
        user.medical_conditions = medical_conditions

        # Commit changes to the database
        db.session.commit()

    return redirect(url_for('main'))


# Route to check symptoms
@app.route('/check_symptoms', methods=['GET', 'POST'])
def check_symptoms():
    return redirect(url_for('predict'))

# Suggest a doctor for the predicted disease
def suggest_doctor_for_disease(disease):
    # Filter doctors_df for the specific disease
    doctors_for_disease = doctors_df[doctors_df['Condition'] == disease]
    if not doctors_for_disease.empty:
        # Since there's only one doctor per disease, directly retrieve the doctor
        doctor_info = doctors_for_disease.iloc[0]
        doctor_name = doctor_info['DocName']
        doctor_url = doctor_info['Link']
        return doctor_name, doctor_url
    else:
        # Return default response if no doctor is found for the disease
        return "No Doctors Available", "#"

# Route to predict disease based on symptoms
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        if not selected_symptoms:
            return render_template('predict.html', symptoms=list(training_df.columns[:-1])) 
        
        symptoms = [symptom.strip().replace(' ', '_') for symptom in selected_symptoms]
        predicted_disease = predict_disease(symptoms)
        disease_description = disease_descriptions.get(predicted_disease, "Description not available.")
        drugs = disease_drugs.get(predicted_disease, "Drugs not Available")

        # Suggest a doctor specific to the disease
        doctor_name, doctor_url = suggest_doctor_for_disease(predicted_disease)

        return render_template('predict.html', prediction=predicted_disease, description=disease_description, drugs=drugs, doctor_name=doctor_name, doctor_url=doctor_url, symptoms=list(training_df.columns[:-1]))
    
    return render_template('predict.html', symptoms=list(training_df.columns[:-1]))

# Route to get symptoms for autocomplete feature
@app.route('/get_symptoms', methods=['GET'])
def get_symptoms():
    query = request.args.get('query', '').lower()
    all_symptoms = list(training_df.columns[:-1])
    filtered_symptoms = [symptom for symptom in all_symptoms if query in symptom.lower()]
    return jsonify({'symptoms': filtered_symptoms})

# Route to display common diseases
@app.route('/common-diseases')
def common_diseases():
    return render_template('common-diseases.html')

# Route to display emergency contacts
@app.route('/emergency-contacts')
def emergency_contacts():   
    return render_template('emergency-contacts.html')

# Route to display first aid tips
@app.route('/first-aid-tips')
def first_aid_tips():
    return render_template('first-aid-tips.html')

# Main function to create tables and run the application
if __name__ == "__main__":
    create_tables()
    app.run(debug=True)



API_KEY = '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'
API_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    payload = {
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "messages": [
            {"role": "system", "content": "You are a health chatbot."},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 1000
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "https://final-year-project-4o2a.onrender.com",
        "X-Title": "health_bot"
    }
    try:
        res = requests.post(API_ENDPOINT, json=payload, headers=headers)
        res.raise_for_status()
        return jsonify(res.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500