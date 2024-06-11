from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import logging

app = Flask(__name__)
app.secret_key = 'supersecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

db_path = 'patients.db'

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

def create_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        password TEXT NOT NULL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS patients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        symptoms TEXT NOT NULL,
                        diagnosis TEXT)''')
    conn.commit()
    conn.close()

create_tables()

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id=?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[2])
    return None

class MedicalDiagnosis:
    def __init__(self):
        self.symptoms = ['temperature', 'headache', 'cough', 'fatigue', 'sore_throat']
        self.diagnostics = ['Healthy', 'Common Cold', 'Flu', 'Allergy', 'Bronchitis', 'Pneumonia']
        self.setup_fuzzy_system()

    def setup_fuzzy_system(self):
        self.temperature = ctrl.Antecedent(np.arange(35, 42, 1), 'temperature')
        self.headache = ctrl.Antecedent(np.arange(0, 11, 1), 'headache')
        self.cough = ctrl.Antecedent(np.arange(0, 11, 1), 'cough')
        self.fatigue = ctrl.Antecedent(np.arange(0, 11, 1), 'fatigue')
        self.sore_throat = ctrl.Antecedent(np.arange(0, 11, 1), 'sore_throat')

        self.diagnosis = ctrl.Consequent(np.arange(0, 101, 1), 'diagnosis')

        self.temperature.automf(3)
        self.headache.automf(3)
        self.cough.automf(3)
        self.fatigue.automf(3)
        self.sore_throat.automf(3)

        self.diagnosis['healthy'] = fuzz.trimf(self.diagnosis.universe, [0, 0, 20])
        self.diagnosis['common_cold'] = fuzz.trimf(self.diagnosis.universe, [20, 30, 40])
        self.diagnosis['flu'] = fuzz.trimf(self.diagnosis.universe, [40, 50, 60])
        self.diagnosis['allergy'] = fuzz.trimf(self.diagnosis.universe, [60, 70, 80])
        self.diagnosis['bronchitis'] = fuzz.trimf(self.diagnosis.universe, [80, 90, 100])
        self.diagnosis['pneumonia'] = fuzz.trimf(self.diagnosis.universe, [90, 100, 100])

        rule1 = ctrl.Rule(self.temperature['poor'] & self.cough['poor'], self.diagnosis['common_cold'])
        rule2 = ctrl.Rule(self.temperature['average'] & self.headache['average'], self.diagnosis['flu'])
        rule3 = ctrl.Rule(self.temperature['good'] & self.fatigue['good'], self.diagnosis['bronchitis'])
        rule4 = ctrl.Rule(self.temperature['poor'] & self.sore_throat['poor'], self.diagnosis['allergy'])

        self.diagnosis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
        self.diagnosis_sim = ctrl.ControlSystemSimulation(self.diagnosis_ctrl)

    def diagnose(self, patient):
        symptoms = patient[2].split(',')
        if len(symptoms) != len(self.symptoms):
            raise ValueError("Incorrect number of symptoms provided")

        symptoms_dict = {self.symptoms[i]: float(symptoms[i]) for i in range(len(self.symptoms))}
        for symptom, value in symptoms_dict.items():
            self.diagnosis_sim.input[symptom] = value

        self.diagnosis_sim.compute()
        diagnosis_value = self.diagnosis_sim.output['diagnosis']
        diagnosis_label = self.get_diagnosis_label(diagnosis_value)

        return diagnosis_label

    def get_diagnosis_label(self, diagnosis_value):
        for label in self.diagnostics:
            if diagnosis_value >= fuzz.trimf(self.diagnosis.universe, [self.diagnostics.index(label) * 20, self.diagnostics.index(label) * 20 + 10, self.diagnostics.index(label) * 20 + 20])[1]:
                return label
        return "Unknown"

class IntuitionisticFuzzyDiagnosis:
    def __init__(self):
        self.symptoms = ['temperature', 'headache', 'cough', 'fatigue', 'sore_throat']
        self.diagnoses = ['Healthy', 'Common Cold', 'Flu', 'Allergy', 'Bronchitis', 'Pneumonia']

    def max_min_max_composition(self, A, R):
        B_membership = np.max(np.minimum(A[0][:, :, None], R[0]), axis=1)
        B_non_membership = np.min(np.maximum(A[1][:, :, None], R[1]), axis=1)
        return B_membership, B_non_membership

    def calculate_SR(self, R):
        membership_diff = 1 - (R[0] + R[1])
        non_membership_diff = 1 - membership_diff
        return membership_diff - non_membership_diff

    def diagnose(self, patient_symptoms):
        Q_membership = np.array([patient_symptoms])
        Q_non_membership = 1 - Q_membership

        R_membership = np.array([
            [0.4, 0.7, 0.3, 0.1, 0.1],  # Temperature
            [0.3, 0.2, 0.6, 0.2, 0.0],  # Headache
            [0.1, 0.0, 0.2, 0.8, 0.2],  # Cough
            [0.4, 0.7, 0.2, 0.2, 0.2],  # Fatigue
            [0.1, 0.1, 0.1, 0.2, 0.8]   # Sore Throat
        ])
        R_non_membership = 1 - R_membership

        T_membership, T_non_membership = self.max_min_max_composition((Q_membership, Q_non_membership), (R_membership, R_non_membership))
        SR = self.calculate_SR((T_membership, T_non_membership))

        diagnosis_result = []
        for i in range(T_membership.shape[1]):
            diagnosis_result.append({
                'diagnosis': self.diagnoses[i],
                'membership': T_membership[0, i],
                'non_membership': T_non_membership[0, i],
                'SR': SR[0, i]
            })

        return diagnosis_result

md = MedicalDiagnosis()
ifd = IntuitionisticFuzzyDiagnosis()

@app.route('/')
@login_required
def index():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients')
    patients = cursor.fetchall()
    conn.close()
    return render_template('index.html', patients=patients, symptoms=md.symptoms, diagnostics=md.diagnostics)

@app.route('/add_patient', methods=['POST'])
@login_required
def add_patient():
    name = request.form['name']
    symptoms = [request.form.get(symptom, '') for symptom in md.symptoms]
    if '' in symptoms:
        flash('Please provide all symptom values', 'danger')
        return redirect(url_for('index'))

    try:
        symptoms = list(map(float, symptoms))
    except ValueError:
        flash('Please provide valid numerical values for symptoms', 'danger')
        return redirect(url_for('index'))

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO patients (name, symptoms) VALUES (?, ?)', (name, ','.join(map(str, symptoms))))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/remove_patient/<int:patient_id>')
@login_required
def remove_patient(patient_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM patients WHERE id=?', (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for('index'))

@app.route('/diagnose_patient/<int:patient_id>')
@login_required
def diagnose_patient(patient_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients WHERE id=?', (patient_id,))
    patient = cursor.fetchone()
    conn.close()

    if patient:
        try:
            diagnosis_result = md.diagnose(patient)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('index'))

        patient_symptoms = [float(value) for value in patient[2].split(',')]
        symptoms_values = dict(zip(md.symptoms, patient_symptoms))
        try:
            ifd_diagnosis_results = ifd.diagnose(patient_symptoms)
        except ValueError as e:
            flash(str(e), 'danger')
            return redirect(url_for('index'))

        return render_template('diagnosis_result.html', patient_id=patient_id, patient=patient[1], diagnosis_result=diagnosis_result, ifd_diagnosis_results=ifd_diagnosis_results, symptoms=md.symptoms, symptoms_values=symptoms_values)
    return redirect(url_for('index'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            user_obj = User(user[0], user[1], user[2])
            login_user(user_obj)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=8)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        conn.close()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
