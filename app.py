import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash  
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
from dotenv import load_dotenv
from datetime import datetime
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import io


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///diabetes_risk.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pregnancies = db.Column(db.Float)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree = db.Column(db.Float)
    age = db.Column(db.Float)
    risk_score = db.Column(db.Float, nullable=False)
    recommendation = db.Column(db.String(300))
    risk_factors = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='predictions')

# Helper functions
def validate_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    return True

# Load and preprocess dataset
try:
    # Load the Excel file 
    diabetes_dataset = pd.read_excel("diabetes.xls", sheet_name='diabetes') 
    
    # Validate dataset structure
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
       
    # Feature engineering
    diabetes_dataset['FutureRisk'] = diabetes_dataset.apply(lambda row: 
        1 if (row['BMI'] > 30) or 
           (row['Glucose'] > 126) or
           (row['DiabetesPedigreeFunction'] > 0.8) or 
           (row['Age'] > 45 and row['BMI'] > 25)
        else 0, axis=1)

    # Prepare features and target
    X = diabetes_dataset[['Pregnancies', 'Glucose', 'BloodPressure', 
                         'SkinThickness', 'Insulin', 'BMI', 
                         'DiabetesPedigreeFunction', 'Age']]
    y = diabetes_dataset['FutureRisk']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Feature scaling
    global scaler, model  # Make them global
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    print("Model training completed successfully")
    print(f"Train score: {model.score(X_train_scaled, y_train):.2f}")
    print(f"Test score: {model.score(X_test_scaled, y_test):.2f}")

except Exception as e:
    print(f"Error initializing dataset: {str(e)}")
    raise
   
# user login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def home():
    return render_template('index.html', user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Username and password are required!', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        
        if not validate_password(password):
            flash('Password must be at least 8 characters with uppercase, lowercase, and numbers', 'danger')
            return redirect(url_for('register'))
        
        try:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password_hash=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Registration failed. Please try again.', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Ensure this redirect is present
        else:
            flash('Invalid username or password!', 'danger')
    
    return render_template('login.html')
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.json
        print("Received prediction request:", data)  # Debugging
        
        # Create a DataFrame with proper feature names to match training data
        input_df = pd.DataFrame([[
            float(data.get('pregnancies', 0)),
            float(data.get('glucose', 0)),
            float(data.get('blood_pressure', 0)),
            float(data.get('skin_thickness', 0)),
            float(data.get('insulin', 0)),
            float(data.get('bmi', 0)),
            float(data.get('diabetes_pedigree', 0)),
            float(data.get('age', 0))
        ]], columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 
            'SkinThickness', 'Insulin', 'BMI', 
            'DiabetesPedigreeFunction', 'Age'
        ])
        
        print("Input DataFrame:\n", input_df)  # Debugging

        # Validate input ranges
        ranges = {
            'Pregnancies': (0, 17),
            'Glucose': (0, 200),
            'BloodPressure': (0, 130),
            'SkinThickness': (0, 100),
            'Insulin': (0, 850),
            'BMI': (0, 70),
            'DiabetesPedigreeFunction': (0, 2.5),
            'Age': (0, 120)
        }

        for col, (min_val, max_val) in ranges.items():
            val = input_df[col].iloc[0]
            if not (min_val <= val <= max_val):
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid {col} value: {val}. Must be between {min_val} and {max_val}'
                }), 400

        # Scale and predict
        scaled_input = scaler.transform(input_df)
        print("Scaled input:\n", scaled_input)  # Debugging
        
        risk_probability = model.predict_proba(scaled_input)[0][1] * 100
        print("Risk probability:", risk_probability)  # Debugging

     
        risk_factors = []
        thresholds = {
            'Glucose': 126,
            'BMI': 30,
            'Age': 45,
            'DiabetesPedigreeFunction': 0.8
        }

        if input_df['Glucose'].iloc[0] > thresholds['Glucose']:
            risk_factors.append(f"High Glucose ({input_df['Glucose'].iloc[0]:.1f} > {thresholds['Glucose']})")
        if input_df['BMI'].iloc[0] > thresholds['BMI']:
            risk_factors.append(f"High BMI ({input_df['BMI'].iloc[0]:.1f} > {thresholds['BMI']})")
        if input_df['Age'].iloc[0] > thresholds['Age']:
            risk_factors.append(f"Age ({input_df['Age'].iloc[0]:.1f} > {thresholds['Age']})")
        if input_df['DiabetesPedigreeFunction'].iloc[0] > thresholds['DiabetesPedigreeFunction']:
            risk_factors.append(f"Family History ({input_df['DiabetesPedigreeFunction'].iloc[0]:.2f} > {thresholds['DiabetesPedigreeFunction']})")

        recommendation, risk_category = generate_recommendation(risk_probability, risk_factors)

        # Save prediction
        new_prediction = Prediction(
            user_id=current_user.id,
            pregnancies=input_df['Pregnancies'].iloc[0],
            glucose=input_df['Glucose'].iloc[0],
            blood_pressure=input_df['BloodPressure'].iloc[0],
            skin_thickness=input_df['SkinThickness'].iloc[0],
            insulin=input_df['Insulin'].iloc[0],
            bmi=input_df['BMI'].iloc[0],
            diabetes_pedigree=input_df['DiabetesPedigreeFunction'].iloc[0],
            age=input_df['Age'].iloc[0],
            risk_score=risk_probability,
            recommendation=recommendation,
            risk_factors=", ".join(risk_factors) if risk_factors else "No significant factors"
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'risk_score': round(risk_probability, 2),
            'risk_category': risk_category,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'prediction_id': new_prediction.id
        })

    except Exception as e:
        print(f"Prediction error: {str(e)}", exc_info=True)
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def generate_recommendation(risk_score, risk_factors):
    if risk_score > 75:
        category = "Very High"
        recommendation = (
            "Immediate action recommended. High likelihood of developing diabetes. "
            "Consult a healthcare provider immediately. Consider comprehensive "
            "lifestyle changes including diet modification and regular exercise."
        )
    elif risk_score > 50:
        category = "High"
        recommendation = (
            "High risk of developing diabetes. Schedule a doctor's appointment soon. "
            "Focus on weight management and reducing sugar intake. "
            "Begin regular blood glucose monitoring."
        )
    elif risk_score > 25:
        category = "Moderate"
        recommendation = (
            "Moderate risk. Maintain healthy habits. Increase physical activity "
            "to at least 150 minutes per week. Consider annual glucose screening."
        )
    else:
        category = "Low"
        recommendation = (
            "Low current risk. Maintain healthy lifestyle. Continue balanced diet "
            "and regular exercise. Periodic check-ups recommended."
        )

    # Add factor-specific advice
    factor_advice = []
    if any('High Glucose' in factor for factor in risk_factors):
        factor_advice.append("Reduce sugar and refined carbs in your diet.")
    if any('High BMI' in factor for factor in risk_factors):
        factor_advice.append("Aim for gradual weight loss through diet and exercise.")
    if any('Age' in factor for factor in risk_factors):
        factor_advice.append("As you age, regular health screenings become more important.")
    if any('Family History' in factor for factor in risk_factors):
        factor_advice.append("Family history increases risk - be vigilant with monitoring.")
    
    if factor_advice:
        recommendation += " Additional advice: " + " ".join(factor_advice)

    return recommendation, category

# Initialize database
def initialize_database():
    with app.app_context():
        db.create_all()
@app.route('/history/<int:id>')
@login_required
def view_prediction(id):
    prediction = Prediction.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    # Ensure risk_category is set
    if not hasattr(prediction, 'risk_category'):
        _, prediction.risk_category = generate_recommendation(prediction.risk_score, 
                                                           prediction.risk_factors.split(', ') if prediction.risk_factors else [])
    return render_template('history_detail.html', prediction=prediction)

@app.route('/history/<int:id>/download')
@login_required
def download_prediction(id):
    # Get the prediction
    prediction = Prediction.query.filter_by(id=id, user_id=current_user.id).first_or_404()
    
    # Ensure risk_category is set
    if not hasattr(prediction, 'risk_category'):
        _, prediction.risk_category = generate_recommendation(prediction.risk_score, 
                                                           prediction.risk_factors.split(', ') if prediction.risk_factors else [])
    
    # Create a PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # PDF content
    elements = []
    
    # Title
    elements.append(Paragraph("Diabetes Risk Assessment Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Assessment info
    elements.append(Paragraph(f"Assessment Date: {prediction.created_at.strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    elements.append(Paragraph(f"Risk Score: {prediction.risk_score:.2f}% ({prediction.risk_category})", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Risk factors
    elements.append(Paragraph("Primary Risk Factors:", styles['Heading3']))
    factors = prediction.risk_factors.split(', ') if prediction.risk_factors else ["No significant factors"]
    for factor in factors:
        elements.append(Paragraph(f"• {factor}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Health parameters
    elements.append(Paragraph("Health Parameters:", styles['Heading3']))
    data = [
        ["Parameter", "Value"],
        ["Pregnancies", str(prediction.pregnancies)],
        ["Glucose (mg/dL)", str(prediction.glucose)],
        ["Blood Pressure (mmHg)", str(prediction.blood_pressure)],
        ["Skin Thickness (mm)", str(prediction.skin_thickness)],
        ["Insulin (μU/mL)", str(prediction.insulin)],
        ["BMI", str(prediction.bmi)],
        ["Diabetes Pedigree", str(prediction.diabetes_pedigree)],
        ["Age", str(prediction.age)],
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))
    
    # Recommendations
    elements.append(Paragraph("Recommendations:", styles['Heading3']))
    elements.append(Paragraph(prediction.recommendation, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"diabetes_assessment_{prediction.created_at.strftime('%Y%m%d')}.pdf",
        mimetype='application/pdf'
    )

# Update the existing history route to ensure risk_category is set
@app.route('/history')
@login_required
def prediction_history():
    predictions = Prediction.query.filter_by(user_id=current_user.id)\
                       .order_by(Prediction.created_at.desc())\
                       .limit(10).all()
    
    # Add risk_category to each prediction if not already set
    for pred in predictions:
        if not hasattr(pred, 'risk_category'):
            _, pred.risk_category = generate_recommendation(pred.risk_score, pred.risk_factors.split(', ') if pred.risk_factors else [])
    
    return render_template('history.html', 
                         predictions=predictions, 
                         user=current_user)

@app.route('/create-admin')
def create_admin():
    if not User.query.filter_by(username='admin').first():
        hashed_password = generate_password_hash('admin123', method='pbkdf2:sha256')
        admin = User(username='admin', password_hash=hashed_password, is_admin=True)
        db.session.add(admin)
        db.session.commit()
        return 'Admin created!'
    return 'Admin already exists!'
@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('You do not have permission to access this page', 'danger')
        return redirect(url_for('home'))
    
    # Get statistics
    total_users = User.query.count()
    total_assessments = Prediction.query.count()
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    
    return render_template('admin/dashboard.html', 
                         total_users=total_users,
                         total_assessments=total_assessments,
                         recent_users=recent_users)
# Admin Users Management
@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=users)

@app.route('/admin/users/<int:user_id>')
@login_required
def admin_user_detail(user_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    user = User.query.get_or_404(user_id)
    # Fix: Use the relationship with .query to get order_by capability
    assessments = Prediction.query.filter_by(user_id=user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('admin/user_detail.html', user=user, assessments=assessments)

# Admin Assessments Management
@app.route('/admin/assessments')
@login_required
def admin_assessments():
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    assessments = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template('admin/assessments.html', assessments=assessments)
# Add these routes to your app.py

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    user = User.query.get_or_404(user_id)
    if user.is_admin:
        flash('Cannot delete other admins', 'danger')
        return redirect(url_for('admin_users'))
    
    # Delete all user's predictions first
    Prediction.query.filter_by(user_id=user_id).delete()
    db.session.delete(user)
    db.session.commit()
    
    flash('User and all their assessments deleted successfully', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/assessments/delete/<int:assessment_id>', methods=['POST'])
@login_required
def admin_delete_assessment(assessment_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    assessment = Prediction.query.get_or_404(assessment_id)
    db.session.delete(assessment)
    db.session.commit()
    
    flash('Assessment deleted successfully', 'success')
    return redirect(url_for('admin_assessments'))
# Add this route for admin assessment view
@app.route('/admin/assessments/<int:assessment_id>')
@login_required
def admin_view_assessment(assessment_id):
    if not current_user.is_admin:
        flash('Access denied', 'danger')
        return redirect(url_for('home'))
    
    assessment = Prediction.query.get_or_404(assessment_id)
    return render_template('admin/assessment_detail.html', 
                         assessment=assessment,
                         user=current_user)
if __name__ == '__main__':
    initialize_database()
    app.run(debug=os.getenv('FLASK_DEBUG', 'False') == 'True')