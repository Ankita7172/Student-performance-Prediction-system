import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
file_path = 'Students_data_May2024 (1).csv'
df = pd.read_csv(file_path)

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Check for non-numeric columns
non_numeric_columns = df.select_dtypes(include=['object']).columns

# Encode categorical variables
label_encoders = {}
for col in non_numeric_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split the data into features and target variable
X = df.drop('Final Grade', axis=1)
y = df['Final Grade']

# Save the feature names for later use
feature_names = X.columns.tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Fit models and calculate accuracy
accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

# Streamlit app
def register():
    st.title("Student Performance Prediction System")
    st.subheader("Registration Page")
    
    username = st.text_input("Username", key='register_username')
    email = st.text_input("Email", key='register_email')
    password = st.text_input("Password", type='password', key='register_password')
    confirm_password = st.text_input("Confirm Password", type='password', key='register_confirm_password')
    
    if st.button("Register"):
        if password == confirm_password:
            st.success("Registration successful! Proceed to login.")
            st.session_state['page'] = 'login'
            st.session_state['username'] = username
            st.session_state['email'] = email
        else:
            st.error("Passwords do not match!")

def login():
    st.title("Student Performance Prediction System")
    st.subheader("Login Page")
    
    username = st.text_input("Username or Email", key='login_username')
    password = st.text_input("Password", type='password', key='login_password')
    
    if st.button("Login"):
        st.success("Login successful! Proceed to dashboard.")
        st.session_state['page'] = 'dashboard'

def dashboard():
    st.title("Student Dashboard")

    student_id = st.number_input("Student ID", min_value=0, max_value=395, key='dashboard_student_id')
    gender_options = df['Gender'].unique()  # Get unique gender labels from the dataset
    gender = st.selectbox("Gender", gender_options, key='dashboard_gender')  # Use dataset unique labels
    age = st.number_input("Age", min_value=10, max_value=100, value=18, key='dashboard_age')
    ethnicity = st.number_input("Ethnicity", min_value=0, max_value=len(df['Ethnicity'].unique()) - 1, key='dashboard_ethnicity')
    parental_education = st.number_input("Parental Education", min_value=0, max_value=len(df['Parental Education'].unique()) - 1, key='dashboard_parental_education')
    income_level = st.number_input("Income Level", min_value=0, max_value=len(df['Income Level'].unique()) - 1, key='dashboard_income_level')
    previous_grade = st.number_input("Previous Grade", min_value=0, max_value=100, value=75, key='dashboard_previous_grade')
    test_score = st.number_input("Test Score", min_value=0, max_value=100, value=75, key='dashboard_test_score')
    attendance = st.number_input("Attendance", min_value=0, max_value=100, value=75, key='dashboard_attendance')
    study_hours = st.number_input("Study Hours", min_value=0, max_value=24, value=5, key='dashboard_study_hours')
    access_to_resources = st.number_input("Access to Resources", min_value=0, max_value=1, key='dashboard_access_to_resources')
    transportation = st.number_input("Transportation", min_value=0, max_value=1, key='dashboard_transportation')
    extracurricular_activities = st.number_input("Extracurricular Activities", min_value=0, max_value=1, key='dashboard_extracurricular_activities')
    mental_health = st.number_input("Mental Health", min_value=0, max_value=2, key='dashboard_mental_health')
    family_support = st.number_input("Family Support", min_value=0, max_value=1, key='dashboard_family_support')
    peer_relationships = st.number_input("Peer Relationships", min_value=0, max_value=2, key='dashboard_peer_relationships')
    technology_access = st.number_input("Technology Access", min_value=0, max_value=1, key='dashboard_technology_access')
    
    if st.button("Predict Performance"):
        input_data = pd.DataFrame({
            'Student ID': [student_id],
            'Gender': [gender],  # Include Gender in the input data
            'Age': [age],
            'Ethnicity': [ethnicity],
            'Parental Education': [parental_education],
            'Income Level': [income_level],
            'Previous Grade': [previous_grade],
            'Test Score': [test_score],
            'Attendance': [attendance],
            'Study Hours': [study_hours],
            'Access to Resources': [access_to_resources],
            'Transportation': [transportation],
            'Extracurricular Activities': [extracurricular_activities],
            'Mental Health': [mental_health],
            'Family Support': [family_support],
            'Peer Relationships': [peer_relationships],
            'Technology Access': [technology_access]
        })
        
        for col in non_numeric_columns:
            input_data[col] = label_encoders[col].transform(input_data[col])
        
        input_data = scaler.transform(input_data)
        
        predictions = {name: model.predict(input_data)[0] for name, model in models.items()}
        
        st.session_state['predictions'] = predictions
        st.session_state['page'] = 'predicted_results'

def predicted_results():
    st.title("Predicted Results")
    
    predictions = st.session_state.get('predictions', {})
    displayed = False
    
    for model, result in predictions.items():
        if not displayed:
            cgpa = calculate_cgpa(result)
            percentage = calculate_percentage(cgpa)
            grade, remarks = get_grade_remarks(cgpa)
            
            st.write("Predicted Performance:")
            st.write(f"CGPA: {cgpa}")
            st.write(f"Percentage: {percentage}%")
            st.write(f"Grade: {grade} ({remarks})")
            if grade == "F":
                st.write("Remarks: Fail")
            else:
                st.write("Remarks: Pass")
            displayed = True
    
        st.write(f"{model} Prediction: {result}")

    # Add Student Name and Email ID columns
    st.write("Student Name:", st.session_state['username'])
    st.write("Email ID:", st.session_state['email'])
    
    if st.button("Print Report Card"):
        st.write("Printing the report card...")
        st.session_state['page'] = 'report_card'

def report_card():
    st.title("Report Card")

    # Display the report card details here
    st.write("Report card details will be displayed here.")

    # Add an option to go back to the register
    if st.button("Back to Register"):
        st.session_state['page'] = 'register'

# Define helper functions
def calculate_cgpa(grade):
    # Assuming a 10-point CGPA scale
    if grade >= 90:
        return 10
    elif grade >= 80:
        return 9
    elif grade >= 70:
        return 8
    elif grade >= 60:
        return 7
    else:
        return 6

def calculate_percentage(cgpa):
    # Assuming simple conversion formula: Percentage
    return (cgpa - 0.75) * 10

def get_grade_remarks(cgpa):
    if cgpa >= 9.0:
        return "O", "Outstanding"
    elif cgpa >= 8.0:
        return "A+", "Excellent"
    elif cgpa >= 7.0:
        return "A", "Very Good"
    elif cgpa >= 6.0:
        return "B+", "Good"
    else:
        return "F", "Fail"

# Main logic to switch between pages
if 'page' not in st.session_state:
    st.session_state['page'] = 'register'

if st.session_state['page'] == 'register':
    register()
elif st.session_state['page'] == 'login':
    login()
elif st.session_state['page'] == 'dashboard':
    dashboard()
elif st.session_state['page'] == 'predicted_results':
        predicted_results()
elif st.session_state['page'] == 'report_card':
    report_card()
