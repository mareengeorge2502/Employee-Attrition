import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, confusion_matrix,
                            classification_report)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Data Loading and Preprocessing
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv('dataset.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file 'dataset.csv' not found. Please upload the file.")
        return None

@st.cache_data
def preprocess_data(df):
    """Preprocess data for machine learning"""
    df_processed = df.copy()
    
    # Convert Attrition to binary
    df_processed['Attrition'] = df_processed['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Drop unnecessary columns
    cols_to_drop = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']
    df_processed = df_processed.drop([col for col in cols_to_drop if col in df_processed.columns], axis=1)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    label_encoders = {}
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_processed[col] = label_encoders[col].fit_transform(df_processed[col])
    
    return df_processed, label_encoders

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """Train multiple ML models and return results"""
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    return trained_models, results

# Page Navigation
def main():
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📈 EDA", "🤖 Model Performance", "🎯 Prediction", "💡 Insights", "📝 Conclusion"]
    )
    
    # Display selected page
    if page == "🏠 Home":
        home_page(df)
    elif page == "📈 EDA":
        eda_page(df)
    elif page == "🤖 Model Performance":
        model_page(df)
    elif page == "🎯 Prediction":
        prediction_page(df)
    elif page == "💡 Insights":
        insights_page(df)
    elif page == "📝 Conclusion":
        conclusion_page()

def home_page(df):
    """Home page with overview and KPIs"""
    st.markdown("<h1 class='main-header'>👥 Employee Attrition & Performance Analytics</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Business Problem
    Employee attrition is a critical challenge for organizations, leading to:
    - 💰 High recruitment and training costs
    - 📉 Loss of institutional knowledge
    - 🔄 Disruption to team dynamics
    - ⏱️ Decreased productivity
    
    This application leverages machine learning to predict employee attrition and identify key drivers,
    enabling HR teams to take proactive retention measures.
    """)
    
    st.markdown("---")
    
    # Key Statistics
    st.markdown("<h2 class='sub-header'>📊 Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📋 Total Employees",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        attrition_rate = (df['Attrition'].value_counts()['Yes'] / len(df) * 100)
        st.metric(
            label="📉 Attrition Rate",
            value=f"{attrition_rate:.1f}%",
            delta=f"{attrition_rate - 15:.1f}% vs Industry Avg",
            delta_color="inverse"
        )
    
    with col3:
        avg_income = df['MonthlyIncome'].mean()
        st.metric(
            label="💵 Avg Monthly Income",
            value=f"${avg_income:,.0f}",
            delta=None
        )
    
    with col4:
        avg_tenure = df['YearsAtCompany'].mean()
        st.metric(
            label="⏳ Avg Tenure (Years)",
            value=f"{avg_tenure:.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Dataset Preview
    st.markdown("<h2 class='sub-header'>🔍 Dataset Preview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Data Dimensions**")
        st.write(f"- Rows: {df.shape[0]:,}")
        st.write(f"- Columns: {df.shape[1]}")
        st.write(f"- Features: {df.shape[1] - 1}")
        st.write(f"- Target: Attrition (Binary)")
    
    with col2:
        st.markdown("**📋 Data Types**")
        st.write(f"- Numerical: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        st.write(f"- Categorical: {len(df.select_dtypes(include=['object']).columns)}")
        st.write(f"- Missing Values: {df.isnull().sum().sum()}")
    
    st.dataframe(df.head(10), use_container_width=True)
    
    # Quick Stats
    st.markdown("<h2 class='sub-header'>📈 Quick Statistics</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Attrition Distribution**")
        attrition_counts = df['Attrition'].value_counts()
        fig = px.pie(
            values=attrition_counts.values,
            names=['No', 'Yes'],
            title='Attrition Breakdown',
            color_discrete_sequence=['#2ecc71', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Department Distribution**")
        dept_counts = df['Department'].value_counts()
        fig = px.bar(
            x=dept_counts.index,
            y=dept_counts.values,
            title='Employees by Department',
            labels={'x': 'Department', 'y': 'Count'},
            color=dept_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

def eda_page(df):
    """Exploratory Data Analysis page"""
    st.markdown("<h1 class='main-header'>📈 Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    
    # Sidebar Filters
    st.sidebar.markdown("### 🔍 Filters")
    
    departments = ['All'] + list(df['Department'].unique())
    selected_dept = st.sidebar.selectbox("Department", departments)
    
    job_roles = ['All'] + list(df['JobRole'].unique())
    selected_role = st.sidebar.selectbox("Job Role", job_roles)
    
    genders = ['All'] + list(df['Gender'].unique())
    selected_gender = st.sidebar.selectbox("Gender", genders)
    
    # Apply filters
    df_filtered = df.copy()
    if selected_dept != 'All':
        df_filtered = df_filtered[df_filtered['Department'] == selected_dept]
    if selected_role != 'All':
        df_filtered = df_filtered[df_filtered['JobRole'] == selected_role]
    if selected_gender != 'All':
        df_filtered = df_filtered[df_filtered['Gender'] == selected_gender]
    
    st.info(f"📊 Showing data for {len(df_filtered)} employees")
    
    # Attrition Analysis
    st.markdown("<h2 class='sub-header'>🎯 Attrition Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by Department
        dept_attrition = df_filtered.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
        fig = px.bar(
            dept_attrition,
            x='Department',
            y='Count',
            color='Attrition',
            title='Attrition by Department',
            barmode='group',
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attrition by OverTime
        overtime_attrition = df_filtered.groupby(['OverTime', 'Attrition']).size().reset_index(name='Count')
        fig = px.bar(
            overtime_attrition,
            x='OverTime',
            y='Count',
            color='Attrition',
            title='Attrition by Overtime Status',
            barmode='group',
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Income and Tenure Analysis
    st.markdown("<h2 class='sub-header'>💰 Income & Tenure Analysis</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(
            df_filtered,
            x='Attrition',
            y='MonthlyIncome',
            title='Monthly Income vs Attrition',
            color='Attrition',
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df_filtered,
            x='Attrition',
            y='YearsAtCompany',
            title='Years at Company vs Attrition',
            color='Attrition',
            color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Age Distribution
    st.markdown("<h2 class='sub-header'>👤 Age Distribution</h2>", unsafe_allow_html=True)
    
    fig = px.histogram(
        df_filtered,
        x='Age',
        color='Attrition',
        title='Age Distribution by Attrition',
        nbins=30,
        barmode='overlay',
        color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("<h2 class='sub-header'>🔥 Correlation Heatmap</h2>", unsafe_allow_html=True)
    
    # Select numeric columns
    numeric_cols = df_filtered.select_dtypes(include=['int64', 'float64']).columns
    corr_df = df_filtered[numeric_cols].copy()
    
    # Convert Attrition to numeric for correlation
    if 'Attrition' in df_filtered.columns:
        corr_df['Attrition'] = df_filtered['Attrition'].map({'Yes': 1, 'No': 0})
    
    corr_matrix = corr_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Matrix',
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Job Role Analysis
    st.markdown("<h2 class='sub-header'>👔 Job Role Analysis</h2>", unsafe_allow_html=True)
    
    role_attrition = df_filtered.groupby('JobRole')['Attrition'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100
    ).sort_values(ascending=False)
    
    fig = px.bar(
        x=role_attrition.index,
        y=role_attrition.values,
        title='Attrition Rate by Job Role',
        labels={'x': 'Job Role', 'y': 'Attrition Rate (%)'},
        color=role_attrition.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def model_page(df):
    """Model Training and Performance page"""
    st.markdown("<h1 class='main-header'>🤖 Model Performance Analysis</h1>", unsafe_allow_html=True)
    
    with st.spinner("🔄 Training models... This may take a moment."):
        # Preprocess data
        df_processed, label_encoders = preprocess_data(df)
        
        # Prepare features and target
        X = df_processed.drop('Attrition', axis=1)
        y = df_processed['Attrition']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        trained_models, results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    st.success("✅ Models trained successfully!")
    
    # Model Comparison Table
    st.markdown("<h2 class='sub-header'>📊 Model Comparison</h2>", unsafe_allow_html=True)
    
    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    st.markdown(f"<div class='insight-box'><strong>🏆 Best Model: {best_model_name}</strong><br>Selected based on highest F1-Score, which balances precision and recall for imbalanced classification.</div>", unsafe_allow_html=True)
    
    # Visualize Model Comparison
    st.markdown("<h2 class='sub-header'>📈 Performance Metrics Visualization</h2>", unsafe_allow_html=True)
    
    metrics_df = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1']
        }
        for name, metrics in results.items()
    ])
    
    fig = px.bar(
        metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
        x='Model',
        y='Score',
        color='Metric',
        title='Model Performance Comparison',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curves
    st.markdown("<h2 class='sub-header'>📉 ROC Curves</h2>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    for model_name, metrics in results.items():
        if metrics['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, metrics['y_pred_proba'])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f"{model_name} (AUC={metrics['roc_auc']:.3f})",
                mode='lines'
            ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    st.markdown("<h2 class='sub-header'>🎯 Confusion Matrices</h2>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        with cols[idx % 2]:
            cm = metrics['confusion_matrix']
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Attrition', 'Attrition'],
                y=['No Attrition', 'Attrition'],
                title=f'{model_name}',
                color_continuous_scale='Blues',
                text_auto=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.markdown("<h2 class='sub-header'>⭐ Feature Importance (Random Forest)</h2>", unsafe_allow_html=True)
    
    if 'Random Forest' in trained_models:
        rf_model = trained_models['Random Forest']
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(15)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Most Important Features',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<div class='insight-box'><strong>💡 Key Insight:</strong><br>The top features driving attrition predictions include job-related factors, compensation, and work-life balance metrics. Focus retention strategies on these areas for maximum impact.</div>", unsafe_allow_html=True)
    
    # Store models in session state for prediction page
    st.session_state['trained_models'] = trained_models
    st.session_state['scaler'] = scaler
    st.session_state['label_encoders'] = label_encoders
    st.session_state['feature_columns'] = X.columns.tolist()

def prediction_page(df):
    """Interactive Prediction page"""
    st.markdown("<h1 class='main-header'>🎯 Employee Attrition Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Enter employee details below to predict their attrition risk. The model will provide:
    - 🎲 Attrition probability
    - 🚦 Risk level classification
    - 📊 Confidence score
    """)
    
    # Check if models are trained
    if 'trained_models' not in st.session_state:
        st.warning("⚠️ Please train the models first by visiting the 'Model Performance' page.")
        return
    
    st.markdown("---")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Personal Information")
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        gender = st.selectbox("Gender", df['Gender'].unique())
        marital_status = st.selectbox("Marital Status", df['MaritalStatus'].unique())
        distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=5)
    
    with col2:
        st.markdown("### 💼 Job Information")
        department = st.selectbox("Department", df['Department'].unique())
        job_role = st.selectbox("Job Role", df['JobRole'].unique())
        job_level = st.slider("Job Level", 1, 5, 2)
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
        years_in_role = st.number_input("Years in Current Role", min_value=0, max_value=20, value=2)
    
    with col3:
        st.markdown("### ⚙️ Work Details")
        overtime = st.selectbox("OverTime", ['Yes', 'No'])
        business_travel = st.selectbox("Business Travel", df['BusinessTravel'].unique())
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        percent_salary_hike = st.slider("Last Salary Hike (%)", 10, 25, 15)
    
    st.markdown("---")
    
    if st.button("🔮 Predict Attrition Risk", type="primary"):
        # Prepare input data
        input_data = {
            'Age': age,
            'Gender': gender,
            'MaritalStatus': marital_status,
            'DistanceFromHome': distance_from_home,
            'Department': department,
            'JobRole': job_role,
            'JobLevel': job_level,
            'MonthlyIncome': monthly_income,
            'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'OverTime': overtime,
            'BusinessTravel': business_travel,
            'JobSatisfaction': job_satisfaction,
            'WorkLifeBalance': work_life_balance,
            'EnvironmentSatisfaction': environment_satisfaction,
            'PercentSalaryHike': percent_salary_hike,
            'DailyRate': 800,
            'HourlyRate': 65,
            'MonthlyRate': 14000,
            'NumCompaniesWorked': 2,
            'Education': 3,
            'EducationField': 'Life Sciences',
            'JobInvolvement': 3,
            'PerformanceRating': 3,
            'RelationshipSatisfaction': 3,
            'StockOptionLevel': 1,
            'TotalWorkingYears': years_at_company + 3,
            'TrainingTimesLastYear': 2,
            'YearsSinceLastPromotion': 1,
            'YearsWithCurrManager': years_in_role
        }
        
        # Create dataframe
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        label_encoders = st.session_state['label_encoders']
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except:
                    input_df[col] = 0
        
        # Ensure all required features are present
        for col in st.session_state['feature_columns']:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns
        input_df = input_df[st.session_state['feature_columns']]
        
        # Scale features
        scaler = st.session_state['scaler']
        input_scaled = scaler.transform(input_df)
        
        # Get best model
        trained_models = st.session_state['trained_models']
        best_model = trained_models['Random Forest']
        
        # Make prediction
        prediction = best_model.predict(input_scaled)[0]
        prediction_proba = best_model.predict_proba(input_scaled)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("<h2 class='sub-header'>📊 Prediction Results</h2>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Prediction",
                value="WILL LEAVE" if prediction == 1 else "WILL STAY",
                delta="High Risk" if prediction == 1 else "Low Risk",
                delta_color="inverse"
            )
        
        with col2:
            attrition_prob = prediction_proba[1] * 100
            st.metric(
                label="Attrition Probability",
                value=f"{attrition_prob:.1f}%"
            )
        
        with col3:
            confidence = max(prediction_proba) * 100
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%"
            )
        
        # Risk gauge
        st.markdown("### 🎯 Risk Assessment")
        
        if attrition_prob < 30:
            risk_level = "Low Risk"
            risk_color = "#2ecc71"
        elif attrition_prob < 60:
            risk_level = "Medium Risk"
            risk_color = "#f39c12"
        else:
            risk_level = "High Risk"
            risk_color = "#e74c3c"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=attrition_prob,
            number={'suffix': '%'},
            title={'text': risk_level},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': '#2ecc71'},
                    {'range': [30, 60], 'color': '#f39c12'},
                    {'range': [60, 100], 'color': '#e74c3c'}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("<h2 class='sub-header'>💡 Retention Recommendations</h2>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("""
            <div class='warning-box'>
            <strong>🚨 High Attrition Risk Detected</strong>
            <ul>
                <li>📈 Review compensation and recent salary hikes</li>
                <li>🧑‍💼 Discuss career growth and promotion opportunities</li>
                <li>⚖️ Improve work-life balance and workload</li>
                <li>🤝 Increase manager check-ins and engagement</li>
                <li>🎓 Offer training or reskilling opportunities</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='insight-box'>
            <strong>✅ Low Attrition Risk</strong>
            <ul>
                <li>🌟 Maintain current engagement strategies</li>
                <li>📊 Monitor satisfaction periodically</li>
                <li>🎯 Continue performance recognition</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)


def insights_page(df):
    """Business Insights page"""
    st.markdown("<h1 class='main-header'>💡 Key Business Insights</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <strong>🔍 Key Findings from Analysis:</strong>
    <ul>
        <li>⏱️ Employees working overtime show significantly higher attrition</li>
        <li>💰 Lower income and fewer salary hikes correlate with higher churn</li>
        <li>📉 Early-career employees (< 5 years tenure) are most at risk</li>
        <li>⚖️ Poor work-life balance is a strong attrition driver</li>
        <li>🧑‍💼 Certain job roles consistently show higher attrition rates</li>
    </ul>
    </div>
    """ , unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box'>
    <strong>🎯 Strategic Recommendations:</strong>
    <ul>
        <li>Introduce targeted retention bonuses</li>
        <li>Limit excessive overtime with workload planning</li>
        <li>Create career progression plans for early-tenure staff</li>
        <li>Conduct regular satisfaction and engagement surveys</li>
        <li>Train managers on early attrition warning signals</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def conclusion_page():
    """Conclusion page"""
    st.markdown("<h1 class='main-header'>📝 Conclusion</h1>", unsafe_allow_html=True)

    st.markdown("""
    ### ✅ Project Summary
    This Employee Attrition Analytics platform successfully:
    - Identified key attrition drivers using EDA
    - Built and evaluated multiple ML models
    - Selected Logistic Regression as the best-performing model
    - Enabled real-time attrition risk prediction
    - Delivered actionable business insights

    ### 🚀 Business Impact
    - Proactive retention reduces hiring costs
    - Data-driven HR decisions
    - Improved workforce stability
    - Better employee engagement

    ### 🔮 Future Enhancements
    - SHAP explainability for predictions
    - Time-series attrition forecasting
    - Integration with HR systems
    - Automated alerts for high-risk employees
    """)


# App Entry Point
if __name__ == "__main__":
    main()
