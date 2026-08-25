import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR, SVC
from imodels import HSTreeClassifier, HSTreeRegressor
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.compose import TransformedTargetRegressor

class VANETMLPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = {}
        self.model_dir = 'models'
        self.results_dir = 'results'
        
        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_and_preprocess_data(self, filepath, is_train=True):
        """Load and preprocess the dataset"""
        df = pd.read_csv(filepath)
        
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        if is_train:
            # Handle categorical columns
            for col in df.select_dtypes(include='object').columns:
                if col not in ['GPS Location (Lat, Long)']:  # Skip GPS coordinates
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Handle GPS coordinates
            if 'GPS Location (Lat, Long)' in df.columns:
                df[['Latitude', 'Longitude']] = df['GPS Location (Lat, Long)'].str.extract(r'\(([^,]+),\s*([^)]+)\)').astype(float)
                df = df.drop('GPS Location (Lat, Long)', axis=1)
            
            # Fill missing values
            df = df.fillna(df.mean(numeric_only=True))
            
            # Separate features and targets
            X = df.drop(columns=['Priority Level (1-5)', 'Traffic Density (vehicles/km²)'])
            y_class = df['Priority Level (1-5)']
            y_reg = df['Traffic Density (vehicles/km²)']
            
            return X, y_class, y_reg
        else:
            # Apply same preprocessing for test data
            for col in df.select_dtypes(include='object').columns:
                if col not in ['GPS Location (Lat, Long)'] and col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
            
            # Handle GPS coordinates
            if 'GPS Location (Lat, Long)' in df.columns:
                df[['Latitude', 'Longitude']] = df['GPS Location (Lat, Long)'].str.extract(r'\(([^,]+),\s*([^)]+)\)').astype(float)
                df = df.drop('GPS Location (Lat, Long)', axis=1)
            
            # Fill missing values
            df = df.fillna(df.mean(numeric_only=True))
            
            return df
    
    def perform_eda(self, X, y_class, y_reg):
        """Perform Exploratory Data Analysis"""
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.patch.set_facecolor('white')
        
        # 1. Priority Level Distribution
        axes[0, 0].hist(y_class, bins=5, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_title('Distribution of Priority Levels')
        axes[0, 0].set_xlabel('Priority Level (1-5)')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Traffic Density Distribution
        axes[0, 1].hist(y_reg, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Distribution of Traffic Density')
        axes[0, 1].set_xlabel('Traffic Density (vehicles/km²)')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Speed vs Priority Level
        priority_groups = [X[y_class == i]['Speed (km/h)'] for i in range(1, 6)]
        axes[0, 2].boxplot(priority_groups, labels=[1, 2, 3, 4, 5])
        axes[0, 2].set_title('Speed by Priority Level')
        axes[0, 2].set_xlabel('Priority Level')
        axes[0, 2].set_ylabel('Speed (km/h)')
        
        # 4. Signal Strength vs Priority Level
        signal_groups = [X[y_class == i]['Signal Strength (dBm)'] for i in range(1, 6)]
        axes[1, 0].boxplot(signal_groups, labels=[1, 2, 3, 4, 5])
        axes[1, 0].set_title('Signal Strength by Priority Level')
        axes[1, 0].set_xlabel('Priority Level')
        axes[1, 0].set_ylabel('Signal Strength (dBm)')
        
        # 5. Packet Delay vs Traffic Density
        axes[1, 1].scatter(y_reg, X['Packet Delay (ms)'], alpha=0.6, color='coral')
        axes[1, 1].set_title('Packet Delay vs Traffic Density')
        axes[1, 1].set_xlabel('Traffic Density (vehicles/km²)')
        axes[1, 1].set_ylabel('Packet Delay (ms)')
        
        # 6. Congestion Level Distribution
        axes[1, 2].hist(X['Congestion Level (%)'], bins=20, color='orange', edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('Congestion Level Distribution')
        axes[1, 2].set_xlabel('Congestion Level (%)')
        axes[1, 2].set_ylabel('Count')
        
        # 7. Available Bandwidth vs Priority Level
        bandwidth_groups = [X[y_class == i]['Available Bandwidth (MHz)'] for i in range(1, 6)]
        axes[2, 0].boxplot(bandwidth_groups, labels=[1, 2, 3, 4, 5])
        axes[2, 0].set_title('Available Bandwidth by Priority Level')
        axes[2, 0].set_xlabel('Priority Level')
        axes[2, 0].set_ylabel('Available Bandwidth (MHz)')
        
        # 8. RSU Connection Distribution
        rsu_counts = X['RSU Connection (Yes/No)'].value_counts()
        axes[2, 1].bar(rsu_counts.index, rsu_counts.values, color=['lightblue', 'lightcoral'])
        axes[2, 1].set_title('RSU Connection Distribution')
        axes[2, 1].set_xlabel('RSU Connection')
        axes[2, 1].set_ylabel('Count')
        
        # 9. Speed vs Available Bandwidth
        axes[2, 2].scatter(X['Speed (km/h)'], X['Available Bandwidth (MHz)'], 
                          c=y_class, cmap='viridis', alpha=0.6)
        axes[2, 2].set_title('Speed vs Available Bandwidth (colored by Priority)')
        axes[2, 2].set_xlabel('Speed (km/h)')
        axes[2, 2].set_ylabel('Available Bandwidth (MHz)')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/eda_analysis.png', dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return f'{self.results_dir}/eda_analysis.png'
    
    def train_hybrid_RF(self, X_train, y_train, X_test, y_test, task='classification'):
        """Train hybrid RF + HSTree model using Stacking"""

        if task == 'classification':
            # Hybrid stacking classifier (RF + HSTree)
            stack_clf = StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier()),
                    ('hs', HSTreeClassifier())
                ],
                final_estimator=LogisticRegression(),
                passthrough=False
            )
            
            stack_clf.fit(X_train, y_train)
            pred1 = stack_clf.predict(X_test)
            return pred1, stack_clf

        else:  # regression
            # Hybrid stacking regressor (RF + HSTree)
            hs_reg = TransformedTargetRegressor(regressor=HSTreeRegressor())

            stack_reg = StackingRegressor(
                estimators=[
                    ('rf', RandomForestRegressor()),
                    ('hs', hs_reg)
                ],
                final_estimator=LinearRegression(),
                passthrough=False
            )
            
            stack_reg.fit(X_train, y_train)
            pred1 = stack_reg.predict(X_test)
            return pred1, stack_reg
            
    def train_models(self, X, y_class, y_reg):
        """Train all models and save them"""
        # Split data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, f'{self.model_dir}/scaler.pkl')
        joblib.dump(self.label_encoders, f'{self.model_dir}/label_encoders.pkl')
        
        results = {
            'classification': {},
            'regression': {}
        }
        
        # Classification Models
        print("Training Classification Models...")
        
        # SVM Classifier
        svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
        svm_clf.fit(X_train_scaled, y_class_train)
        svm_pred = svm_clf.predict(X_test_scaled)
        
        results['classification']['SVM'] = {
            'accuracy': accuracy_score(y_class_test, svm_pred) * 100,
            'precision': precision_score(y_class_test, svm_pred, average='macro') * 100,
            'recall': recall_score(y_class_test, svm_pred, average='macro') * 100,
            'f1_score': f1_score(y_class_test, svm_pred, average='macro') * 100
        }
        
        joblib.dump(svm_clf, f'{self.model_dir}/svm_classifier.pkl')
        
        # Random Forest Classifier
        rf_clf = RandomForestClassifier(
            n_estimators=10,         
            max_depth=3,             
            min_samples_split=10,    
            min_samples_leaf=5,     
            max_features=2,        
            random_state=42
        )
        rf_clf.fit(X_train, y_class_train)
        rf_pred = rf_clf.predict(X_test)
        
        results['classification']['Random Forest'] = {
            'accuracy': accuracy_score(y_class_test, rf_pred) * 100,
            'precision': precision_score(y_class_test, rf_pred, average='macro') * 100,
            'recall': recall_score(y_class_test, rf_pred, average='macro') * 100,
            'f1_score': f1_score(y_class_test, rf_pred, average='macro') * 100
        }
        
        joblib.dump(rf_clf, f'{self.model_dir}/rf_classifier.pkl')
        
        # Hybrid Random Forest Classifier
        hybrid_pred,clf = self.train_hybrid_RF(
            X_train, y_class_train, X_test, y_class_test, 'classification'
        )
        
        results['classification']['Stacked RF-HST-LLR'] = {
            'accuracy': accuracy_score(y_class_test, hybrid_pred) * 100,
            'precision': precision_score(y_class_test, hybrid_pred, average='macro') * 100,
            'recall': recall_score(y_class_test, hybrid_pred, average='macro') * 100,
            'f1_score': f1_score(y_class_test, hybrid_pred, average='macro') * 100
        }
        
        joblib.dump(clf, f'{self.model_dir}/hybrid_classifier.pkl')
        
        # Regression Models
        print("Training Regression Models...")
        
        # SVM Regressor
        svm_reg = SVR(kernel='rbf')
        svm_reg.fit(X_train_scaled, y_reg_train)
        svm_reg_pred = svm_reg.predict(X_test_scaled)
        
        results['regression']['SVM'] = {
            'mae': mean_absolute_error(y_reg_test, svm_reg_pred),
            'mse': mean_squared_error(y_reg_test, svm_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, svm_reg_pred)),
            'r2_score': r2_score(y_reg_test, svm_reg_pred)
        }
        
        joblib.dump(svm_reg, f'{self.model_dir}/svm_regressor.pkl')
        
        # Random Forest Regressor
        rf_reg = RandomForestRegressor(
            n_estimators=10,         
            max_depth=3,             
            min_samples_split=10,    
            min_samples_leaf=5,     
            max_features=2,        
            random_state=42
        )
        rf_reg.fit(X_train, y_reg_train)
        rf_reg_pred = rf_reg.predict(X_test)
        
        results['regression']['Random Forest'] = {
            'mae': mean_absolute_error(y_reg_test, rf_reg_pred),
            'mse': mean_squared_error(y_reg_test, rf_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, rf_reg_pred)),
            'r2_score': r2_score(y_reg_test, rf_reg_pred)
        }
        
        joblib.dump(rf_reg, f'{self.model_dir}/rf_regressor.pkl')
        
        # Hybrid Random Forest Regressor
        hybrid_reg_pred, reg = self.train_hybrid_RF(
            X_train, y_reg_train, X_test, y_reg_test, 'regression'
        )
        
        results['regression']['Stacked RF-HST-LLR'] = {
            'mae': mean_absolute_error(y_reg_test, hybrid_reg_pred),
            'mse': mean_squared_error(y_reg_test, hybrid_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, hybrid_reg_pred)),
            'r2_score': r2_score(y_reg_test, hybrid_reg_pred)
        }
        joblib.dump(reg, f'{self.model_dir}/hybrid_regressor.pkl')
        
        
        self.save_performance_plots(results)
        
        return results
    
    def save_performance_plots(self, results):
        """Save performance comparison plots"""
        # Classification performance plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
        algorithms = list(results['classification'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(algorithms))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results['classification'][alg][metric] for alg in algorithms]
            ax1.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Algorithms')
        ax1.set_ylabel('Score (%)')
        ax1.set_title('Classification Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(algorithms)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Regression performance plot
        algorithms = list(results['regression'].keys())
        metrics = ['mae', 'rmse', 'r2_score']
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [results['regression'][alg][metric] for alg in algorithms]
            ax2.bar(x + i * width, values, width, label=metric.replace('_', ' ').upper())
        
        ax2.set_xlabel('Algorithms')
        ax2.set_ylabel('Score')
        ax2.set_title('Regression Performance Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(algorithms)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/model_performance.png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
    
    def load_models(self):
        """Load trained models if they exist"""
        try:
            self.scaler = joblib.load(f'{self.model_dir}/scaler.pkl')
            self.label_encoders = joblib.load(f'{self.model_dir}/label_encoders.pkl')
            
            self.models = {
                'svm_classifier': joblib.load(f'{self.model_dir}/svm_classifier.pkl'),
                'rf_classifier': joblib.load(f'{self.model_dir}/rf_classifier.pkl'),
                'hybrid_classifier': joblib.load(f'{self.model_dir}/hybrid_classifier.pkl'),
                'svm_regressor': joblib.load(f'{self.model_dir}/svm_regressor.pkl'),
                'rf_regressor': joblib.load(f'{self.model_dir}/rf_regressor.pkl'),
                'hybrid_regressor': joblib.load(f'{self.model_dir}/hybrid_regressor.pkl'),
            }
            return True
        except FileNotFoundError:
            return False
    
    def predict(self, input_data, algorithm='Hybrid Random Forest'):
        """Make predictions using trained models"""
        # Preprocess input data
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
        
        # Handle categorical encoding
        for col in df.select_dtypes(include='object').columns:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Handle GPS coordinates if present
        if 'GPS Location (Lat, Long)' in df.columns:
            df[['Latitude', 'Longitude']] = df['GPS Location (Lat, Long)'].str.extract(r'\(([^,]+),\s*([^)]+)\)').astype(float)
            df = df.drop('GPS Location (Lat, Long)', axis=1)
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        if algorithm == 'SVM':
            class_pred = self.models['svm_classifier'].predict(X_scaled)
            reg_pred = self.models['svm_regressor'].predict(X_scaled)
        elif algorithm == 'Random Forest':
            class_pred = self.models['rf_classifier'].predict(df)
            reg_pred = self.models['rf_regressor'].predict(df)
        else:  # Hybrid Random Forest
            # Classification - Majority voting between two RF models
            class_pred = self.models['hybrid_classifier'].predict(df)
            reg_pred   = self.models['hybrid_regressor'].predict(df)
        
        return class_pred, reg_pred
