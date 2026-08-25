import os
import json
from flask import render_template, request, jsonify, flash, redirect, url_for
from app import app
from ml_pipeline import VANETMLPipeline
import pandas as pd

# Initialize ML pipeline
pipeline = VANETMLPipeline()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/eda')
def eda():
    # Check if EDA results exist
    eda_image = 'results/eda_analysis.png'
    if os.path.exists(eda_image):
        return render_template('eda.html', eda_image=eda_image)
    else:
        return render_template('eda.html', eda_image=None, 
                             message="No EDA analysis found. Please train models first.")

@app.route('/classification')
def classification():
    # Check if performance results exist
    performance_image = 'results/model_performance.png'
    if os.path.exists(performance_image):
        return render_template('classification.html', performance_image=performance_image)
    else:
        return render_template('classification.html', performance_image=None,
                             message="No classification results found. Please train models first.")

@app.route('/regression')
def regression():
    # Check if performance results exist
    performance_image = 'results/model_performance.png'
    if os.path.exists(performance_image):
        return render_template('regression.html', performance_image=performance_image)
    else:
        return render_template('regression.html', performance_image=None,
                             message="No regression results found. Please train models first.")

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        # Check if main dataset exists
        main_dataset = 'Dataset/dataset.csv'
        if not os.path.exists(main_dataset):
            return jsonify({'status': 'error', 'message': 'Main dataset not found. Please upload the dataset first.'})
        
        # Load and preprocess data
        X, y_class, y_reg = pipeline.load_and_preprocess_data(main_dataset, is_train=True)
        
        # Perform EDA
        eda_image = pipeline.perform_eda(X, y_class, y_reg)
        
        # Copy EDA image to static folder
        if os.path.exists(eda_image):
            import shutil
            static_eda_path = 'static/images/eda_analysis.png'
            shutil.copy2(eda_image, static_eda_path)
        
        # Train models
        results = pipeline.train_models(X, y_class, y_reg)
        
        # Copy performance image to static folder
        performance_image = 'results/model_performance.png'
        if os.path.exists(performance_image):
            static_perf_path = 'static/images/model_performance.png'
            shutil.copy2(performance_image, static_perf_path)
        
        return jsonify({'status': 'success', 'message': 'Models trained successfully!'})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error training models: {str(e)}'})

@app.route('/load_models', methods=['POST'])
def load_models():
    try:
        success = pipeline.load_models()
        if success:
            flash('Models loaded successfully!', 'success')
            return jsonify({'status': 'success', 'message': 'Models loaded successfully!'})
        else:
            flash('No trained models found. Please train models first.', 'warning')
            return jsonify({'status': 'warning', 'message': 'No trained models found. Please train models first.'})
    except Exception as e:
        flash(f'Error loading models: {str(e)}', 'error')
        return jsonify({'status': 'error', 'message': f'Error loading models: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_data = {
            'Vehicle ID': request.form['vehicle_id'],
            'Speed (km/h)': float(request.form['speed']),
            'GPS Location (Lat, Long)': request.form['gps_location'],
            'Signal Strength (dBm)': float(request.form['signal_strength']),
            'Packet Delay (ms)': float(request.form['packet_delay']),
            'Congestion Level (%)': float(request.form['congestion_level']),
            'Available Bandwidth (MHz)': float(request.form['bandwidth']),
            'Contention Window Size': float(request.form['window_size']),
            'RSU Connection (Yes/No)': request.form['rsu_connection']
        }
        
        algorithm = request.form.get('algorithm', 'Hybrid Random Forest')
        
        # Check if models are loaded
        if not pipeline.models:
            success = pipeline.load_models()
            if not success:
                return jsonify({
                    'status': 'error', 
                    'message': 'No trained models found. Please train models first.'
                })
        
        # Make predictions
        class_pred, reg_pred = pipeline.predict(input_data, algorithm)
        
        return jsonify({
            'status': 'success',
            'predictions': {
                'priority_level': int(class_pred[0]),
                'traffic_density': float(reg_pred[0]),
                'algorithm_used': algorithm
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Prediction error: {str(e)}'})

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # Check if test dataset exists
        test_dataset = 'Dataset/test_data.csv'
        if not os.path.exists(test_dataset):
            return jsonify({
                'status': 'error', 
                'message': 'Test dataset not found. Please upload the test dataset first.'
            })
        
        algorithm = request.form.get('algorithm', 'Hybrid Random Forest')
        
        # Check if models are loaded
        if not pipeline.models:
            success = pipeline.load_models()
            if not success:
                return jsonify({
                    'status': 'error', 
                    'message': 'No trained models found. Please train models first.'
                })
        
        # Load test data
        test_df = pipeline.load_and_preprocess_data(test_dataset, is_train=False)
        
        # Make predictions
        class_pred, reg_pred = pipeline.predict(test_df, algorithm)
        
        # Prepare results
        results = []
        original_df = pd.read_csv(test_dataset)
        for i in range(len(class_pred)):
            results.append({
                'vehicle_id': original_df.iloc[i]['Vehicle ID'],
                'predicted_priority': int(class_pred[i]),
                'predicted_traffic_density': float(reg_pred[i])
            })
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'algorithm_used': algorithm,
            'total_predictions': len(results)
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Batch prediction error: {str(e)}'})

@app.route('/check_models')
def check_models():
    """Check if trained models exist"""
    model_files = [
        'scaler.pkl', 'label_encoders.pkl',
        'svm_classifier.pkl', 'rf_classifier.pkl', 'rf_hybrid_classifier1.pkl', 'rf_hybrid_classifier2.pkl',
        'svm_regressor.pkl', 'rf_regressor.pkl', 'rf_hybrid_regressor1.pkl', 'rf_hybrid_regressor2.pkl'
    ]
    
    existing_models = []
    for model_file in model_files:
        if os.path.exists(f'models/{model_file}'):
            existing_models.append(model_file)
    
    return jsonify({
        'models_exist': len(existing_models) == len(model_files),
        'existing_models': existing_models,
        'total_required': len(model_files)
    })
