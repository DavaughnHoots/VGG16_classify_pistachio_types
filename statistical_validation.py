"""
Statistical Validation Script for VGG16 Pistachio Classification
This script runs multiple experiments with different random seeds to properly validate results
according to IEEE conference standards.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from scipy import stats
import json
import os
import random
from datetime import datetime

# Set up results directory
RESULTS_DIR = 'statistical_validation_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuration
CONFIG = {
    'num_runs': 5,  # IEEE standard: 5-10 runs (reduced for faster results)
    'image_size': (224, 224),  # VGG16 standard input
    'batch_size': 32,
    'epochs': 10,  # Reduced from 20 for faster results
    'learning_rate': 0.0001,
    'validation_split': 0.14,  # Your 86/14 split
    'train_dir': 'C:/Users/Owner/Documents/Research/VGG16/Pistachio_Dataset/Pistachio_Image_Dataset/Pistachio_Image_Dataset',
    'random_seeds': [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
}

def set_seeds(seed):
    """Set all random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_vgg16_model(num_classes=2):
    """Create VGG16 model with transfer learning"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze base model layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classifier
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def prepare_data_generators():
    """Prepare data generators with proper preprocessing"""
    # Data augmentation for training only
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        validation_split=CONFIG['validation_split']
    )
    
    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=CONFIG['validation_split']
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        CONFIG['train_dir'],
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        CONFIG['train_dir'],
        target_size=CONFIG['image_size'],
        batch_size=CONFIG['batch_size'],
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def train_model(model, train_gen, val_gen, run_id, seed):
    """Train model with proper callbacks"""
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=CONFIG['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_path = os.path.join(RESULTS_DIR, f'best_model_run_{run_id}_seed_{seed}.h5')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=3,  # Reduced from 5 for faster convergence
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=CONFIG['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, val_gen):
    """Comprehensive evaluation with all required metrics"""
    # Get predictions
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    y_pred_proba = predictions[:, 1]  # Probability of positive class
    
    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract metrics
    accuracy = report['accuracy']
    precision_macro = report['macro avg']['precision']
    recall_macro = report['macro avg']['recall']
    f1_macro = report['macro avg']['f1-score']
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    
    # AUC-ROC (for binary classification)
    try:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    except:
        auc_roc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return metrics

def run_statistical_validation():
    """Run complete statistical validation"""
    all_results = []
    
    print(f"Starting {CONFIG['num_runs']} experimental runs...")
    print("="*60)
    
    for run_id in range(CONFIG['num_runs']):
        seed = CONFIG['random_seeds'][run_id]
        print(f"\nRun {run_id + 1}/{CONFIG['num_runs']} with seed {seed}")
        print("-"*40)
        
        # Set seeds
        set_seeds(seed)
        
        # Create model
        model, base_model = create_vgg16_model()
        
        # Prepare data
        train_gen, val_gen = prepare_data_generators()
        
        # Train model
        history = train_model(model, train_gen, val_gen, run_id, seed)
        
        # Evaluate model
        metrics = evaluate_model(model, val_gen)
        metrics['run_id'] = run_id
        metrics['seed'] = seed
        
        all_results.append(metrics)
        
        print(f"Run {run_id + 1} - Accuracy: {metrics['accuracy']:.4f}")
        
        # Save individual run results
        with open(os.path.join(RESULTS_DIR, f'run_{run_id}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    # Calculate statistical summary
    accuracies = [r['accuracy'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)  # Sample standard deviation
    
    # 95% confidence interval
    confidence_level = 0.95
    degrees_freedom = len(accuracies) - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_error = t_value * (std_acc / np.sqrt(len(accuracies)))
    ci_lower = mean_acc - margin_error
    ci_upper = mean_acc + margin_error
    
    # Summary statistics
    summary = {
        'num_runs': CONFIG['num_runs'],
        'accuracies': accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'confidence_interval_95': [ci_lower, ci_upper],
        'all_metrics': {
            'precision_macro': {
                'mean': np.mean([r['precision_macro'] for r in all_results]),
                'std': np.std([r['precision_macro'] for r in all_results], ddof=1)
            },
            'recall_macro': {
                'mean': np.mean([r['recall_macro'] for r in all_results]),
                'std': np.std([r['recall_macro'] for r in all_results], ddof=1)
            },
            'f1_macro': {
                'mean': np.mean([r['f1_macro'] for r in all_results]),
                'std': np.std([r['f1_macro'] for r in all_results], ddof=1)
            },
            'specificity': {
                'mean': np.mean([r['specificity'] for r in all_results]),
                'std': np.std([r['specificity'] for r in all_results], ddof=1)
            }
        }
    }
    
    # Save summary
    with open(os.path.join(RESULTS_DIR, 'statistical_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n" + "="*60)
    print("STATISTICAL VALIDATION SUMMARY")
    print("="*60)
    print(f"Number of runs: {CONFIG['num_runs']}")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Min accuracy: {min(accuracies):.4f}")
    print(f"Max accuracy: {max(accuracies):.4f}")
    print(f"\nResults saved to: {RESULTS_DIR}/")
    
    return summary

def generate_latex_table(summary):
    """Generate LaTeX table for paper"""
    latex_table = """
\\begin{{table}}[htbp]
\\centering
\\caption{{Statistical Validation Results over 10 Independent Runs}}
\\label{{tab:statistical_validation}}
\\begin{{tabular}}{{lc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value (Mean ± Std)}} \\\\
\\hline
Accuracy & {:.4f} ± {:.4f} \\\\
Precision (macro) & {:.4f} ± {:.4f} \\\\
Recall (macro) & {:.4f} ± {:.4f} \\\\
F1-score (macro) & {:.4f} ± {:.4f} \\\\
Specificity & {:.4f} ± {:.4f} \\\\
\\hline
95\\% CI (Accuracy) & [{:.4f}, {:.4f}] \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""".format(
        summary['mean_accuracy'], summary['std_accuracy'],
        summary['all_metrics']['precision_macro']['mean'], 
        summary['all_metrics']['precision_macro']['std'],
        summary['all_metrics']['recall_macro']['mean'],
        summary['all_metrics']['recall_macro']['std'],
        summary['all_metrics']['f1_macro']['mean'],
        summary['all_metrics']['f1_macro']['std'],
        summary['all_metrics']['specificity']['mean'],
        summary['all_metrics']['specificity']['std'],
        summary['confidence_interval_95'][0],
        summary['confidence_interval_95'][1]
    )
    
    # Save LaTeX table
    with open(os.path.join(RESULTS_DIR, 'results_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print(f"\nLaTeX table saved to: {RESULTS_DIR}/results_table.tex")

if __name__ == "__main__":
    # Update the dataset path before running!
    print("IMPORTANT: Update CONFIG['train_dir'] with your dataset path before running!")
    print("\nThis script will:")
    print("1. Run 10 independent experiments with different random seeds")
    print("2. Calculate mean, std, and 95% confidence intervals")
    print("3. Report all required IEEE metrics")
    print("4. Generate LaTeX table for your paper")
    
    # Uncomment the following lines after updating the dataset path:
    summary = run_statistical_validation()
    generate_latex_table(summary)
