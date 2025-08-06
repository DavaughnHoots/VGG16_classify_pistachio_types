"""
Statistical Comparison Script: VGG16 vs MLP
This script performs proper statistical tests to compare CNN and MLP models
according to IEEE standards for machine learning papers.
"""

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedStratifiedKFold
import json

def mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    McNemar's test for comparing two classifiers on the same dataset.
    This is the recommended test for comparing classification models.
    """
    # Create contingency table
    # a: both models correct
    # b: model1 correct, model2 wrong  
    # c: model1 wrong, model2 correct
    # d: both models wrong
    
    a = np.sum((y_true == y_pred_model1) & (y_true == y_pred_model2))
    b = np.sum((y_true == y_pred_model1) & (y_true != y_pred_model2))
    c = np.sum((y_true != y_pred_model1) & (y_true == y_pred_model2))
    d = np.sum((y_true != y_pred_model1) & (y_true != y_pred_model2))
    
    # McNemar's test statistic
    if b + c == 0:
        print("Warning: b + c = 0, cannot perform McNemar's test")
        return None, None
    
    # Use continuity correction for small samples
    if b + c < 25:
        chi2 = (abs(b - c) - 1)**2 / (b + c)
    else:
        chi2 = (b - c)**2 / (b + c)
    
    # p-value from chi-square distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return chi2, p_value

def paired_t_test_cv(model1_scores, model2_scores):
    """
    Paired t-test for comparing cross-validation scores.
    Use this when you have paired accuracy scores from k-fold CV.
    """
    differences = np.array(model1_scores) - np.array(model2_scores)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
    
    # Effect size (Cohen's d)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    
    return t_stat, p_value, cohens_d

def five_by_two_cv_test(model1_scores, model2_scores):
    """
    5x2 Cross-Validation paired t-test (Dietterich, 1998)
    More robust than standard paired t-test for model comparison.
    
    Expects 10 scores (5 repetitions x 2 folds) for each model.
    """
    if len(model1_scores) != 10 or len(model2_scores) != 10:
        raise ValueError("5x2 CV test requires exactly 10 scores per model")
    
    # Reshape to 5x2
    scores1 = np.array(model1_scores).reshape(5, 2)
    scores2 = np.array(model2_scores).reshape(5, 2)
    
    # Calculate differences for each fold
    differences = scores1 - scores2
    
    # Calculate variance for each repetition
    variances = []
    for i in range(5):
        diff_i = differences[i]
        mean_diff_i = np.mean(diff_i)
        var_i = np.sum((diff_i - mean_diff_i)**2) / 1  # df = 1 for 2 folds
        variances.append(var_i)
    
    # First repetition difference
    first_diff = differences[0, 0] - differences[0, 1]
    
    # 5x2cv t-statistic
    mean_variance = np.mean(variances)
    if mean_variance == 0:
        print("Warning: Zero variance, cannot compute test statistic")
        return None, None
    
    t_stat = first_diff / np.sqrt(mean_variance)
    
    # Approximate with t-distribution (df=5)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=5))
    
    return t_stat, p_value

def generate_comparison_report(vgg16_results, mlp_results):
    """
    Generate comprehensive comparison report between VGG16 and MLP.
    """
    report = {
        'model_comparison': {
            'VGG16': {
                'mean_accuracy': float(np.mean(vgg16_results['accuracies'])),
                'std_accuracy': float(np.std(vgg16_results['accuracies'], ddof=1)),
                'ci_95': vgg16_results.get('ci_95', 'Not computed')
            },
            'MLP': {
                'mean_accuracy': float(np.mean(mlp_results['accuracies'])),
                'std_accuracy': float(np.std(mlp_results['accuracies'], ddof=1)),
                'ci_95': mlp_results.get('ci_95', 'Not computed')
            }
        }
    }
    
    # Perform statistical tests
    if len(vgg16_results['accuracies']) == len(mlp_results['accuracies']):
        t_stat, p_value, effect_size = paired_t_test_cv(
            vgg16_results['accuracies'], 
            mlp_results['accuracies']
        )
        
        report['statistical_tests'] = {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'significant': bool(p_value < 0.05)
            }
        }
    
    # Calculate improvement
    mean_vgg = np.mean(vgg16_results['accuracies'])
    mean_mlp = np.mean(mlp_results['accuracies'])
    improvement = ((mean_vgg - mean_mlp) / mean_mlp) * 100
    
    report['improvement'] = {
        'absolute': float(mean_vgg - mean_mlp),
        'relative_percent': float(improvement)
    }
    
    return report

def create_comparison_visualizations(vgg16_results, mlp_results, save_dir='./'):
    """
    Create publication-ready comparison plots.
    """
    # Set style for publication
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # 1. Box plot comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    data = pd.DataFrame({
        'Accuracy': vgg16_results['accuracies'] + mlp_results['accuracies'],
        'Model': ['VGG16'] * len(vgg16_results['accuracies']) + 
                 ['MLP'] * len(mlp_results['accuracies'])
    })
    
    sns.boxplot(x='Model', y='Accuracy', data=data, ax=ax)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim([0.8, 1.0])  # Adjust based on your results
    
    # Add mean markers
    means = data.groupby('Model')['Accuracy'].mean()
    for i, (model, mean) in enumerate(means.items()):
        ax.plot(i, mean, marker='D', color='red', markersize=8, 
                label=f'Mean' if i == 0 else '')
    
    if ax.get_legend():
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_comparison_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_dir}/model_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Paired accuracy plot (if same number of runs)
    if len(vgg16_results['accuracies']) == len(mlp_results['accuracies']):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        runs = range(1, len(vgg16_results['accuracies']) + 1)
        ax.plot(runs, vgg16_results['accuracies'], 'o-', label='VGG16', linewidth=2, markersize=8)
        ax.plot(runs, mlp_results['accuracies'], 's-', label='MLP', linewidth=2, markersize=8)
        
        ax.set_xlabel('Run Number', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy Across Independent Runs', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/accuracy_across_runs.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_dir}/accuracy_across_runs.png', dpi=300, bbox_inches='tight')
        plt.close()

def generate_latex_comparison_table(report):
    """
    Generate LaTeX table for model comparison.
    """
    vgg_mean = report['model_comparison']['VGG16']['mean_accuracy']
    vgg_std = report['model_comparison']['VGG16']['std_accuracy']
    mlp_mean = report['model_comparison']['MLP']['mean_accuracy']
    mlp_std = report['model_comparison']['MLP']['std_accuracy']
    
    p_value = report['statistical_tests']['paired_t_test']['p_value']
    significant = report['statistical_tests']['paired_t_test']['significant']
    
    latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Statistical Comparison of VGG16 and MLP Performance}}
\\label{{tab:model_comparison}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Model}} & \\textbf{{Accuracy (Mean ± Std)}} & \\textbf{{Parameters}} \\\\
\\hline
VGG16 (Proposed) & \\textbf{{{vgg_mean:.4f} ± {vgg_std:.4f}}} & 138.4M \\\\
MLP (Baseline) & {mlp_mean:.4f} ± {mlp_std:.4f} & ~2M \\\\
\\hline
\\multicolumn{{3}}{{l}}{{Improvement: {report['improvement']['absolute']:.4f} ({report['improvement']['relative_percent']:.2f}\\%)}} \\\\
\\multicolumn{{3}}{{l}}{{Statistical significance (paired t-test): p = {p_value:.4f}{' *' if significant else ''}}} \\\\
\\hline
\\multicolumn{{3}}{{l}}{{\\textit{{* indicates p < 0.05}}}} \\\\
\\end{{tabular}}
\\end{{table}}
"""
    
    return latex_table

# Example usage
if __name__ == "__main__":
    # Example data - replace with your actual results
    vgg16_results = {
        'accuracies': [0.9916, 0.9890, 0.9923, 0.9901, 0.9895, 
                       0.9912, 0.9908, 0.9920, 0.9889, 0.9905],
        'ci_95': [0.9895, 0.9917]
    }
    
    mlp_results = {
        'accuracies': [0.8766, 0.8734, 0.8789, 0.8751, 0.8772,
                       0.8745, 0.8798, 0.8756, 0.8741, 0.8763],
        'ci_95': [0.8745, 0.8787]
    }
    
    # Generate comparison report
    report = generate_comparison_report(vgg16_results, mlp_results)
    
    # Save report
    with open('model_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate visualizations
    create_comparison_visualizations(vgg16_results, mlp_results)
    
    # Generate LaTeX table
    latex_table = generate_latex_comparison_table(report)
    with open('comparison_table.tex', 'w') as f:
        f.write(latex_table)
    
    # Print summary
    print("Model Comparison Summary")
    print("="*50)
    print(f"VGG16: {report['model_comparison']['VGG16']['mean_accuracy']:.4f} ± "
          f"{report['model_comparison']['VGG16']['std_accuracy']:.4f}")
    print(f"MLP: {report['model_comparison']['MLP']['mean_accuracy']:.4f} ± "
          f"{report['model_comparison']['MLP']['std_accuracy']:.4f}")
    print(f"\nImprovement: {report['improvement']['absolute']:.4f} "
          f"({report['improvement']['relative_percent']:.2f}%)")
    print(f"Statistical significance: p = {report['statistical_tests']['paired_t_test']['p_value']:.4f}")
    
    if report['statistical_tests']['paired_t_test']['significant']:
        print("Result: VGG16 is SIGNIFICANTLY better than MLP (p < 0.05)")
    else:
        print("Result: No significant difference between models (p >= 0.05)")
