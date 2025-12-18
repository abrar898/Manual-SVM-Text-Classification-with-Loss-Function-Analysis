"""
Generate comprehensive comparison visualizations for Manual vs Library SVM models
Outputs: Bar charts, Radar charts, Tables, Confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import os
import sys

# Add scripts directory to path to allow importing ManualSVM
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.models import ManualSVM

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create report directory if it doesn't exist
REPORT_DIR = "report"
os.makedirs(REPORT_DIR, exist_ok=True)

# Model performance data (from actual training results)
models_data = {
    'Model': [
        'Manual Hinge', 'Manual Squared Hinge', 'Manual Logistic',
        'Library Hinge', 'Library Squared Hinge', 'Library Logistic'
    ],
    'Type': ['Manual', 'Manual', 'Manual', 'Library', 'Library', 'Library'],
    'Accuracy': [88.68, 90.34, 87.65, 88.89, 90.19, 86.58],
    'Precision': [87.41, 89.50, 86.60, 87.62, 89.20, 85.50],
    'Recall': [90.45, 91.46, 89.17, 90.51, 91.30, 88.10],
    'F1-Score': [88.90, 90.47, 87.86, 89.04, 90.24, 86.78]
}

df = pd.DataFrame(models_data)

print("=" * 60)
print("Generating SVM Model Comparison Visualizations")
print("=" * 60)

# ============================================================================
# 1. BAR CHART COMPARISON
# ============================================================================
print("\n1. Creating bar chart comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Manual vs Library SVM Performance Comparison', fontsize=18, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_manual = ['#FF6B6B', '#4ECDC4', '#45B7D1']
colors_library = ['#FFA07A', '#98D8C8', '#6DD5ED']

for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    manual_data = df[df['Type'] == 'Manual']
    library_data = df[df['Type'] == 'Library']
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, manual_data[metric].values, width, 
                   label='Manual', color=colors_manual, alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, library_data[metric].values, width,
                   label='Library', color=colors_library, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel(f'{metric} (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Hinge', 'Squared Hinge', 'Logistic'], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([80, 95])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'bar_chart_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {REPORT_DIR}/bar_chart_comparison.png")
plt.close()

# ============================================================================
# 2. RADAR CHART COMPARISON
# ============================================================================
print("\n2. Creating radar chart comparison...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=dict(projection='polar'))
fig.suptitle('Manual vs Library SVM - Radar Chart Comparison', fontsize=18, fontweight='bold')

loss_functions = ['Hinge', 'Squared Hinge', 'Logistic']
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
num_vars = len(categories)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for idx, (ax, loss_func) in enumerate(zip(axes, loss_functions)):
    manual_row = df[(df['Type'] == 'Manual') & (df['Model'].str.contains(loss_func))]
    library_row = df[(df['Type'] == 'Library') & (df['Model'].str.contains(loss_func))]
    
    if len(manual_row) > 0 and len(library_row) > 0:
        manual_values = [manual_row[cat].values[0] for cat in categories]
        library_values = [library_row[cat].values[0] for cat in categories]
        
        manual_values += manual_values[:1]
        library_values += library_values[:1]
        
        ax.plot(angles, manual_values, 'o-', linewidth=2, label='Manual', color='#FF6B6B')
        ax.fill(angles, manual_values, alpha=0.25, color='#FF6B6B')
        
        ax.plot(angles, library_values, 's-', linewidth=2, label='Library', color='#4ECDC4')
        ax.fill(angles, library_values, alpha=0.25, color='#4ECDC4')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(80, 95)
        ax.set_title(f'{loss_func} Loss', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {REPORT_DIR}/radar_chart_comparison.png")
plt.close()

# ============================================================================
# 3. COMPARISON TABLE VISUALIZATION
# ============================================================================
print("\n3. Creating comparison table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
for _, row in df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Accuracy']:.2f}%",
        f"{row['Precision']:.2f}%",
        f"{row['Recall']:.2f}%",
        f"{row['F1-Score']:.2f}"
    ])

# Create table
table = ax.table(cellText=table_data,
                colLabels=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(5):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Style rows - alternate colors and highlight best
for i in range(1, 7):
    row_color = '#FFE5E5' if i % 2 == 0 else '#E5F5FF'
    for j in range(5):
        table[(i, j)].set_facecolor(row_color)
    
    # Highlight best model (Manual Squared Hinge - row 3)
    if i == 3:
        for j in range(5):
            table[(i, j)].set_facecolor('#90EE90')
            table[(i, j)].set_text_props(weight='bold')

plt.title('Manual vs Library SVM - Performance Comparison Table', 
          fontsize=16, fontweight='bold', pad=20)
plt.savefig(os.path.join(REPORT_DIR, 'comparison_table.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {REPORT_DIR}/comparison_table.png")
plt.close()

# ============================================================================
# 4. CONFUSION MATRICES
# ============================================================================
print("\n4. Creating confusion matrices...")

# Load test data
try:
    X_test, y_test = joblib.load('data/test_data.joblib')
    
    # Load models
    models_to_plot = [
        ('best_manual_svm.joblib', 'Manual Squared Hinge\n(Best Model - 90.34%)'),
        ('models/library_squared_hinge.joblib', 'Library Squared Hinge\n(90.19%)'),
        ('models/manual_hinge.joblib', 'Manual Hinge\n(88.68%)'),
        ('models/library_hinge.joblib', 'Library Hinge\n(88.89%)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Confusion Matrices: Manual vs Library SVM', fontsize=18, fontweight='bold')
    
    for idx, (model_path, title) in enumerate(models_to_plot):
        ax = axes.flat[idx]
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar_kws={'label': 'Count'},
                       square=True, linewidths=2, linecolor='black')
            
            # Add percentage annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                                 ha="center", va="center", color="red", fontsize=10)
            
            ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_xticklabels(['Negative', 'Positive'], fontsize=10)
            ax.set_yticklabels(['Negative', 'Positive'], fontsize=10, rotation=0)
            
            # Add accuracy text
            accuracy = (cm[0,0] + cm[1,1]) / cm.sum() * 100
            ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2f}%',
                   transform=ax.transAxes, ha='center', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'Model not found:\n{model_path}',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {REPORT_DIR}/confusion_matrices.png")
    plt.close()
    
except FileNotFoundError as e:
    print(f"⚠ Warning: Could not create confusion matrices - {e}")
    print("  Make sure test data and models are available")

# ============================================================================
# 5. OVERALL PERFORMANCE SUMMARY
# ============================================================================
print("\n5. Creating overall performance summary...")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle('Manual vs Library SVM - Overall Performance Summary', 
             fontsize=18, fontweight='bold')

# Left: Grouped bar chart for all metrics
ax1 = axes[0]
manual_avg = df[df['Type'] == 'Manual'][metrics].mean()
library_avg = df[df['Type'] == 'Library'][metrics].mean()

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, manual_avg.values, width, label='Manual (Avg)',
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax1.bar(x + width/2, library_avg.values, width, label='Library (Avg)',
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=2)

ax1.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax1.set_title('Average Performance: Manual vs Library', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([85, 92])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: Best model per loss function
ax2 = axes[1]
loss_funcs = ['Hinge', 'Squared Hinge', 'Logistic']
manual_best = [88.68, 90.34, 87.65]
library_best = [88.89, 90.19, 86.58]

x = np.arange(len(loss_funcs))
bars1 = ax2.bar(x - width/2, manual_best, width, label='Manual',
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax2.bar(x + width/2, library_best, width, label='Library',
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=2)

ax2.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax2.set_title('Best Accuracy by Loss Function', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(loss_funcs, fontsize=12)
ax2.legend(fontsize=12)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([85, 92])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'overall_summary.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {REPORT_DIR}/overall_summary.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("VISUALIZATION GENERATION COMPLETE")
print("=" * 60)
print(f"\nGenerated files in '{REPORT_DIR}/' directory:")
print("  1. bar_chart_comparison.png      - Bar charts for all metrics")
print("  2. radar_chart_comparison.png    - Radar charts by loss function")
print("  3. comparison_table.png          - Performance comparison table")
print("  4. confusion_matrices.png        - Confusion matrices (4 models)")
print("  5. overall_summary.png           - Overall performance summary")
print("\n" + "=" * 60)

# Print best model summary
print("\n🏆 BEST MODEL: Manual Squared Hinge SVM")
print(f"   Accuracy:  90.34%")
print(f"   Precision: 89.50%")
print(f"   Recall:    91.46%")
print(f"   F1-Score:  0.9047")
print("=" * 60)
