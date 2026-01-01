import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for professional presentation
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'bold'

# ============================================================================
# VISUALIZATION 1: Training Time Comparison (Before vs After)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Training Time\n(hours)']
before = [20]  # 20+ hours
after = [0.25]  # 15 minutes = 0.25 hours

x = np.arange(len(categories))
width = 0.35

# Create bars
bars1 = ax.barh(x - width/2, before, width, label='Before Optimization', 
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.barh(x + width/2, after, width, label='After Optimization', 
                color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bar, value in zip(bars1, before):
    ax.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{value} hrs', va='center', fontweight='bold', fontsize=14)

for bar, value in zip(bars2, after):
    ax.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
            f'{value*60:.0f} min', va='center', fontweight='bold', fontsize=14, color='#2ecc71')

# Add improvement percentage
improvement = ((before[0] - after[0]) / before[0]) * 100
ax.text(10, 0, f'🚀 {improvement:.0f}% FASTER!', 
        fontsize=20, fontweight='bold', color='#2ecc71',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax.set_ylabel('', fontsize=14, fontweight='bold')
ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
ax.set_title('Training Time: Before vs After Optimization', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_yticks(x)
ax.set_yticklabels(categories, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.set_xlim(0, 22)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: training_time_comparison.png")
plt.show()

# ============================================================================
# VISUALIZATION 2: Model Performance Metrics (F1-Score Before vs After)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['F1-Score']
before_scores = [0.35]  # Before fixing multi-label
after_scores = [0.72]   # After fixing

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, before_scores, width, label='Before Fix\n(Wrong Config)', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax.bar(x + width/2, after_scores, width, label='After Fix\n(Multi-Label)', 
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, value in zip(bars1, before_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

for bar, value in zip(bars2, after_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=14)

# Add improvement percentage
improvement = ((after_scores[0] - before_scores[0]) / before_scores[0]) * 100
ax.text(0, 0.85, f'📈 {improvement:.0f}% BETTER!', 
        fontsize=20, fontweight='bold', color='#2ecc71',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
        ha='center')

ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
ax.set_title('Model Performance: Wrong Config vs Correct Multi-Label', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Add reference line for good performance
ax.axhline(y=0.7, color='gray', linestyle='--', linewidth=2, alpha=0.5)
ax.text(0.5, 0.71, 'Good Performance Threshold', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: performance_comparison.png")
plt.show()

# ============================================================================
# VISUALIZATION 3: Multi-Metric Comparison Dashboard
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Before vs After: Complete Comparison Dashboard', 
             fontsize=18, fontweight='bold', y=0.98)

# --- Subplot 1: Training Time ---
ax1 = axes[0, 0]
categories = ['Training\nTime']
before_time = [1200]  # 20 hours in minutes
after_time = [15]     # 15 minutes

bars = ax1.bar(['Before', 'After'], [before_time[0], after_time[0]], 
               color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Time (minutes)', fontweight='bold', fontsize=12)
ax1.set_title('Training Time Reduction', fontweight='bold', fontsize=14, pad=10)
ax1.set_ylim(0, 1400)
for bar, value in zip(bars, [before_time[0], after_time[0]]):
    ax1.text(bar.get_x() + bar.get_width()/2., value + 50,
            f'{value} min' if value < 100 else f'{value//60} hrs', 
            ha='center', fontweight='bold', fontsize=12)
ax1.text(0.5, 800, '93% Faster ⚡', ha='center', fontsize=16, 
         fontweight='bold', color='#2ecc71',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# --- Subplot 2: F1-Score ---
ax2 = axes[0, 1]
bars = ax2.bar(['Before\n(Wrong)', 'After\n(Fixed)'], [0.35, 0.72], 
               color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
ax2.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
ax2.set_title('Model Performance (F1-Score)', fontweight='bold', fontsize=14, pad=10)
ax2.set_ylim(0, 1)
ax2.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, linewidth=2)
for bar, value in zip(bars, [0.35, 0.72]):
    ax2.text(bar.get_x() + bar.get_width()/2., value + 0.03,
            f'{value:.2f}', ha='center', fontweight='bold', fontsize=12)
ax2.text(0.5, 0.55, '105% Better 📈', ha='center', fontsize=16, 
         fontweight='bold', color='#2ecc71',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# --- Subplot 3: Memory Usage ---
ax3 = axes[1, 0]
bars = ax3.bar(['BERT\n(Before)', 'DistilBERT\n(After)'], [8, 3], 
               color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('Memory (GB)', fontweight='bold', fontsize=12)
ax3.set_title('Memory Requirements', fontweight='bold', fontsize=14, pad=10)
ax3.set_ylim(0, 10)
for bar, value in zip(bars, [8, 3]):
    ax3.text(bar.get_x() + bar.get_width()/2., value + 0.3,
            f'{value} GB', ha='center', fontweight='bold', fontsize=12)
ax3.text(0.5, 6, '62% Less\nMemory 💾', ha='center', fontsize=16, 
         fontweight='bold', color='#2ecc71',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# --- Subplot 4: Prediction Capability ---
ax4 = axes[1, 1]
categories = ['Single-Label\n(Before)', 'Multi-Label\n(After)']
labels_predicted = [1, 3]  # Can predict 1 vs 3 categories
bars = ax4.bar(categories, labels_predicted, 
               color=['#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=2)
ax4.set_ylabel('Max Categories', fontweight='bold', fontsize=12)
ax4.set_title('Classification Capability', fontweight='bold', fontsize=14, pad=10)
ax4.set_ylim(0, 4)
for bar, value in zip(bars, labels_predicted):
    ax4.text(bar.get_x() + bar.get_width()/2., value + 0.15,
            f'{value}', ha='center', fontweight='bold', fontsize=14)
ax4.text(0.5, 2.5, 'Now Realistic! ✓', ha='center', fontsize=16, 
         fontweight='bold', color='#2ecc71',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

plt.tight_layout()
plt.savefig('complete_comparison_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Saved: complete_comparison_dashboard.png")
plt.show()

# ============================================================================
# VISUALIZATION 4: Problem → Solution → Result Flow Chart
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.5, 'Our Problem-Solving Journey', 
        fontsize=20, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#667eea', 
                 edgecolor='black', linewidth=3, alpha=0.8))

# Problem boxes (RED)
problems = [
    "❌ 20+ Hour Training",
    "❌ Wrong Config\n(Single-Label)",
    "❌ 8GB Memory\nRequired"
]

for i, problem in enumerate(problems):
    y_pos = 7.5 - i * 2.5
    rect = Rectangle((0.2, y_pos - 0.4), 2.5, 0.8, 
                     facecolor='#e74c3c', edgecolor='black', 
                     linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(1.45, y_pos, problem, ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')

# Solution boxes (BLUE)
solutions = [
    "🔧 DistilBERT +\nSmart Sampling",
    "🔧 Added:\nproblem_type=\n'multi_label'",
    "🔧 Optimized\nArchitecture"
]

for i, solution in enumerate(solutions):
    y_pos = 7.5 - i * 2.5
    rect = Rectangle((3.5, y_pos - 0.4), 2.5, 0.8, 
                     facecolor='#3498db', edgecolor='black', 
                     linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(4.75, y_pos, solution, ha='center', va='center', 
           fontsize=10, fontweight='bold', color='white')
    
    # Arrow from problem to solution
    ax.annotate('', xy=(3.4, y_pos), xytext=(2.8, y_pos),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Result boxes (GREEN)
results = [
    "✅ 15 Minutes\n(93% Faster)",
    "✅ F1: 0.72\n(105% Better)",
    "✅ 3GB Only\n(62% Less)"
]

for i, result in enumerate(results):
    y_pos = 7.5 - i * 2.5
    rect = Rectangle((7, y_pos - 0.4), 2.5, 0.8, 
                     facecolor='#2ecc71', edgecolor='black', 
                     linewidth=2, alpha=0.8)
    ax.add_patch(rect)
    ax.text(8.25, y_pos, result, ha='center', va='center', 
           fontsize=11, fontweight='bold', color='white')
    
    # Arrow from solution to result
    ax.annotate('', xy=(6.9, y_pos), xytext=(6.1, y_pos),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Add section labels
ax.text(1.45, 8.5, 'PROBLEMS', ha='center', fontsize=14, 
       fontweight='bold', color='#e74c3c')
ax.text(4.75, 8.5, 'SOLUTIONS', ha='center', fontsize=14, 
       fontweight='bold', color='#3498db')
ax.text(8.25, 8.5, 'RESULTS', ha='center', fontsize=14, 
       fontweight='bold', color='#2ecc71')

plt.tight_layout()
plt.savefig('problem_solution_flow.png', dpi=300, bbox_inches='tight')
print("✅ Saved: problem_solution_flow.png")
plt.show()

# ============================================================================
# VISUALIZATION 5: The Critical Bug Fix (Code Visualization)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9, 'The One Line That Fixed Everything', 
        fontsize=18, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))

# BEFORE code (WRONG)
before_code = """❌ BEFORE (WRONG):
model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=6
)
# Missing critical parameter! ⚠️
# Treats as multi-CLASS (only 1 label)"""

rect1 = Rectangle((0.5, 5), 4.5, 2.5, facecolor='#ffe6e6', 
                 edgecolor='#e74c3c', linewidth=3)
ax.add_patch(rect1)
ax.text(2.75, 6.25, before_code, ha='center', va='center', 
       fontsize=10, fontfamily='monospace', color='black')

# AFTER code (CORRECT)
after_code = """✅ AFTER (FIXED):
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=6,
    problem_type="multi_label_classification"  ⭐
)
# Now predicts MULTIPLE labels! ✓"""

rect2 = Rectangle((5, 5), 4.5, 2.5, facecolor='#e6ffe6', 
                 edgecolor='#2ecc71', linewidth=3)
ax.add_patch(rect2)
ax.text(7.25, 6.25, after_code, ha='center', va='center', 
       fontsize=10, fontfamily='monospace', color='black')

# Arrow between them
ax.annotate('', xy=(4.9, 6.25), xytext=(5.1, 6.25),
            arrowprops=dict(arrowstyle='<->', lw=4, color='#f39c12'))

# Impact statement
impact_text = """
IMPACT: F1-Score jumped from 0.35 → 0.72 (105% improvement!)
Now correctly predicts papers with MULTIPLE categories
"""
ax.text(5, 3.5, impact_text, ha='center', va='center', 
       fontsize=12, fontweight='bold', color='#2ecc71',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.3))

# Bottom stats
ax.text(2.75, 2, 'Single Label\nOnly', ha='center', fontsize=14, 
       fontweight='bold', color='#e74c3c')
ax.text(7.25, 2, 'Multi-Label\nCapable', ha='center', fontsize=14, 
       fontweight='bold', color='#2ecc71')

plt.tight_layout()
plt.savefig('critical_bug_fix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: critical_bug_fix.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("📊 VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print("\n✅ Generated 5 presentation-ready visualizations:")
print("\n1️⃣  training_time_comparison.png")
print("    → Shows 93% training time reduction")
print("\n2️⃣  performance_comparison.png")
print("    → Shows 105% F1-Score improvement")
print("\n3️⃣  complete_comparison_dashboard.png")
print("    → 4-panel dashboard with all metrics")
print("\n4️⃣  problem_solution_flow.png")
print("    → Problem → Solution → Result flowchart")
print("\n5️⃣  critical_bug_fix.png")
print("    → Code comparison showing the critical fix")
print("\n" + "="*70)
print("🎤 PRESENTATION TIPS:")
print("="*70)
print("• Use #3 (dashboard) as your main slide")
print("• Use #5 (code fix) to show technical depth")
print("• Use #4 (flowchart) to tell the story")
print("• Keep #1 and #2 as backup/detail slides")
print("="*70)