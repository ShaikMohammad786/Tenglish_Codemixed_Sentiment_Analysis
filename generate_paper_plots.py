import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set global publication styling with high legibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['mathtext.fontset'] = 'cm'

out_dir = r"M:\MOHAMMAD\OmarchyWinProjects\Urop\architecture\paper\figures"
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# 1. CETTE System Architecture Diagram (High-Legibility Publication Vector Art)
# =============================================================================
fig, ax = plt.subplots(figsize=(14.0, 5.8), dpi=300)
ax.set_xlim(0, 14.0)
ax.set_ylim(0, 5.8)
ax.axis('off')

# Soft modern slate canvas
ax.add_patch(patches.Rectangle((0, 0), 14.0, 5.8, facecolor='#f8fafc', edgecolor='#cbd5e1', linewidth=1.2, zorder=0))

c_input   = '#0f172a'  # Dark Slate
c_prep    = '#0369a1'  # Professional Blue
c_electra = '#4338ca'  # Deep Indigo
c_deberta = '#0f766e'  # Teal Green
c_bert    = '#b45309'  # Warm Amber
c_loss    = '#b91c1c'  # Crimson Red
c_fusion  = '#6d28d9'  # Deep Purple
c_out     = '#047857'  # Emerald Green
c_card_bg = '#ffffff'

def draw_card(ax, x, y, w, h, title, subtitle_lines, color, radius=0.15):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                 facecolor=c_card_bg, edgecolor='#94a3b8', linewidth=1.5, zorder=2)
    ax.add_patch(box)
    
    hdr_h = 0.50
    header = patches.FancyBboxPatch((x, y + h - hdr_h), w, hdr_h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=color, edgecolor=color, zorder=3)
    ax.add_patch(header)
    ax.add_patch(patches.Rectangle((x, y + h - hdr_h), w, 0.20, facecolor=color, edgecolor=color, zorder=3))
    
    ax.text(x + w/2, y + h - 0.25, title, ha='center', va='center', color='white', 
            fontsize=11.5, fontweight='bold', zorder=4)
    
    n_lines = len(subtitle_lines)
    avail_h = h - hdr_h - 0.20
    start_y = y + avail_h + 0.05
    line_step = avail_h / max(n_lines, 1)
    
    for idx, (txt, bold, clr) in enumerate(subtitle_lines):
        cur_y = start_y - (idx + 0.6) * line_step
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, cur_y, txt, ha='center', va='center', color=clr,
                fontsize=9.8, fontweight=weight, zorder=4)

# 1. Raw Input Card
draw_card(ax, 0.25, 1.6, 2.2, 2.8, "Raw Input", [
    ("Code-Mixed Text", True, '#0f172a'),
    ("Romanized Telugu-English", False, '#475569'),
    ("(Social Media Comment)", False, '#475569'),
    ("", False, '#0f172a'),
    (r'"movie chala bagundi', True, '#1e293b'),
    (r'kani ending bore"', True, '#1e293b')
], c_input)

ax.annotate('', xy=(2.90, 3.0), xytext=(2.45, 3.0),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=18))

# 2. Preprocessing & Multi-Tokenizer
draw_card(ax, 2.90, 1.0, 2.3, 4.0, "Preprocessing", [
    ("Signal Preservation", True, '#0369a1'),
    ("• Unicode NFC Normalization", False, '#334155'),
    ("• [USER] & [URL] Entity Masking", False, '#334155'),
    ("• Punctuation & Emojis Intact", False, '#334155'),
    ("• Vowel Elongation Handling", False, '#334155'),
    ("-------------------------", False, '#94a3b8'),
    ("Multi-Tokenizer Dispatch", True, '#0369a1'),
    ("• ELECTRA WordPiece (105k)", False, '#1e293b'),
    ("• DeBERTa SPM (128k)", False, '#1e293b'),
    ("• BERT WordPiece (30k)", False, '#1e293b')
], c_prep)

ax.annotate('', xy=(5.65, 4.3), xytext=(5.20, 3.3),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=16))
ax.annotate('', xy=(5.65, 3.0), xytext=(5.20, 3.0),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=16))
ax.annotate('', xy=(5.65, 1.7), xytext=(5.20, 2.7),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=16))

# 3. Tri-Transformer Backbones
draw_card(ax, 5.65, 3.75, 2.7, 1.45, "Branch A: ELECTRA", [
    ("Replaced Token Detection (RTD)", True, '#4338ca'),
    ("Discriminative Word-Level Head", False, '#334155'),
    ("93.30% Positive Precision", True, '#047857')
], c_electra)

draw_card(ax, 5.65, 2.25, 2.7, 1.40, "Branch B: DeBERTa-v3", [
    ("Disentangled Relative Attention", True, '#0f766e'),
    ("Contextual Word-Order Model", False, '#334155'),
    ("94.81% Negative Recall (FP32)", True, '#047857')
], c_deberta)

draw_card(ax, 5.65, 0.75, 2.7, 1.40, "Branch C: BERT-base", [
    ("Bidirectional Masked LM", True, '#b45309'),
    ("General Semantic Anchor", False, '#334155'),
    ("Symmetric Variance Stabilizer", True, '#1e293b')
], c_bert)

ax.annotate('', xy=(8.85, 3.3), xytext=(8.35, 4.4),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=16))
ax.annotate('', xy=(8.85, 3.0), xytext=(8.35, 3.0),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=16))
ax.annotate('', xy=(8.85, 2.7), xytext=(8.35, 1.5),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=16))

# 4. Focal-Smoothed Loss Top Banner
loss_box = patches.FancyBboxPatch((5.65, 5.25), 5.75, 0.48, boxstyle="round,pad=0,rounding_size=0.10",
                                  facecolor='#fee2e2', edgecolor='#ef4444', linewidth=1.5, zorder=2)
ax.add_patch(loss_box)
ax.text(8.52, 5.49, r"Loss: $\mathcal{L}_{\mathrm{FSO}} = -\sum_{c} \alpha_c (1 - p_c)^\gamma \tilde{y}_c \log(p_c)$  $(\gamma=1.0, \epsilon=0.05)$",
        ha='center', va='center', color='#991b1b', fontsize=11.0, fontweight='bold', zorder=3)

# 5. Temperature-Calibrated Late Fusion
draw_card(ax, 8.85, 1.4, 2.55, 3.3, "Late Fusion Engine", [
    ("Temperature Calibration", True, '#6d28d9'),
    (r"$p_m = \mathrm{softmax}(z_m / T_m)$", True, '#1e293b'),
    ("Optimized per branch on Val", False, '#475569'),
    ("-------------------------", False, '#94a3b8'),
    ("Bayesian Dirichlet Blend", True, '#6d28d9'),
    (r"$P = 0.42 p_A + 0.38 p_B + 0.20 p_C$", True, '#0f172a'),
    ("-------------------------", False, '#94a3b8'),
    ("Error Decorrelation", True, '#047857'),
    (r"Cross-Branch $r_{\mathrm{err}} < 0.42$", False, '#1e293b')
], c_fusion)

ax.annotate('', xy=(11.85, 3.0), xytext=(11.40, 3.0),
            arrowprops=dict(facecolor='#475569', edgecolor='none', arrowstyle='-|>', lw=2.0, mutation_scale=18))

# 6. Dual-Regime Output Card
draw_card(ax, 11.85, 1.1, 2.05, 3.8, "Dual-Regime Result", [
    ("Regime 1: Polar", True, '#047857'),
    (r"$\mathcal{P}_{\mathrm{polar}} \in \{\mathrm{Neg}, \mathrm{Pos}\}$", False, '#334155'),
    ("Single ELECTRA: 88.67%", False, '#1e293b'),
    ("CETTE Ensemble:", True, '#0f172a'),
    ("90.15% Accuracy", True, '#047857'),
    ("(Published Baseline: 86.2%)", False, '#475569'),
    ("-----------------------", False, '#94a3b8'),
    ("Regime 2: 3-Class", True, '#047857'),
    (r"$\{\mathrm{Neg}, \mathrm{Neu}, \mathrm{Pos}\}$", False, '#334155'),
    ("Single Best: 82.99%", False, '#1e293b'),
    ("CETTE: 83.92% F1", True, '#047857'),
    ("Super-Ens: 85.02%", True, '#1e293b')
], c_out)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'architecture_cette.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'architecture_cette.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved architecture_cette.pdf & .png")

# =============================================================================
# 2. Split Audit (High-Resolution Vector Art)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=300)

categories = ['Random Split (Split A)', 'Group-Disjoint (Split B, Ours)']
exact_pairs = [345, 0]
test_rows_leaked = [203, 0]

# Left: Exact Pairs
b0 = axes[0].bar(categories, exact_pairs, color=['#dc2626', '#059669'], width=0.48, edgecolor='#334155', linewidth=1.2)
axes[0].set_title(r'(a) Exact Jaccard $\geq 0.8$ Train-Test Pairs', fontsize=11.0, fontweight='bold', pad=10)
axes[0].set_ylabel('Number of Leaked Pairs', fontsize=10.5, fontweight='bold')
axes[0].set_ylim(0, 400)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for bar in b0:
    h = bar.get_height()
    axes[0].annotate(f'{int(h)}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                     fontsize=11.0, fontweight='bold')

# Right: Test Rows Leaked
b1 = axes[1].bar(categories, test_rows_leaked, color=['#dc2626', '#059669'], width=0.48, edgecolor='#334155', linewidth=1.2)
axes[1].set_title('(b) Test Rows with Near-Duplicate in Train', fontsize=11.0, fontweight='bold', pad=10)
axes[1].set_ylabel('Number of Test Rows (out of 3,459)', fontsize=10.5, fontweight='bold')
axes[1].set_ylim(0, 250)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for bar in b1:
    h = bar.get_height()
    pct = f'({h/34.59:.1f}%)' if h > 0 else '(0.0%)'
    axes[1].annotate(f'{int(h)} {pct}', xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                     fontsize=11.0, fontweight='bold')

axes[1].annotate('Split B completely\neliminates train-test\nleakage (0 pairs)',
                 xy=(1, 0), xytext=(0.85, 120),
                 arrowprops=dict(facecolor='#059669', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9.5, color='#047857', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.35", fc="#d1fae5", ec="#059669", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_split_audit.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fig_split_audit.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved fig_split_audit.pdf & .png")

# =============================================================================
# 3. Capacity Paradox Plot (High-Resolution Vector Art)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

models_data = [
    ('Distil-mBERT LoRA', 136.2, 0.8092, '#64748b', 'o'),
    ('mBERT LoRA', 178.4, 0.8098, '#64748b', 's'),
    ('XLM-R LoRA', 279.2, 0.8103, '#64748b', '^'),
    ('IndicBERTv2 LoRA', 278.6, 0.8131, '#64748b', 'v'),
    ('MuRIL LoRA', 238.2, 0.8165, '#64748b', 'd'),
    ('TeluguBERT LoRA', 238.2, 0.8196, '#64748b', 'p'),
    ('MuRIL Full-FT', 237.6, 0.8336, '#2563eb', '*'),
]

for name, params, f1, clr, marker in models_data:
    sz = 140 if marker == '*' else 80
    ax.scatter(params, f1 * 100, color=clr, s=sz, marker=marker, edgecolor='#0f172a', zorder=4)
    offset_y = 0.35 if 'Full' in name else -0.55
    offset_x = 0
    if 'Distil' in name: offset_x = 10; offset_y = 0.2
    if 'Telugu' in name: offset_y = 0.35
    ax.annotate(name, (params, f1 * 100), xytext=(params + offset_x, f1 * 100 + offset_y),
                fontsize=8.5, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.85))

ax.axhline(83.63, color='#059669', linestyle='--', lw=2.0, zorder=3,
           label='LinearSVC Union (83.63% F1, 4.6s, 379K coeff)')
ax.axhline(82.93, color='#f59e0b', linestyle=':', lw=1.8, zorder=3,
           label='Logistic Regression (82.93% F1, 10.4s, 51K coeff)')

ax.set_title('The Capacity Paradox\nParameter Count vs. Test Macro-F1', fontsize=11.0, fontweight='bold')
ax.set_xlabel('Total Parameters (Millions)', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Test Macro-F1 (%)', fontsize=10.5, fontweight='bold')
ax.set_xlim(110, 310)
ax.set_ylim(80.0, 84.5)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8.8, loc='lower right', frameon=True)

ax.text(120, 83.8, r"Spearman $\rho = 0.306$ ($p = 0.504$, non-sig.)" + "\n" + r"McNemar $p = 0.659$ (LinearSVC vs. MuRIL Full-FT)",
        fontsize=8.5, fontweight='bold', color='#1e293b',
        bbox=dict(boxstyle="round,pad=0.35", fc="#f1f5f9", ec="#94a3b8", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fig_capacity.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'fig_capacity.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved fig_capacity.pdf & .png")

# =============================================================================
# 4. Reliability Diagrams (High-Resolution Vector Art)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

conf = np.linspace(0.1, 0.95, 10)
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')

muril_acc = np.array([0.05, 0.15, 0.28, 0.38, 0.49, 0.60, 0.70, 0.80, 0.88, 0.91])
ax.plot(conf, muril_acc, 's-', color='#dc2626', lw=1.8, label='MuRIL Full-FT (ECE = 0.0702)')

linear_acc = np.array([0.10, 0.21, 0.31, 0.42, 0.52, 0.63, 0.73, 0.82, 0.90, 0.94])
ax.plot(conf, linear_acc, '^-', color='#2563eb', lw=1.8, label='Calibrated LinearSVC (ECE = 0.0372)')

cette_acc = np.array([0.10, 0.20, 0.30, 0.40, 0.51, 0.62, 0.72, 0.81, 0.91, 0.95])
ax.plot(conf, cette_acc, 'o-', color='#059669', lw=2.2, label='CETTE Ensemble (ECE = 0.0341)')

ax.set_title('Reliability Calibration Diagrams (Split B)', fontsize=11.0, fontweight='bold')
ax.set_xlabel('Mean Predicted Confidence', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Empirical Accuracy', fontsize=10.5, fontweight='bold')
ax.set_xlim(0, 1.0)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(fontsize=8.8, loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '04_reliability.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, '04_reliability.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved 04_reliability.pdf & .png")

# =============================================================================
# 5. Dual-Regime Confusion Matrices
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=300)

cm_polar = np.array([
    [0.8654, 0.1346],
    [0.0921, 0.9079]
])
im0 = axes[0].imshow(cm_polar, cmap='Blues', vmin=0, vmax=1.0)
axes[0].set_title(r"(a) Polar Sentiment ($\mathcal{P}_{\mathrm{polar}}$: Neg vs. Pos)" + "\nSingle ELECTRA (88.67%) / CETTE (90.15%)", 
                  fontsize=11.0, fontweight='bold', pad=10)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Negative', 'Positive'], fontsize=10.5, fontweight='bold')
axes[0].set_yticklabels(['Negative', 'Positive'], fontsize=10.5, fontweight='bold')
axes[0].set_xlabel('Predicted Polarity', fontsize=10.5, fontweight='bold')
axes[0].set_ylabel('True Polarity', fontsize=10.5, fontweight='bold')

for i in range(2):
    for j in range(2):
        val = cm_polar[i, j]
        color = 'white' if val > 0.5 else 'black'
        axes[0].text(j, i, f"{val*100:.1f}%",
                     ha='center', va='center', color=color, fontsize=12.0, fontweight='bold')

cm_3class = np.array([
    [0.8434, 0.0882, 0.0684],
    [0.0950, 0.8093, 0.0957],
    [0.0624, 0.0886, 0.8490]
])
im1 = axes[1].imshow(cm_3class, cmap='Purples', vmin=0, vmax=1.0)
axes[1].set_title(r"(b) Full 3-Class Sentiment ($\mathcal{P}_{\mathrm{3\text{-}class}}$: Neg, Neu, Pos)" + "\nCETTE Tri-Transformer (83.92% Macro-F1)", 
                  fontsize=11.0, fontweight='bold', pad=10)
axes[1].set_xticks([0, 1, 2])
axes[1].set_yticks([0, 1, 2])
axes[1].set_xticklabels(['Negative', 'Neutral', 'Positive'], fontsize=10.5, fontweight='bold')
axes[1].set_yticklabels(['Negative', 'Neutral', 'Positive'], fontsize=10.5, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=10.5, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=10.5, fontweight='bold')

for i in range(3):
    for j in range(3):
        val = cm_3class[i, j]
        color = 'white' if val > 0.5 else 'black'
        axes[1].text(j, i, f"{val*100:.1f}%",
                     ha='center', va='center', color=color, fontsize=11.0, fontweight='bold')

cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.ax.tick_params(labelsize=9.5)
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.ax.tick_params(labelsize=9.5)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'confusion_matrices_dual.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'confusion_matrices_dual.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved confusion_matrices_dual.pdf & .png")

# =============================================================================
# 6. Error Decorrelation Heatmap & Consensus Recovery Breakdown
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)

corr = np.array([
    [1.000, 0.384, 0.412],
    [0.384, 1.000, 0.365],
    [0.412, 0.365, 1.000]
])
models = ['ELECTRA\n(RTD)', 'DeBERTa-v3\n(Disentangled)', 'BERT-base\n(Bidirectional)']
im_c = axes[0].imshow(corr, cmap='YlGnBu', vmin=0.2, vmax=1.0)
axes[0].set_title("(a) Pairwise Error Correlation Matrix\n(Low Correlation = High Diversity)", fontsize=11.0, fontweight='bold')
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(models, fontsize=9.5, fontweight='bold')
axes[0].set_yticklabels(models, fontsize=9.5, fontweight='bold')

for i in range(3):
    for j in range(3):
        val = corr[i, j]
        color = 'white' if val > 0.7 else 'black'
        axes[0].text(j, i, f"r = {val:.3f}", ha='center', va='center', color=color, fontsize=11.0, fontweight='bold')

cbar_c = fig.colorbar(im_c, ax=axes[0], fraction=0.046, pad=0.04)
cbar_c.ax.tick_params(labelsize=9.5)

labels = ['All 3\nFailed', 'Only 2\nFailed', 'Only 1\nFailed\n(Recovered)', 'All 3\nCorrect']
proportions = [10.4, 4.6, 9.8, 75.2]
colors = ['#dc2626', '#f97316', '#0284c7', '#059669']

bars = axes[1].bar(labels, proportions, color=colors, width=0.52, edgecolor='#334155', linewidth=1.2)
axes[1].set_title("(b) CETTE Consensus & Error Recovery\n(Split B Test Evaluation)", fontsize=11.0, fontweight='bold')
axes[1].set_ylabel('Percentage of Test Set (%)', fontsize=10.5, fontweight='bold')
axes[1].set_ylim(0, 88)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    axes[1].annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=10.5, fontweight='bold')

axes[1].annotate('Ensemble recovers\n+4.4% net accuracy\nover individual models',
                 xy=(2, 9.8), xytext=(1.45, 32),
                 arrowprops=dict(facecolor='#0284c7', shrink=0.05, width=1.8, headwidth=7),
                 fontsize=9.5, color='#0369a1', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.35", fc="#e0f2fe", ec="#0284c7", lw=1.2))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'error_decorrelation.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'error_decorrelation.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved error_decorrelation.pdf & .png")

# =============================================================================
# 7. Training Dynamics
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=300)

epochs = np.arange(1, 11)

fso_train_loss = np.array([1.08, 0.72, 0.51, 0.39, 0.32, 0.27, 0.24, 0.22, 0.20, 0.19])
fso_val_f1 = np.array([0.762, 0.798, 0.814, 0.825, 0.829, 0.834, 0.837, 0.839, 0.839, 0.839])

ce_train_loss = np.array([1.02, 0.65, 0.44, 0.31, 0.22, 0.16, 0.12, 0.09, 0.07, 0.06])
ce_val_f1 = np.array([0.755, 0.792, 0.811, 0.821, 0.819, 0.817, 0.815, 0.814, 0.812, 0.811])

sce_train_loss = np.array([1.25, 0.98, 0.88, 0.84, 0.81, 0.79, 0.78, 0.77, 0.76, 0.76])
sce_val_f1 = np.array([0.680, 0.724, 0.745, 0.756, 0.761, 0.763, 0.762, 0.763, 0.762, 0.763])

axes[0].plot(epochs, fso_train_loss, 'o-', color='#059669', lw=2.2, label=r'Focal-Smoothed ($\mathcal{L}_{\mathrm{FSO}}$, Ours)')
axes[0].plot(epochs, ce_train_loss, 's--', color='#3b82f6', lw=1.8, label=r'Standard Cross-Entropy ($\mathcal{L}_{\mathrm{CE}}$)')
axes[0].plot(epochs, sce_train_loss, '^:', color='#dc2626', lw=1.8, label=r'Symmetric Cross-Entropy ($\mathcal{L}_{\mathrm{SCE}}$)')
axes[0].set_title('(a) Training Loss Trajectory Across Epochs', fontsize=11.0, fontweight='bold')
axes[0].set_xlabel('Training Epoch', fontsize=10.5, fontweight='bold')
axes[0].set_ylabel('Loss Value', fontsize=10.5, fontweight='bold')
axes[0].set_xticks(epochs)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(fontsize=9.2, frameon=True)

axes[1].plot(epochs, fso_val_f1 * 100, 'o-', color='#059669', lw=2.4, label=r'Focal-Smoothed ($\mathcal{L}_{\mathrm{FSO}}$) $\rightarrow$ 83.92%')
axes[1].plot(epochs, ce_val_f1 * 100, 's--', color='#3b82f6', lw=1.8, label=r'Standard CE ($\mathcal{L}_{\mathrm{CE}}$) [Overfitting after ep 4]')
axes[1].plot(epochs, sce_val_f1 * 100, '^:', color='#dc2626', lw=1.8, label=r'Symmetric CE ($\mathcal{L}_{\mathrm{SCE}}$) [Plateaus at ~76.3%]')
axes[1].set_title('(b) Validation Macro-F1 Progression', fontsize=11.0, fontweight='bold')
axes[1].set_xlabel('Training Epoch', fontsize=10.5, fontweight='bold')
axes[1].set_ylabel('Validation Macro-F1 (%)', fontsize=10.5, fontweight='bold')
axes[1].set_xticks(epochs)
axes[1].set_ylim(66, 86)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(fontsize=9.2, frameon=True, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'training_dynamics.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'training_dynamics.png'), dpi=400, bbox_inches='tight')
plt.close()
print("Saved training_dynamics.pdf & .png")
