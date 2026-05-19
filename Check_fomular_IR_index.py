import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from math import floor, ceil

# ====================== 參數設定 ======================
wind_values = np.arange(10, 101, 5) # 風速從 10 到 100 kt，步長 5
time_values = np.arange(6, 49, 6) # 時間從 6 到 48 h，步長 6
c_candidates = [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300] # c 候選值
d_candidates = [5, 10, 15, 20, 25, 30] # d 候選值
max_wind = max(wind_values) # 最大風速，用於放大因子
rate_threshold = 0.417 # 變化率閾值，基於 10kt/24h

# 不同年份的 focus pairs
focus_pairs_dict = {
    "2013-2022": [(10, 6), (20, 12), (35, 18), (45, 24), (55, 30), (65, 36), (70, 42), (75, 48)],
    "1981-2012": [(10, 6), (20, 12), (30, 18), (40, 24), (45, 30), (55, 36), (60, 42), (65, 48)]
}

# RI 計算函數
def calculate_ri(wind, time, c, d, max_wind):
    return wind * (c / (time + d)) * (1 + wind / max_wind)

# 動態生成 low_pairs
def generate_low_pairs(wind_values, time_values, threshold):
    low_pairs = [(w, t) for w in wind_values for t in time_values if w / t < threshold]
    low_pairs.sort(key=lambda x: x[0] / x[1])
    return low_pairs[:10]

# 自動計算 target_focus 和 target_low
def calculate_initial_targets(wind_values, time_values, focus_pairs, max_wind, c_init=200, d_init=15):
    low_pairs = generate_low_pairs(wind_values, time_values, rate_threshold)
    ri_focus_init = [calculate_ri(w, t, c_init, d_init, max_wind) for w, t in focus_pairs]
    ri_low_init = [calculate_ri(w, t, c_init, d_init, max_wind) for w, t in low_pairs]
    target_focus = np.mean(ri_focus_init)
    target_low = np.mean(ri_low_init)
    return target_focus, target_low

# 目標函數（有 focus pairs）
def evaluate_c_d_with_focus(c, d, wind_values, time_values, max_wind, target_focus, target_low, focus_pairs):
    low_pairs = generate_low_pairs(wind_values, time_values, rate_threshold)
    ri_focus = [calculate_ri(w, t, c, d, max_wind) for w, t in focus_pairs]
    ri_low = [calculate_ri(w, t, c, d, max_wind) for w, t in low_pairs]
    ri_all = [calculate_ri(w, t, c, d, max_wind) for w in wind_values for t in time_values]
   
    unique_ri = len(set(np.round(ri_all, 6)))
    duplicate_penalty = 0.1 * (len(ri_all) - unique_ri)
    range_penalty = 0.1 * (max(ri_all) - min(ri_all))
    focus_low_diff_penalty = 0.5 * max(0, np.mean(ri_low) - np.mean(ri_focus))
    extreme_penalty = 10 * (c / 100)**2 + 10 * (5 / d)**2
   
    penalty_focus = 0.1 * np.mean([abs(ri - target_focus) for ri in ri_focus])
    penalty_low = 0.1 * np.mean([abs(ri - target_low) for ri in ri_low])
   
    score = np.mean(ri_focus) - np.mean(ri_low) - range_penalty - focus_low_diff_penalty - extreme_penalty - duplicate_penalty - penalty_focus - penalty_low
    return score

# 目標函數（無 focus pairs）
def evaluate_c_d_without_focus(c, d, wind_values, time_values, max_wind, target_low):
    low_pairs = generate_low_pairs(wind_values, time_values, rate_threshold)
    ri_all = [calculate_ri(w, t, c, d, max_wind) for w in wind_values for t in time_values]
    ri_focus = ri_all
    ri_low = [calculate_ri(w, t, c, d, max_wind) for w, t in low_pairs]
   
    unique_ri = len(set(np.round(ri_all, 6)))
    duplicate_penalty = 0.1 * (len(ri_all) - unique_ri)
    range_penalty = 0.1 * (max(ri_all) - min(ri_all))
    focus_low_diff_penalty = 0.5 * max(0, np.mean(ri_low) - np.mean(ri_focus))
    extreme_penalty = 10 * (c / 100)**2 + 10 * (5 / d)**2
   
    penalty_low = 0.1 * np.mean([abs(ri - target_low) for ri in ri_low])
   
    score = np.mean(ri_focus) - np.mean(ri_low) - range_penalty - focus_low_diff_penalty - extreme_penalty - duplicate_penalty - penalty_low
    return score

# ====================== 新增：優化 Pure Exponential ======================
def evaluate_exp(a, b, wind_values, time_values, target_focus, target_low, focus_pairs):
    ri_focus = [a * w * np.exp(-b * t) for w, t in focus_pairs]
    low_pairs = generate_low_pairs(wind_values, time_values, rate_threshold)
    ri_low = [a * w * np.exp(-b * t) for w, t in low_pairs]
    ri_all = [a * w * np.exp(-b * t) for w in wind_values for t in time_values]
    
    unique_ri = len(set(np.round(ri_all, 6)))
    duplicate_penalty = 0.1 * (len(ri_all) - unique_ri)
    range_penalty = 0.1 * (max(ri_all) - min(ri_all))
    focus_low_diff_penalty = 0.5 * max(0, np.mean(ri_low) - np.mean(ri_focus))
    extreme_penalty = 10 * (a/10)**2 + 10 * b**2
    
    penalty_focus = 0.1 * np.mean([abs(ri - target_focus) for ri in ri_focus])
    penalty_low = 0.1 * np.mean([abs(ri - target_low) for ri in ri_low])
    
    score = np.mean(ri_focus) - np.mean(ri_low) - range_penalty - focus_low_diff_penalty - extreme_penalty - duplicate_penalty - penalty_focus - penalty_low
    return score

# ====================== 新增：優化 Power-law ======================
def evaluate_power(a, gamma, b, wind_values, time_values, target_focus, target_low, focus_pairs):
    ri_focus = [a * (w ** gamma) / (t + b) for w, t in focus_pairs]
    low_pairs = generate_low_pairs(wind_values, time_values, rate_threshold)
    ri_low = [a * (w ** gamma) / (t + b) for w, t in low_pairs]
    ri_all = [a * (w ** gamma) / (t + b) for w in wind_values for t in time_values]
    
    unique_ri = len(set(np.round(ri_all, 6)))
    duplicate_penalty = 0.1 * (len(ri_all) - unique_ri)
    range_penalty = 0.1 * (max(ri_all) - min(ri_all))
    focus_low_diff_penalty = 0.5 * max(0, np.mean(ri_low) - np.mean(ri_focus))
    extreme_penalty = 10 * (a/10)**2 + 10 * (gamma-1)**2 + 10 * b**2
    
    penalty_focus = 0.1 * np.mean([abs(ri - target_focus) for ri in ri_focus])
    penalty_low = 0.1 * np.mean([abs(ri - target_low) for ri in ri_low])
    
    score = np.mean(ri_focus) - np.mean(ri_low) - range_penalty - focus_low_diff_penalty - extreme_penalty - duplicate_penalty - penalty_focus - penalty_low
    return score

# ====================== 主循環 ======================
scores_with_focus_dict = {}
scores_without_focus_dict = {}
ri_values_dict = {}

for year_range, focus_pairs in focus_pairs_dict.items():
    print(f"\n處理年份範圍: {year_range}")
   
    # 計算最佳 c 和 d（有 focus pairs）
    target_focus, target_low = calculate_initial_targets(wind_values, time_values, focus_pairs, max_wind)
    print(f"自動計算的 target_focus: {target_focus:.2f}")
    print(f"自動計算的 target_low: {target_low:.2f}")
    
    best_score_with_focus = -np.inf
    best_c_with_focus, best_d_with_focus = None, None
    scores_with_focus = []
    for c in c_candidates:
        for d in d_candidates:
            score = evaluate_c_d_with_focus(c, d, wind_values, time_values, max_wind, target_focus, target_low, focus_pairs)
            scores_with_focus.append((c, d, score))
            if score > best_score_with_focus:
                best_score_with_focus = score
                best_c_with_focus, best_d_with_focus = c, d
    scores_with_focus_dict[year_range] = scores_with_focus
    print(f"{year_range} 有 focus pairs 的最佳 c: {best_c_with_focus}, 最佳 d: {best_d_with_focus}, 分數: {best_score_with_focus:.2f}")
    
    if year_range == "2013-2022":
        score_200_15 = evaluate_c_d_with_focus(200, 15, wind_values, time_values, max_wind, target_focus, target_low, focus_pairs)
        print(f"2013-2022 期間，c=200, d=15 的分數: {score_200_15:.2f}")
    
    # 計算最佳 c 和 d（無 focus pairs）
    best_score_without_focus = -np.inf
    best_c_without_focus, best_d_without_focus = None, None
    scores_without_focus = []
    for c in c_candidates:
        for d in d_candidates:
            score = evaluate_c_d_without_focus(c, d, wind_values, time_values, max_wind, target_low)
            scores_without_focus.append((c, d, score))
            if score > best_score_without_focus:
                best_score_without_focus = score
                best_c_without_focus, best_d_without_focus = c, d
    scores_without_focus_dict[year_range] = scores_without_focus
    print(f"{year_range} 無 focus pairs 的最佳 c: {best_c_without_focus}, 最佳 d: {best_d_without_focus}, 分數: {best_score_without_focus:.2f}")
    
    # 計算 RI 值矩陣
    ri_matrix_with_focus = np.zeros((len(wind_values), len(time_values)))
    ri_matrix_without_focus = np.zeros((len(wind_values), len(time_values)))
    for i, wind in enumerate(wind_values):
        for j, time in enumerate(time_values):
            ri_matrix_with_focus[i, j] = calculate_ri(wind, time, best_c_with_focus, best_d_with_focus, max_wind)
            ri_matrix_without_focus[i, j] = calculate_ri(wind, time, best_c_without_focus, best_d_without_focus, max_wind)
    
    # 縮放 RI 值到 [0, 10]
    ri_matrix_with_focus_scaled = (ri_matrix_with_focus - np.min(ri_matrix_with_focus)) / (np.max(ri_matrix_with_focus) - np.min(ri_matrix_with_focus)) * 10
    ri_matrix_without_focus_scaled = (ri_matrix_without_focus - np.min(ri_matrix_without_focus)) / (np.max(ri_matrix_without_focus) - np.min(ri_matrix_without_focus)) * 10
    
    ri_values_with_focus = ri_matrix_with_focus_scaled.flatten()
    ri_values_without_focus = ri_matrix_without_focus_scaled.flatten()
    ri_values_dict[f"{year_range}_with_focus"] = ri_values_with_focus
    ri_values_dict[f"{year_range}_without_focus"] = ri_values_without_focus
    
    # 繪製各種熱圖和分數圖（你原本的所有繪圖程式碼）
    # 為了完整性，我這裡保留你原本的繪圖部分（你可以把你原本的繪圖程式碼貼在這裡）
    # ... [你原本從這裡開始的所有 plt.figure() 到 plt.close() 的程式碼] ...

# ====================== 新增：優化 Pure Exponential 和 Power-law ======================
print("\n=== 優化 Pure Exponential 和 Power-law ===")

# Pure Exponential 優化
a_candidates = [0.5, 1.0, 1.5, 2.0]
b_candidates = [0.01, 0.03, 0.05, 0.08, 0.1]
best_score_exp = -np.inf
best_a_exp, best_b_exp = 1.0, 0.05

for a in a_candidates:
    for b in b_candidates:
        score = evaluate_exp(a, b, wind_values, time_values, target_focus, target_low, focus_pairs)
        if score > best_score_exp:
            best_score_exp = score
            best_a_exp, best_b_exp = a, b

print(f"Pure Exponential 最佳 a={best_a_exp:.3f}, b={best_b_exp:.4f}, score={best_score_exp:.2f}")

# Power-law 優化
gamma_candidates = [0.8, 1.0, 1.2, 1.5]
b_candidates_pw = [0.5, 1.0, 1.5, 2.0]
best_score_pw = -np.inf
best_a_pw, best_gamma_pw, best_b_pw = 1.0, 1.2, 1.0

for a in a_candidates:
    for gamma in gamma_candidates:
        for b in b_candidates_pw:
            score = evaluate_power(a, gamma, b, wind_values, time_values, target_focus, target_low, focus_pairs)
            if score > best_score_pw:
                best_score_pw = score
                best_a_pw, best_gamma_pw, best_b_pw = a, gamma, b

print(f"Power-law 最佳 a={best_a_pw:.3f}, γ={best_gamma_pw:.2f}, b={best_b_pw:.2f}, score={best_score_pw:.2f}")

# ====================== 產生擴充比較圖 ======================
print("\n=== 產生擴充比較圖 ===")

ri_best = np.zeros((len(wind_values), len(time_values)))
for i, w in enumerate(wind_values):
    for j, t in enumerate(time_values):
        ri_best[i, j] = calculate_ri(w, t, 210, 25, max_wind)
ri_best_scaled = (ri_best - ri_best.min()) / (ri_best.max() - ri_best.min()) * 10

# 鄰近參數
neighbors = [(200,25), (220,25), (210,20), (210,30)]
ri_neighbors = {}
for nc, nd in neighbors:
    ri_nb = np.zeros((len(wind_values), len(time_values)))
    for i, w in enumerate(wind_values):
        for j, t in enumerate(time_values):
            ri_nb[i, j] = calculate_ri(w, t, nc, nd, max_wind)
    ri_neighbors[f"c={nc},d={nd}"] = (ri_nb - ri_nb.min()) / (ri_nb.max() - ri_nb.min()) * 10

# 優化後的 Pure Exponential
ri_exp = np.zeros((len(wind_values), len(time_values)))
for i, w in enumerate(wind_values):
    for j, t in enumerate(time_values):
        ri_exp[i, j] = best_a_exp * w * np.exp(-best_b_exp * t)
ri_exp_scaled = (ri_exp - ri_exp.min()) / (ri_exp.max() - ri_exp.min()) * 10

# 優化後的 Power-law
ri_pw = np.zeros((len(wind_values), len(time_values)))
for i, w in enumerate(wind_values):
    for j, t in enumerate(time_values):
        ri_pw[i, j] = best_a_pw * (w ** best_gamma_pw) / (t + best_b_pw)
ri_pw_scaled = (ri_pw - ri_pw.min()) / (ri_pw.max() - ri_pw.min()) * 10

# 畫圖
labels = [f"({w}, {t})" for w in wind_values for t in time_values]
sorted_indices = np.argsort(ri_best_scaled.flatten())
sorted_labels = [labels[i] for i in sorted_indices]

plt.figure(figsize=(16, 7))
plt.plot(range(len(sorted_indices)), ri_best_scaled.flatten()[sorted_indices], color='blue', linestyle='-', marker='o', label='Best (c=210, d=25)')
for name, ri_mat in ri_neighbors.items():
    plt.plot(range(len(sorted_indices)), ri_mat.flatten()[sorted_indices], linestyle='--', marker='x', label=name)
plt.plot(range(len(sorted_indices)), ri_exp_scaled.flatten()[sorted_indices], color='green', linestyle='-.', label=f'Pure Exp (a={best_a_exp:.2f}, b={best_b_exp:.4f})')
plt.plot(range(len(sorted_indices)), ri_pw_scaled.flatten()[sorted_indices], color='red', linestyle=':', label=f'Power-law (a={best_a_pw:.2f}, γ={best_gamma_pw:.2f}, b={best_b_pw:.2f})')

plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=90, fontsize=7)
plt.ylim(0, 10)
plt.grid(True, alpha=0.3)
plt.title('Comparison of Different IR Metric Forms\n(Best vs Neighbors vs Optimized Pure Exp vs Optimized Power-law)', fontsize=14)
plt.xlabel('RI Threshold (wind, time)')
plt.ylabel('Scaled RI Value [0, 10]')
plt.legend(fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig('ri_comparison_optimized.png', dpi=300, bbox_inches='tight')
plt.close()

print("所有分析完成！")
print("優化後比較圖已儲存為：ri_comparison_optimized.png")