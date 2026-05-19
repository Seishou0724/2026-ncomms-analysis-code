import os
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error   # ← 這一行已補上

# ====================== 設定 ======================
year_range = "1981-2022"   # ←←← 改成你要畫的年代 (1959-1980 / 1981-2012 / 2013-2022)
region = "upper_inner"
output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{year_range}-continue"

# 設置日誌
log_file_path = os.path.join(output_dir, f"heatmap_plot_only_{year_range}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
              logging.StreamHandler()]
)
logging.info(f"開始獨立繪製熱圖 - {year_range}")

# ====================== RI 計算相關參數 (與訓練程式碼完全相同) ======================
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
max_wind = max(wind_thresholds)
rate_threshold = 0.417
focus_pairs = [(10, 6), (20, 12), (35, 18), (45, 24), (55, 30), (65, 36), (70, 42), (75, 48)]

c_candidates = [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300]
d_candidates = [5, 10, 15, 20, 25, 30]

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

# 目標函數
def evaluate_c_d(c, d, wind_values, time_values, max_wind, target_focus, target_low):
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

# 計算最佳 c 和 d
def find_best_c_d(wind_values, time_values, focus_pairs, max_wind):
    target_focus, target_low = calculate_initial_targets(wind_values, time_values, focus_pairs, max_wind)
    best_score = -np.inf
    best_c, best_d = None, None
    for c in c_candidates:
        for d in d_candidates:
            score = evaluate_c_d(c, d, wind_values, time_values, max_wind, target_focus, target_low)
            if score > best_score:
                best_score = score
                best_c, best_d = c, d
    logging.info(f"最佳 c: {best_c}, 最佳 d: {best_d}, 分數: {best_score:.2f}")
    return best_c, best_d

# ====================== 載入檔案 ======================
try:
    best_c, best_d = find_best_c_d(wind_thresholds, time_intervals, focus_pairs, max_wind)
    
    X_test = joblib.load(os.path.join(output_dir, f"PVTHE_X_test_rf_onset_{region}_{year_range}.pkl"))
    combo_test = joblib.load(os.path.join(output_dir, f"PVTHE_combo_test_rf_onset_{region}_{year_range}.pkl"))
    
    model = joblib.load(os.path.join(output_dir, f"PVTHE_rf_model_onset_{region}_{year_range}.pkl"))
    feature_scaler = joblib.load(os.path.join(output_dir, f"PVTHE_rf_feature_scaler_onset_{region}_{year_range}.pkl"))
    target_scaler = joblib.load(os.path.join(output_dir, f"PVTHE_rf_target_scaler_onset_{region}_{year_range}.pkl"))
    
    logging.info("所有檔案載入成功")
except Exception as e:
    logging.error(f"載入檔案失敗: {e}")
    exit()

# ====================== 預測與計算 ======================
X_scaled = feature_scaler.transform(X_test)
y_pred = model.predict(X_scaled)
y_pred = np.clip(y_pred, 0, 10)

# 計算 target（使用重新計算的 best_c, best_d）
y_target_raw = np.array([calculate_ri(wind, time, best_c, best_d, max_wind) for wind, time in combo_test])
y_target_scaled = target_scaler.transform(y_target_raw.reshape(-1, 1)).ravel()

test_results = pd.DataFrame({
    'y_pred': y_pred,
    'y_target': y_target_scaled,
    'wind': [c[0] for c in combo_test],
    'time': [c[1] for c in combo_test]
})

# 建立矩陣
ri_index_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
mse_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
diff_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)

for wind_idx, wind in enumerate(wind_thresholds):
    for time_idx, time in enumerate(time_intervals):
        combo_data = test_results[(test_results['wind'] == wind) & (test_results['time'] == time)]
        if len(combo_data) > 0:
            ri_index = np.mean(combo_data['y_pred'])
            mse_val = mean_squared_error(combo_data['y_target'], combo_data['y_pred'])
            difference = ri_index - np.mean(combo_data['y_target'])
            
            ri_index_matrix[wind_idx, time_idx] = ri_index
            mse_matrix[wind_idx, time_idx] = mse_val
            diff_matrix[wind_idx, time_idx] = difference

# ====================== 第一張圖：只顯示 Accessed IR Index ======================
annot_matrix1 = np.array([[f"{ri_index_matrix[i, j]:.2f}" 
                           if not np.isnan(ri_index_matrix[i, j]) else ""
                           for j in range(len(time_intervals))]
                          for i in range(len(wind_thresholds))])

plt.figure(figsize=(14, 10))
ax1 = sns.heatmap(ri_index_matrix[::-1], 
                  xticklabels=time_intervals, 
                  yticklabels=wind_thresholds[::-1],
                  cmap="YlOrRd", 
                  annot=annot_matrix1[::-1], 
                  fmt="", 
                  linewidths=0.5, 
                  linecolor='black',
                  cbar_kws={'label': f"Predicted IR Index"}, 
                  annot_kws={"size": 11},
                  vmin=0, vmax=10)
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(2)
plt.title(f"Predicted IR Index by RF - {region} ({year_range})")
plt.xlabel("Time Interval (h)")
plt.ylabel("Wind Speed Increment Threshold (kt)")
heatmap_path1 = os.path.join(output_dir, f"RI_Index_RF_heatmap_predicted_only_{year_range}_{region}.png")
plt.savefig(heatmap_path1, bbox_inches='tight', dpi=300)
plt.close()
logging.info(f"主要熱圖 (只顯示 Predicted IR Index) 已儲存至 {heatmap_path1}")

# ====================== 第二張圖：顯示 difference 和 MSE ======================
annot_matrix2 = np.array([[f"{diff_matrix[i, j]:.2f}\n({mse_matrix[i, j]:.2f})" 
                           if not np.isnan(diff_matrix[i, j]) else ""
                           for j in range(len(time_intervals))]
                          for i in range(len(wind_thresholds))])

plt.figure(figsize=(14, 10))
ax2 = sns.heatmap(diff_matrix[::-1], 
                  xticklabels=time_intervals, 
                  yticklabels=wind_thresholds[::-1],
                  cmap="RdBu_r",
                  center=0,
                  annot=annot_matrix2[::-1], 
                  fmt="", 
                  linewidths=0.5, 
                  linecolor='black',
                  cbar_kws={'label': f"Difference  (Predicted - Target) IR Index "}, 
                  annot_kws={"size": 11},
                  vmin=-10, vmax=10)
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(2)
plt.title(f"Predicted IR Index by RF with Difference & MSE - {region} ({year_range})")
plt.xlabel("Time Interval (h)")
plt.ylabel("Wind Speed Increment Threshold (kt)")
heatmap_path2 = os.path.join(output_dir, f"RI_Index_RF_heatmap_with_diff_mse_{year_range}_{region}.png")
plt.savefig(heatmap_path2, bbox_inches='tight', dpi=300)
plt.close()
logging.info(f"補充熱圖 (包含 difference 和 MSE) 已儲存至 {heatmap_path2}")

logging.info("兩張熱圖繪製完成！")