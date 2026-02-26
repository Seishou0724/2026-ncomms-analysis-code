import os
import numpy as np
import pandas as pd
import joblib
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.patches import Rectangle
# 定義年份範圍（評估用，可改成 "2013-2022" 或 "1981-2022"）
eval_year_range = "2013-2022"
# 訓練年份範圍（用來載入模型）
train_year_range = "2013-2022"
# 設置輸出目錄
train_output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{train_year_range}-continue"
eval_output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{eval_year_range}-eval"
os.makedirs(eval_output_dir, exist_ok=True)
# 設置日誌
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)
log_file_path = os.path.join(eval_output_dir, f"ANO_RII_randomforest_continue_{eval_year_range}_eval.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, delay=False), logging.StreamHandler()],
    force=True
)
# 基本參數
base_path = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{eval_year_range}"
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
}
# RI 計算相關參數
c_candidates = [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300]
d_candidates = [5, 10, 15, 20, 25, 30]
max_wind = max(wind_thresholds)
rate_threshold = 0.417
focus_pairs = [(10, 6), (20, 12), (35, 18), (45, 24), (55, 30), (65, 36), (70, 42), (75, 48)]
# Best track CSV 路徑
best_track_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/ibtracs.WP.list.v04r01.csv"
# 訓練測試集資料的儲存路徑
test_data_files = {
    "rf": {
        "X_test": os.path.join(train_output_dir, "PVTHE_X_test_rf_kfold_upper_inner_2013-2022.pkl"),
        "y_test": os.path.join(train_output_dir, "PVTHE_y_test_rf_kfold_upper_inner_2013-2022.pkl"),
        "combo_test": os.path.join(train_output_dir, "PVTHE_combo_test_rf_kfold_upper_inner_2013-2022.pkl"),
        "delta_winds": os.path.join(train_output_dir, "PVTHE_delta_winds_rf_kfold_upper_inner_2013-2022.pkl")
    },
    "svr": {
        "X_test": os.path.join(train_output_dir, "PVTHE_X_test_svr_kfold_upper_inner_2013-2022.pkl"),
        "y_test": os.path.join(train_output_dir, "PVTHE_y_test_svr_kfold_upper_inner_2013-2022.pkl"),
        "combo_test": os.path.join(train_output_dir, "PVTHE_combo_test_svr_kfold_upper_inner_2013-2022.pkl"),
        "delta_winds": os.path.join(train_output_dir, "PVTHE_delta_winds_svr_kfold_upper_inner_2013-2022.pkl")
    },
    "ann": {
        "X_test": os.path.join(train_output_dir, "PVTHE_X_test_ann_kfold_upper_inner_2013-2022.pkl"),
        "y_test": os.path.join(train_output_dir, "PVTHE_y_test_ann_kfold_upper_inner_2013-2022.pkl"),
        "combo_test": os.path.join(train_output_dir, "PVTHE_combo_test_ann_kfold_upper_inner_2013-2022.pkl"),
        "delta_winds": os.path.join(train_output_dir, "PVTHE_delta_winds_ann_kfold_upper_inner_2013-2022.pkl")
    }
}
# 數據調整函數 (保留，但評估不用)
def adjust_data(data, var, region_rows, region_cols):
    if data.shape != (27, 21):
        return np.array([])
    outer_cols = list(range(18, 21))
    if not all(0 <= col < data.shape[1] for col in outer_cols):
        return np.array([])
    if np.all(np.isnan(data)) or np.all(data == 0):
        return np.array([])
    adjusted_data = data.copy()
    for row in range(data.shape[0]):
        outer_data_row = data[row, outer_cols]
        outer_avg = 0 if np.all(np.isnan(outer_data_row)) else np.nanmean(outer_data_row)
        adjusted_data[row, :] -= outer_avg
    var_rows = variables[var]["rows"]
    var_cols = variables[var]["cols"]
    intersect_rows = sorted(set(var_rows) & set(region_rows))
    intersect_cols = sorted(set(var_cols) & set(region_cols))
    if not intersect_rows or not intersect_cols:
        return np.array([])
    selected_data = adjusted_data[np.ix_(intersect_rows, intersect_cols)]
    if selected_data.size == 0 or np.all(np.isnan(selected_data)) or np.all(selected_data == 0):
        return np.array([])
    return selected_data.flatten()
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
# 修改: 加載測試資料函數 (從 pkl 載入測試集資料)
def load_test_data(model_type):
    logging.info(f"開始載入 {model_type} 的測試資料")
    X_path = test_data_files[model_type]["X_test"]
    delta_path = test_data_files[model_type]["delta_winds"]
    combo_path = test_data_files[model_type]["combo_test"]
    if not os.path.exists(X_path) or not os.path.exists(delta_path) or not os.path.exists(combo_path):
        logging.error(f"測試資料檔案不存在 for {model_type}")
        raise FileNotFoundError(f"測試資料檔案不存在 for {model_type}")
    X_test = joblib.load(X_path)
    delta_winds = joblib.load(delta_path)
    combo_test = joblib.load(combo_path)
    # 如果 X_test 是 DataFrame，轉成 np.array
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    logging.info(f"載入 X_test shape: {X_test.shape}, delta_winds shape: {delta_winds.shape}, combo_test len: {len(combo_test)}")
    return X_test, delta_winds, combo_test
# 主函數
def main():
    logging.info("評估程式開始執行")
    best_c, best_d = find_best_c_d(wind_thresholds, time_intervals, focus_pairs, max_wind)
    region = "upper_inner"
    logging.info(f"開始分析區域: {region}")
    # 載入三個模型 (假設 SVR 和 ANN 有相應 pkl)
    models = {
        'rf': joblib.load(os.path.join(train_output_dir, f"PVTHE_rf_model_kfold_{region}_{train_year_range}.pkl")),
        'svr': joblib.load(os.path.join(train_output_dir, f"PVTHE_svr_model_kfold_{region}_{train_year_range}.pkl")),
        'ann': joblib.load(os.path.join(train_output_dir, f"PVTHE_ann_model_kfold_{region}_{train_year_range}.pkl"))
    }
    logging.info("載入模型和縮放器...")
    # 為每個模型處理
    for model_type, model in models.items():
        logging.info(f"處理模型: {model_type}")
        # 載入每個模型的測試資料
        X_test, delta_winds, combo_test = load_test_data(model_type)
        # 載入該模型的 scaler (修改為 per model)
        scaler = joblib.load(os.path.join(train_output_dir, f"PVTHE_{model_type}_feature_scaler_kfold_{region}_{train_year_range}.pkl"))
        scaler_y = joblib.load(os.path.join(train_output_dir, f"PVTHE_{model_type}_target_scaler_kfold_{region}_{train_year_range}.pkl"))
  
        # 為每個 time 計算 confusion
        confusion_matrix = np.zeros((len(wind_thresholds), len(time_intervals), 4)) # [hit, miss, fp, tn]
  
        for t_idx, time in enumerate(time_intervals):
            # 根據 combo_test 群組樣本
            mask = np.array([c[1] == time for c in combo_test])
            if np.sum(mask) == 0:
                logging.warning(f"For time={time}: 無樣本")
                continue
            X_for_t = X_test[mask] # (n_for_t, 180)
            delta_for_t = delta_winds[mask, t_idx] # 該 time 的 actual delta
            if len(X_for_t) == 0:
                logging.warning(f"For time={time}: 無序列資料")
                continue
            # 預測: 每個樣本單點預測 (無序列)
            y_pred_scaled = []
            X_for_t_scaled = scaler.transform(X_for_t)
            preds = model.predict(X_for_t_scaled)
            y_pred_scaled = np.clip(preds, 0, 10)
      
            for w_idx, wind in enumerate(wind_thresholds):
                cutoff = scaler_y.transform([[calculate_ri(wind, time, best_c, best_d, max_wind)]])[0][0]
          
                hit, miss, fp, tn = 0, 0, 0, 0
                if wind == 30 and time == 24:
                    logging.info(f"開始記錄 30kt/24h 詳細資料 (模型: {model_type})")
                for i in range(len(delta_for_t)):
                    actual_delta = delta_for_t[i]
                    is_actual_ri = (actual_delta >= wind)
                    is_pred_ri = (y_pred_scaled[i] >= cutoff)
                    if wind == 30 and time == 24:
                        category = ""
                        if is_actual_ri and is_pred_ri:
                            category = "Hit"
                        elif is_actual_ri and not is_pred_ri:
                            category = "Miss"
                        elif not is_actual_ri and is_pred_ri:
                            category = "FA"
                        elif not is_actual_ri and not is_pred_ri:
                            category = "CR"
                        logging.info(f"樣本 {i}: delta_wind={actual_delta}, RI_index={y_pred_scaled[i]:.4f}, cutoff={cutoff:.4f}, actual_ri={is_actual_ri}, pred_ri={is_pred_ri}, category={category}")
                    if is_actual_ri and is_pred_ri:
                        hit += 1
                    elif is_actual_ri and not is_pred_ri:
                        miss += 1
                    elif not is_actual_ri and is_pred_ri:
                        fp += 1
                    elif not is_actual_ri and not is_pred_ri:
                        tn += 1
                if wind == 30 and time == 24:
                    logging.info(f"30kt/24h 總結: Hit={hit}, Miss={miss}, FA={fp}, CR={tn}")
                confusion_matrix[w_idx, t_idx] = [hit, miss, fp, tn]
  
        # 建擴大 matrix 為 sub-box，以百分比為單位
        num_winds = len(wind_thresholds)
        num_times = len(time_intervals)
        expanded_matrix = np.full((2 * num_winds, 2 * num_times), np.nan)
        annot_matrix = np.full((2 * num_winds, 2 * num_times), "", dtype=object)
        balance_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
   
        for w_idx, wind in enumerate(wind_thresholds):
            for t_idx, time in enumerate(time_intervals):
                conf = confusion_matrix[w_idx, t_idx]
                hit, miss, fp, tn = conf
                total_ri = hit + miss
                total_nonri = fp + tn
          
                # 如果 total_ri == 0 且 CR == 100% (fp == 0)，則空白整個組合
                if total_ri == 0 and total_nonri > 0 and fp == 0:
                    expanded_matrix[2*w_idx:2*w_idx+2, 2*t_idx:2*t_idx+2] = np.nan
                    annot_matrix[2*w_idx:2*w_idx+2, 2*t_idx:2*t_idx+2] = ""
                    continue
          
                # 否則正常顯示，包括 0% 的情況
                hit_pct = (hit / total_ri * 100) if total_ri > 0 else 0.0
                miss_pct = (miss / total_ri * 100) if total_ri > 0 else 0.0
                expanded_matrix[2*w_idx, 2*t_idx] = hit_pct
                expanded_matrix[2*w_idx, 2*t_idx+1] = miss_pct
                annot_matrix[2*w_idx, 2*t_idx] = f"Recall: {hit_pct:.1f}%"
                annot_matrix[2*w_idx, 2*t_idx+1] = f"Miss: {miss_pct:.1f}%"
          
                fp_pct = (fp / total_nonri * 100) if total_nonri > 0 else 0.0
                tn_pct = (tn / total_nonri * 100) if total_nonri > 0 else 0.0
                expanded_matrix[2*w_idx+1, 2*t_idx] = fp_pct
                expanded_matrix[2*w_idx+1, 2*t_idx+1] = tn_pct
                annot_matrix[2*w_idx+1, 2*t_idx] = f"FA: {fp_pct:.1f}%"
                annot_matrix[2*w_idx+1, 2*t_idx+1] = f"CR: {tn_pct:.1f}%"
               
                # 計算 Balance Indicator: 1 - |TPR - TNR|
                tpr = hit / total_ri if total_ri > 0 else 0
                tnr = tn / total_nonri if total_nonri > 0 else 0
                balance = 1 - abs(tpr - tnr)
                balance_matrix[w_idx, t_idx] = balance
       
        # 畫 confusion 熱圖
        plt.figure(figsize=(28, 20))
        ax = sns.heatmap(expanded_matrix[::-1], xticklabels=False, yticklabels=False,
                          cmap="Blues", annot=annot_matrix[::-1], fmt="", linewidths=3, linecolor='black',
                          cbar_kws={'label': "Percentage"})
        # 自訂 ticks
        xticks = [1 + 2*i for i in range(num_times)]
        ax.set_xticks(xticks)
        ax.set_xticklabels(time_intervals, rotation=0)
        yticks = [1 + 2*i for i in range(num_winds)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(wind_thresholds[::-1], rotation=0)
 
        # 加粗主要分界線
        for i in range(2, expanded_matrix.shape[0], 2):
            ax.axhline(i, color='black', lw=6)
        for j in range(2, expanded_matrix.shape[1], 2):
            ax.axvline(j, color='black', lw=6)
 
        # 強化外框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(6)
            spine.set_color('black')
       
        # 加紅框給 Balance > 0.8 的 2x2 子格
        for t_idx in range(num_times):
            max_balance_in_col = np.nanmax(balance_matrix[:, t_idx])
            for w_idx in range(num_winds):
                balance = balance_matrix[w_idx, t_idx]
                if np.isnan(balance):
                    continue
                # 熱圖 y 軸倒序，所以 2x2 子格的左上角 y = (num_winds - w_idx - 1) * 2
                y_pos = (num_winds - w_idx - 1) * 2
                x_pos = t_idx * 2
                if balance > 0.8:
                    color = 'darkred' if balance == max_balance_in_col else 'red'
                    ax.add_patch(Rectangle((x_pos, y_pos), 2, 2, fill=False, edgecolor=color, lw=6, zorder=10))
 
# plt.title(f"RI Confusion Matrix (Hit/Miss/FA/CR) - {region} ({eval_year_range}) - Model: {model_type.upper()}")
        plt.xlabel("Time Interval (h)", fontsize=20)
        plt.ylabel("Wind Speed Increment Threshold (kt)", fontsize=20)
        # 加大 colorbar 的 label 和 tick labels 字體大小
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(20)  # Colorbar label 大小
        cbar.ax.tick_params(labelsize=16)  # Colorbar tick labels 大小
        heatmap_path = os.path.join(eval_output_dir, f"RI_confusion_heatmap_{model_type}_{region}_{eval_year_range}_{train_year_range}.png")
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"熱圖已儲存至 {heatmap_path}")
       
        # 畫 Balance 熱圖 (簡單 annot Balance 值)
        plt.figure(figsize=(14, 10))
        ax_balance = sns.heatmap(balance_matrix[::-1], xticklabels=time_intervals, yticklabels=wind_thresholds[::-1],
                                 cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.5,
                                 cbar_kws={'label': "Balance Indicator"})
# plt.title(f"Balance Indicator Heatmap - {region} ({eval_year_range}) - Model: {model_type.upper()}")
        plt.xlabel("Time Interval (h)", fontsize=20)
        plt.ylabel("Wind Speed Increment Threshold (kt)", fontsize=20)
        # 加大 colorbar 的 label 和 tick labels 字體大小
        cbar = ax_balance.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(20)  # Colorbar label 大小
        cbar.ax.tick_params(labelsize=16)  # Colorbar tick labels 大小
        balance_heatmap_path = os.path.join(eval_output_dir, f"RI_balance_heatmap_{model_type}_{region}_{eval_year_range}_{train_year_range}.png")
        plt.savefig(balance_heatmap_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"Balance 熱圖已儲存至 {balance_heatmap_path}")
if __name__ == "__main__":
    try:
        main()
        logging.info("程式執行完畢")
    except Exception as e:
        logging.exception("程式執行時發生嚴重錯誤：")
    finally:
        logging.shutdown()