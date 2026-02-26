import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# 定義年份範圍
year_range = "1981-2022"
# 設置輸出目錄
output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{year_range}-continue"
os.makedirs(output_dir, exist_ok=True)
# 設置日誌
logging.getLogger('matplotlib').setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)
log_file_path = os.path.join(output_dir, f"ANO_RII_randomforest_continue_{year_range}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, delay=False), logging.StreamHandler()],
    force=True
)
# 基本參數
base_path = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{year_range}"
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
    # "W": {"rows": range(0, 27), "cols": range(0, 21)}
}
regions = {
    # "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
    # "upper_outer": {"rows": range(2, 12), "cols": range(9, 18), "name": "Upper Level Outer Area"},
    # "middle_inner": {"rows": range(12, 21), "cols": range(0, 9), "name": "Middle Level Inner-Core"},
    # "middle_outer": {"rows": range(12, 21), "cols": range(9, 18), "name": "Middle Level Outer Area"},
    # "lower_inner": {"rows": range(21, 27), "cols": range(0, 9), "name": "Lower Level Inner-Core"},
    # "lower_outer": {"rows": range(21, 27), "cols": range(9, 18), "name": "Lower Level Outer Area"},
    # "midlower_inner": {"rows": range(12, 27), "cols": range(0, 9), "name": "Mid-Lower Level Inner-Core"},
    # "midlower_outer": {"rows": range(12, 27), "cols": range(9, 18), "name": "Mid-Lower Level Outer Area"}
}
# RI 計算相關參數
c_candidates = [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300]
d_candidates = [5, 10, 15, 20, 25, 30]
max_wind = max(wind_thresholds)
rate_threshold = 0.417
focus_pairs = [(10, 6), (20, 12), (35, 18), (45, 24), (55, 30), (65, 36), (70, 42), (75, 48)]
# 數據調整函數
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
# 載入所有組合的數據並記錄 (wind, time) 及計算 delta_winds
def load_all_data(wind_thresholds, time_intervals, base_path, regions, best_c, best_d, max_files=None):
    X_all, y_all, combo_labels, delta_winds_list = [], [], [], []
    bt_cache = {} # 緩存 best_track_data
    delta_cache = {} # 緩存 delta_winds 和 wind_current
    for region in regions.keys():
        region_rows = regions[region]["rows"]
        region_cols = regions[region]["cols"]
        for wind in wind_thresholds:
            for time in time_intervals:
                ri_value = calculate_ri(wind, time, best_c, best_d, max_wind)
                pv_path_ri = os.path.join(base_path, "Azi_PV", "Individual", f"Azi_PV-RI-{wind}-{time:02d}", "removenan")
                the_path_ri = os.path.join(base_path, "Azi_THE", "Individual", f"Azi_THE-RI-{wind}-{time:02d}", "removenan")
                if os.path.exists(pv_path_ri) and os.path.exists(the_path_ri):
                    sid_files = defaultdict(list) # {sid: list of (dt, basename)}
                    pv_files_ri = [os.path.join(pv_path_ri, f) for f in os.listdir(pv_path_ri) if f.endswith(".txt")]
                    for pv_file in pv_files_ri:
                        basename = os.path.basename(pv_file)
                        parts = basename.replace('.txt', '').split('-')
                        if len(parts) != 3 or parts[0] != 'Azi_PV':
                            logging.warning(f"無效檔案名格式: {basename}")
                            continue
                        sid = parts[2]
                        timestr = parts[1]
                        if len(timestr) != 10:
                            logging.warning(f"無效時間字符串: {timestr}")
                            continue
                        try:
                            year = int(timestr[0:4])
                            month = int(timestr[4:6])
                            day = int(timestr[6:8])
                            hour = int(timestr[8:10])
                            file_dt = datetime(year, month, day, hour, 0, 0)
                            sid_files[sid].append((file_dt, basename))
                        except ValueError:
                            logging.warning(f"無效時間格式: {timestr}")
                            continue
                    # 對於每個 sid，取最早的檔案
                    for sid, files_list in sid_files.items():
                        if not files_list:
                            continue
                        files_list.sort(key=lambda x: x[0])
                        first_entry = files_list[0]
                        first_dt, basename = first_entry
                        pv_file = os.path.join(pv_path_ri, basename)
                        the_basename = basename.replace('Azi_PV', 'Azi_THE')
                        the_file = os.path.join(the_path_ri, the_basename)
                        if not os.path.exists(the_file):
                            logging.warning(f"對應 the 檔案不存在 for {the_basename}")
                            continue
                        try:
                            pv_data = np.loadtxt(pv_file)
                            the_data = np.loadtxt(the_file)
                            pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                            the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                            if pv_selected.size == 0 or the_selected.size == 0:
                                continue
                            combined_features = np.concatenate([pv_selected, the_selected])
                            timestr = basename.split('-')[1]
                            year = int(timestr[0:4])
                            month = int(timestr[4:6])
                            day = int(timestr[6:8])
                            hour = int(timestr[8:10])
                            current_dt = datetime(year, month, day, hour, 0, 0)
                            cache_key = (sid, str(current_dt))
                            if cache_key in delta_cache:
                                wind_current, delta_winds = delta_cache[cache_key]
                                logging.info(f"使用緩存 delta_winds: {timestr}_{sid}")
                            else:
                                files = os.listdir(base_path)
                                matching_files = [f for f in files if f.endswith('.txt') and f.startswith('JTWC-') and str(year) in f and sid in f]
                                if len(matching_files) != 1:
                                    logging.warning(f"未找到或多個 best track 檔案 for year {year}, SID {sid}: {matching_files}")
                                    continue
                                best_track_file = matching_files[0]
                                best_track_path = os.path.join(base_path, best_track_file)
                                if not os.path.exists(best_track_path):
                                    logging.warning(f"Best track 檔案不存在: {best_track_path}")
                                    continue
                                if best_track_path in bt_cache:
                                    bt_data = bt_cache[best_track_path]
                                    logging.info(f"使用緩存 best track 數據: {best_track_path}")
                                else:
                                    bt_data = pd.read_csv(best_track_path, sep='\s+', header=None,
                                                          names=['sid', 'year', 'month', 'day', 'hour', 'lat', 'lon', 'wind', 'pressure', 'source'])
                                    bt_data['datetime'] = pd.to_datetime(bt_data[['year', 'month', 'day', 'hour']])
                                    bt_cache[best_track_path] = bt_data
                                if current_dt not in bt_data['datetime'].values:
                                    closest_idx = bt_data['datetime'].sub(current_dt).abs().idxmin()
                                    current_row = bt_data.iloc[closest_idx]
                                    logging.warning(f"當前時間不在 best track 中: {current_dt}, 使用最近時間: {current_row['datetime']}")
                                else:
                                    current_row = bt_data[bt_data['datetime'] == current_dt].iloc[0]
                                wind_current = current_row['wind']
                                delta_winds = []
                                for t in time_intervals:
                                    future_dt = current_dt + timedelta(hours=t)
                                    if future_dt not in bt_data['datetime'].values:
                                        closest_idx = bt_data['datetime'].sub(future_dt).abs().idxmin()
                                        future_row = bt_data.iloc[closest_idx]
                                        logging.warning(f"未來時間不在 best track 中: {future_dt}, 使用最近時間: {future_row['datetime']}")
                                    else:
                                        future_row = bt_data[bt_data['datetime'] == future_dt].iloc[0]
                                    wind_future = future_row['wind']
                                    delta = wind_future - wind_current
                                    delta_winds.append(delta)
                                delta_cache[cache_key] = (wind_current, delta_winds)
                            X_all.append(combined_features)
                            y_all.append(ri_value)
                            combo_labels.append((wind, time))
                            delta_winds_list.append(delta_winds)
                        except Exception as e:
                            logging.error(f"處理 RI 檔案時出錯 {pv_file}, {the_file}: {e}")
                            continue
    return np.array(X_all), np.array(y_all), combo_labels, np.array(delta_winds_list)
# 主函數
def main():
    logging.info("程式開始執行")
    best_c, best_d = find_best_c_d(wind_thresholds, time_intervals, focus_pairs, max_wind)
    region = "upper_inner"
    logging.info(f"開始分析區域: {region}")
    X_all, y_all, combo_labels, delta_winds_list = load_all_data(wind_thresholds, time_intervals, base_path, {region: regions[region]}, best_c, best_d)
    if len(X_all) < 10:
        logging.warning(f"區域 {region}: 數據不足")
        return
    stratify_labels = [f"{wind}_{time}" for wind, time in combo_labels]
    
    # 新增: 處理樣本數 < 2 的類別
    label_counts = Counter(stratify_labels)
    # 收集只有 1 個樣本的類別，將它們全放進訓練集
    single_sample_indices = [i for i, label in enumerate(stratify_labels) if label_counts[label] == 1]
    multi_sample_indices = [i for i, label in enumerate(stratify_labels) if label_counts[label] >= 2]
    
    if single_sample_indices:
        logging.info(f"發現 {len(single_sample_indices)} 個單一樣本類別，將它們全部分配到訓練集")
    
    # 對多樣本類別進行 stratified split
    if multi_sample_indices:
        X_multi = X_all[multi_sample_indices]
        y_multi = y_all[multi_sample_indices]
        combo_multi = [combo_labels[i] for i in multi_sample_indices]
        delta_multi = [delta_winds_list[i] for i in multi_sample_indices]
        stratify_multi = [stratify_labels[i] for i in multi_sample_indices]
        
        X_df_multi = pd.DataFrame(X_multi)
        X_train_multi, X_test_multi, y_train_multi, y_test_multi, combo_train_multi, combo_test_multi = train_test_split(
            X_df_multi, y_multi, combo_multi, test_size=0.2, random_state=42, stratify=stratify_multi
        )
        test_indices_multi = X_df_multi.index[X_df_multi.index.isin(X_test_multi.index)].tolist()
        delta_winds_test_multi = np.array(delta_multi)[test_indices_multi]
    else:
        logging.warning("無多樣本類別，無法進行 stratified split")
        return
    
    # 處理單一樣本：全放訓練集
    if single_sample_indices:
        X_single = X_all[single_sample_indices]
        y_single = y_all[single_sample_indices]
        combo_single = [combo_labels[i] for i in single_sample_indices]
        delta_single = [delta_winds_list[i] for i in single_sample_indices]
        
        # 合併到訓練集
        X_train = np.concatenate([X_train_multi, X_single])
        y_train = np.concatenate([y_train_multi, y_single])
        combo_train = combo_train_multi + combo_single
        # 測試集保持原樣（無單一樣本）
        X_test = X_test_multi
        y_test = y_test_multi
        combo_test = combo_test_multi
        delta_winds_test = delta_winds_test_multi
    else:
        X_train = X_train_multi
        y_train = y_train_multi
        combo_train = combo_train_multi
        X_test = X_test_multi
        y_test = y_test_multi
        combo_test = combo_test_multi
        delta_winds_test = delta_winds_test_multi
    
    # 儲存測試集（delta_winds_test 已調整）
    X_test_path = os.path.join(output_dir, f"PVTHE_X_test_rf_kfold_{region}_{year_range}.pkl")
    y_test_path = os.path.join(output_dir, f"PVTHE_y_test_rf_kfold_{region}_{year_range}.pkl")
    combo_test_path = os.path.join(output_dir, f"PVTHE_combo_test_rf_kfold_{region}_{year_range}.pkl")
    delta_winds_path = os.path.join(output_dir, f"PVTHE_delta_winds_rf_kfold_{region}_{year_range}.pkl")
    joblib.dump(X_test, X_test_path)
    joblib.dump(y_test, y_test_path)
    joblib.dump(combo_test, combo_test_path)
    joblib.dump(delta_winds_test, delta_winds_path)
    logging.info(f"已儲存測試集資料至: {X_test_path}, {y_test_path}, {combo_test_path}, {delta_winds_path}")
    scaler_y = MinMaxScaler(feature_range=(0, 10))
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    stratify_train_labels = [f"{wind}_{time}" for wind, time in combo_train]
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {'mse': [], 'r2': [], 'ri_mean': []}
    fold_best_params = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, stratify_train_labels)):
        logging.info(f"開始第 {fold+1} fold")
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train_scaled[train_idx], y_train_scaled[val_idx]
        param_grid = {
            'n_estimators': [100, 500],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train_fold, y_train_fold)
        best_rf = grid_search.best_estimator_
        best_params = grid_search.best_params_
        fold_best_params.append(best_params)
        logging.info(f"Fold {fold+1} - 最佳參數: {best_params}")
        y_val_pred = best_rf.predict(X_val_fold)
        y_val_pred = np.clip(y_val_pred, 0, 10)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        r2 = r2_score(y_val_fold, y_val_pred)
        ri_mean = np.mean(y_val_pred)
        cv_results['mse'].append(mse)
        cv_results['r2'].append(r2)
        cv_results['ri_mean'].append(ri_mean)
        logging.info(f"Fold {fold+1}: MSE={mse:.4f}, R2={r2:.4f}, RI_mean={ri_mean:.4f}")
    avg_mse = np.mean(cv_results['mse'])
    std_mse = np.std(cv_results['mse'])
    avg_r2 = np.mean(cv_results['r2'])
    std_r2 = np.std(cv_results['r2'])
    avg_ri = np.mean(cv_results['ri_mean'])
    std_ri = np.std(cv_results['ri_mean'])
    logging.info(f"K-fold CV 摘要 (5 folds) - 平均 MSE: {avg_mse:.4f} (+/- {std_mse:.4f})")
    logging.info(f"平均 R2: {avg_r2:.4f} (+/- {std_r2:.4f})")
    logging.info(f"平均 RI Index: {avg_ri:.4f} (+/- {std_ri:.4f})")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(range(1, n_splits+1), cv_results['mse'])
    axes[0].axhline(y=avg_mse, color='r', linestyle='--', label=f'Mean: {avg_mse:.4f}')
    axes[0].set_title('MSE per Fold')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[1].bar(range(1, n_splits+1), cv_results['r2'])
    axes[1].axhline(y=avg_r2, color='r', linestyle='--', label=f'Mean: {avg_r2:.4f}')
    axes[1].set_title('R² per Fold')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('R²')
    axes[1].legend()
    axes[2].bar(range(1, n_splits+1), cv_results['ri_mean'])
    axes[2].axhline(y=avg_ri, color='r', linestyle='--', label=f'Mean: {avg_ri:.4f}')
    axes[2].set_title('RI Index Mean per Fold')
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('RI Mean')
    axes[2].legend()
    plt.tight_layout()
    cv_plot_path = os.path.join(output_dir, f"RF_Kfold_CV_results_{region}_{year_range}.png")
    plt.savefig(cv_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"CV 結果圖表已儲存至 {cv_plot_path}")
    best_params_dict = {}
    for k in fold_best_params[0].keys():
        values = [p[k] for p in fold_best_params]
        mode_val = max(set(values), key=values.count)
        best_params_dict[k] = mode_val
    logging.info(f"最終模型參數 (mode from folds): {best_params_dict}")
    rf_final = RandomForestRegressor(**best_params_dict, random_state=42)
    rf_final.fit(X_train_scaled, y_train_scaled)
    model_path = os.path.join(output_dir, f"PVTHE_rf_model_kfold_{region}_{year_range}.pkl")
    scaler_path = os.path.join(output_dir, f"PVTHE_rf_feature_scaler_kfold_{region}_{year_range}.pkl")
    scaler_y_path = os.path.join(output_dir, f"PVTHE_rf_target_scaler_kfold_{region}_{year_range}.pkl")
    joblib.dump(rf_final, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(scaler_y, scaler_y_path)
    logging.info(f"已儲存 Random Forest 模型至 {model_path}")
    logging.info(f"已儲存特徵縮放器至 {scaler_path}")
    logging.info(f"已儲存目標縮放器至 {scaler_y_path}")
    y_test_pred = rf_final.predict(X_test_scaled)
    y_test_pred = np.clip(y_test_pred, 0, 10)
    ri_index = np.mean(y_test_pred)
    mse = mean_squared_error(y_test_scaled, y_test_pred)
    logging.info(f"區域 {region} - RI Index: {ri_index:.2f}, MSE: {mse:.2f}")
    test_results = pd.DataFrame({
        'y_true': y_test_scaled,
        'y_pred': y_test_pred,
        'wind': [combo[0] for combo in combo_test],
        'time': [combo[1] for combo in combo_test]
    })
    ri_index_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
    mse_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
    diff_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan) # 新增 diff_matrix
    for wind_idx, wind in enumerate(wind_thresholds):
        for time_idx, time in enumerate(time_intervals):
            combo_data = test_results[(test_results['wind'] == wind) & (test_results['time'] == time)]
            if len(combo_data) > 0:
                y_true_combo = combo_data['y_true']
                y_pred_combo = combo_data['y_pred']
                ri_index = np.mean(y_pred_combo)
                mse = mean_squared_error(y_true_combo, y_pred_combo)
                # 計算真實 RI metric（縮放後）
                ri_metric = calculate_ri(wind, time, best_c, best_d, max_wind)
                ri_metric_scaled = scaler_y.transform([[ri_metric]])[0][0]
                # 計算差異
                difference = ri_index - ri_metric_scaled
                ri_index_matrix[wind_idx, time_idx] = ri_index
                mse_matrix[wind_idx, time_idx] = mse
                diff_matrix[wind_idx, time_idx] = difference
            else:
                logging.info(f"區域 {region}, 風速 {wind} kt, 時間 {time} h: 無測試數據")
    # ri_index_values = ri_index_matrix[~np.isnan(ri_index_matrix)]
    # ri_index_mean = np.mean(ri_index_values)
    # ri_index_std = np.std(ri_index_values)
    # high_threshold = ri_index_mean + ri_index_std
    # low_threshold = ri_index_mean - ri_index_std
    # annot_matrix = np.array([[f"{ri_index_matrix[i, j]:.2f}\n({mse_matrix[i, j]:.2f})"
    annot_matrix = np.array([[f"{ri_index_matrix[i, j]:.2f}, {diff_matrix[i, j]:.2f} ({mse_matrix[i, j]:.2f})"
                             if not np.isnan(ri_index_matrix[i, j]) else ""
                             for j in range(len(time_intervals))]
                            for i in range(len(wind_thresholds))])
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(ri_index_matrix[::-1], xticklabels=time_intervals, yticklabels=wind_thresholds[::-1],
                     cmap="YlOrRd", annot=annot_matrix[::-1], fmt="", linewidths=0.5, linecolor='black',
                     cbar_kws={'label': f"Assessed RI Index - {region}"}, annot_kws={"size": 8},
                     vmin=0, vmax=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(2)
    plt.title(f"Assessed RI Index by Random Forest (MSE) - {region} ({year_range})")
    plt.xlabel("Time Interval (h)")
    plt.ylabel("Wind Speed Increment Threshold (kt)")
    # threshold_text = f"High RI index > {high_threshold:.2f}\nLow RI index < {low_threshold:.2f}"
    # plt.text(0.5, -0.1, threshold_text, ha='center', va='top', transform=ax.transAxes, fontsize=10)
    heatmap_path = os.path.join(output_dir, f"PV_THE_ANO_ri_index_heatmap_rf_kfold_{year_range}_{region}.png")
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()
    logging.info(f"區域 {region} - 熱圖已儲存至 {heatmap_path}")
if __name__ == "__main__":
    try:
        main()
        logging.info("程式執行完畢")
    except Exception as e:
        logging.exception("程式執行時發生嚴重錯誤：")
    finally:
        logging.shutdown()