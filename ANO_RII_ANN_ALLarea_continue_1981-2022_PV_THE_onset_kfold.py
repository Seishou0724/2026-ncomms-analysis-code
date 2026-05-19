import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# ====================== 基本設定 ======================
year_range = "1981-2022"
output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{year_range}-continue"
os.makedirs(output_dir, exist_ok=True)

logging.getLogger('matplotlib').setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)

log_file_path = os.path.join(output_dir, f"ANO_RII_ann_continue_{year_range}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, delay=False), logging.StreamHandler()],
    force=True
)

base_path = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{year_range}"
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]

variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
}

regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
    # "upper_outer": {"rows": range(2, 12), "cols": range(9, 18), "name": "Upper Level Outer Area"},
    # "middle_inner": {"rows": range(12, 21), "cols": range(0, 9), "name": "Middle Level Inner-Core"},
    # "middle_outer": {"rows": range(12, 21), "cols": range(9, 18), "name": "Middle Level Outer Area"},
    #"lower_inner": {"rows": range(21, 27), "cols": range(0, 9), "name": "Lower Level Inner-Core"},
    # "lower_outer": {"rows": range(21, 27), "cols": range(9, 18), "name": "Lower Level Outer Area"},
    # "midlower_inner": {"rows": range(12, 27), "cols": range(0, 9), "name": "Mid-Lower Level Inner-Core"},
    # "midlower_outer": {"rows": range(12, 27), "cols": range(9, 18), "name": "Mid-Lower Level Outer Area"}
}

c_candidates = [50, 100, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300]
d_candidates = [5, 10, 15, 20, 25, 30]
max_wind = max(wind_thresholds)
rate_threshold = 0.417
focus_pairs = [(10, 6), (20, 12), (35, 18), (45, 24), (55, 30), (65, 36), (70, 42), (75, 48)]

# ====================== 函數（完全不變） ======================
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

def calculate_ri(wind, time, c, d, max_wind):
    return wind * (c / (time + d)) * (1 + wind / max_wind)

def generate_low_pairs(wind_values, time_values, threshold):
    low_pairs = [(w, t) for w in wind_values for t in time_values if w / t < threshold]
    low_pairs.sort(key=lambda x: x[0] / x[1])
    return low_pairs[:10]

def calculate_initial_targets(wind_values, time_values, focus_pairs, max_wind, c_init=200, d_init=15):
    low_pairs = generate_low_pairs(wind_values, time_values, rate_threshold)
    ri_focus_init = [calculate_ri(w, t, c_init, d_init, max_wind) for w, t in focus_pairs]
    ri_low_init = [calculate_ri(w, t, c_init, d_init, max_wind) for w, t in low_pairs]
    target_focus = np.mean(ri_focus_init)
    target_low = np.mean(ri_low_init)
    return target_focus, target_low

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

# ====================== load_all_data（使用你最新成功的版本，完全保留） ======================
def load_all_data(wind_thresholds, time_intervals, base_path, regions, best_c, best_d,
                  train_sids=None, test_sids=None, collect_unique_sids=False):
    X_all, y_all, combo_labels, delta_winds_list, sid_list = [], [], [], [], []
    bt_cache = {}
    delta_cache = {}
    all_sids = set()
    for region in regions.keys():
        region_rows = regions[region]["rows"]
        region_cols = regions[region]["cols"]
        for wind in wind_thresholds:
            for time in time_intervals:
                ri_value = calculate_ri(wind, time, best_c, best_d, max_wind)
                pv_path_ri = os.path.join(base_path, "Azi_PV", "Individual", f"Azi_PV-RI-{wind}-{time:02d}", "removenan")
                the_path_ri = os.path.join(base_path, "Azi_THE", "Individual", f"Azi_THE-RI-{wind}-{time:02d}", "removenan")
               
                if not (os.path.exists(pv_path_ri) and os.path.exists(the_path_ri)):
                    continue
                sid_files = defaultdict(list)
                pv_files_ri = [os.path.join(pv_path_ri, f) for f in os.listdir(pv_path_ri) if f.endswith(".txt")]
                for pv_file in pv_files_ri:
                    basename = os.path.basename(pv_file)
                    parts = basename.replace('.txt', '').split('-')
                    if len(parts) != 3 or parts[0] != 'Azi_PV':
                        continue
                    sid = parts[2]
                    timestr = parts[1]
                    try:
                        year = int(timestr[0:4])
                        month = int(timestr[4:6])
                        day = int(timestr[6:8])
                        hour = int(timestr[8:10])
                        file_dt = datetime(year, month, day, hour, 0, 0)
                        sid_files[sid].append((file_dt, basename))
                        all_sids.add(sid)
                    except ValueError:
                        continue
                logging.info(f"組合 ({wind},{time}) 找到 {len(sid_files)} 個 unique TC")
                success_count = 0
                skipped_count = 0
                skip_reasons = {"THE不存在": 0, "adjust_data空": 0, "BestTrack不符": 0, "其他錯誤": 0}
                for sid, files_list in sid_files.items():
                    if not files_list:
                        continue
                    files_list.sort(key=lambda x: x[0])
                    loaded = False
                    for file_dt, basename in files_list:
                        if loaded:
                            break
                        pv_file = os.path.join(pv_path_ri, basename)
                        the_basename = basename.replace('Azi_PV', 'Azi_THE')
                        the_file = os.path.join(the_path_ri, the_basename)
                        if not os.path.exists(the_file):
                            skipped_count += 1
                            skip_reasons["THE不存在"] += 1
                            continue
                        try:
                            pv_data = np.loadtxt(pv_file)
                            the_data = np.loadtxt(the_file)
                            pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                            the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                            if pv_selected.size == 0 or the_selected.size == 0:
                                skipped_count += 1
                                skip_reasons["adjust_data空"] += 1
                                continue
                            combined_features = np.concatenate([pv_selected, the_selected])
                            current_dt = file_dt
                            cache_key = (sid, str(current_dt))
                            if cache_key in delta_cache:
                                wind_current, delta_winds = delta_cache[cache_key]
                            else:
                                files = os.listdir(base_path)
                                candidate_files = [f for f in files if f.endswith('.txt') and 'JTWC' in f and sid in f]
                                best_track_file = None
                                matched_dt = None
                                for cand in candidate_files:
                                    cand_path = os.path.join(base_path, cand)
                                    try:
                                        bt_data = pd.read_csv(cand_path, sep=r'\s+', header=None,
                                                              names=['sid', 'year', 'month', 'day', 'hour', 'lat', 'lon', 'wind', 'pressure', 'source'],
                                                              engine='python')
                                        bt_data['datetime'] = pd.to_datetime(bt_data[['year','month','day','hour']])
                                        exact_match = bt_data[bt_data['datetime'] == current_dt]
                                        if not exact_match.empty:
                                            best_track_file = cand
                                            matched_dt = current_dt
                                            break
                                        time_diff = (bt_data['datetime'] - current_dt).abs()
                                        if time_diff.min() <= pd.Timedelta(hours=3):
                                            closest_idx = time_diff.idxmin()
                                            best_track_file = cand
                                            matched_dt = bt_data['datetime'].iloc[closest_idx]
                                            break
                                    except Exception:
                                        continue
                                if best_track_file is None:
                                    skipped_count += 1
                                    skip_reasons["BestTrack不符"] += 1
                                    continue
                                best_track_path = os.path.join(base_path, best_track_file)
                                if best_track_path in bt_cache:
                                    bt_data = bt_cache[best_track_path]
                                else:
                                    bt_data = pd.read_csv(best_track_path, sep=r'\s+', header=None,
                                                          names=['sid', 'year', 'month', 'day', 'hour', 'lat', 'lon', 'wind', 'pressure', 'source'],
                                                          engine='python')
                                    bt_data['datetime'] = pd.to_datetime(bt_data[['year','month','day','hour']])
                                    bt_cache[best_track_path] = bt_data
                                if matched_dt in bt_data['datetime'].values:
                                    current_row = bt_data[bt_data['datetime'] == matched_dt].iloc[0]
                                else:
                                    closest_idx = (bt_data['datetime'] - current_dt).abs().idxmin()
                                    current_row = bt_data.iloc[closest_idx]
                                wind_current = float(current_row['wind'])
                                delta_winds = []
                                for t in time_intervals:
                                    future_dt = current_dt + timedelta(hours=t)
                                    future_idx = (bt_data['datetime'] - future_dt).abs().idxmin()
                                    future_row = bt_data.iloc[future_idx]
                                    delta = float(future_row['wind']) - wind_current
                                    delta_winds.append(delta)
                                delta_cache[cache_key] = (wind_current, delta_winds)
                            X_all.append(combined_features)
                            y_all.append(ri_value)
                            combo_labels.append((wind, time))
                            delta_winds_list.append(delta_winds)
                            sid_list.append(sid)
                            success_count += 1
                            loaded = True
                        except Exception as e:
                            skipped_count += 1
                            skip_reasons["其他錯誤"] += 1
                            continue
                logging.info(f" → 【總結】成功 {success_count} 筆 | 跳過 {skipped_count} 筆")
                for reason, cnt in skip_reasons.items():
                    if cnt > 0:
                        logging.info(f" 跳過原因: {reason} = {cnt} 筆")
    if not X_all:
        logging.warning("load_all_data 沒有任何成功資料")
        return np.array([]), np.array([]), [], np.array([]), [], set()
    return (np.array(X_all), np.array(y_all), combo_labels,
            np.array(delta_winds_list), sid_list, all_sids)

# ====================== 主函數（ANN 版本，完全保留你提供的超參數） ======================
def main():
    logging.info("程式開始執行 - ANN (MLPRegressor) - 每個 (wind,time) 組合獨立 80/20 split")
    best_c, best_d = find_best_c_d(wind_thresholds, time_intervals, focus_pairs, max_wind)
    region = "upper_inner"
    logging.info(f"開始分析區域: {region}")
    X_train_all = []
    y_train_all = []
    combo_train_all = []
    delta_train_all = []
    X_test_all = []
    y_test_all = []
    combo_test_all = []
    delta_test_all = []
    for wind in wind_thresholds:
        for time in time_intervals:
            logging.info(f"處理組合 wind={wind} kt, time={time} h")
           
            X_combo, y_combo, combo_combo, delta_combo, sid_combo, _ = load_all_data(
                [wind], [time], base_path, {region: regions[region]},
                best_c, best_d, collect_unique_sids=True
            )
           
            logging.info(f" → 該組合共有 {len(sid_combo)} 筆第一筆資料，{len(set(sid_combo))} 個 unique TC")
           
            if len(sid_combo) == 0:
                continue
            n_samples = len(sid_combo)
            if n_samples < 2:
                train_mask = np.ones(n_samples, dtype=bool)
            else:
                np.random.seed(42)
                test_size = max(1, int(0.2 * n_samples))
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                test_idx = indices[:test_size]
                train_idx = indices[test_size:]
                train_mask = np.zeros(n_samples, dtype=bool)
                train_mask[train_idx] = True
            test_count = np.sum(~train_mask)
            logging.info(f" → Test Set 實際分配 {test_count} 筆資料 (總共 {n_samples} 筆)")
            X_train_all.extend(X_combo[train_mask])
            y_train_all.extend(y_combo[train_mask])
            combo_train_all.extend([combo_combo[i] for i in range(len(combo_combo)) if train_mask[i]])
            delta_train_all.extend(delta_combo[train_mask])
            X_test_all.extend(X_combo[~train_mask])
            y_test_all.extend(y_combo[~train_mask])
            combo_test_all.extend([combo_combo[i] for i in range(len(combo_combo)) if not train_mask[i]])
            delta_test_all.extend(delta_combo[~train_mask])
    X_train = np.array(X_train_all)
    y_train = np.array(y_train_all)
    combo_train = combo_train_all
    delta_train = np.array(delta_train_all)
    X_test = np.array(X_test_all)
    y_test = np.array(y_test_all)
    combo_test = combo_test_all
    delta_test = np.array(delta_test_all)
    logging.info(f"最終樣本數 → Train: {len(X_train)} | Test: {len(X_test)}")
    if len(X_train) == 0:
        logging.error("Training Set 為空！")
        return
    # ====================== 縮放 ======================
    scaler_y = MinMaxScaler(feature_range=(0, 10))
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # ====================== ANN + GridSearchCV（超參數完全保留） ======================
    if len(X_train) < 5:
        logging.warning("樣本太少，使用簡單訓練")
        ann_final = MLPRegressor(hidden_layer_sizes=(100,), activation='relu',
                                 solver='adam', alpha=0.001, max_iter=500,
                                 early_stopping=True, validation_fraction=0.1, random_state=42)
        ann_final.fit(X_train_scaled, y_train_scaled)
    else:
        skf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {'mse': [], 'r2': [], 'ri_mean': []}
        fold_best_params = []
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.005, 0.01]
        }
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled)):
            logging.info(f"開始第 {fold+1} fold")
            X_train_fold = X_train_scaled[train_idx]
            X_val_fold = X_train_scaled[val_idx]
            y_train_fold = y_train_scaled[train_idx]
            y_val_fold = y_train_scaled[val_idx]
            ann = MLPRegressor(solver='adam', max_iter=500,
                               early_stopping=True, validation_fraction=0.1, random_state=42)
           
            grid_search = GridSearchCV(ann, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train_fold, y_train_fold)
            best_ann = grid_search.best_estimator_
            fold_best_params.append(grid_search.best_params_)
            y_val_pred = best_ann.predict(X_val_fold)
            y_val_pred = np.clip(y_val_pred, 0, 10)
            mse = mean_squared_error(y_val_fold, y_val_pred)
            r2 = r2_score(y_val_fold, y_val_pred)
            cv_results['mse'].append(mse)
            cv_results['r2'].append(r2)
            logging.info(f"Fold {fold+1}: MSE={mse:.4f}, R2={r2:.4f}")
        # 最終模型
        best_params_dict = {}
        for k in fold_best_params[0].keys():
            values = [p[k] for p in fold_best_params]
            best_params_dict[k] = max(set(values), key=values.count)
        ann_final = MLPRegressor(**best_params_dict, solver='adam', max_iter=500,
                                 early_stopping=True, validation_fraction=0.1, random_state=42)
        ann_final.fit(X_train_scaled, y_train_scaled)
    # Test 評估
    y_test_pred = ann_final.predict(X_test_scaled)
    y_test_pred = np.clip(y_test_pred, 0, 10)
    final_mse = mean_squared_error(y_test_scaled, y_test_pred)
    final_r2 = r2_score(y_test_scaled, y_test_pred)
    logging.info(f"Test Set 最終驗證 - MSE: {final_mse:.4f}, R2: {final_r2:.4f}")
    # ====================== 儲存 ======================
    test_prefix = "PVTHE"
    joblib.dump(X_test, os.path.join(output_dir, f"{test_prefix}_X_test_ann_onset_{region}_{year_range}.pkl"))
    joblib.dump(y_test, os.path.join(output_dir, f"{test_prefix}_y_test_ann_onset_{region}_{year_range}.pkl"))
    joblib.dump(combo_test, os.path.join(output_dir, f"{test_prefix}_combo_test_ann_onset_{region}_{year_range}.pkl"))
    joblib.dump(delta_test, os.path.join(output_dir, f"{test_prefix}_delta_winds_ann_onset_{region}_{year_range}.pkl"))
    model_path = os.path.join(output_dir, f"PVTHE_ann_model_onset_{region}_{year_range}.pkl")
    scaler_path = os.path.join(output_dir, f"PVTHE_ann_feature_scaler_onset_{region}_{year_range}.pkl")
    scaler_y_path = os.path.join(output_dir, f"PVTHE_ann_target_scaler_onset_{region}_{year_range}.pkl")
    joblib.dump(ann_final, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(scaler_y, scaler_y_path)
    logging.info(f"ANN 模型已儲存至 {model_path}")
    logging.info("程式執行完畢 - ANN 版本")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("程式執行時發生嚴重錯誤：")
    finally:
        logging.shutdown()