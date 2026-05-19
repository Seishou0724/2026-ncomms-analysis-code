import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

# ====================== 設定 ======================
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/Temporal_Stability_Analysis"
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "temporal_stability_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
        logging.StreamHandler()
    ]
)
logging.info("=== 跨年代穩定性分析開始執行 ===")

# ====================== 參數 ======================
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"}
}
periods = ["1959-1980", "1981-2012", "2013-2022"]
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/2013-2022-continue"
best_c = 210
best_d = 25
max_wind = max(wind_thresholds)

# ====================== 模型與縮放器路徑 ======================
model_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_model_kfold_upper_inner_2013-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_model_kfold_upper_inner_2013-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_model_kfold_upper_inner_2013-2022.pkl")
}
feature_scaler_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_feature_scaler_kfold_upper_inner_2013-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_feature_scaler_kfold_upper_inner_2013-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_feature_scaler_kfold_upper_inner_2013-2022.pkl")
}
target_scaler_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_target_scaler_kfold_upper_inner_2013-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_target_scaler_kfold_upper_inner_2013-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_target_scaler_kfold_upper_inner_2013-2022.pkl")
}

# ====================== 載入模型 ======================
logging.info("開始載入模型和縮放器...")
models = {}
scalers = {"feature": {}, "target": {}}
for name in ["random_forest", "svr", "ann"]:
    try:
        logging.info(f"正在載入 {name} ...")
        models[name] = joblib.load(model_files[name])
        scalers["feature"][name] = joblib.load(feature_scaler_files[name])
        scalers["target"][name] = joblib.load(target_scaler_files[name])
        logging.info(f"{name.upper()} 載入成功")
    except Exception as e:
        logging.error(f"{name.upper()} 載入失敗: {e}")

# ====================== 載入 2013-2022 的測試集 ======================
def load_2013_test_set():
    test_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/2013-2022-continue"
    X_test = joblib.load(os.path.join(test_dir, "PVTHE_X_test_rf_kfold_upper_inner_2013-2022.pkl"))
    logging.info(f"已成功載入 2013-2022 測試集，shape: {X_test.shape}")
    return X_test

# ====================== 函數（移到最前面） ======================
def calculate_ri(wind, time, c, d, max_wind):
    return wind * (c / (time + d)) * (1 + wind / max_wind)

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

# ====================== 【已修正】載入測試資料 ======================
def load_test_data(wind_thresholds, time_intervals, test_base_path, regions, best_c, best_d):
    X_test, y_test, combo_labels, delta_winds_list = [], [], [], []
    region = "upper_inner"
    if region not in regions:
        logging.error(f"區域 {region} 未定義")
        return np.array([]), np.array([]), [], []
    region_rows = regions[region]["rows"]
    region_cols = regions[region]["cols"]
    logging.info(f"開始載入測試資料，區域: {regions[region]['name']}")
    bt_cache = {}
    delta_cache = {}
    for wind in wind_thresholds:
        for time in time_intervals:
            ri_value = calculate_ri(wind, time, best_c, best_d, max_wind)
            pv_path_ri = os.path.join(test_base_path, "Azi_PV", "Individual", f"Azi_PV-RI-{wind}-{time:02d}", "removenan")
            the_path_ri = os.path.join(test_base_path, "Azi_THE", "Individual", f"Azi_THE-RI-{wind}-{time:02d}", "removenan")
            if os.path.exists(pv_path_ri) and os.path.exists(the_path_ri):
                pv_files_ri = [os.path.join(pv_path_ri, f) for f in os.listdir(pv_path_ri) if f.endswith(".txt")]
                the_files_ri = [os.path.join(the_path_ri, f) for f in os.listdir(the_path_ri) if f.endswith(".txt")]
                min_files_ri = min(len(pv_files_ri), len(the_files_ri))
                if min_files_ri == 0:
                    continue
                for pv_file, the_file in zip(pv_files_ri[:min_files_ri], the_files_ri[:min_files_ri]):
                    try:
                        pv_data = np.loadtxt(pv_file)
                        the_data = np.loadtxt(the_file)
                        pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                        the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                        if pv_selected.size == 0 or the_selected.size == 0:
                            continue
                        combined_features = np.concatenate([pv_selected, the_selected])
                        f = os.path.basename(pv_file)
                        parts = f.replace('.txt', '').split('-')
                        if len(parts) != 3 or parts[0] != 'Azi_PV':
                            continue
                        sid = parts[2]
                        timestr = parts[1]
                        if len(timestr) != 10:
                            continue
                        year = int(timestr[0:4])
                        month = int(timestr[4:6])
                        day = int(timestr[6:8])
                        hour = int(timestr[8:10])
                        current_dt = datetime(year, month, day, hour, 0, 0)
                        cache_key = (sid, str(current_dt))
                        if cache_key in delta_cache:
                            wind_current, delta_winds = delta_cache[cache_key]
                        else:
                            files = os.listdir(test_base_path)
                            matching_files = [f for f in files if f.endswith('.txt') and f.startswith('JTWC-') and sid in f]
                            # === 修正重點：處理多檔案 ===
                            if len(matching_files) > 1:
                                year_str = str(current_dt.year)
                                matching_files = [f for f in matching_files if year_str in f]
                            if len(matching_files) != 1:
                                continue
                            best_track_path = os.path.join(test_base_path, matching_files[0])
                            if best_track_path not in bt_cache:
                                bt_data = pd.read_csv(best_track_path, sep='\s+', header=None,
                                                      names=['sid', 'year', 'month', 'day', 'hour', 'lat', 'lon', 'wind', 'pressure', 'source'])
                                bt_data['datetime'] = pd.to_datetime(bt_data[['year', 'month', 'day', 'hour']])
                                bt_cache[best_track_path] = bt_data
                            else:
                                bt_data = bt_cache[best_track_path]
                            if current_dt not in bt_data['datetime'].values:
                                closest_idx = bt_data['datetime'].sub(current_dt).abs().idxmin()
                                current_row = bt_data.iloc[closest_idx]
                            else:
                                current_row = bt_data[bt_data['datetime'] == current_dt].iloc[0]
                            wind_current = current_row['wind']
                            delta_winds = []
                            for t in time_intervals:
                                future_dt = current_dt + timedelta(hours=t)
                                if future_dt not in bt_data['datetime'].values:
                                    closest_idx = bt_data['datetime'].sub(future_dt).abs().idxmin()
                                    future_row = bt_data.iloc[closest_idx]
                                else:
                                    future_row = bt_data[bt_data['datetime'] == future_dt].iloc[0]
                                delta_winds.append(future_row['wind'] - wind_current)
                            delta_cache[cache_key] = (wind_current, delta_winds)
                        X_test.append(combined_features)
                        y_test.append(ri_value)
                        combo_labels.append((wind, time))
                        delta_winds_list.append(delta_winds)
                    except Exception as e:
                        logging.error(f"處理檔案時出錯 {pv_file}: {e}")
                        continue
            else:
                logging.warning(f"路徑不存在: wind={wind}, time={time}")
    logging.info(f"總測試樣本數: {len(X_test)}")
    return np.array(X_test), np.array(y_test), combo_labels, np.array(delta_winds_list)

# ====================== 跨年代穩定性分析 ======================
def run_temporal_stability():
    mean_ri_dict = {"RF": [], "SVR": [], "ANN": []}
    std_ri_dict = {"RF": [], "SVR": [], "ANN": []}
    pred_dict = {"RF": [], "SVR": [], "ANN": []}
   
    for period in periods:
        logging.info(f"\n=== 處理時期: {period} ===")
       
        if period == "2013-2022":
            X_test = load_2013_test_set()
        else:
            test_base_path = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{period}"
            X_test, _, _, _ = load_test_data(
                wind_thresholds, time_intervals, test_base_path, regions, best_c, best_d
            )
       
        if len(X_test) == 0:
            logging.warning(f"{period} 無有效資料")
            continue
       
        for orig_name, display_name in [("random_forest", "RF"), ("svr", "SVR"), ("ann", "ANN")]:
            if orig_name not in models:
                continue
            feature_scaler = scalers["feature"][orig_name]
            model = models[orig_name]
           
            X_scaled = feature_scaler.transform(X_test)
            y_pred = model.predict(X_scaled)
            y_pred = np.clip(y_pred, 0, 10)
           
            mean_ri = np.mean(y_pred)
            std_ri = np.std(y_pred)
           
            mean_ri_dict[display_name].append(mean_ri)
            std_ri_dict[display_name].append(std_ri)
            pred_dict[display_name].append(y_pred.copy())
           
            logging.info(f"{display_name:<6} Mean RI = {mean_ri:.3f} ± {std_ri:.3f}")
   
    # ====================== 時間序列趨勢圖 ======================
    plt.figure(figsize=(10, 6))
    for name in ["RF", "SVR", "ANN"]:
        means = mean_ri_dict[name]
        stds = std_ri_dict[name]
        plt.plot(periods, means, marker='o', label=name)
        plt.fill_between(periods, np.array(means) - np.array(stds),
                         np.array(means) + np.array(stds), alpha=0.3)
    plt.title("Mean RI Index 跨年代趨勢 (含標準差)", fontsize=16)
    plt.xlabel("時期")
    plt.ylabel("Mean Predicted RI Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "Mean_RI_Index_Time_Series_with_Std.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("時間序列趨勢圖已儲存")

    # ====================== Pearson Correlation ======================
    logging.info("\n=== Pearson Correlation between Periods ===")
    for name in ["RF", "SVR", "ANN"]:
        preds = pred_dict[name]
        corr_matrix = np.zeros((3, 3))
       
        for i in range(3):
            for j in range(3):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    min_len = min(len(preds[i]), len(preds[j]))
                    if min_len < 10 or np.std(preds[i][:min_len]) < 1e-5 or np.std(preds[j][:min_len]) < 1e-5:
                        corr_matrix[i, j] = np.nan
                    else:
                        corr, _ = pearsonr(preds[i][:min_len], preds[j][:min_len])
                        corr_matrix[i, j] = corr
       
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
                    xticklabels=periods, yticklabels=periods, vmin=-1, vmax=1, center=0)
        plt.title(f"Pearson Correlation of RI Index - {name}", fontsize=14)
        plt.savefig(os.path.join(output_dir, f"Pearson_Correlation_3x3_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
       
        logging.info(f"{name} Pearson Correlation 熱圖已儲存")
        logging.info(f"{name} Correlation Matrix:\n{np.round(corr_matrix, 3)}")
    logging.info("跨年代穩定性分析完成！請檢查輸出目錄中的圖片")

if __name__ == "__main__":
    run_temporal_stability()
    logging.info("腳本執行結束")