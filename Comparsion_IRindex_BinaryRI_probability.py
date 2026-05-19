import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ====================== 基本設定 ======================
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/M1981-2022_onset_test_for_binary_comparison"
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "validation_log_1981-2022_test.txt")
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
                              logging.StreamHandler()])
logging.info("腳本開始執行")

wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {"PV": {"rows": range(0, 27), "cols": range(0, 21)}, "THE": {"rows": range(0, 27), "cols": range(0, 21)}}
regions = {"upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"}}

test_year_range = "1981-2022"
test_base_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-1981-2022"
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/1981-2022-continue-forcomparsionBinaryRI"
best_track_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/ibtracs.WP.list.v04r01.csv"

# ====================== 載入模型和縮放器 ======================
model_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_model_onset_upper_inner_1981-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_model_onset_upper_inner_1981-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_model_onset_upper_inner_1981-2022.pkl")
}
feature_scaler_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_feature_scaler_onset_upper_inner_1981-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_feature_scaler_onset_upper_inner_1981-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_feature_scaler_onset_upper_inner_1981-2022.pkl")
}

rf_model = rf_feature_scaler = None
svr_model = svr_feature_scaler = None
ann_model = ann_feature_scaler = None

for name in ["random_forest", "svr", "ann"]:
    try:
        model_var = joblib.load(model_files[name])
        scaler_var = joblib.load(feature_scaler_files[name])
        if name == "random_forest":
            rf_model, rf_feature_scaler = model_var, scaler_var
        elif name == "svr":
            svr_model, svr_feature_scaler = model_var, scaler_var
        else:
            ann_model, ann_feature_scaler = model_var, scaler_var
        logging.info(f"{name.capitalize()} 模型與 feature_scaler 載入成功")
    except Exception as e:
        logging.error(f"{name} 載入失敗: {e}")

# ====================== adjust_data ======================
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

# ====================== generate_test_data ======================
def generate_test_data(time_intervals, test_base_path, regions):
    X_test = []
    delta_winds_list = []
    region = "upper_inner"
    if region not in regions:
        logging.error(f"區域 {region} 未定義")
        return np.array([]), []
    region_rows = regions[region]["rows"]
    region_cols = regions[region]["cols"]
    logging.info(f"開始生成測試資料，區域: {regions[region]['name']}")

    delta_cache = {}
    bt_data_cache = {}
    bt_data = pd.read_csv(best_track_path, low_memory=False)
    whole_path = os.path.join(test_base_path, "Whole")
    logging.info(f"Whole 路徑: {whole_path}")

    if os.path.exists(whole_path):
        pv_files = [os.path.join(whole_path, f) for f in os.listdir(whole_path) if f.startswith('Azi_PV-') and f.endswith(".txt")]
        logging.info(f"找到 PV 檔案數: {len(pv_files)}")

        sid_files = {}
        for pv_file in pv_files:
            f = os.path.basename(pv_file)
            parts = f.replace('.txt', '').split('-')
            if len(parts) != 3 or parts[0] != 'Azi_PV':
                continue
            timestr = parts[1]
            sid = parts[2]
            if len(timestr) != 10:
                continue
            year = int(timestr[0:4])
            month = int(timestr[4:6])
            day = int(timestr[6:8])
            hour = int(timestr[8:10])
            current_dt = datetime(year, month, day, hour, 0, 0)
            if sid not in sid_files:
                sid_files[sid] = []
            sid_files[sid].append((current_dt, pv_file))

        for sid in sid_files:
            sid_files[sid].sort(key=lambda x: x[0])

        for sid, files in sid_files.items():
            if len(files) < 2:
                continue
            if sid not in bt_data_cache:
                bt_data_sid = bt_data[bt_data['SID'].str.endswith(sid)].copy()
                bt_data_sid['ISO_TIME'] = pd.to_datetime(bt_data_sid['ISO_TIME'])
                if bt_data_sid.empty:
                    continue
                bt_data_cache[sid] = bt_data_sid

            for idx in range(1, len(files)):
                current_dt, pv_file = files[idx]
                the_file = os.path.join(whole_path, os.path.basename(pv_file).replace('Azi_PV', 'Azi_THE'))
                if not os.path.exists(the_file):
                    continue
                try:
                    pv_data = np.genfromtxt(pv_file, delimiter='', invalid_raise=False, filling_values=np.nan, missing_values=' ')
                    the_data = np.genfromtxt(the_file, delimiter='', invalid_raise=False, filling_values=np.nan, missing_values=' ')

                    pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                    the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                    if pv_selected.size == 0 or the_selected.size == 0:
                        continue

                    combined_features = np.concatenate([pv_selected, the_selected])

                    cache_key = (sid, str(current_dt))
                    if cache_key in delta_cache:
                        wind_current, delta_winds = delta_cache[cache_key]
                    else:
                        bt_data_sid = bt_data_cache[sid]
                        current_row = bt_data_sid[bt_data_sid['ISO_TIME'] == current_dt]
                        if current_row.empty:
                            closest_idx = (bt_data_sid['ISO_TIME'] - current_dt).abs().argmin()
                            current_row = bt_data_sid.iloc[closest_idx]
                        else:
                            current_row = current_row.iloc[0]
                        wind_current = float(current_row.get('USA_WIND', 0)) if not pd.isna(current_row.get('USA_WIND')) else 0

                        delta_winds = []
                        for t in time_intervals:
                            future_dt = current_dt + timedelta(hours=t)
                            future_row = bt_data_sid[bt_data_sid['ISO_TIME'] == future_dt]
                            if future_row.empty:
                                closest_idx = (bt_data_sid['ISO_TIME'] - future_dt).abs().argmin()
                                future_row = bt_data_sid.iloc[closest_idx]
                            else:
                                future_row = future_row.iloc[0]
                            wind_future = float(future_row.get('USA_WIND', 0)) if not pd.isna(future_row.get('USA_WIND')) else 0
                            delta_winds.append(wind_future - wind_current)
                        delta_cache[cache_key] = (wind_current, delta_winds)

                    X_test.append(combined_features)
                    delta_winds_list.append(delta_winds)

                except:
                    continue   # 隱藏錯誤訊息

    logging.info(f"總測試樣本數: {len(X_test)}")
    return np.array(X_test), delta_winds_list

# ====================== process_model（已加入 refinement distribution 虛線） ======================
def process_model(model_name, X_test, delta_winds_list, feature_scaler, model):
    if model is None or feature_scaler is None:
        logging.error(f"{model_name} 模型或 feature_scaler 未就緒，跳過")
        return

    logging.info(f"開始處理 {model_name} ...")

    try:
        if model_name == "ANN":
            y_test_pred = model.predict(feature_scaler.transform(X_test)).flatten()
        else:
            y_test_pred = model.predict(feature_scaler.transform(X_test))
        y_test_pred = np.clip(y_test_pred, 0, 10)

        logging.info(f"{model_name} 預測 IR Index 範圍: {y_test_pred.min():.2f} ~ {y_test_pred.max():.2f}")

        # ====================== 圖1: IR Index vs 傳統 RI 發生率 ======================
        n_bins = 10
        bins = np.linspace(0, 10, n_bins + 1)
        bin_indices = np.digitize(y_test_pred, bins[1:])
        observed_ri_prob = np.zeros(n_bins)
        delta_winds_array = np.array(delta_winds_list)

        for i in range(n_bins):
            mask = (bin_indices == i)
            if np.sum(mask) == 0: continue
            time_24_idx = time_intervals.index(24) if 24 in time_intervals else 3
            delta_24h = delta_winds_array[mask, time_24_idx]
            valid_delta = delta_24h[~np.isnan(delta_24h)]
            if len(valid_delta) > 0:
                occurred_ri = (valid_delta >= 30).sum()
                observed_ri_prob[i] = occurred_ri / len(valid_delta)

        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, observed_ri_prob, 'o-', color='blue', linewidth=2, markersize=8,
                 label='Observed RI Probability (≥30 kt/24h)')
        plt.axhline(y=0.05, color='red', linestyle='--', label='5% baseline')
        plt.xlabel('Predicted IR Index Bin')
        plt.ylabel('Observed Probability of RI (≥ 30 kt / 24 h)')
        plt.title(f'IR Index vs. Traditional RI Occurrence Probability ({model_name})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'IR_Index_vs_Traditional_RI_Prob_{model_name}_{test_year_range}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"已儲存 {model_name} 的 IR Index 比較圖")

        # ====================== 圖2: Reliability Diagram（含 refinement 虛線） ======================
        time_24_idx = time_intervals.index(24) if 24 in time_intervals else 3
        delta_24h_all = delta_winds_array[:, time_24_idx]
        valid_mask = ~np.isnan(delta_24h_all)

        if np.sum(valid_mask) > 10:
            X_cal = y_test_pred[valid_mask].reshape(-1, 1)
            y_cal = (delta_24h_all[valid_mask] >= 30).astype(int)

            if len(np.unique(y_cal)) > 1:
                calib = LogisticRegression()
                calib.fit(X_cal, y_cal)
                calibrated_prob = calib.predict_proba(X_cal)[:, 1]
            else:
                calibrated_prob = np.zeros(len(X_cal))

            # 計算 observed frequency 和 refinement distribution
            n_bins_cal = 10
            bin_edges = np.linspace(0, 1, n_bins_cal + 1)
            bin_centers_cal = (bin_edges[:-1] + bin_edges[1:]) / 2
            observed_freq = np.zeros(n_bins_cal)
            refinement = np.zeros(n_bins_cal)   # 虛線用

            for i in range(n_bins_cal):
                mask = (calibrated_prob >= bin_edges[i]) & (calibrated_prob < bin_edges[i+1])
                count = np.sum(mask)
                if count > 0:
                    observed_freq[i] = np.mean(y_cal[mask])
                    refinement[i] = count / len(calibrated_prob)   # 該區間樣本比例

            # 畫圖
            plt.figure(figsize=(8, 8))
            plt.plot(bin_centers_cal, observed_freq, 'o-', color='blue', label='Observed Frequency')
            plt.plot(bin_centers_cal, refinement, 's--', color='gray', linewidth=1.5, label='Refinement Distribution (dashed)')
            plt.plot([0, 1], [0, 1], 'k--', label='Perfect Reliability')
            plt.xlabel('Forecasted RI Probability')
            plt.ylabel('Frequency / Observed RI Frequency')
            plt.title(f'Reliability Diagram with Refinement - {model_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'Reliability_Diagram_{model_name}_{test_year_range}.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"已儲存 {model_name} 的 Reliability Diagram（含 refinement 虛線）")

    except Exception as e:
        logging.error(f"{model_name} 處理過程中發生錯誤: {e}")

    logging.info(f"{model_name} 處理完成")

# ====================== 主程式 ======================
def main():
    logging.info(f"開始驗證 {test_year_range} 測試集資料")
    X_test, delta_winds_list = generate_test_data(time_intervals, test_base_path, regions)
    if len(X_test) == 0:
        logging.warning("測試數據生成失敗")
        return
    logging.info(f"成功生成測試數據，樣本數: {len(X_test)}")

    if rf_model and rf_feature_scaler:
        process_model("Random_Forest", X_test, delta_winds_list, rf_feature_scaler, rf_model)
    else:
        logging.warning("Random_Forest 模型或 feature_scaler 未就緒，跳過")

    if svr_model and svr_feature_scaler:
        process_model("SVR", X_test, delta_winds_list, svr_feature_scaler, svr_model)
    else:
        logging.warning("SVR 模型或 feature_scaler 未就緒，跳過")

    if ann_model and ann_feature_scaler:
        process_model("ANN", X_test, delta_winds_list, ann_feature_scaler, ann_model)
    else:
        logging.warning("ANN 模型或 feature_scaler 未就緒，跳過")

    logging.info("所有分析與圖表已完成！請檢查輸出目錄。")

if __name__ == "__main__":
    main()