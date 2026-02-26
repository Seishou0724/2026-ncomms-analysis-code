import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
from datetime import datetime, timedelta

# 設置輸出目錄
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/M1981-2022_onset_test"
os.makedirs(output_dir, exist_ok=True)

# 設置日誌，確保立即寫入
log_file_path = os.path.join(output_dir, "validation_log_1981-2022_test.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
        logging.StreamHandler()
    ]
)
logging.info("腳本開始執行")

# 定義參數 (與訓練程式碼一致)
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)}
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"}
}
test_year_range = "1981-2022"  # 使用訓練年的測試集
test_base_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-1981-2022"
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/1981-2022-continue"
best_track_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/ibtracs.WP.list.v04r01.csv"

# 假設最佳 c 和 d 與訓練一致
best_c = 210
best_d = 25
max_wind = max(wind_thresholds)

# 載入訓練模型和縮放器 .pkl 檔案
model_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_model_kfold_upper_inner_1981-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_model_kfold_upper_inner_1981-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_model_kfold_upper_inner_1981-2022.pkl")
}
feature_scaler_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_feature_scaler_kfold_upper_inner_1981-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_feature_scaler_kfold_upper_inner_1981-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_feature_scaler_kfold_upper_inner_1981-2022.pkl")
}
target_scaler_files = {
    "random_forest": os.path.join(train_pkl_dir, "PVTHE_rf_target_scaler_kfold_upper_inner_1981-2022.pkl"),
    "svr": os.path.join(train_pkl_dir, "PVTHE_svr_target_scaler_kfold_upper_inner_1981-2022.pkl"),
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_target_scaler_kfold_upper_inner_1981-2022.pkl")
}

logging.info("載入模型和縮放器...")
# 載入訓練模型和縮放器
rf_model = None
svr_model = None
ann_model = None
try:
    rf_model = joblib.load(model_files["random_forest"])
    rf_feature_scaler = joblib.load(feature_scaler_files["random_forest"])
    rf_target_scaler = joblib.load(target_scaler_files["random_forest"])
    logging.info("Random Forest 模型和縮放器載入成功")
except FileNotFoundError as e:
    logging.error(f"Random Forest 模型載入失敗: {e}")
except Exception as e:
    logging.error(f"Random Forest 模型載入異常: {e}")
try:
    svr_model = joblib.load(model_files["svr"])
    svr_feature_scaler = joblib.load(feature_scaler_files["svr"])
    svr_target_scaler = joblib.load(target_scaler_files["svr"])
    logging.info("SVR 模型和縮放器載入成功")
except FileNotFoundError as e:
    logging.error(f"SVR 模型載入失敗: {e}")
except Exception as e:
    logging.error(f"SVR 模型載入異常: {e}")
try:
    ann_model = joblib.load(model_files["ann"])
    ann_feature_scaler = joblib.load(feature_scaler_files["ann"])
    ann_target_scaler = joblib.load(target_scaler_files["ann"])
    logging.info("ANN 模型和縮放器載入成功")
except FileNotFoundError as e:
    logging.error(f"ANN 模型載入失敗: {e}")
except Exception as e:
    logging.error(f"ANN 模型載入異常: {e}")

# RI 計算函數 (與訓練一致，但測試時不需用)
def calculate_ri(wind, time, c, d, max_wind):
    return wind * (c / (time + d)) * (1 + wind / max_wind)

# 數據調整函數 (與訓練一致)
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

# 修改: 生成測試資料函數 (載入所有資料點，排除每個 TC 的第一筆，每個 dt 只處理一次)
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
    delta_cache = {}  # 緩存 delta_winds 和 wind_current，key: (sid, str(current_dt))
    bt_data_cache = {}  # 緩存 per SID 的 bt_data_sid
    # 載入整個 ibtracs CSV 只一次
    bt_data = pd.read_csv(best_track_path, low_memory=False)
    # 資料在 Whole 資料夾
    whole_path = os.path.join(test_base_path, "Whole")
    logging.info(f"Whole 路徑: {whole_path}")
    if os.path.exists(whole_path):
        # 掃描檔案
        pv_files = [os.path.join(whole_path, f) for f in os.listdir(whole_path) if f.startswith('Azi_PV-') and f.endswith(".txt")]
        logging.info(f"找到 PV 檔案數: {len(pv_files)}")
        # 先收集所有檔案的 sid 和 dt 資訊，按 SID 群組並排序
        sid_files = {}
        for pv_file in pv_files:
            f = os.path.basename(pv_file)
            parts = f.replace('.txt', '').split('-')
            if len(parts) != 3 or parts[0] != 'Azi_PV':
                logging.warning(f"無效檔案名格式: {f}")
                continue
            timestr = parts[1]
            sid = parts[2]
            if len(timestr) != 10:
                logging.warning(f"無效時間字符串: {timestr}")
                continue
            year = int(timestr[0:4])
            month = int(timestr[4:6])
            day = int(timestr[6:8])
            hour = int(timestr[8:10])
            current_dt = datetime(year, month, day, hour, 0, 0)
            if sid not in sid_files:
                sid_files[sid] = []
            sid_files[sid].append((current_dt, pv_file))
        # 對每個 SID 排序檔案（按時間）
        for sid in sid_files:
            sid_files[sid].sort(key=lambda x: x[0])
        # 處理每個 SID，從第二筆開始
        for sid, files in sid_files.items():
            if len(files) < 2:
                logging.info(f"跳過 SID {sid}: 只有 {len(files)} 筆資料，無法排除第一筆")
                continue
            # 預載 per SID 的 bt_data
            if sid not in bt_data_cache:
                bt_data_sid = bt_data[bt_data['SID'].str.endswith(sid)].copy()
                bt_data_sid['ISO_TIME'] = pd.to_datetime(bt_data_sid['ISO_TIME'])
                if bt_data_sid.empty:
                    logging.warning(f"未找到 SID {sid} 在 best track CSV")
                    continue
                bt_data_cache[sid] = bt_data_sid
            for idx in range(1, len(files)):  # 從索引 1 開始 (第二筆)
                current_dt, pv_file = files[idx]
                # 假設 the_file 檔名相同
                f_base = os.path.basename(pv_file)
                the_file = os.path.join(whole_path, f_base.replace('Azi_PV', 'Azi_THE'))
                if not os.path.exists(the_file):
                    logging.warning(f"缺少 THE 檔案 for {pv_file}")
                    continue
                try:
                    # 用 genfromtxt 處理無效值 (空字串變 NaN)
                    pv_data = np.genfromtxt(pv_file, delimiter='', invalid_raise=False, filling_values=np.nan)
                    the_data = np.genfromtxt(the_file, delimiter='', invalid_raise=False, filling_values=np.nan)
                    logging.info(f"載入檔案成功: {pv_file} (SID {sid}, 非第一筆)")
                    pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                    the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                    if pv_selected.size == 0 or the_selected.size == 0:
                        logging.warning(f"選取數據為空: {pv_file}")
                        continue
                    combined_features = np.concatenate([pv_selected, the_selected])
                    cache_key = (sid, str(current_dt))
                    if cache_key in delta_cache:
                        wind_current, delta_winds = delta_cache[cache_key]
                        logging.info(f"使用緩存數據: {current_dt}_{sid}")
                    else:
                        bt_data_sid = bt_data_cache[sid]
                        current_row = bt_data_sid[bt_data_sid['ISO_TIME'] == current_dt]
                        if current_row.empty:
                            closest_idx = (bt_data_sid['ISO_TIME'] - current_dt).abs().argmin()
                            current_row = bt_data_sid.iloc[closest_idx]
                            logging.warning(f"當前時間不在 best track 中: {current_dt}, 使用最近時間: {current_row['ISO_TIME']}")
                        else:
                            current_row = current_row.iloc[0]
                        wind_current = float(current_row['USA_WIND']) if not pd.isna(current_row['USA_WIND']) else float(current_row['USA_WIND'])
                        delta_winds = []
                        for t in time_intervals:
                            future_dt = current_dt + timedelta(hours=t)
                            future_row = bt_data_sid[bt_data_sid['ISO_TIME'] == future_dt]
                            if future_row.empty:
                                closest_idx = (bt_data_sid['ISO_TIME'] - future_dt).abs().argmin()
                                future_row = bt_data_sid.iloc[closest_idx]
                                logging.warning(f"未來時間不在 best track 中: {future_dt}, 使用最近時間: {future_row['ISO_TIME']}")
                            else:
                                future_row = future_row.iloc[0]
                            wind_future = float(future_row['USA_WIND']) if not pd.isna(future_row['USA_WIND']) else float(future_row['USA_WIND'])
                            delta = wind_future - wind_current
                            delta_winds.append(delta)
                        delta_cache[cache_key] = (wind_current, delta_winds)
                    # 成功計算 delta 後，才 append X
                    X_test.append(combined_features)
                    delta_winds_list.append(delta_winds)
                except Exception as e:
                    logging.error(f"處理檔案時出錯 {pv_file}: {e}")
                    continue
    else:
        logging.warning("資料夾不存在: Whole 等")
    logging.info(f"總測試樣本數: {len(X_test)}")
    return np.array(X_test), delta_winds_list

# 主程式 (無變)
def main():
    logging.info(f"開始驗證 {test_year_range} 測試集資料與 1981–2022 模型的預測偏差")
    if not any([rf_model, svr_model, ann_model]):
        logging.error("所有模型載入失敗，程式終止")
        return
    # 生成測試資料 (所有資料點，排除每個 TC 的第一筆)
    logging.info("開始生成測試資料...")
    X_test, delta_winds_list = generate_test_data(time_intervals, test_base_path, regions)
    if len(X_test) == 0:
        logging.warning("測試數據生成失敗或數量不足，無法進行驗證")
        return
    logging.info(f"成功生成測試數據，樣本數: {len(X_test)}")
    # 定義通用函數來處理每個模型的計算和繪圖
    def process_model(model_name, X_test, delta_winds_list, feature_scaler, target_scaler, model):
        # 計算預測值
        if model_name == "ANN":
            y_test_pred = model.predict(feature_scaler.transform(X_test)).flatten()
        else:
            y_test_pred = model.predict(feature_scaler.transform(X_test))
        logging.info(f"{model_name} y_test_pred min: {np.min(y_test_pred):.2f}, max: {np.max(y_test_pred):.2f}")
        y_test_pred = np.clip(y_test_pred, 0, 10)  # 限制在 MinMaxScaler(0,10) 範圍
        logging.info(f"{model_name} y_test_pred after clip min: {np.min(y_test_pred):.2f}, max: {np.max(y_test_pred):.2f}")
        mean_ri = np.mean(y_test_pred)
        std_ri = np.std(y_test_pred)
        high_threshold = mean_ri + std_ri
        low_threshold = mean_ri - std_ri
        logging.info(f"{model_name} on {test_year_range}: Mean RI Index = {mean_ri:.2f}, "
                     f"High Threshold = {high_threshold:.2f}, Low Threshold = {low_threshold:.2f}")
        # 計算機率部分
        y_pred_ri = y_test_pred
        bins = np.linspace(np.min(y_pred_ri), np.max(y_pred_ri), 11)
        bin_indices = np.digitize(y_pred_ri, bins) - 1
        # 新增: 檢查 y_pred_ri 分布 - 保存 histogram 圖
        logging.info(f"y_pred_ri 平均: {np.mean(y_pred_ri):.2f}, 標準差: {np.std(y_pred_ri):.2f}")
        plt.figure(figsize=(8, 6))
        plt.hist(y_pred_ri, bins=20, edgecolor='black')
        plt.title(f"RI Index Prediction Distribution ({model_name}, {test_year_range})")
        plt.xlabel("Predicted RI Index")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"RI_Index_hist_{model_name}_{test_year_range}.png"), dpi=300)
        plt.close()
        logging.info(f"RI Index histogram saved to {os.path.join(output_dir, f'RI_Index_hist_{model_name}_{test_year_range}.png')}")
        # 新增: 計算每個 bin 的樣本數 (用 np.histogram)
        counts, _ = np.histogram(y_pred_ri, bins=bins)
        for i, count in enumerate(counts):
            logging.info(f"Bin {i+1} [{bins[i]:.2f}-{bins[i+1]:.2f}) 的樣本數: {count}")
        # 修改 bin_labels 加 (n=count)
        bin_labels = [f"Bin {k+1} [{bins[k]:.2f}-{bins[k+1]:.2f}) (n={counts[k]})" for k in range(10)]
        num_bins = 10
        num_times = len(time_intervals)
        num_winds = len(wind_thresholds)
        avg_delta_matrix = np.full((num_bins, num_times), np.nan)
        prob_matrix = np.full((num_bins, num_times, num_winds), np.nan)
        delta_winds_array = np.array(delta_winds_list)  # 轉成 array (num_samples, num_times)
        for i in range(num_bins):
            mask = (bin_indices == i)
            if np.any(mask):
                delta_sub = delta_winds_array[mask]
                for j in range(num_times):
                    valid_deltas = delta_sub[:, j][~np.isnan(delta_sub[:, j])]
                    if len(valid_deltas) > 0:
                        avg_delta_matrix[i, j] = np.mean(valid_deltas)
                    for k in range(num_winds):
                        thresh = wind_thresholds[k]
                        occurred = (delta_sub[:, j] >= thresh) & (~np.isnan(delta_sub[:, j]))
                        num_valid = np.sum(~np.isnan(delta_sub[:, j]))
                        if num_valid > 0:
                            prob_matrix[i, j, k] = np.sum(occurred) / num_valid
        # 平均風變化熱圖 (使用新 bin_labels)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_delta_matrix, xticklabels=time_intervals, yticklabels=bin_labels,
                    cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Average Wind Change by RI Index Bin and Time Interval ({model_name}, {test_year_range})", fontsize=16)
        plt.xlabel("Time Interval (h)", fontsize=14)
        plt.ylabel("RI Index Bin (Range with Count)", fontsize=14)
        plt.tick_params(axis='both', labelsize=10)
        plt.savefig(os.path.join(output_dir, f"Avg_Delta_Wind_heatmap_{model_name}_{test_year_range}.png"), dpi=300)
        plt.close()
        logging.info(f"Average Delta Wind heatmap saved to {os.path.join(output_dir, f'Avg_Delta_Wind_heatmap_{model_name}_{test_year_range}.png')}")
        # 新增: 機率熱圖 (為每個風速閾值產生一個熱圖，使用新 bin_labels)
        for k, wind_thresh in enumerate(wind_thresholds):
            plt.figure(figsize=(12, 8))
            sns.heatmap(prob_matrix[:, :, k], xticklabels=time_intervals, yticklabels=bin_labels,
                        cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1)
            plt.title(f"RI Probability (Wind >= {wind_thresh} kt) by Bin and Time ({model_name}, {test_year_range})", fontsize=16)
            plt.xlabel("Time Interval (h)", fontsize=14)
            plt.ylabel("RI Index Bin (Range with Count)", fontsize=14)
            plt.tick_params(axis='both', labelsize=10)
            plt.savefig(os.path.join(output_dir, f"RI_Prob_heatmap_wind{wind_thresh}_{model_name}_{test_year_range}.png"), dpi=300)
            plt.close()
            logging.info(f"RI Probability heatmap for wind {wind_thresh} saved to {os.path.join(output_dir, f'RI_Prob_heatmap_wind{wind_thresh}_{model_name}_{test_year_range}.png')}")
        # 繪製 KDE 圖
        data = []
        for s in range(len(delta_winds_list)):
            bin_idx = bin_indices[s]
            if bin_idx < 0 or bin_idx >= num_bins:
                continue
            bin_label = bin_labels[bin_idx]
            for j, t in enumerate(time_intervals):
                delta = delta_winds_list[s][j]
                if np.isnan(delta):
                    continue
                data.append({'bin': bin_label, 'time': t, 'delta_wind': delta})
        df = pd.DataFrame(data)
        # 新增: 檢查 df 中每個 bin 的數據點數 (考慮 NaN 過濾後)
        if not df.empty:
            for i in range(num_bins):
                bin_df = df[df['bin'] == bin_labels[i]]
                logging.info(f"{bin_labels[i]} 的 delta_wind 數據點數 (after NaN filter): {len(bin_df)}")
        if not df.empty:
            df['time_jitter'] = df['time'] + np.random.uniform(-0.4, 0.4, len(df))
            plt.figure(figsize=(10, 10))
            threshold = 0.5
            levels = np.linspace(0.5, 0.9, 5)
            linewidths = [1.0, 1.5, 2.0, 2.5, 3.0]
            colors = sns.color_palette("coolwarm", num_bins)
            for i in range(num_bins):
                bin_df = df[df['bin'] == bin_labels[i]]
                if bin_df.empty:
                    logging.info(f"Skipping KDE for {bin_labels[i]}: no data points")
                    continue
                sns.kdeplot(data=bin_df, x='time_jitter', y='delta_wind', levels=levels, thresh=threshold,
                            color=colors[i], linewidths=linewidths, common_norm=False)
            plt.xticks(time_intervals)
            plt.yticks(np.arange(-70, max(wind_thresholds) + 10, 10))
            plt.ylim(-70, max(wind_thresholds) + 10)
            bin_legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=bin_labels[i]) for i in range(num_bins)]
            bin_legend = plt.legend(handles=bin_legend_elements, title='Bins (Ranges with Count)', loc='upper right', fontsize=10, title_fontsize=12)
            plt.gca().add_artist(bin_legend)
            thickness_legend_elements = [Line2D([0], [0], color='black', lw=linewidths[k], label=f'{levels[k]*100:.0f}%') for k in range(len(linewidths))]
            plt.legend(handles=thickness_legend_elements, title='Probability Levels', loc='upper left', fontsize=10, title_fontsize=12)
            plt.xlabel("Time Interval (h)", fontsize=14)
            plt.ylabel("Wind Change (kt)", fontsize=14)
            plt.title(f"2D KDE and RI Index Bin ({model_name}, {test_year_range})", fontsize=16)
            plt.tick_params(axis='both', labelsize=10)
            plt.savefig(os.path.join(output_dir, f"Wind_Change_KDE_by_Bin_and_Time_{model_name}_{test_year_range}.png"), dpi=300)
            plt.close()
            logging.info(f"Wind Change KDE plot saved to {os.path.join(output_dir, f'Wind_Change_KDE_by_Bin_and_Time_{model_name}_{test_year_range}.png')}")
        return high_threshold, low_threshold
    # 進行預測
    results = {}
    if rf_model:
        try:
            if len(X_test) == 0:
                logging.error("Random Forest 測試集資料為空，跳過處理")
            else:
                high_threshold_rf, low_threshold_rf = process_model(
                    "Random_Forest", X_test, delta_winds_list,
                    rf_feature_scaler, rf_target_scaler, rf_model
                )
                results["random_forest"] = (high_threshold_rf, low_threshold_rf)
        except Exception as e:
            logging.error(f"Random Forest 預測錯誤: {e}")
    if svr_model:
        try:
            if len(X_test) == 0:
                logging.error("SVR 測試集資料為空，跳過處理")
            else:
                high_threshold_svr, low_threshold_svr = process_model(
                    "SVR", X_test, delta_winds_list,
                    svr_feature_scaler, svr_target_scaler, svr_model
                )
                results["svr"] = (high_threshold_svr, low_threshold_svr)
        except Exception as e:
            logging.error(f"SVR 預測錯誤: {e}")
    if ann_model:
        try:
            if len(X_test) == 0:
                logging.error("ANN 測試集資料為空，跳過處理")
            else:
                high_threshold_ann, low_threshold_ann = process_model(
                    "ANN", X_test, delta_winds_list,
                    ann_feature_scaler, ann_target_scaler, ann_model
                )
                results["ann"] = (high_threshold_ann, low_threshold_ann)
        except Exception as e:
            logging.error(f"ANN 預測錯誤: {e}")
    # 計算與訓練閾值的偏差
    rf_train_high = 4.29
    rf_train_low = 1.71
    svr_train_high = 4.63
    svr_train_low = 1.34
    ann_train_high = 4.72
    ann_train_low = 1.33
    if "random_forest" in results:
        high_dev_rf = ((results["random_forest"][0] - rf_train_high) / rf_train_high) * 100
        low_dev_rf = ((results["random_forest"][1] - rf_train_low) / rf_train_low) * 100
        logging.info(f"Random Forest High Threshold Deviation: {high_dev_rf:.2f}%, Low Threshold Deviation: {low_dev_rf:.2f}%")
        print(f"Random Forest High Threshold Deviation: {high_dev_rf:.2f}%")
        print(f"Random Forest Low Threshold Deviation: {low_dev_rf:.2f}%")
    if "svr" in results:
        high_dev_svr = ((results["svr"][0] - svr_train_high) / svr_train_high) * 100
        low_dev_svr = ((results["svr"][1] - svr_train_low) / svr_train_low) * 100
        logging.info(f"SVR High Threshold Deviation: {high_dev_svr:.2f}%, Low Threshold Deviation: {low_dev_svr:.2f}%")
        print(f"SVR High Threshold Deviation: {high_dev_svr:.2f}%")
        print(f"SVR Low Threshold Deviation: {low_dev_svr:.2f}%")
    if "ann" in results:
        high_dev_ann = ((results["ann"][0] - ann_train_high) / ann_train_high) * 100
        low_dev_ann = ((results["ann"][1] - ann_train_low) / ann_train_low) * 100
        logging.info(f"ANN High Threshold Deviation: {high_dev_ann:.2f}%, Low Threshold Deviation: {low_dev_ann:.2f}%")
        print(f"ANN High Threshold Deviation: {high_dev_ann:.2f}%")
        print(f"ANN Low Threshold Deviation: {low_dev_ann:.2f}%")

if __name__ == "__main__":
    main()
    logging.info("驗證完成。請檢查日誌檔案以獲取詳細結果。")