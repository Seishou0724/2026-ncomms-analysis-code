import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D
# 設置輸出目錄
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/M2013-2022_T1981-2012"
os.makedirs(output_dir, exist_ok=True)
# 設置日誌，確保立即寫入
log_file_path = os.path.join(output_dir, "validation_log_1981-2012.txt")
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
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
    # "W": {"rows": range(0, 27), "cols": range(0, 21)}
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"}
}
test_year_range = "1959-1980" # 可改為 "1959-1980"
test_base_path = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{test_year_range}"
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/2013-2022-continue"
# 假設最佳 c 和 d 與訓練一致
best_c = 210
best_d = 25
max_wind = max(wind_thresholds)
# 載入訓練模型和縮放器 .pkl 檔案
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
# RI 計算函數 (與訓練一致)
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
# 載入測試資料函數 (計算所有 time_intervals 的 delta_wind)
def load_test_data(wind_thresholds, time_intervals, test_base_path, regions, best_c, best_d):
    X_test, y_test, combo_labels, delta_winds_list = [], [], [], []
    region = "upper_inner"
    if region not in regions:
        logging.error(f"區域 {region} 未定義")
        return np.array([]), np.array([]), [], []
    region_rows = regions[region]["rows"]
    region_cols = regions[region]["cols"]
    logging.info(f"開始載入測試資料，區域: {regions[region]['name']}")
    bt_cache = {} # 緩存 best_track_data，key: best_track_path
    delta_cache = {} # 緩存 delta_winds 和 wind_current，key: (sid, str(current_dt))
    for wind in wind_thresholds:
        for time in time_intervals:
            ri_value = calculate_ri(wind, time, best_c, best_d, max_wind)
            pv_path_ri = os.path.join(test_base_path, "Azi_PV", "Individual", f"Azi_PV-RI-{wind}-{time:02d}", "removenan")
            the_path_ri = os.path.join(test_base_path, "Azi_THE", "Individual", f"Azi_THE-RI-{wind}-{time:02d}", "removenan")
            # w_path_ri = os.path.join(test_base_path, "Azi_W", "Individual", f"Azi_W-RI-{wind}-{time:02d}", "removenan")
            logging.info(f"檢查路徑: wind={wind}, time={time}")
            logging.info(f"PV 路徑: {pv_path_ri}")
            logging.info(f"THE 路徑: {the_path_ri}")
            # logging.info(f"W 路徑: {w_path_ri}")
            if os.path.exists(pv_path_ri) and os.path.exists(the_path_ri): # and os.path.exists(w_path_ri):
                pv_files_ri = [os.path.join(pv_path_ri, f) for f in os.listdir(pv_path_ri) if f.endswith(".txt")]
                the_files_ri = [os.path.join(the_path_ri, f) for f in os.listdir(the_path_ri) if f.endswith(".txt")]
                # w_files_ri = [os.path.join(w_path_ri, f) for f in os.listdir(w_path_ri) if f.endswith(".txt")]
                logging.info(f"找到 PV 檔案數: {len(pv_files_ri)}")
                logging.info(f"找到 THE 檔案數: {len(the_files_ri)}")
                # logging.info(f"找到 W 檔案數: {len(w_files_ri)}")
                min_files_ri = min(len(pv_files_ri), len(the_files_ri)) # , len(w_files_ri))
                logging.info(f"最小檔案數: {min_files_ri}")
                if min_files_ri == 0:
                    logging.warning(f"無有效檔案: wind={wind}, time={time}")
                    continue
                for pv_file, the_file in zip(pv_files_ri[:min_files_ri], the_files_ri[:min_files_ri]): # , w_file in zip(pv_files_ri[:min_files_ri], the_files_ri[:min_files_ri], w_files_ri[:min_files_ri]):
                    try:
                        pv_data = np.loadtxt(pv_file)
                        the_data = np.loadtxt(the_file)
                        # w_data = np.loadtxt(w_file)
                        logging.info(f"載入檔案成功: {pv_file}")
                        pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                        the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                        # w_selected = adjust_data(w_data, "W", region_rows, region_cols)
                        if pv_selected.size == 0 or the_selected.size == 0: # or w_selected.size == 0:
                            logging.warning(f"選取數據為空: wind={wind}, time={time}")
                            continue
                        combined_features = np.concatenate([pv_selected, the_selected]) # , w_selected])
                        # 解析檔案名以獲取 SID 和 timestamp
                        f = os.path.basename(pv_file)
                        parts = f.replace('.txt', '').split('-')
                        if len(parts) != 3 or parts[0] != 'Azi_PV':
                            logging.warning(f"無效檔案名格式: {f}")
                            continue
                        sid = parts[2]
                        timestr = parts[1]
                        if len(timestr) != 10:
                            logging.warning(f"無效時間字符串: {timestr}")
                            continue
                        year = int(timestr[0:4])
                        month = int(timestr[4:6])
                        day = int(timestr[6:8])
                        hour = int(timestr[8:10])
                        current_dt = datetime(year, month, day, hour, 0, 0) # 明確指定分秒為0
                        cache_key = (sid, str(current_dt))
                        if cache_key in delta_cache:
                            wind_current, delta_winds = delta_cache[cache_key]
                            logging.info(f"使用緩存數據: {timestr}_{sid}")
                        else:
                            # 找到匹配的 best-track 檔案，使用年份和 SID過濾
                            files = os.listdir(test_base_path)
                            matching_files = [f for f in files if f.endswith('.txt') and f.startswith('JTWC-') and str(year) in f and sid in f]
                            if len(matching_files) != 1:
                                logging.warning(f"未找到或多個 best track 檔案 for year {year}, SID {sid}: {matching_files}")
                                continue
                            best_track_file = matching_files[0]
                            best_track_path = os.path.join(test_base_path, best_track_file)
                            if not os.path.exists(best_track_path):
                                logging.warning(f"Best track 檔案不存在: {best_track_path}")
                                continue
                            # 檢查緩存中是否有 bt_data
                            if best_track_path in bt_cache:
                                bt_data = bt_cache[best_track_path]
                                logging.info(f"使用緩存 best track 數據: {best_track_path}")
                            else:
                                # 載入 best-track 資料，使用 sep='\s+' 代替 delim_whitespace
                                bt_data = pd.read_csv(best_track_path, sep='\s+', header=None,
                                                      names=['sid', 'year', 'month', 'day', 'hour', 'lat', 'lon', 'wind', 'pressure', 'source'])
                                bt_data['datetime'] = pd.to_datetime(bt_data[['year', 'month', 'day', 'hour']])
                                bt_cache[best_track_path] = bt_data
                            # 如果 current_dt 不存在，找最近的時間點
                            if current_dt not in bt_data['datetime'].values:
                                closest_idx = bt_data['datetime'].sub(current_dt).abs().idxmin()
                                current_row = bt_data.iloc[closest_idx]
                                logging.warning(f"當前時間不在 best track 中: {current_dt}, 使用最近時間: {current_row['datetime']}")
                            else:
                                current_row = bt_data[bt_data['datetime'] == current_dt].iloc[0]
                            wind_current = current_row['wind']
                            # 為每個 time_interval 計算 delta_wind
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
                            # 緩存結果
                            delta_cache[cache_key] = (wind_current, delta_winds)
                        # 添加樣本
                        X_test.append(combined_features)
                        y_test.append(ri_value)
                        combo_labels.append((wind, time))
                        delta_winds_list.append(delta_winds)
                        logging.info(f"成功添加樣本: wind={wind}, time={time}, delta_winds={delta_winds}")
                    except Exception as e:
                        logging.error(f"處理 RI 檔案時出錯 {pv_file}, {the_file}: {e}") # , {w_file}: {e}")
                        continue
            else:
                logging.warning(f"路徑不存在: wind={wind}, time={time}")
    logging.info(f"總測試樣本數: {len(X_test)}")
    return np.array(X_test), np.array(y_test), combo_labels, np.array(delta_winds_list)
# 主程式
def main():
    logging.info(f"開始驗證 {test_year_range} 資料與 2013–2022 模型的預測偏差")
    if not any([rf_model, svr_model, ann_model]):
        logging.error("所有模型載入失敗，程式終止")
        return
    # 載入測試資料
    logging.info("開始載入測試資料...")
    X_test, y_test, combo_labels, delta_winds_list = load_test_data(wind_thresholds, time_intervals, test_base_path, regions, best_c, best_d)
    if len(X_test) == 0 or len(y_test) == 0:
        logging.warning("測試數據載入失敗或數量不足，無法進行驗證")
        return
    logging.info(f"成功載入測試數據，樣本數: {len(X_test)}")
    # 定義通用函數來處理每個模型的計算和繪圖
    def process_model(model_name, y_test_pred, target_scaler, delta_winds_list):
        # 記錄原始預測值的範圍
        logging.info(f"{model_name} y_test_pred min: {np.min(y_test_pred):.2f}, max: {np.max(y_test_pred):.2f}")
        # 對所有模型預測值進行剪切
        y_test_pred = np.clip(y_test_pred, 0, 10) # 限制在 MinMaxScaler(0,10) 範圍
        logging.info(f"{model_name} y_test_pred after clip min: {np.min(y_test_pred):.2f}, max: {np.max(y_test_pred):.2f}")
        y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
        mse = mean_squared_error(y_test_scaled, y_test_pred)
        mean_ri = np.mean(y_test_pred)
        std_ri = np.std(y_test_pred)
        high_threshold = mean_ri + std_ri
        low_threshold = mean_ri - std_ri
        logging.info(f"{model_name} on {test_year_range}: MSE = {mse:.2f}, Mean RI Index = {mean_ri:.2f}, "
                     f"High Threshold = {high_threshold:.2f}, Low Threshold = {low_threshold:.2f}")
        # 計算機率部分
        y_pred_ri = y_test_pred
        # 等分為10個bins
        bins = np.linspace(np.min(y_pred_ri), np.max(y_pred_ri), 11)
        bin_indices = np.digitize(y_pred_ri, bins) - 1
        # 修改 bin_labels 為顯示範圍
        bin_labels = [f"Bin {k+1} [{bins[k]:.2f}-{bins[k+1]:.2f})" for k in range(10)]
        num_bins = 10
        num_times = len(time_intervals)
        num_winds = len(wind_thresholds)
        # 計算平均 delta_wind 3D 矩陣 (bins x times)
        avg_delta_matrix = np.full((num_bins, num_times), np.nan)
        # 計算機率 3D 矩陣 (bins x times x winds)
        prob_matrix = np.full((num_bins, num_times, num_winds), np.nan)
        for i in range(num_bins):
            mask = (bin_indices == i)
            if np.any(mask):
                delta_sub = delta_winds_list[mask]
                for j in range(num_times):
                    valid_deltas = delta_sub[:, j][~np.isnan(delta_sub[:, j])]
                    if len(valid_deltas) > 0:
                        avg_delta_matrix[i, j] = np.mean(valid_deltas)
                    for k in range(num_winds):
                        thresh = wind_thresholds[k]
                        occurred = (delta_sub[:, j] >= thresh) & (~np.isnan(delta_sub[:, j]))
                        if np.any(~np.isnan(delta_sub[:, j])):
                            prob = np.sum(occurred) / np.sum(~np.isnan(delta_sub[:, j]))
                            prob_matrix[i, j, k] = prob
        # 畫平均 delta_wind 熱圖 (bins x times)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_delta_matrix, xticklabels=time_intervals, yticklabels=bin_labels,
                    cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Average Wind Change by RI Index Bin and Time Interval ({model_name}, {test_year_range})", fontsize=16)
        plt.xlabel("Time Interval (h)", fontsize=14)
        plt.ylabel("RI Index Bin (Range)", fontsize=14)
        plt.tick_params(axis='both', labelsize=12)
        plt.savefig(os.path.join(output_dir, f"Avg_Delta_Wind_heatmap_{model_name}_{test_year_range}.png"), dpi=300)
        plt.close()
        logging.info(f"Average Delta Wind heatmap saved to {os.path.join(output_dir, f'Avg_Delta_Wind_heatmap_{model_name}_{test_year_range}.png')}")
        # 新增: 機率熱圖 (為每個風速閾值產生一個熱圖)
        for k, wind_thresh in enumerate(wind_thresholds):
            plt.figure(figsize=(12, 8))
            sns.heatmap(prob_matrix[:, :, k], xticklabels=time_intervals, yticklabels=bin_labels,
                        cmap="YlGnBu", annot=True, fmt=".2f", vmin=0, vmax=1)
            plt.title(f"RI Probability (Wind >= {wind_thresh} kt) by Bin and Time ({model_name}, {test_year_range})", fontsize=16)
            plt.xlabel("Time Interval (h)", fontsize=14)
            plt.ylabel("RI Index Bin (Range)", fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(os.path.join(output_dir, f"RI_Prob_heatmap_wind{wind_thresh}_{model_name}_{test_year_range}.png"), dpi=300)
            plt.close()
            logging.info(f"RI Probability heatmap for wind {wind_thresh} saved to {os.path.join(output_dir, f'RI_Prob_heatmap_wind{wind_thresh}_{model_name}_{test_year_range}.png')}")
        # 準備數據框用於 KDE 圖
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
        # 畫 2D KDE 圖，x=time (jitter), y=delta_wind, hue=bin
        if not df.empty:
            df['time_jitter'] = df['time'] + np.random.uniform(-0.4, 0.4, len(df))
            plt.figure(figsize=(10, 10))
            threshold = 0.5 # 過濾50%以下的密度
            levels = np.linspace(0.5, 0.9, 5) # 50%, 60%, 70%, 80%, 90%, 100%
            linewidths = [1.0, 1.5, 2.0, 2.5, 3.0] # 從細到粗，對應低到高水平
            colors = sns.color_palette("coolwarm", num_bins) # 藍到紅，低bin藍，高bin紅
            for i in range(num_bins):
                bin_df = df[df['bin'] == bin_labels[i]]
                if bin_df.empty:
                    continue
                sns.kdeplot(data=bin_df, x='time_jitter', y='delta_wind', levels=levels, thresh=threshold,
                            color=colors[i], linewidths=linewidths, common_norm=False)
            plt.xticks(time_intervals)
            plt.yticks(np.arange(-70, max(wind_thresholds) + 10, 10)) # 每 25kt 一個刻度
            plt.ylim(-70, max(wind_thresholds) + 10) # 設置 Y 軸範圍
            # 添加 bin 圖例
            bin_legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=bin_labels[i]) for i in range(num_bins)]
            bin_legend = plt.legend(handles=bin_legend_elements, title='Bins (Ranges)', loc='upper right', fontsize=14, title_fontsize=16)
            plt.gca().add_artist(bin_legend)
            # 添加機率水平粗細圖例，使用黑色線
            thickness_legend_elements = [Line2D([0], [0], color='black', lw=linewidths[k], label=f'{levels[k]*100:.0f}%') for k in range(len(linewidths))]
            plt.legend(handles=thickness_legend_elements, title='Probability Levels', loc='upper left', fontsize=14, title_fontsize=16)
            plt.xlabel("Time Interval (h)", fontsize=14)
            plt.ylabel("Wind Change (kt)", fontsize=14)
            plt.title(f"2D KDE and RI Index Bin ({model_name}, {test_year_range})", fontsize=16)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(os.path.join(output_dir, f"Wind_Change_KDE_by_Bin_and_Time_{model_name}_{test_year_range}.png"), dpi=300)
            plt.close()
            logging.info(f"Wind Change KDE plot saved to {os.path.join(output_dir, f'Wind_Change_KDE_by_Bin_and_Time_{model_name}_{test_year_range}.png')}")
        # 生成 RI Index 熱圖
        if not np.all(np.isnan(y_test_pred)):
            ri_index_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
            mse_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
            combo_data = pd.DataFrame({
                'y_pred': y_test_pred,
                'y_true': target_scaler.transform(y_test.reshape(-1, 1)).ravel(),
                'wind': [c[0] for c in combo_labels],
                'time': [c[1] for c in combo_labels],
            })
            for wind_idx, wind in enumerate(wind_thresholds):
                for time_idx, time in enumerate(time_intervals):
                    combo_group = combo_data[(combo_data['wind'] == wind) & (combo_data['time'] == time)]
                    if not combo_group.empty:
                        ri_index_matrix[wind_idx, time_idx] = np.mean(combo_group['y_pred'])
                        mse_matrix[wind_idx, time_idx] = mean_squared_error(combo_group['y_true'], combo_group['y_pred'])
            plt.figure(figsize=(14, 10))
            ax = sns.heatmap(ri_index_matrix[::-1], xticklabels=time_intervals, yticklabels=wind_thresholds[::-1],
                             cmap="YlOrRd", annot=True, fmt=".2f", linewidths=0.5, linecolor='black',
                             cbar_kws={'label': f"Predicted RI Index - {regions['upper_inner']['name']} ({test_year_range})"})
            plt.title(f"Validation RI Index - {model_name} ({test_year_range})", fontsize=16)
            plt.xlabel("Time Interval (h)", fontsize=14)
            plt.ylabel("Wind Speed Threshold (kt)", fontsize=14)
            plt.tick_params(axis='both', labelsize=12)
            plt.savefig(os.path.join(output_dir, f"RI_Index_heatmap_validation_{model_name}_{test_year_range}.png"), bbox_inches='tight', dpi=300)
            plt.close()
            logging.info(f"{model_name} 熱圖已儲存至 {os.path.join(output_dir, f'RI_Index_heatmap_validation_{model_name}_{test_year_range}.png')}")
        return high_threshold, low_threshold
    # 預測並計算偏差 (Random Forest)
    if rf_model:
        try:
            y_test_pred_rf = rf_model.predict(rf_feature_scaler.transform(X_test))
            y_test_pred_rf = np.clip(y_test_pred_rf, 0, 10) # 限制 RF 預測值範圍
            high_threshold_rf, low_threshold_rf = process_model("Random_Forest", y_test_pred_rf, rf_target_scaler, delta_winds_list)
        except Exception as e:
            logging.error(f"Random Forest 預測錯誤: {e}")
    # 預測並計算偏差 (SVR)
    if svr_model:
        try:
            y_test_pred_svr = svr_model.predict(svr_feature_scaler.transform(X_test))
            y_test_pred_svr = np.clip(y_test_pred_svr, 0, 10) # 限制 SVR 預測值範圍
            high_threshold_svr, low_threshold_svr = process_model("SVR", y_test_pred_svr, svr_target_scaler, delta_winds_list)
        except Exception as e:
            logging.error(f"SVR 預測錯誤: {e}")
    # 預測並計算偏差 (ANN)
    if ann_model:
        try:
            y_test_pred_ann = ann_model.predict(ann_feature_scaler.transform(X_test)).flatten()
            y_test_pred_ann = np.clip(y_test_pred_ann, 0, 10) # 限制 ANN 預測值範圍
            high_threshold_ann, low_threshold_ann = process_model("ANN", y_test_pred_ann, ann_target_scaler, delta_winds_list)
        except Exception as e:
            logging.error(f"ANN 預測錯誤: {e}")
    # 計算與訓練閾值的偏差
    # Random Forest 閾值
    rf_train_high = 4.29
    rf_train_low = 1.71
    # SVR 閾值
    svr_train_high = 4.63
    svr_train_low = 1.34
    # ANN 閾值
    ann_train_high = 4.72
    ann_train_low = 1.33
    if 'high_threshold_rf' in locals():
        high_dev_rf = ((high_threshold_rf - rf_train_high) / rf_train_high) * 100
        low_dev_rf = ((low_threshold_rf - rf_train_low) / rf_train_low) * 100
        logging.info(f"Random Forest High Threshold Deviation: {high_dev_rf:.2f}%, Low Threshold Deviation: {low_dev_rf:.2f}%")
        print(f"Random Forest High Threshold Deviation: {high_dev_rf:.2f}%")
        print(f"Random Forest Low Threshold Deviation: {low_dev_rf:.2f}%")
    if 'high_threshold_svr' in locals():
        high_dev_svr = ((high_threshold_svr - svr_train_high) / svr_train_high) * 100
        low_dev_svr = ((low_threshold_svr - svr_train_low) / svr_train_low) * 100
        logging.info(f"SVR High Threshold Deviation: {high_dev_svr:.2f}%, Low Threshold Deviation: {low_dev_svr:.2f}%")
        print(f"SVR High Threshold Deviation: {high_dev_svr:.2f}%")
        print(f"SVR Low Threshold Deviation: {low_dev_svr:.2f}%")
    if 'high_threshold_ann' in locals():
        high_dev_ann = ((high_threshold_ann - ann_train_high) / ann_train_high) * 100
        low_dev_ann = ((low_threshold_ann - ann_train_low) / ann_train_low) * 100
        logging.info(f"ANN High Threshold Deviation: {high_dev_ann:.2f}%, Low Threshold Deviation: {low_dev_ann:.2f}%")
        print(f"ANN High Threshold Deviation: {high_dev_ann:.2f}%")
        print(f"ANN Low Threshold Deviation: {low_dev_ann:.2f}%")
if __name__ == "__main__":
    main()
    logging.info("驗證完成。請檢查日誌檔案以獲取詳細結果。")