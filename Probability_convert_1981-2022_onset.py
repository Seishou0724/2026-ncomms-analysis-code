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
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
    "W": {"rows": range(0, 27), "cols": range(0, 21)}
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"}
}
test_year_range = "1981-2022" # 使用訓練年的測試集
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/1981-2022-continue"

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

# 訓練測試集資料的儲存路徑
test_data_files = {
    "random_forest": {
        "X_test": os.path.join(train_pkl_dir, "PVTHE_X_test_rf_kfold_upper_inner_1981-2022.pkl"),
        "y_test": os.path.join(train_pkl_dir, "PVTHE_y_test_rf_kfold_upper_inner_1981-2022.pkl"),
        "combo_test": os.path.join(train_pkl_dir, "PVTHE_combo_test_rf_kfold_upper_inner_1981-2022.pkl"),
        "delta_winds": os.path.join(train_pkl_dir, "PVTHE_delta_winds_rf_kfold_upper_inner_1981-2022.pkl")
    },
    "svr": {
        "X_test": os.path.join(train_pkl_dir, "PVTHE_X_test_svr_kfold_upper_inner_1981-2022.pkl"),
        "y_test": os.path.join(train_pkl_dir, "PVTHE_y_test_svr_kfold_upper_inner_1981-2022.pkl"),
        "combo_test": os.path.join(train_pkl_dir, "PVTHE_combo_test_svr_kfold_upper_inner_1981-2022.pkl"),
        "delta_winds": os.path.join(train_pkl_dir, "PVTHE_delta_winds_svr_kfold_upper_inner_1981-2022.pkl")
    },
    "ann": {
        "X_test": os.path.join(train_pkl_dir, "PVTHE_X_test_ann_kfold_upper_inner_1981-2022.pkl"),
        "y_test": os.path.join(train_pkl_dir, "PVTHE_y_test_ann_kfold_upper_inner_1981-2022.pkl"),
        "combo_test": os.path.join(train_pkl_dir, "PVTHE_combo_test_ann_kfold_upper_inner_1981-2022.pkl"),
        "delta_winds": os.path.join(train_pkl_dir, "PVTHE_delta_winds_ann_kfold_upper_inner_1981-2022.pkl")
    }
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

# 載入訓練測試集資料函數
def load_test_data(model_name):
    try:
        # 嘗試載入儲存的 20% 測試集資料
        X_test = joblib.load(test_data_files[model_name]["X_test"])
        y_test = joblib.load(test_data_files[model_name]["y_test"])
        combo_labels = joblib.load(test_data_files[model_name]["combo_test"])
        delta_winds_list = joblib.load(test_data_files[model_name]["delta_winds"])
        logging.info(f"成功載入 {model_name} 的 20% 測試集資料，樣本數: {len(X_test)}")
        return X_test, y_test, combo_labels, delta_winds_list
    except FileNotFoundError as e:
        logging.error(f"無法載入 {model_name} 的測試集資料: {e}")
        return np.array([]), np.array([]), [], []

# 主程式
def main():
    logging.info(f"開始驗證 {test_year_range} 測試集資料與 1981–2022 模型的預測偏差")
    if not any([rf_model, svr_model, ann_model]):
        logging.error("所有模型載入失敗，程式終止")
        return

    # 定義通用函數來處理每個模型的計算和繪圖
    def process_model(model_name, X_test, y_test, combo_labels, delta_winds_list, feature_scaler, target_scaler, model):
        # 計算預測值
        if model_name == "ANN":
            y_test_pred = model.predict(feature_scaler.transform(X_test)).flatten()
        else:
            y_test_pred = model.predict(feature_scaler.transform(X_test))
        
        logging.info(f"{model_name} y_test_pred min: {np.min(y_test_pred):.2f}, max: {np.max(y_test_pred):.2f}")
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
        bins = np.linspace(np.min(y_pred_ri), np.max(y_pred_ri), 11)
        bin_indices = np.digitize(y_pred_ri, bins) - 1
        
        # 新增: 檢查 y_pred_ri 分布 - 保存 histogram 圖
        logging.info(f"y_pred_ri 平均: {np.mean(y_pred_ri):.2f}, 標準差: {np.std(y_pred_ri):.2f}")
        plt.figure(figsize=(8, 6))
        plt.hist(y_pred_ri, bins=20, edgecolor='black')
        plt.title(f"IR Index Prediction Distribution ({model_name}, {test_year_range})")
        plt.xlabel("Predicted IR Index")
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

        # 平均風變化熱圖 (使用新 bin_labels)
        plt.figure(figsize=(12, 8))
        sns.heatmap(avg_delta_matrix, xticklabels=time_intervals, yticklabels=bin_labels,
                    cmap="coolwarm", annot=True, fmt=".2f")
        plt.title(f"Average Wind Change by RI Index Bin and Time Interval ({model_name}, {test_year_range})", fontsize=16)
        plt.xlabel("Time Interval (h)", fontsize=14)
        plt.ylabel("IR Index Bin (Range with Count)", fontsize=14)
        plt.tick_params(axis='both', labelsize=10)  # 稍小以避免蓋圖
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
            plt.tick_params(axis='both', labelsize=10)  # 稍小
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
                
                
                
                
                # 1. 水平參考線：Zero wind speed change
                plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, 
                            label='Zero Wind Change (0 kt)')

                # 2. 垂直參考線：Traditional 30 kt / 24 h RI threshold
                plt.axvline(x=24, color='red', linestyle='--', linewidth=1.8, alpha=0.85, 
                            label='Traditional RI Threshold\n(30 kt / 24 h)')
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
            plt.title(f"2D KDE and IR Index Bin ({model_name}, {test_year_range})", fontsize=16)
            plt.tick_params(axis='both', labelsize=10)  # 稍小以避免蓋圖
            plt.savefig(os.path.join(output_dir, f"Wind_Change_KDE_by_Bin_and_Time_{model_name}_{test_year_range}.png"), dpi=300)
            plt.close()
            logging.info(f"Wind Change KDE plot saved to {os.path.join(output_dir, f'Wind_Change_KDE_by_Bin_and_Time_{model_name}_{test_year_range}.png')}")

        # 生成 RI Index 熱圖
        if not np.all(np.isnan(y_test_pred)):
            ri_index_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
            mse_matrix = np.full((len(wind_thresholds), len(time_intervals)), np.nan)
            combo_data = pd.DataFrame({
                'y_pred': y_test_pred,
                'y_true': y_test_scaled,
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

    # 載入測試資料並進行預測
    logging.info("開始載入測試資料...")
    results = {}
    if rf_model:
        try:
            X_test_rf, y_test_rf, combo_labels_rf, delta_winds_rf = load_test_data("random_forest")
            if len(X_test_rf) == 0:
                logging.error("Random Forest 測試集資料為空，跳過處理")
            else:
                high_threshold_rf, low_threshold_rf = process_model(
                    "Random_Forest", X_test_rf, y_test_rf, combo_labels_rf, delta_winds_rf,
                    rf_feature_scaler, rf_target_scaler, rf_model
                )
                results["random_forest"] = (high_threshold_rf, low_threshold_rf)
        except Exception as e:
            logging.error(f"Random Forest 預測錯誤: {e}")
    if svr_model:
        try:
            X_test_svr, y_test_svr, combo_labels_svr, delta_winds_svr = load_test_data("svr")
            if len(X_test_svr) == 0:
                logging.error("SVR 測試集資料為空，跳過處理")
            else:
                high_threshold_svr, low_threshold_svr = process_model(
                    "SVR", X_test_svr, y_test_svr, combo_labels_svr, delta_winds_svr,
                    svr_feature_scaler, svr_target_scaler, svr_model
                )
                results["svr"] = (high_threshold_svr, low_threshold_svr)
        except Exception as e:
            logging.error(f"SVR 預測錯誤: {e}")
    if ann_model:
        try:
            X_test_ann, y_test_ann, combo_labels_ann, delta_winds_ann = load_test_data("ann")
            if len(X_test_ann) == 0:
                logging.error("ANN 測試集資料為空，跳過處理")
            else:
                high_threshold_ann, low_threshold_ann = process_model(
                    "ANN", X_test_ann, y_test_ann, combo_labels_ann, delta_winds_ann,
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