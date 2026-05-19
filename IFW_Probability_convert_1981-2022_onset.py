import os
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# ====================== 基本設定（使用 classweight 版本） ======================
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/M1981-2022_onset_test-classweight"
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "validation_log_1981-2022_test.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
              logging.StreamHandler()]
)

def flush_log():
    for handler in logging.getLogger().handlers:
        handler.flush()

logging.info("=== 腳本開始執行 - 只驗證 ANN ClassWeight 版本 ===")
flush_log()

# ====================== RI 計算參數 ======================
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)}
}
regions = {
    #"upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
    # "upper_outer": {"rows": range(2, 12), "cols": range(9, 18), "name": "Upper Level Outer Area"},
    # "middle_inner": {"rows": range(12, 21), "cols": range(0, 9), "name": "Middle Level Inner-Core"},
    # "middle_outer": {"rows": range(12, 21), "cols": range(9, 18), "name": "Middle Level Outer Area"},
    "lower_inner": {"rows": range(21, 27), "cols": range(0, 9), "name": "Lower Level Inner-Core"},
    # "lower_outer": {"rows": range(21, 27), "cols": range(9, 18), "name": "Lower Level Outer Area"},
    # "midlower_inner": {"rows": range(12, 27), "cols": range(0, 9), "name": "Mid-Lower Level Inner-Core"},
    # "midlower_outer": {"rows": range(12, 27), "cols": range(9, 18), "name": "Mid-Lower Level Outer Area"}
}

test_year_range = "1981-2022"
test_base_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-1981-2022"
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/1981-2022-continue-classweight"
best_track_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/ibtracs.WP.list.v04r01.csv"

c = 210
d = 25
max_wind = max(wind_thresholds)

# ====================== RI 計算函數 ======================
def calculate_ri(wind, time_h, c=210, d=25, max_wind=100):
    return wind * (c / (time_h + d)) * (1 + wind / max_wind)

# ====================== 只載入 ANN 模型與 feature_scaler（移除 target_scaler） ======================
model_files = {
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_model_onset_weighted_lower_inner_1981-2022.pkl")
}
feature_scaler_files = {
    "ann": os.path.join(train_pkl_dir, "PVTHE_ann_feature_scaler_onset_weighted_lower_inner_1981-2022.pkl")
}

ann_model = None
ann_feature_scaler = None

try:
    ann_model = joblib.load(model_files["ann"])
    ann_feature_scaler = joblib.load(feature_scaler_files["ann"])
    logging.info("✅ ANN 模型與 feature_scaler 載入成功")
    flush_log()
except Exception as e:
    logging.error(f"❌ ANN 載入失敗: {e}")
    flush_log()

# ====================== 資料調整函數 ======================
def adjust_data(data, var, region_rows, region_cols):
    if not isinstance(data, np.ndarray):
        logging.error(f"adjust_data 收到非 numpy array 類型: {type(data)}")
        return np.array([])
   
    data = np.asarray(data, dtype=np.float64)
   
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

# ====================== 生成測試資料 ======================
def generate_test_data(time_intervals, test_base_path, regions):
    X_test = []
    delta_winds_list = []
    region = "lower_inner"
    if region not in regions:
        logging.error(f"區域 {region} 未定義")
        flush_log()
        return np.array([]), []
    region_rows = regions[region]["rows"]
    region_cols = regions[region]["cols"]
    logging.info(f"開始生成測試資料，區域: {regions[region]['name']}")
    flush_log()
    delta_cache = {}
    bt_data_cache = {}
    bt_data = pd.read_csv(best_track_path, low_memory=False)
    logging.info("已載入 ibtracs 最佳路徑資料")
    flush_log()
    whole_path = os.path.join(test_base_path, "Whole")
    logging.info(f"Whole 路徑: {whole_path}")
    flush_log()
    if os.path.exists(whole_path):
        pv_files = [os.path.join(whole_path, f) for f in os.listdir(whole_path)
                    if f.startswith('Azi_PV-') and f.endswith(".txt")]
        logging.info(f"找到 PV 檔案數: {len(pv_files)}")
        flush_log()
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
            sid_files.setdefault(sid, []).append((current_dt, pv_file))
        for sid in sid_files:
            sid_files[sid].sort(key=lambda x: x[0])
        for sid, files in sid_files.items():
            if len(files) < 2:
                logging.info(f"跳過 SID {sid}: 只有 {len(files)} 筆資料")
                flush_log()
                continue
            if sid not in bt_data_cache:
                bt_data_sid = bt_data[bt_data['SID'].str.endswith(sid)].copy()
                bt_data_sid['ISO_TIME'] = pd.to_datetime(bt_data_sid['ISO_TIME'])
                if bt_data_sid.empty:
                    logging.warning(f"未找到 SID {sid} 在 best track CSV")
                    flush_log()
                    continue
                bt_data_cache[sid] = bt_data_sid
            for idx in range(1, len(files)):
                current_dt, pv_file = files[idx]
                the_file = os.path.join(whole_path, os.path.basename(pv_file).replace('Azi_PV', 'Azi_THE'))
                if not os.path.exists(the_file):
                    logging.warning(f"缺少 THE 檔案: {the_file}")
                    flush_log()
                    continue
                try:
                    pv_df = pd.read_csv(pv_file, sep=r'\s+', header=None, dtype=float, engine='python', na_values=['', ' ', '-'])
                    the_df = pd.read_csv(the_file, sep=r'\s+', header=None, dtype=float, engine='python', na_values=['', ' ', '-'])
                    pv_data = pv_df.values
                    the_data = the_df.values
                    if pv_data.shape != (27, 21) or the_data.shape != (27, 21):
                        logging.warning(f"形狀異常 {os.path.basename(pv_file)}，跳過")
                        continue
                    pv_data = np.nan_to_num(pv_data, nan=0.0).astype(np.float64)
                    the_data = np.nan_to_num(the_data, nan=0.0).astype(np.float64)
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
                        wind_current = float(current_row.get('USA_WIND', 0))
                        delta_winds = []
                        for t in time_intervals:
                            future_dt = current_dt + timedelta(hours=t)
                            future_row = bt_data_sid[bt_data_sid['ISO_TIME'] == future_dt]
                            if future_row.empty:
                                closest_idx = (bt_data_sid['ISO_TIME'] - future_dt).abs().argmin()
                                future_row = bt_data_sid.iloc[closest_idx]
                            else:
                                future_row = future_row.iloc[0]
                            wind_future = float(future_row.get('USA_WIND', 0))
                            delta_winds.append(wind_future - wind_current)
                        delta_cache[cache_key] = (wind_current, delta_winds)
                    X_test.append(combined_features)
                    delta_winds_list.append(delta_winds)
                    logging.info(f"✅ 成功處理完成: {os.path.basename(pv_file)}")
                except Exception as e:
                    logging.error(f"處理檔案時出錯 {pv_file}: {type(e).__name__} - {e}")
                    flush_log()
                    continue
    else:
        logging.warning(f"資料夾不存在: {whole_path}")
        flush_log()
    logging.info(f"總測試樣本數: {len(X_test)}")
    flush_log()
    return np.array(X_test), delta_winds_list

# ====================== 主程式 ======================
def main():
    logging.info(f"開始驗證 {test_year_range} 測試集資料與 ANN ClassWeight 模型")
    flush_log()
    if ann_model is None:
        logging.error("ANN 模型載入失敗，程式終止")
        flush_log()
        return

    logging.info("開始生成測試資料...")
    flush_log()
    X_test, delta_winds_list = generate_test_data(time_intervals, test_base_path, regions)

    if len(X_test) == 0:
        logging.warning("測試數據生成失敗或數量不足")
        flush_log()
        return

    logging.info(f"成功生成測試數據，樣本數: {len(X_test)}")
    flush_log()

    # ====================== 處理 ANN 模型 ======================
    def process_model(model_name, X_test, delta_winds_list, feature_scaler, model):
        try:
            y_test_pred = model.predict(feature_scaler.transform(X_test)).flatten()
            y_test_pred = np.clip(y_test_pred, 0, 10)

            mean_ri = np.mean(y_test_pred)
            std_ri = np.std(y_test_pred)
            high_threshold = mean_ri + std_ri
            low_threshold = mean_ri - std_ri

            logging.info(f"{model_name} Mean RI Index = {mean_ri:.3f} ± {std_ri:.3f} | High = {high_threshold:.3f}, Low = {low_threshold:.3f}")

            # ====================== 計算真實 target IR 並 scale 到 0~10 ======================
            scaler_y = MinMaxScaler(feature_range=(0, 10))
            y_true_raw = []
            for s in range(len(delta_winds_list)):
                idx_24 = time_intervals.index(24) if 24 in time_intervals else 3
                delta_24 = delta_winds_list[s][idx_24]
                if np.isnan(delta_24):
                    delta_24 = 0.0
                true_ir_raw = calculate_ri(delta_24, 24, c, d, max_wind)
                y_true_raw.append(true_ir_raw)

            y_true_raw = np.array(y_true_raw).reshape(-1, 1)
            y_true = scaler_y.fit_transform(y_true_raw).ravel()

            # 計算 MSE 和 R²
            mse = np.mean((y_test_pred - y_true) ** 2)
            ss_res = np.sum((y_test_pred - y_true) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            logging.info(f"{model_name} MSE = {mse:.4f}, R² = {r2:.4f}")
            flush_log()

            # ====================== KDE 繪圖（n<5 用虛線 + 參考線） ======================
            bins = np.linspace(np.min(y_test_pred), np.max(y_test_pred), 11)
            #bin_indices = np.digitize(y_test_pred, bins) - 1
            #counts, _ = np.histogram(y_test_pred, bins=bins)
            
            # 使用 pd.cut 更穩健地分配 bin
            bin_labels = [f"Bin {k+1} [{bins[k]:.2f}-{bins[k+1]:.2f})" for k in range(10)]
        
            # 強制把 y_test_pred 分配到正確的 bin
            df_temp = pd.DataFrame({'y_pred': y_test_pred})
            df_temp['bin_idx'] = pd.cut(df_temp['y_pred'], bins=bins, labels=False, include_lowest=True)
        
            counts = np.zeros(10, dtype=int)
            for i in range(10):
                counts[i] = (df_temp['bin_idx'] == i).sum()
            bin_labels = [f"Bin {k+1} [{bins[k]:.2f}-{bins[k+1]:.2f}) (n={counts[k]})" for k in range(10)]

            data = []
            for s in range(len(delta_winds_list)):
                #bin_idx = bin_indices[s]
                bin_idx = df_temp['bin_idx'].iloc[s]
                #if bin_idx < 0 or bin_idx >= 10:
                if pd.isna(bin_idx) or bin_idx < 0 or bin_idx >= 10:
                    continue
                bin_label = bin_labels[bin_idx]
                for j, t in enumerate(time_intervals):
                    delta = delta_winds_list[s][j]
                    if np.isnan(delta):
                        continue
                    data.append({'bin': bin_label, 'time': t, 'delta_wind': delta})

            df = pd.DataFrame(data)
            if not df.empty:
                df['time_jitter'] = df['time'] + np.random.uniform(-0.4, 0.4, len(df))

                plt.figure(figsize=(11, 11))
                levels = np.linspace(0.5, 0.9, 5)
                colors = sns.color_palette("coolwarm", 10)
                
                # 設定不同機率等值線的粗細（越高機率越粗）
                linewidths = [1.0, 1.5, 2.0, 2.5, 3.0]

                collection_indices = []
                for i in range(10):
                    bin_df = df[df['bin'] == bin_labels[i]]
                    n_points = len(bin_df)
                    if n_points == 0:
                        collection_indices.append(None)
                        continue
                    
                    
                    current_collections_count = len(plt.gca().collections)
                    
                    try:
                        sns.kdeplot(data=bin_df, x='time_jitter', y='delta_wind',
                                levels=levels, thresh=0.05, color=colors[i],
                                linewidths=linewidths,
                                common_norm=False)
                        logging.info(f"Bin {i+1} (n={n_points}) 產生 KDE collection")
                   # 檢查是否真的產生了新的 collection
                        if len(plt.gca().collections) > current_collections_count:
                            collection_indices.append(len(plt.gca().collections) - 1)
                            logging.info(f"Bin {i+1} (n={n_points}) 產生 KDE collection2")
                        else:
                                collection_indices.append(None)
                                logging.info(f"Bin {i+1} (n={n_points}) 沒有產生 KDE collection")

                    except:
                        collection_indices.append(None)
                        logging.info(f"Bin {i+1} KDE 繪製失敗 (n={n_points})")
                # 根據樣本數設定虛線
                for i, coll_idx in enumerate(collection_indices):
                    if coll_idx is not None and counts[i] < 5:
                        plt.gca().collections[coll_idx].set_linestyle('--')

                # 參考線
                plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Wind Change (0 kt)')
                plt.axhline(y=30, color='red', linestyle='--', linewidth=1.8, alpha=0.85, label='Traditional RI Threshold (30 kt)')
                plt.axvline(x=24, color='red', linestyle='--', linewidth=1.8, alpha=0.85, label='24 hours')

                # 圖例
                bin_legend_elements = []
                for i in range(10):
                    ls = '--' if counts[i] < 5 else '-'
                    bin_legend_elements.append(
                        Line2D([0], [0], color=colors[i], lw=2, linestyle=ls, label=bin_labels[i])
                    )

                bin_legend = plt.legend(handles=bin_legend_elements,
                                        title='Predicted IR Index Bins',
                                        loc='upper right', fontsize=9, title_fontsize=11)
                plt.gca().add_artist(bin_legend)
                
                # 機率等值線粗細圖例（左上方，避免與 Bin 圖例重疊）
                thickness_legend_elements = []
                for i, lw in enumerate(linewidths):
                    thickness_legend_elements.append(
                        Line2D([0], [0], color='gray', lw=lw, linestyle='-', label=f'{levels[i]*100:.0f}%')
                    )

                thickness_legend = plt.legend(handles=thickness_legend_elements,
                                              title='Probability Levels',
                                              loc='upper left', fontsize=9, title_fontsize=11)
                plt.gca().add_artist(thickness_legend)
                # 參考線圖例（右下方，避免重疊）
                plt.legend(loc='lower right', fontsize=9)


                plt.xticks(time_intervals)
                plt.yticks(np.arange(-70, max(wind_thresholds) + 10, 10))
                plt.ylim(-70, max(wind_thresholds) + 10)
                plt.xlabel("Time Interval (h)", fontsize=14)
                plt.ylabel("Wind Change (kt)", fontsize=14)
                plt.title(f"2D KDE by Predicted IR Index Bins (ANN ClassWeight, {test_year_range})", fontsize=16)
                plt.tick_params(axis='both', labelsize=10)

                save_path = os.path.join(output_dir, f"Wind_Change_KDE_by_Bin_ANN_lower_inner_{test_year_range}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logging.info(f"✅ ANN KDE 圖已儲存至 {save_path} (總樣本數: {len(X_test)})")
                flush_log()

                return high_threshold, low_threshold

        except Exception as e:
            logging.error(f"❌ ANN 處理錯誤: {e}")
            flush_log()

    # 執行 ANN
    if ann_model is not None and len(X_test) > 0:
        logging.info("=== 開始處理 ANN ===")
        high, low = process_model("ANN", X_test, delta_winds_list, ann_feature_scaler, ann_model)

    logging.info("驗證完成！請檢查 log 檔案與 KDE 圖檔")
    flush_log()

if __name__ == "__main__":
    main()
    logging.info("驗證完成！請檢查 log 檔案與 KDE 圖檔")
    flush_log()