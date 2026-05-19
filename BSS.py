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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

# ====================== 基本設定 ======================
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/M1981-2022_onset_test"
os.makedirs(output_dir, exist_ok=True)

log_file_path = os.path.join(output_dir, "validation_log_1981-2022_test_bss.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
                              logging.StreamHandler()])

def flush_log():
    for handler in logging.getLogger().handlers:
        handler.flush()

logging.info("=== 開始執行 BSS 計算 (RF + SVR + ANN) ===")
flush_log()

# ====================== 必要變數定義 ======================
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)}
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"}
}
test_base_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-1981-2022"
best_track_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/ibtracs.WP.list.v04r01.csv"
train_pkl_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/1981-2022-continue"

# ====================== RI 定義 ======================
ri_definitions = [
    (20, 12), (25, 24), (30, 24), (35, 24),
    (40, 24), (45, 36), (55, 48)
]

# ====================== RI 計算參數 ======================
c = 210
d = 25
max_wind = 100

def calculate_ri(wind, time_h):
    return wind * (c / (time_h + d)) * (1 + wind / max_wind)

# ====================== 載入三個模型 ======================
models = {}
for m in ["rf", "svr", "ann"]:
    model_path = os.path.join(train_pkl_dir, f"PVTHE_{m}_model_onset_upper_inner_1981-2022.pkl")
    scaler_path = os.path.join(train_pkl_dir, f"PVTHE_{m}_feature_scaler_onset_upper_inner_1981-2022.pkl")
    try:
        models[m] = {
            "model": joblib.load(model_path),
            "scaler": joblib.load(scaler_path)
        }
        logging.info(f"✅ {m.upper()} 模型載入成功")
    except Exception as e:
        logging.error(f"❌ {m.upper()} 載入失敗: {e}")

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
    logging.info("已載入 ibtracs 最佳路徑資料")
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
            sid_files.setdefault(sid, []).append((current_dt, pv_file))
        for sid in sid_files:
            sid_files[sid].sort(key=lambda x: x[0])
        for sid, files in sid_files.items():
            if len(files) < 2:
                logging.info(f"跳過 SID {sid}: 只有 {len(files)} 筆資料")
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
                    pv_df = pd.read_csv(pv_file, sep=r'\s+', header=None, dtype=float, engine='python', na_values=['', ' ', '-'])
                    the_df = pd.read_csv(the_file, sep=r'\s+', header=None, dtype=float, engine='python', na_values=['', ' ', '-'])
                    pv_data = pv_df.values
                    the_data = the_df.values
                    if pv_data.shape != (27, 21) or the_data.shape != (27, 21):
                        continue
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
                    continue
    logging.info(f"總測試樣本數: {len(X_test)}")
    return np.array(X_test), delta_winds_list

# ====================== 主程式 ======================
def main():
    logging.info("開始生成測試資料...")
    X_test, delta_winds_list = generate_test_data(time_intervals, test_base_path, regions)
    if len(X_test) == 0:
        logging.error("測試數據生成失敗")
        return

    logging.info(f"成功生成測試數據，樣本數: {len(X_test)}")

    # 預測三個模型的 IR Index
    predictions = {}
    for name, data in models.items():
        y_pred = data["model"].predict(data["scaler"].transform(X_test))
        if name == "ann":
            y_pred = y_pred.flatten()
        predictions[name] = np.clip(y_pred, 0, 10)

    # ====================== 計算 BSS ======================
    results = {name: [] for name in ["rf_bin", "rf_lr", "svr_bin", "svr_lr", "ann_bin", "ann_lr"]}
    n_total = []
    n_ri = []
    labels = []

    for thresh, lead_time in ri_definitions:
        labels.append(f"{thresh}kt/{lead_time}h")

        idx = time_intervals.index(lead_time)
        ri_binary = np.array([1 if delta[idx] >= thresh else 0 for delta in delta_winds_list])
        climo_prob = np.mean(ri_binary)
        bs_climo = brier_score_loss(ri_binary, np.full_like(ri_binary, climo_prob))

        n_total.append(len(ri_binary))
        n_ri.append(np.sum(ri_binary))

        for model_name in ["rf", "svr", "ann"]:
            y_pred = predictions[model_name]

            # Bin Calibration
            bins = np.linspace(0, 10, 11)
            bin_idx = np.digitize(y_pred, bins) - 1
            prob_bin = np.zeros(10)
            for i in range(10):
                mask = (bin_idx == i)
                if np.any(mask):
                    prob_bin[i] = np.mean(ri_binary[mask])
            forecast_prob_bin = prob_bin[bin_idx]
            bs_bin = brier_score_loss(ri_binary, forecast_prob_bin)
            bss_bin = 1 - bs_bin / bs_climo if bs_climo > 0 else 0
            results[f"{model_name}_bin"].append(bss_bin * 100)

            # Logistic Regression
            lr = LogisticRegression()
            lr.fit(y_pred.reshape(-1, 1), ri_binary)
            forecast_prob_lr = lr.predict_proba(y_pred.reshape(-1, 1))[:, 1]
            bs_lr = brier_score_loss(ri_binary, forecast_prob_lr)
            bss_lr = 1 - bs_lr / bs_climo if bs_climo > 0 else 0
            results[f"{model_name}_lr"].append(bss_lr * 100)

    # ====================== 畫圖 ======================
    x = np.arange(len(labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['skyblue', 'deepskyblue', 'lightgreen', 'limegreen', 'salmon', 'red']
    methods = ["rf_bin", "rf_lr", "svr_bin", "svr_lr", "ann_bin", "ann_lr"]
    labels_legend = ["RF Bin", "RF LR", "SVR Bin", "SVR LR", "ANN Bin", "ANN LR"]

    for i, method in enumerate(methods):
        ax.bar(x + (i-2.5)*width, results[method], width, label=labels_legend[i], color=colors[i], edgecolor='black')

    # 在 x 軸下方加上 N 和 NRI 數量（位置調整，避免被切掉）
    for i in range(len(labels)):
        ax.text(x[i], -4, f'N={n_total[i]}\nNRI={n_ri[i]}', 
                ha='center', va='top', fontsize=9)

    ax.set_ylabel('Brier Skill Score (%)', fontsize=14)
    ax.set_xlabel('RI Threshold', fontsize=14)
    ax.set_title('Brier Skill Score Comparison for Different RI Definitions\n(RF, SVR, ANN)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)   # 不旋轉
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # 調整 Y 軸範圍，讓文字不會被切掉
    ax.set_ylim(-8, max(max(results[m]) for m in methods) * 1.15)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"BSS_Comparison_RF_SVR_ANN.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"✅ BSS 比較圖已儲存至 {save_path}")

    # 輸出數值
    for i, label in enumerate(labels):
        logging.info(f"{label} (N={n_total[i]}, NRI={n_ri[i]}):")
        for j, method in enumerate(methods):
            logging.info(f"   {labels_legend[j]}: {results[method][i]:.2f}%")

if __name__ == "__main__":
    main()