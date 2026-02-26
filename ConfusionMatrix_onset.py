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
# 定義年份範圍（評估用，可改成 "2013-2022" 或 "1981-2022"）
eval_year_range = "1981-2022"
# 訓練年份範圍（用來載入模型）
train_year_range = "1981-2022"
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
# 修改: 生成測試資料函數 (載入所有資料點，排除每個 TC 的第一筆，每個 dt 只處理一次)
def load_test_data(base_path, regions, time_intervals, best_track_path):
    X_dict = {} # {time: list of concat_features}
    delta_dict = {} # {time: list of delta for the segment}
    region = "upper_inner"
    region_rows = regions[region]["rows"]
    region_cols = regions[region]["cols"]
    logging.info(f"開始生成測試資料，區域: {regions[region]['name']}")
    bt_data_cache = {} # 緩存 per SID 的 bt_data_sid
    # 載入整個 ibtracs CSV 只一次
    bt_data = pd.read_csv(best_track_path, low_memory=False)
    # 資料在 Whole 資料夾
    whole_path = os.path.join(base_path, "Whole")
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
        # 預載 per SID 的 bt_data
        for sid in sid_files:
            bt_data_sid = bt_data[bt_data['SID'].str.endswith(sid)].copy()
            bt_data_sid['ISO_TIME'] = pd.to_datetime(bt_data_sid['ISO_TIME'])
            if bt_data_sid.empty:
                logging.warning(f"未找到 SID {sid} 在 best track CSV")
                continue
            bt_data_cache[sid] = bt_data_sid
        # 為每個 time_interval 收集序列
        for time in time_intervals:
            steps = time // 6 + 1 # 假設每6h一筆，steps = number of points
            X_dict[time] = []
            delta_dict[time] = []
            for sid, files in sid_files.items():
                if sid not in bt_data_cache:
                    continue
                if len(files) < steps:
                    continue
                for start_idx in range(1, len(files) - steps + 1): # 從1開始，排除每個 TC 的第一筆
                    sequence = sid_files[sid][start_idx:start_idx + steps]
                    concat_features = []
                    for dt, pv_file in sequence:
                        f_base = os.path.basename(pv_file)
                        the_file = os.path.join(whole_path, f_base.replace('Azi_PV', 'Azi_THE'))
                        if not os.path.exists(the_file):
                            logging.warning(f"缺少 THE 檔案 for {pv_file}")
                            break
                        try:
                            pv_data = np.genfromtxt(pv_file, delimiter='', invalid_raise=False, filling_values=np.nan)
                            the_data = np.genfromtxt(the_file, delimiter='', invalid_raise=False, filling_values=np.nan)
                            pv_selected = adjust_data(pv_data, "PV", region_rows, region_cols)
                            the_selected = adjust_data(the_data, "THE", region_rows, region_cols)
                            if pv_selected.size == 0 or the_selected.size == 0:
                                logging.warning(f"選取數據為空: {pv_file}")
                                break
                            combined = np.concatenate([pv_selected, the_selected])
                            concat_features.append(combined)
                        except Exception as e:
                            logging.error(f"處理檔案時出錯 {pv_file}: {e}")
                            break
                    if len(concat_features) == steps:
                        concat_features = np.concatenate(concat_features)
                        X_dict[time].append(concat_features)
                        # 計算 delta = wind_last - wind_first
                        first_dt = sequence[0][0]
                        last_dt = sequence[-1][0]
                        bt_data_sid = bt_data_cache[sid]
                        first_row = bt_data_sid[bt_data_sid['ISO_TIME'] == first_dt]
                        last_row = bt_data_sid[bt_data_sid['ISO_TIME'] == last_dt]
                        if first_row.empty or last_row.empty:
                            logging.warning(f"無法找到 wind for {sid} at {first_dt} or {last_dt}")
                            continue
                        wind_first_str = first_row.iloc[0]['USA_WIND'] if not pd.isna(first_row.iloc[0]['USA_WIND']) else first_row.iloc[0]['USA_WIND']
                        wind_last_str = last_row.iloc[0]['USA_WIND'] if not pd.isna(last_row.iloc[0]['USA_WIND']) else last_row.iloc[0]['USA_WIND']
                        if wind_first_str == ' ' or wind_last_str == ' ':
                            logging.warning(f"風速值為空格，跳過樣本 {sid} at {first_dt}-{last_dt}")
                            continue
                        wind_first = float(wind_first_str)
                        wind_last = float(wind_last_str)
                        delta = wind_last - wind_first
                        delta_dict[time].append(delta)
            logging.info(f"For time={time}: 收集到 {len(X_dict[time])} 個序列樣本")
    else:
        logging.warning("資料夾不存在: Whole 等")
    return X_dict, delta_dict
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
    # 載入數據 (只載入一次)
    X_dict, delta_dict = load_test_data(base_path, regions, time_intervals, best_track_path)
    # 為每個模型處理
    for model_type, model in models.items():
        logging.info(f"處理模型: {model_type}")
        # 載入該模型的 scaler (修改為 per model)
        scaler = joblib.load(os.path.join(train_output_dir, f"PVTHE_{model_type}_feature_scaler_kfold_{region}_{train_year_range}.pkl"))
        scaler_y = joblib.load(os.path.join(train_output_dir, f"PVTHE_{model_type}_target_scaler_kfold_{region}_{train_year_range}.pkl"))
    
        # 為每個 time 計算 confusion
        confusion_matrix = np.zeros((len(wind_thresholds), len(time_intervals), 4)) # [hit, miss, fp, tn]
        feature_size = 180 # PV + THE for each time point: 90 + 90 = 180
    
        for t_idx, time in enumerate(time_intervals):
            X_sequence_list = X_dict.get(time, [])
            delta_for_t = np.array(delta_dict.get(time, []))
            if len(X_sequence_list) == 0:
                logging.warning(f"For time={time}: 無序列資料")
                continue
            # 預測: 分開每個時間點預測，然後平均
            y_pred_scaled = []
            for sequence in X_sequence_list:
                steps = time // 6 + 1
                if len(sequence) != steps * feature_size:
                    logging.warning(f"序列長度不匹配 for time={time}: {len(sequence)} != {steps} * {feature_size}")
                    continue
                preds = []
                for i in range(steps):
                    chunk = sequence[i * feature_size : (i + 1) * feature_size].reshape(1, -1)
                    chunk_scaled = scaler.transform(chunk)
                    pred = model.predict(chunk_scaled)[0]
                    preds.append(pred)
                avg_pred = np.mean(preds)
                y_pred_scaled.append(np.clip(avg_pred, 0, 10))
            y_pred_scaled = np.array(y_pred_scaled)
        
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
    
        for w_idx, wind in enumerate(wind_thresholds):
            for t_idx, time in enumerate(time_intervals):
                conf = confusion_matrix[w_idx, t_idx]
                hit, miss, fp, tn = conf
                total_ri = hit + miss
                total_nonri = fp + tn
            
                # 如果 CR == 100% (fp == 0)，則空白整個組合
                if fp == 0 and total_nonri > 0:
                    expanded_matrix[2*w_idx:2*w_idx+2, 2*t_idx:2*t_idx+2] = np.nan
                    annot_matrix[2*w_idx:2*w_idx+2, 2*t_idx:2*t_idx+2] = ""
                    continue
            
                # 否則正常顯示，包括 0% 的情況
                hit_pct = (hit / total_ri * 100) if total_ri > 0 else 0.0
                miss_pct = (miss / total_ri * 100) if total_ri > 0 else 0.0
                expanded_matrix[2*w_idx, 2*t_idx] = hit_pct
                expanded_matrix[2*w_idx, 2*t_idx+1] = miss_pct
                annot_matrix[2*w_idx, 2*t_idx] = f"Hit: {hit_pct:.1f}%"
                annot_matrix[2*w_idx, 2*t_idx+1] = f"Miss: {miss_pct:.1f}%"
            
                fp_pct = (fp / total_nonri * 100) if total_nonri > 0 else 0.0
                tn_pct = (tn / total_nonri * 100) if total_nonri > 0 else 0.0
                expanded_matrix[2*w_idx+1, 2*t_idx] = fp_pct
                expanded_matrix[2*w_idx+1, 2*t_idx+1] = tn_pct
                annot_matrix[2*w_idx+1, 2*t_idx] = f"FA: {fp_pct:.1f}%"
                annot_matrix[2*w_idx+1, 2*t_idx+1] = f"CR: {tn_pct:.1f}%"
        # 畫熱圖
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
   
        plt.title(f"RI Confusion Matrix (Hit/Miss/FA/CR) - {region} ({eval_year_range}) - Model: {model_type.upper()}")
        plt.xlabel("Time Interval (h)")
        plt.ylabel("Wind Speed Threshold (kt)")
        heatmap_path = os.path.join(eval_output_dir, f"RI_confusion_heatmap_{model_type}_{region}_{eval_year_range}_{train_year_range}_onset.png")
        plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
        plt.close()
        logging.info(f"熱圖已儲存至 {heatmap_path}")
if __name__ == "__main__":
    try:
        main()
        logging.info("程式執行完畢")
    except Exception as e:
        logging.exception("程式執行時發生嚴重錯誤：")
    finally:
        logging.shutdown()