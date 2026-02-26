import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re
import matplotlib.ticker as ticker
# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
# 定義年份範圍
year_ranges = ["2013-2022", "1981-2022"]
# 基本參數
base_path_template = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{}"
best_track_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/ibtracs.WP.list.v04r01.csv"
whole_folder = "Whole"
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]
# 方案3: 8 時間點
variables = {
    "PV": {"rows": range(0, 27), "cols": range(0, 21)},
    "THE": {"rows": range(0, 27), "cols": range(0, 21)},
}
regions = {
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
}
# 數據調整函數（放寬：移除全0檢查，只留全NaN）
def adjust_data(data, var, region_rows, region_cols):
    if data.shape != (27, 21):
        return np.array([])
    outer_cols = list(range(18, 21))
    if not all(0 <= col < data.shape[1] for col in outer_cols):
        return np.array([])
    if np.all(np.isnan(data)):
        return np.array([]) # 只檢查全NaN，移除 || np.all(data == 0)
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
    if selected_data.size == 0 or np.all(np.isnan(selected_data)):
        # 移除 || np.all(selected_data == 0)
        return np.array([])
    return selected_data.flatten()
# 解析檔案名：提取時間和 SID (後六碼)
def parse_filename(filename):
    parts = filename.replace('.txt', '').split('-')
    if len(parts) != 3:
        return None, None
    timestr = parts[1]
    sid_full = parts[2]
    sid_short = sid_full[-6:] if len(sid_full) > 6 else sid_full
    try:
        year = int(timestr[:4])
        month = int(timestr[4:6])
        day = int(timestr[6:8])
        hour = int(timestr[8:10])
        dt = datetime(year, month, day, hour)
        return dt, sid_short
    except:
        return None, None
# 載入 best track 資料（指定格式以避免警告）
def load_best_track():
    df = pd.read_csv(best_track_path, low_memory=False) # 指定格式以避免警告
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['SID_SHORT'] = df['SID'].astype(str).str[-6:]
    df['USA_WIND'] = pd.to_numeric(df['USA_WIND'], errors='coerce')
    return df
# 為每個檔案找到未來 lead_hours 的 USA_WIND
def get_future_wind(df_bt, current_dt, sid_short, lead_hours):
    future_dt = current_dt + timedelta(hours=lead_hours)
    mask = (df_bt['SID_SHORT'] == sid_short) & (df_bt['ISO_TIME'] == future_dt)
    if mask.any():
        return df_bt.loc[mask, 'USA_WIND'].iloc[0]
    # 找最近的未來時間 (容忍 ±1h)
    time_diff = (df_bt['ISO_TIME'] - future_dt).abs()
    mask_close = (df_bt['SID_SHORT'] == sid_short) & (time_diff <= timedelta(hours=1)) & (df_bt['ISO_TIME'] > current_dt)
    if mask_close.any():
        return df_bt.loc[mask_close, 'USA_WIND'].iloc[0]
    return np.nan
# 載入 Whole 資料夾的資料，按 SID 分組（加追蹤print）
def load_data_from_whole(base_path, df_bt, region):
    region_rows = regions[region]["rows"]
    region_cols = regions[region]["cols"]
    whole_path = os.path.join(base_path, whole_folder)
    if not os.path.exists(whole_path):
        raise ValueError(f"Whole 資料夾不存在: {whole_path}")
    # 按 SID 分組的資料 {sid: [(dt, features, current_wind, future_winds)]}
    sid_data = defaultdict(list)
    total_txt = 0
    pv_files = 0
    no_the = 0
    parse_fail = 0
    pv_adjust_fail = 0
    the_adjust_fail = 0
    no_current = 0
    no_future = 0
    success = 0
    for filename in os.listdir(whole_path):
        if not filename.endswith('.txt'):
            continue
        total_txt += 1
        # 只處理 PV 檔案，避免重複載入 THE 對應檔案
        if 'Azi_PV' not in filename:
            continue
        pv_files += 1
        var = 'PV'
        the_file = filename.replace('Azi_PV', 'Azi_THE')
        pv_path = os.path.join(whole_path, filename)
        the_path = os.path.join(whole_path, the_file)
        if not os.path.exists(the_path):
            no_the += 1
            continue
        dt, sid_short = parse_filename(filename)
        if dt is None or sid_short is None:
            parse_fail += 1
            continue
        try:
            # 載入並調整 PV
            pv_data = np.loadtxt(pv_path)
            pv_selected = adjust_data(pv_data, 'PV', region_rows, region_cols)
            if pv_selected.size == 0:
                pv_adjust_fail += 1
                continue
            # 載入並調整 THE
            the_data = np.loadtxt(the_path)
            the_selected = adjust_data(the_data, 'THE', region_rows, region_cols)
            if the_selected.size == 0:
                the_adjust_fail += 1
                continue
            # 合併特徵 (PV + THE)
            features = np.concatenate([pv_selected, the_selected])
            # 當前風速 (最近時間)
            mask_current = (df_bt['SID_SHORT'] == sid_short) & (df_bt['ISO_TIME'] == dt)
            current_wind = df_bt.loc[mask_current, 'USA_WIND'].iloc[0] if mask_current.any() else np.nan
            if pd.isna(current_wind):
                time_diff = (df_bt['ISO_TIME'] - dt).abs()
                mask_close = (df_bt['SID_SHORT'] == sid_short) & (time_diff <= timedelta(hours=1))
                current_wind = df_bt.loc[mask_close, 'USA_WIND'].iloc[0] if mask_close.any() else np.nan
            if pd.isna(current_wind):
                no_current += 1
                continue
            # 為每個 lead time 計算未來風速
            future_winds = {}
            for lead in time_intervals:
                future_wind = get_future_wind(df_bt, dt, sid_short, lead)
                if not pd.isna(future_wind):
                    future_winds[lead] = future_wind
            if len(future_winds) > 0:
                # 只加入至少有一個 lead 的樣本
                sid_data[sid_short].append((dt, features, current_wind, future_winds))
                success += 1
            else:
                no_future += 1
        except Exception as e:
            logging.error(f"載入檔案錯誤 {filename}: {e}")
            continue
    # 排序每個 SID 的時間序列
    for sid in sid_data:
        sid_data[sid].sort(key=lambda x: x[0])
    # 輸出追蹤
    print(f"Load summary: Total .txt={total_txt}, PV files={pv_files}, No THE={no_the}, Parse fail={parse_fail}, PV adjust fail={pv_adjust_fail}, THE adjust fail={the_adjust_fail}, No current_wind={no_current}, No future={no_future}, Success={success}")
    return sid_data
# 生成差值資料
def generate_diff_data(sid_data, delta):
    if delta == 0:
        return sid_data.copy()
    diff_sid_data = defaultdict(list)
    skipped = 0
    success = 0
    for sid in sid_data:
        times_to_features = {entry[0]: entry[1] for entry in sid_data[sid]}
        for dt, features, current_wind, future_winds in sid_data[sid]:
            target_dt = dt + timedelta(hours=delta)
            # 找精確或最近 (容忍 ±1h, 但僅未來)
            candidates = [(abs(t - target_dt), t) for t in times_to_features if t > dt]
            if not candidates:
                skipped += 1
                continue
            min_diff, closest_t = min(candidates)
            if min_diff > timedelta(hours=1):
                skipped += 1
                continue
            features_delta = times_to_features[closest_t]
            diff_features = features_delta - features
            diff_sid_data[sid].append((dt, diff_features, current_wind, future_winds))
            success += 1
    print(f"For delta={delta}h: Success={success}, Skipped={skipped}")
    return diff_sid_data
# 訓練模型：所有資料點 80/20 分割，per-lead models (步驟2: 調 RF param; 步驟3: per lead scaler)
def train_model(sid_data, output_dir, year_range, region, delta=0):
    sids = list(sid_data.keys())
    if len(sids) < 1:
        raise ValueError("SID 數量不足")
    # 收集所有樣本 with at least one lead: (features, current_wind, future_winds, sid, seq_idx)
    all_samples = []
    for sid in sids:
        for seq_idx, (dt, features, current_wind, future_winds) in enumerate(sid_data[sid]):
            if len(future_winds) > 0:
                all_samples.append((features, current_wind, future_winds, sid, seq_idx))
    print(f"Total input samples (with at least one lead) for {year_range} delta{delta}: {len(all_samples)}")
    if len(all_samples) == 0:
        logging.warning(f"No data for any lead in {year_range} delta{delta}")
        return {}, [], None, None, None
    # 隨機分訓練/測試樣本 (80/20)
    train_samples, test_samples = train_test_split(all_samples, test_size=0.2, random_state=42)
    print(f"Train samples for {year_range} delta{delta}: {len(train_samples)}")
    print(f"Test samples for {year_range} delta{delta}: {len(test_samples)}")
    # Per-lead models with per-lead scaler (步驟3)
    models = {}
    scalers = {}
    pcas = {}
    # 新: per lead scaler
    cv_results = {}
    for lead in time_intervals:
        # Train data for this lead: 過濾 RI-only (future > current, 排除減弱)
        ri_threshold = 0 # 調整: >0 為嚴格增加；可設 +10 kt for 強 RI
        train_mask = [lead in s[2] and s[2][lead] > s[1] + ri_threshold for s in train_samples]
        X_train_full = [np.concatenate((s[0], [s[1]])) for s, m in zip(train_samples, train_mask) if m]
        y_train_lead = np.array([s[2][lead] for s, m in zip(train_samples, train_mask) if m])
        print(f"For lead {lead}h delta{delta}, train samples after RI filter: {len(X_train_full)} (orig {sum(lead in s[2] for s in train_samples)})")
        if len(X_train_full) == 0:
            logging.warning(f"No train data for {lead}h in {year_range} delta{delta}")
            continue
        X_train_lead = np.array(X_train_full)
        # Step 1: Clip outliers for training (加強: 從5-95%改成1-99%)
        q_low, q_high = np.percentile(y_train_lead, [1, 99])
        y_train_clipped = np.clip(y_train_lead, q_low, q_high)
        # K-fold CV for this lead (步驟2: 調 RF param, per fold scaler + PCA)
        n_folds = 3 if len(X_train_lead) < 100 else 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_mse = 0
        cv_r2 = 0
        for train_idx, val_idx in kf.split(X_train_lead):
            X_fold_train = X_train_lead[train_idx]
            y_fold_train = y_train_clipped[train_idx] # Use clipped for CV
            X_fold_val = X_train_lead[val_idx]
            y_fold_val = y_train_lead[val_idx] # Original for eval
            # Per fold scaler
            scaler_fold = StandardScaler()
            X_fold_train_scaled = scaler_fold.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler_fold.transform(X_fold_val)
            # Per fold PCA (Step 1: 0.95 variance)
            pca_fold = PCA(n_components=0.95)
            X_fold_train_pca = pca_fold.fit_transform(X_fold_train_scaled)
            X_fold_val_pca = pca_fold.transform(X_fold_val_scaled)
            svr = SVR(kernel='rbf', C=1.0, epsilon=0.1) # 步驟2: 減過擬合，調參
            svr.fit(X_fold_train_pca, y_fold_train)
            y_pred_val = svr.predict(X_fold_val_pca)
            cv_mse += mean_squared_error(y_fold_val, y_pred_val)
            cv_r2 += r2_score(y_fold_val, y_pred_val)
        cv_mse /= n_folds
        cv_r2 /= n_folds
        cv_results[lead] = (cv_mse, cv_r2)
        logging.info(f"{year_range} delta{delta} {lead}h CV: MSE {cv_mse:.2f}, R2 {cv_r2:.2f}")
        # Final model for this lead (full train scaler + PCA)
        scaler_lead = StandardScaler()
        X_train_lead_scaled = scaler_lead.fit_transform(X_train_lead)
        scalers[lead] = scaler_lead
        pca_lead = PCA(n_components=0.95) # Step 1: 0.95 variance
        X_train_lead_pca = pca_lead.fit_transform(X_train_lead_scaled)
        pcas[lead] = pca_lead
        # Step 2: Ensemble with VotingRegressor (調參 + 加 rf4 多樣性)
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr.fit(X_train_lead_pca, y_train_clipped) # Use clipped for training
        models[lead] = svr
        # Save per lead model
        model_path = os.path.join(output_dir, f"svr_model_{lead}h_{region}_{year_range}_delta{delta}.pkl")
        joblib.dump(svr, model_path)
        # Save per lead scaler
        scaler_path = os.path.join(output_dir, f"svr_scaler_{lead}h_{region}_{year_range}_delta{delta}.pkl")
        joblib.dump(scaler_lead, scaler_path)
        # Save per lead PCA
        pca_path = os.path.join(output_dir, f"svr_pca_{lead}h_{region}_{year_range}_delta{delta}.pkl")
        joblib.dump(pca_lead, pca_path)
    logging.info(f"{year_range} delta{delta} CV results: {cv_results}")
    # 加 feature importance (從 rf1 取 top 5)
    for lead in models:
        pass
    return {}, test_samples, models, scalers, pcas # 返回 pcas dict
# 測試評估：per lead metrics, scatter, bias for all leads per sample (步驟1: outlier 檢測)
def evaluate_test(test_samples, models, scalers, pcas, output_dir, year_range, region, delta=0):
    if len(test_samples) == 0 or len(models) == 0:
        logging.warning("No test samples or models, skipping evaluation")
        return None
    print(f"Evaluating {len(test_samples)} test samples for {year_range} delta{delta}")
    # For bias plot (all leads, per sample) with debug (步驟1)
    sample_bias_info = [] # list of (leads_list, biases_list) for each sample
    valid_samples_count = 0
    lead_lengths = [] # for stats
    all_available_leads = [] # for per-lead usage
    large_bias_samples = [] # 步驟1: |bias|>30 的追蹤
    sample_idx = 0
    all_biases_by_lead = defaultdict(list)
    total_sids = set()
    for s in test_samples:
        features = s[0]
        current_wind = s[1]
        future_winds = s[2]
        sid = s[3] # SID is s[3]
        available_leads = [lead for lead in time_intervals if lead in future_winds and lead in models]
        if len(available_leads) > 0:
            biases = []
            leads_for_this_sample = []
            for lead in available_leads:
                features_full = np.concatenate([features, [current_wind]])
                features_np = np.array([features_full])
                features_scaled = scalers[lead].transform(features_np) # 用 per lead scaler
                features_pca = pcas[lead].transform(features_scaled)
                actual = future_winds[lead]
                predicted = models[lead].predict(features_pca)[0]
                bias = predicted - actual
                # 步驟1: 檢測 large bias
                if abs(bias) > 30:
                    feat_mean = np.mean(features_full)
                    feat_std = np.std(features_full)
                    large_bias_samples.append({
                        'sample_idx': sample_idx,
                        'lead': lead,
                        'actual': actual,
                        'predicted': predicted,
                        'bias': bias,
                        'features_mean': feat_mean,
                        'features_std': feat_std
                    })
                    print(f"Large bias alert: Sample {sample_idx}, lead {lead}h delta{delta}: actual={actual:.1f}, pred={predicted:.1f}, bias={bias:.1f}, feat_mean={feat_mean:.2f}, feat_std={feat_std:.2f}")
                biases.append(bias)
                leads_for_this_sample.append(lead)
                all_biases_by_lead[lead].append(bias)
                total_sids.add(sid)
            sample_bias_info.append((leads_for_this_sample, biases))
            valid_samples_count += 1
            lead_lengths.append(len(leads_for_this_sample))
            all_available_leads.extend(leads_for_this_sample)
            # Debug print for first 10 samples
            if sample_idx < 10:
                print(f"Sample {sample_idx} delta{delta}: {len(available_leads)} leads: {available_leads}")
            sample_idx += 1
    print(f"For bias plot delta{delta}: {valid_samples_count} samples with at least one lead")
    print(f"Large bias samples (>30 kt) delta{delta}: {len(large_bias_samples)}") # 步驟1
    if lead_lengths:
        print(f"Lead length distribution delta{delta}: {Counter(lead_lengths)}")
    print(f"Per-lead usage delta{delta}: {Counter(all_available_leads)}")
    # Bias plot (畫所有 samples, 無 subsample/ylim)
    plot_samples = sample_bias_info # 所有
    num_lines_plotted = sum(1 for l, b in plot_samples if len(l) >= 2)
    num_points = sum(1 for l, b in plot_samples if len(l) == 1)
    print(f"Plotted {num_lines_plotted} lines + {num_points} points (all samples) delta{delta}")
    if plot_samples:
        fig, ax = plt.subplots(figsize=(12, 8))
        for leads_list, biases_list in plot_samples:
            ax.plot(leads_list, biases_list, marker='o', color='gray', alpha=0.6) # 所有用 plot (單點也 OK)
        # Average bias per lead (用 all data)
        if all_biases_by_lead:
            mean_biases = []
            mean_pos_biases = []
            mean_neg_biases = []
            mean_leads = sorted(time_intervals)
            for lead in mean_leads:
                lead_biases = all_biases_by_lead[lead]
                mean_b = np.mean(lead_biases) if lead_biases else np.nan
                pos_biases = [b for b in lead_biases if b > 0]
                mean_pos = np.mean(pos_biases) if pos_biases else np.nan
                neg_biases = [b for b in lead_biases if b < 0]
                mean_neg = np.mean(neg_biases) if neg_biases else np.nan
                mean_biases.append(mean_b)
                mean_pos_biases.append(mean_pos)
                mean_neg_biases.append(mean_neg)
            ax.plot(mean_leads, mean_biases, marker='o', linewidth=3, color='black', label='Average Bias')
            ax.plot(mean_leads, mean_pos_biases, marker='o', linewidth=3, color='red', label='Positive Bias Avg')
            ax.plot(mean_leads, mean_neg_biases, marker='o', linewidth=3, color='blue', label='Negative Bias Avg')
        ax.set_xlabel('Lead Time (hours)')
        ax.set_ylabel('Forecast Bias (kt)')
        ax.set_title(f'Forecast Bias of SVR for All Leads per Sample - {region} ({year_range}) delta{delta}')
        ax.set_xlim(0, 50)
        ax.set_xticks(time_intervals)
        ax.set_yticks(np.arange(-100, 101, 10))
        ax.grid(True)
        ax.legend(loc='upper right')
        error_plot_path = os.path.join(output_dir, f"svr_bias_by_sample_all_leads_fixed_{region}_{year_range}_delta{delta}.png")
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"偏差圖已儲存: {error_plot_path}")
    else:
        print(f"No samples to plot; skipping bias plot for delta{delta}.")
    # Per lead evaluation (用 per lead scaler + PCA) - 不filter RI (只用可用lead，公平評估所有情況，包括輕微增加/穩定)
    epsilon = 10 # kt, as default
    for lead in time_intervals:
        if lead not in models:
            logging.warning(f"No model for {lead}h delta{delta}, skipping")
            continue
        test_mask = [lead in s[2] for s in test_samples] # 不加 RI filter (訓練專注 RI，但測試評估所有可用 lead，更公平反映真實應用性能)
        y_test_lead = [s[2][lead] for s, m in zip(test_samples, test_mask) if m]
        features_test_full = [np.concatenate((s[0], [s[1]])) for s, m in zip(test_samples, test_mask) if m]
        if len(y_test_lead) == 0:
            logging.warning(f"No test data for {lead}h delta{delta}")
            continue
        print(f"For lead {lead}h delta{delta}, test samples (no RI filter): {len(y_test_lead)}")
        features_test_np = np.array(features_test_full)
        features_test_scaled = scalers[lead].transform(features_test_np)
        features_test_pca = pcas[lead].transform(features_test_scaled)
        y_pred_lead = models[lead].predict(features_test_pca)
        y_test_lead = np.array(y_test_lead)
        y_pred_lead = np.array(y_pred_lead)
        # Hit Rate calculation
        hit_rate = np.mean(np.abs(y_pred_lead - y_test_lead) < epsilon) * 100
        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_lead, y_pred_lead, alpha=0.6)
        min_val = min(np.min(y_test_lead), np.min(y_pred_lead))
        max_val = max(np.max(y_test_lead), np.max(y_pred_lead))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Future Wind (kt)')
        plt.ylabel('Predicted Future Wind (kt)')
        plt.title(f"Test Prediction {lead}h - {region} ({year_range}) delta{delta}")
        plot_path = os.path.join(output_dir, f"svr_test_scatter_{lead}h_{region}_{year_range}_delta{delta}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        # Metrics
        mse = mean_squared_error(y_test_lead, y_pred_lead)
        mae = mean_absolute_error(y_test_lead, y_pred_lead)
        r2 = r2_score(y_test_lead, y_pred_lead)
        overall_bias = np.mean(y_pred_lead - y_test_lead)
        logging.info(f"{year_range} delta{delta} {lead}h Test: MSE {mse:.2f}, MAE {mae:.2f}, R2 {r2:.2f}, Bias {overall_bias:.2f}, Hit Rate (ε={epsilon}kt) {hit_rate:.2f}%")
        print(f"Final test data points used in metrics for {lead}h {year_range} delta{delta}: {len(y_test_lead)}")
    return all_biases_by_lead, sample_bias_info, len(total_sids)
# 主函數
def main():
    df_bt = load_best_track()
    deltas = [0, 6, 12, 18, 24]
    for year_range in year_ranges:
        base_path = base_path_template.format(year_range)
        output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{year_range}-tc_intensity"
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"svr_tc_intensity_{year_range}.log")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()], force=True)
        logging.info(f"處理 {year_range}")
        region = "upper_inner"
        sid_data = load_data_from_whole(base_path, df_bt, region)
        if len(sid_data) == 0:
            logging.warning(f"無資料 {year_range}")
            continue
        biases_dict = {}
        sample_bias_dict = {}
        tc_dict = {}
        for delta in deltas:
            diff_sid_data = generate_diff_data(sid_data, delta)
            if len(diff_sid_data) == 0:
                logging.warning(f"無差值資料 for delta {delta} in {year_range}")
                continue
            _, test_samples, models, scalers, pcas = train_model(diff_sid_data, output_dir, year_range, region, delta=delta) # pcas dict
            biases_by_lead, sample_bias_info, tc_num = evaluate_test(test_samples, models, scalers, pcas, output_dir, year_range, region, delta=delta)
            if biases_by_lead is not None:
                biases_dict[delta] = biases_by_lead
                sample_bias_dict[delta] = sample_bias_info
                tc_dict[delta] = tc_num
        # 新增: 綜合所有 delta 的圖 (畫所有 delta 的灰線和平均線)
        if biases_dict:
            fig, ax = plt.subplots(figsize=(12, 8))
            # 畫所有 delta 的灰線 (用不同 alpha 或同灰色)
            for delta in deltas:
                if delta in sample_bias_dict:
                    for leads_list, biases_list in sample_bias_dict[delta]:
                        ax.plot(leads_list, biases_list, marker='o', color='gray', alpha=0.3) # 降低 alpha 以避免太亂
            # 畫平均線，調整樣式和 legend 順序
            mean_leads = sorted(time_intervals)
            # 定義顏色和粗細
            fixed_lw = 3 # Positive 和 Negative 的固定粗細
            fixed_sw = 2
            red_colors = plt.cm.Reds(np.linspace(0.3, 1, 5)) # Delta 0~24 紅色漸變
            blue_colors = plt.cm.Blues(np.linspace(0.3, 1, 5)) # Delta 0~24 藍色漸變
            # 先收集所有 Positive, Negative 的線 (for legend 順序)
            pos_lines = []
            neg_lines = []
            tc_lines = []
            for i, delta in enumerate(deltas):
                if delta in biases_dict:
                    all_biases_by_lead = biases_dict[delta]
                    pos_biases_by_lead = [[b for b in all_biases_by_lead[lead] if b > 0] for lead in mean_leads]
                    mean_pos_biases = [np.mean(pos) if pos else np.nan for pos in pos_biases_by_lead]
                    std_pos_biases = [np.std(pos) if pos else np.nan for pos in pos_biases_by_lead]
                    neg_biases_by_lead = [[b for b in all_biases_by_lead[lead] if b < 0] for lead in mean_leads]
                    mean_neg_biases = [np.mean(neg) if neg else np.nan for neg in neg_biases_by_lead]
                    std_neg_biases = [np.std(neg) if neg else np.nan for neg in neg_biases_by_lead]
                    # 計算總個案數 (所有 lead 的 pos/neg 樣本總數)
                    total_pos_n = sum(len(pos) for pos in pos_biases_by_lead)
                    total_neg_n = sum(len(neg) for neg in neg_biases_by_lead)
                    # Positive
                    red_color = red_colors[i]
                    pos_line, = ax.plot(mean_leads, mean_pos_biases, marker='o', linewidth=fixed_lw, color=red_color, label=f'Positive Bias Delta {delta} (n={total_pos_n})')
                    ax.plot(mean_leads, [m + s for m, s in zip(mean_pos_biases, std_pos_biases)], linestyle=':', linewidth=fixed_sw, color=red_color)
                    ax.plot(mean_leads, [m - s for m, s in zip(mean_pos_biases, std_pos_biases)], linestyle=':', linewidth=fixed_sw, color=red_color)
                    pos_lines.append(pos_line)
                    # Negative
                    blue_color = blue_colors[i]
                    neg_line, = ax.plot(mean_leads, mean_neg_biases, marker='o', linewidth=fixed_lw, color=blue_color, label=f'Negative Bias Delta {delta} (n={total_neg_n})')
                    ax.plot(mean_leads, [m + s for m, s in zip(mean_neg_biases, std_neg_biases)], linestyle=':', linewidth=fixed_sw, color=blue_color)
                    ax.plot(mean_leads, [m - s for m, s in zip(mean_neg_biases, std_neg_biases)], linestyle=':', linewidth=fixed_sw, color=blue_color)
                    neg_lines.append(neg_line)
                    # TC number dummy line
                    tc_line, = ax.plot([], [], color='black', label=f'Delta {delta} TC number: {tc_dict[delta]}')
                    tc_lines.append(tc_line)
            # Legend: 先 Positive, then Negative, then TC numbers
            all_lines = pos_lines + neg_lines + tc_lines
            all_labels = [line.get_label() for line in all_lines]
            ax.legend(all_lines, all_labels, loc='upper right', bbox_to_anchor=(1.25, 1), ncol=1, fontsize=8)
            ax.set_xlabel('Lead Time (hours)')
            ax.set_ylabel('Forecast Bias (kt)')
            ax.set_title(f'Combined Forecast Bias for All Deltas - {region} ({year_range})')
            ax.set_xlim(0, 50)
            ax.set_xticks(time_intervals)
            ax.set_yticks(np.arange(-100, 101, 10))
            ax.grid(True)
            combined_plot_path = os.path.join(output_dir, f"svr_bias_combined_all_deltas_{region}_{year_range}.png")
            plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"綜合偏差圖已儲存: {combined_plot_path}")
        logging.info(f"{year_range} 完成")
if __name__ == "__main__":
    main()