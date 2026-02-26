import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr
# 設定日誌
log_file_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/ANO_significantNumber_2013-2022.log"
log_dir = os.path.dirname(log_file_path)
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, delay=False), logging.StreamHandler()],
    force=True
)
base_path = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-2013-2022"
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 48]
# 統一的數據範圍
variables = {
    "PV": {"rows": range(0, 27), "cols": range(9,21)},
    "THE": {"rows": range(0, 27), "cols": range(9, 21)},
    "W": {"rows": range(0, 27), "cols": range(9, 21)}
}
regions = {
    #"upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
    "upper_inner": {"rows": range(2, 12), "cols": range(0, 9), "name": "Upper Level Inner-Core"},
    "upper_outer": {"rows": range(2, 12), "cols": range(9, 18), "name": "Upper Level Outer Area"},
    "middle_inner": {"rows": range(12, 21), "cols": range(0, 9), "name": "Middle Level Inner-Core"},
    "middle_outer": {"rows": range(12, 21), "cols": range(9, 18), "name": "Middle Level Outer Area"},
    "lower_inner": {"rows": range(21, 27), "cols": range(0, 9), "name": "Lower Level Inner-Core"},
    "lower_outer": {"rows": range(21, 27), "cols": range(9, 18), "name": "Lower Level Outer Area"},
    "midlower_inner": {"rows": range(12, 27), "cols": range(0, 9), "name": "Mid-Lower Level Inner-Core"},
    "midlower_outer": {"rows": range(12, 27), "cols": range(9, 18), "name": "Mid-Lower Level Outer Area"}
}
def adjust_data_original(data, var, region_rows, region_cols):
    if data.shape != (27, 21):
        return np.array([])
    if np.all(np.isnan(data)) or np.all(data == 0):
        return np.array([])
    # 不進行調整，直接選取範圍
    intersect_rows = sorted(set(range(data.shape[0])) & set(region_rows))
    intersect_cols = sorted(set(range(data.shape[1])) & set(region_cols))
    if not intersect_rows or not intersect_cols:
        return np.array([])
    selected_data = data[np.ix_(intersect_rows, intersect_cols)]
    if selected_data.size == 0 or np.all(np.isnan(selected_data)) or np.all(selected_data == 0):
        return np.array([])
    return selected_data.flatten()
def adjust_data_adjusted(data, var, region_rows, region_cols):
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
    # 直接使用 region_rows 和 region_cols，無需與 variables 交集
    intersect_rows = sorted(set(range(data.shape[0])) & set(region_rows))
    intersect_cols = sorted(set(range(data.shape[1])) & set(region_cols))
    if not intersect_rows or not intersect_cols:
        return np.array([])
    selected_data = adjusted_data[np.ix_(intersect_rows, intersect_cols)]
    if selected_data.size == 0 or np.all(np.isnan(selected_data)) or np.all(selected_data == 0):
        return np.array([])
    return selected_data.flatten()
def load_all_data(base_path):
    pv_data_orig = {key: [] for key in regions}
    the_data_orig = {key: [] for key in regions}
    pv_data_adj = {key: [] for key in regions}
    the_data_adj = {key: [] for key in regions}
    # 遍歷所有 wind_thresholds 和 time_intervals
    for wind in wind_thresholds:
        for time in time_intervals:
            for label in ["RI", "PreRI"]: # 保留所有標籤，統一處理
                pv_path = os.path.join(base_path, "Azi_PV", "Individual", f"Azi_PV-{label}-{wind}-{time:02d}", "removenan")
                the_path = os.path.join(base_path, "Azi_THE", "Individual", f"Azi_THE-{label}-{wind}-{time:02d}", "removenan")
            
                # 檢查路徑是否存在
                if os.path.exists(pv_path) and os.path.exists(the_path):
                    pv_files = [os.path.join(pv_path, f) for f in os.listdir(pv_path) if f.endswith(".txt")]
                    the_files = [os.path.join(the_path, f) for f in os.listdir(the_path) if f.endswith(".txt")]
                
                    # 確保檔案數量一致
                    min_files = min(len(pv_files), len(the_files))
                    pv_files = pv_files[:min_files]
                    the_files = the_files[:min_files]
                
                    # 處理每個檔案
                    for pv_file, the_file in zip(pv_files, the_files):
                        try:
                            pv_raw = np.loadtxt(pv_file)
                            the_raw = np.loadtxt(the_file)
                            # 檢查數據形狀是否正確
                            if pv_raw.shape != (27, 21) or the_raw.shape != (27, 21):
                                continue
                        
                            # 為每個 region 計算原數據
                            for key, reg in regions.items():
                                pv_orig = adjust_data_original(pv_raw, "PV", reg["rows"], reg["cols"])
                                the_orig = adjust_data_original(the_raw, "THE", reg["rows"], reg["cols"])
                        
                                # 將每個點的數據加入列表，如果不為空
                                if len(pv_orig) > 0 and len(the_orig) > 0:
                                    pv_data_orig[key].extend(pv_orig)
                                    the_data_orig[key].extend(the_orig)
                            
                            # 為每個 region 計算調整後數據
                            for key, reg in regions.items():
                                pv_adj = adjust_data_adjusted(pv_raw, "PV", reg["rows"], reg["cols"])
                                the_adj = adjust_data_adjusted(the_raw, "THE", reg["rows"], reg["cols"])
                                if len(pv_adj) > 0 and len(the_adj) > 0:
                                    pv_data_adj[key].extend(pv_adj)
                                    the_data_adj[key].extend(the_adj)
                        except Exception as e:
                            logging.error(f"處理文件 {pv_file}, {the_file} 時出錯：{e}")
                            continue
    return {key: np.array(v) for key, v in pv_data_orig.items()}, {key: np.array(v) for key, v in the_data_orig.items()}, {key: np.array(v) for key, v in pv_data_adj.items()}, {key: np.array(v) for key, v in the_data_adj.items()}
def standardize_data(data):
    """手動標準化：均值 0, 標準差 1（忽略 NaN）"""
    mask = ~np.isnan(data)
    mean = np.mean(data[mask])
    std = np.std(data[mask])
    if std == 0:
        return data - mean # 避免除 0
    return (data - mean) / std
def main():
    logging.info("程式開始執行")
    # 載入所有資料
    pv_orig_dict, the_orig_dict, pv_adj_dict, the_adj_dict = load_all_data(base_path)
    
    # 為每個 region 生成原數據圖表
    for key in regions:
        pv_orig = pv_orig_dict.get(key, np.array([]))
        the_orig = the_orig_dict.get(key, np.array([]))
        if len(pv_orig) == 0 or len(the_orig) == 0:
            logging.warning(f"沒有載入任何原始數據 for {key}，無法生成圖表。")
            continue
        # 第一張圖：原數據 PV vs Theta E 散點圖
        plt.figure(figsize=(10, 8))
        plt.scatter(pv_orig, the_orig, alpha=0.5)
        # 計算統計指標（忽略 NaN）
        mask = ~np.isnan(pv_orig) & ~np.isnan(the_orig)
        if np.sum(mask) < 2:
            logging.warning(f"原始數據有效點不足 for {key}，無法計算相關係數。")
        else:
            pearson_corr, pearson_p = pearsonr(pv_orig[mask], the_orig[mask])
            spearman_corr, spearman_p = spearmanr(pv_orig[mask], the_orig[mask])
            plt.xlabel('PV')
            plt.ylabel('Theta E')
            plt.title(f'Scatter Plot of PV vs Theta E (Original Data - {regions[key]["name"]})')
            plt.grid(True)
            # 標示統計指標
            stats_text = (f'Pearson Corr: {pearson_corr:.2f} (p={pearson_p:.2e})\n'
                          f'Spearman Corr: {spearman_corr:.2f} (p={spearman_p:.2e})')
            plt.text(0.05, 0.95, stats_text,
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            plt.savefig(f"Scatter_PV_ThetaE_Original_{key}.png")
            plt.show()
            logging.info(f"Original scatter plot of PV vs Theta E for {key} 已生成")
        # 新增：第三張圖 - 標準化版本（基於原始數據）
        if len(pv_orig) > 0 and len(the_orig) > 0:
            pv_std_orig = standardize_data(pv_orig)
            the_std_orig = standardize_data(the_orig)
            plt.figure(figsize=(10, 8))
            plt.scatter(pv_std_orig, the_std_orig, alpha=0.5)
            # 計算統計指標（忽略 NaN）
            mask_std_orig = ~np.isnan(pv_std_orig) & ~np.isnan(the_std_orig)
            if np.sum(mask_std_orig) >= 2:
                pearson_corr_std_orig, pearson_p_std_orig = pearsonr(pv_std_orig[mask_std_orig], the_std_orig[mask_std_orig])
                spearman_corr_std_orig, spearman_p_std_orig = spearmanr(pv_std_orig[mask_std_orig], the_std_orig[mask_std_orig])
                # 畫 y = x 虛線
                lim_min = min(np.min(pv_std_orig[mask_std_orig]), np.min(the_std_orig[mask_std_orig]))
                lim_max = max(np.max(pv_std_orig[mask_std_orig]), np.max(the_std_orig[mask_std_orig]))
                plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', label='y = x (Perfect Correlation)')
                plt.xlabel('Standardized PV')
                plt.ylabel('Standardized Theta E')
                plt.title(f'Scatter Plot of Standardized PV vs Theta E (Original Data - {regions[key]["name"]})')
                plt.grid(True)
                plt.legend()
                # 標示統計指標
                stats_text_std_orig = (f'Pearson Corr: {pearson_corr_std_orig:.2f} (p={pearson_p_std_orig:.2e})\n'
                                       f'Spearman Corr: {spearman_corr_std_orig:.2f} (p={spearman_p_std_orig:.2e})')
                plt.text(0.05, 0.95, stats_text_std_orig,
                         transform=plt.gca().transAxes, fontsize=12,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                plt.savefig(f"Scatter_PV_ThetaE_Standardized_Original_{key}.png")
                plt.show()
                logging.info(f"Standardized scatter plot of PV vs Theta E (Original Data - {key}) 已生成")
    # 為每個 region 生成調整後圖表
    for key in regions:
        pv_adj = pv_adj_dict.get(key, np.array([]))
        the_adj = the_adj_dict.get(key, np.array([]))
        if len(pv_adj) == 0 or len(the_adj) == 0:
            logging.warning(f"沒有載入任何調整後數據 for {key}，無法生成圖表。")
            continue
        # 第二張圖類型：調整後數據 PV vs Theta E 散點圖
        plt.figure(figsize=(10, 8))
        plt.scatter(pv_adj, the_adj, alpha=0.5)
        # 計算統計指標（忽略 NaN）
        mask_adj = ~np.isnan(pv_adj) & ~np.isnan(the_adj)
        if np.sum(mask_adj) < 2:
            logging.warning(f"調整後數據有效點不足 for {key}，無法計算相關係數。")
        else:
            pearson_corr_adj, pearson_p_adj = pearsonr(pv_adj[mask_adj], the_adj[mask_adj])
            spearman_corr_adj, spearman_p_adj = spearmanr(pv_adj[mask_adj], the_adj[mask_adj])
            plt.xlabel('PV')
            plt.ylabel('Theta E')
            plt.title(f'Scatter Plot of PV vs Theta E (Anomaly Subtraction - {regions[key]["name"]})')
            plt.grid(True)
            # 標示統計指標
            stats_text_adj = (f'Pearson Corr: {pearson_corr_adj:.2f} (p={pearson_p_adj:.2e})\n'
                              f'Spearman Corr: {spearman_corr_adj:.2f} (p={spearman_p_adj:.2e})')
            plt.text(0.05, 0.95, stats_text_adj,
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            plt.savefig(f"Scatter_PV_ThetaE_Adjusted_{key}.png")
            plt.show()
            logging.info(f"Adjusted scatter plot of PV vs Theta E for {key} 已生成")
        # 新增：第四張圖類型 - 標準化版本（基於調整後數據）
        pv_std_adj = standardize_data(pv_adj)
        the_std_adj = standardize_data(the_adj)
        plt.figure(figsize=(10, 8))
        plt.scatter(pv_std_adj, the_std_adj, alpha=0.5)
        # 計算統計指標（忽略 NaN）
        mask_std_adj = ~np.isnan(pv_std_adj) & ~np.isnan(the_std_adj)
        if np.sum(mask_std_adj) >= 2:
            pearson_corr_std_adj, pearson_p_std_adj = pearsonr(pv_std_adj[mask_std_adj], the_std_adj[mask_std_adj])
            spearman_corr_std_adj, spearman_p_std_adj = spearmanr(pv_std_adj[mask_std_adj], the_std_adj[mask_std_adj])
            # 畫 y = x 虛線
            lim_min = min(np.min(pv_std_adj[mask_std_adj]), np.min(the_std_adj[mask_std_adj]))
            lim_max = max(np.max(pv_std_adj[mask_std_adj]), np.max(the_std_adj[mask_std_adj]))
            plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', label='y = x (Perfect Correlation)')
            plt.xlabel('Standardized PV')
            plt.ylabel('Standardized Theta E')
            plt.title(f'Scatter Plot of Standardized PV vs Theta E (Anomaly Subtraction - {regions[key]["name"]})')
            plt.grid(True)
            plt.legend()
            # 標示統計指標
            stats_text_std_adj = (f'Pearson Corr: {pearson_corr_std_adj:.2f} (p={pearson_p_std_adj:.2e})\n'
                                  f'Spearman Corr: {spearman_corr_std_adj:.2f} (p={spearman_p_std_adj:.2e})')
            plt.text(0.05, 0.95, stats_text_std_adj,
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            plt.savefig(f"Scatter_PV_ThetaE_Standardized_Adjusted_{key}.png")
            plt.show()
            logging.info(f"Standardized scatter plot of PV vs Theta E (Adjusted Data - {key}) 已生成")
if __name__ == "__main__":
    try:
        main()
        logging.info("程式執行完畢")
    except Exception as e:
        logging.exception("執行過程中發生嚴重錯誤：")
    finally:
        logging.shutdown()