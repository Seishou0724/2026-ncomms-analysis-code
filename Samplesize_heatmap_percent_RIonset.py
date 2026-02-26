import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# 定義年份範圍
year_range = "2013-2022"

# 設置輸出目錄
output_dir = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/PYTHON/RIIndex/{year_range}-Samplesize_RIonset"
os.makedirs(output_dir, exist_ok=True)

# 設置日誌
log_file_path = os.path.join(output_dir, f"sample_size_log_{year_range}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False), logging.StreamHandler()],
    force=True
)

logging.info("腳本開始執行 - 計算 RI Onset 閾值樣本數熱圖")

# 基本參數
base_path = f"/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-{year_range}"
wind_thresholds = list(range(10, 105, 5))
time_intervals = [6, 12, 18, 24, 30, 36, 42, 48]

# 初始化樣本數矩陣
samples_matrix = np.zeros((len(wind_thresholds), len(time_intervals)))

# 迴圈計算每個 (wind, time) 的 PV 樣本數
total_combinations = len(wind_thresholds) * len(time_intervals)
processed_combinations = 0

for i, wind in enumerate(wind_thresholds):
    for j, time in enumerate(time_intervals):
        processed_combinations += 1
        logging.info(f"處理組合 {processed_combinations}/{total_combinations} ({(processed_combinations/total_combinations)*100:.1f}%) - wind={wind}, time={time}")
        
        pv_path_ri = os.path.join(base_path, "Azi_PV", "Individual", f"Azi_PV-RI-{wind}-{time:02d}", "removenan")
        
        if os.path.exists(pv_path_ri):
            pv_files = [f for f in os.listdir(pv_path_ri) if f.endswith(".txt")]
            typhoon_ids = set()
            for f in pv_files:
                parts = f.split('-')
                if len(parts) >= 3:
                    typhoon_id = parts[2].split('.')[0]  # 提取 NXXXXX
                    typhoon_ids.add(typhoon_id)
            sample_count = len(typhoon_ids)
            samples_matrix[i, j] = sample_count
            logging.info(f"Wind {wind} kt, Time {time} h: {sample_count} 個 RI Onset 個案")
        else:
            logging.warning(f"路徑不存在: {pv_path_ri}")
            samples_matrix[i, j] = 0

logging.info(f"樣本數矩陣計算完成，總樣本數: {np.sum(samples_matrix)}")

# 計算總樣本數和百分比矩陣
total_samples = np.sum(samples_matrix)
if total_samples > 0:
    percent_matrix = (samples_matrix / total_samples) * 100
else:
    percent_matrix = np.zeros_like(samples_matrix)

# 繪製熱圖
plt.figure(figsize=(14, 10))
annot_matrix = np.array([[f"{samples_matrix[i, j]:.0f}" if samples_matrix[i, j] > 0 else ""
                          for j in range(len(time_intervals))]
                         for i in range(len(wind_thresholds))])
ax = sns.heatmap(percent_matrix[::-1], xticklabels=time_intervals, yticklabels=wind_thresholds[::-1],
                 cmap="YlOrRd", vmin=0, vmax=3, annot=annot_matrix[::-1], fmt="", linewidths=0.5, linecolor='black',
                 cbar_kws={'label': "Percentage of Total Samples (%)"}, annot_kws={"size": 8})

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color('black')
    spine.set_linewidth(2)

plt.title(f"RI Onset Threshold Sample Size Heatmap - ({year_range})")
plt.text(1, 1.03, f'Total Samples: {total_samples:.0f}', transform=ax.transAxes, ha='right', va='top', fontsize=12)
plt.xlabel("Time Interval (h)")
plt.ylabel("Wind Speed Increment Threshold (kt)")

heatmap_path = os.path.join(output_dir, f"RI_Onset_SampleSize_Heatmap_percent_{year_range}.png")
plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
plt.close()

logging.info(f"熱圖已儲存至 {heatmap_path}")

# 輸出樣本數矩陣摘要
logging.info("樣本數矩陣摘要 (wind x time):")
for i, wind in enumerate(wind_thresholds):
    row_summary = [f"{samples_matrix[i, j]:.0f}" for j in range(len(time_intervals))]
    logging.info(f"Wind {wind} kt: {row_summary}")

logging.info("程式執行完畢")