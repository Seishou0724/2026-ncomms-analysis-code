import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import t

# ====================== 通用函數 ======================
def adjust_data_for_composite(data):
    if data.shape != (27, 21):
        return np.full((27, 21), np.nan)
    outer_cols = list(range(18, 21))
    if not all(0 <= col < 21 for col in outer_cols):
        return np.full((27, 21), np.nan)
    if np.all(np.isnan(data)) or np.all(data == 0):
        return np.full((27, 21), np.nan)
    adjusted_data = data.copy()
    for row in range(data.shape[0]):
        outer_data_row = data[row, outer_cols]
        outer_avg = 0 if np.all(np.isnan(outer_data_row)) else np.nanmean(outer_data_row)
        adjusted_data[row, :] -= outer_avg
    return adjusted_data

# ====================== 載入單一組合資料 ======================
def load_data(var_name, thresh, dur):
    base_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/JTWC-2013-2022/"
    dir1 = os.path.join(base_dir, f"Azi_{var_name}/Individual/Azi_{var_name}-PreRI-{thresh}-{dur}/removenan")
    dir2 = os.path.join(base_dir, f"Azi_{var_name}/Individual/Azi_{var_name}-RI-{thresh}-{dur}/removenan")
   
    if not os.path.exists(dir1) or not os.path.exists(dir2):
        return None
   
    files1 = [f for f in os.listdir(dir1) if f.endswith('.txt')]
    files2 = [f for f in os.listdir(dir2) if f.endswith('.txt')]
    Nifn1 = len(files1)
    Nifn2 = len(files2)
   
    if Nifn1 == 0 or Nifn2 == 0:
        return None
   
    var1 = np.array([np.loadtxt(os.path.join(dir1, f)) for f in files1])
    var2 = np.array([np.loadtxt(os.path.join(dir2, f)) for f in files2])
   
    adjusted_var1 = np.array([adjust_data_for_composite(v) for v in var1])
    adjusted_var2 = np.array([adjust_data_for_composite(v) for v in var2])
   
    mT1 = np.nanmean(var1, axis=0)
    mT2 = np.nanmean(var2, axis=0)
    mT = mT2 - mT1
   
    mT1_a = np.nanmean(adjusted_var1, axis=0)
    mT2_a = np.nanmean(adjusted_var2, axis=0)
    mT_a = mT2_a - mT1_a
   
    return {
        'mT1': mT1, 'mT2': mT2, 'mT': mT,
        'mT1_a': mT1_a, 'mT2_a': mT2_a, 'mT_a': mT_a,
        'N1': Nifn1, 'N2': Nifn2,
        'var1': var1, 'var2': var2,
        'adjusted_var1': adjusted_var1, 'adjusted_var2': adjusted_var2
    }

# ====================== 座標定義 ======================
plev = np.array([100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000])
radius = np.linspace(0, 500, 21)
R, P = np.meshgrid(radius, plev)

# ====================== 輸出目錄 ======================
output_dir = "/Dellwork6/cwusei/RI/ALL_IBTrACS/Program/COMPOSITE_plot/all"
os.makedirs(output_dir, exist_ok=True)

# ====================== 參數範圍 ======================
ri_thresholds = list(range(10, 101, 5))
ri_durations = list(range(6, 49, 6))

# ====================== 第一階段：只用 30kt/24h 和 65kt/24h 這兩組計算固定色階 ======================
print("=== 第一階段：只用 30kt/24h 和 65kt/24h 這兩組資料計算固定色階（避免其他極端 threshold 影響） ===")

reference_cases = [(30, 24), (65, 24)]
data_collect = {'PV': {'comp': [], 'a_comp': [], 'd_abs': [], 'ad_abs': []},
                'THE': {'comp': [], 'a_comp': [], 'd_abs': [], 'ad_abs': []}}

for var_name in ['PV', 'THE']:
    for thresh, dur in reference_cases:
        data = load_data(var_name, thresh, dur)
        if data is None:
            print(f"  Warning: {var_name} {thresh}kt/{dur}h 資料不存在，將使用另一組或預設值")
            continue
        
        mT1 = data['mT1']
        mT2 = data['mT2']
        mT1a = data['mT1_a']
        mT2a = data['mT2_a']
        mT = data['mT']
        mTa = data['mT_a']
        
        if np.all(np.isnan(mT1)) or np.all(np.isnan(mT2)):
            continue
        
        data_collect[var_name]['comp'].append(mT1.ravel())
        data_collect[var_name]['comp'].append(mT2.ravel())
        data_collect[var_name]['a_comp'].append(mT1a.ravel())
        data_collect[var_name]['a_comp'].append(mT2a.ravel())
        data_collect[var_name]['d_abs'].append(np.abs(mT).ravel())
        data_collect[var_name]['ad_abs'].append(np.abs(mTa).ravel())

# ====================== 根據這兩組計算最終固定色階 ======================
global_ranges = {}
for var_name in ['PV', 'THE']:
    dc = data_collect[var_name]
    
    if len(dc['comp']) == 0:
        # 兩組都沒有資料 → 使用合理預設
        if var_name == 'PV':
            c_min, c_max = 0.0, 8.0
            a_min, a_max = -5.0, 5.0
            d_mabs, ad_mabs = 2.0, 2.0
        else:
            c_min, c_max = 330.0, 370.0
            a_min, a_max = -15.0, 15.0
            d_mabs, ad_mabs = 8.0, 8.0
    else:
        # Composite (Pre-RI + RI)
        all_c = np.concatenate(dc['comp'])
        c_min = np.nanmin(all_c)
        c_max = np.nanmax(all_c)
        if var_name == 'PV' or var_name == 'THE':
            c_min = max(0.0, c_min)
        
        # Anomaly Composite
        all_ac = np.concatenate(dc['a_comp'])
        a_min = np.nanmin(all_ac)
        a_max = np.nanmax(all_ac)
        
        # Difference & Anomaly Difference
        all_d = np.concatenate(dc['d_abs'])
        d_mabs = np.nanmax(all_d)
        all_ad = np.concatenate(dc['ad_abs'])
        ad_mabs = np.nanmax(all_ad)
    
    # 加入 5% padding，避免顏色剛好卡邊界
    for vmin, vmax, key in [(c_min, c_max, 'composite'),
                            (a_min, a_max, 'anomaly')]:
        if vmax > vmin:
            pad = 0.05 * (vmax - vmin)
            if key == 'composite':
                c_min -= pad
                c_max += pad
                if var_name in ['PV', 'THE']:
                    c_min = max(0.0, c_min)
            else:
                a_min -= pad
                a_max += pad
    
    if d_mabs > 0:
        d_mabs *= 1.05
    if ad_mabs > 0:
        ad_mabs *= 1.05
    
    global_ranges[var_name] = {
        'composite_min': c_min, 'composite_max': c_max,
        'anomaly_min': a_min,   'anomaly_max': a_max,
        'diff_maxabs': d_mabs,  'anomaly_diff_maxabs': ad_mabs
    }
    
    print(f"{var_name} 固定色階（來自 30kt/24h + 65kt/24h）：")
    print(f"   Composite          : {c_min:.3f} ~ {c_max:.3f}")
    print(f"   Anomaly Composite  : {a_min:.3f} ~ {a_max:.3f}")
    print(f"   Difference maxabs  : {d_mabs:.3f}")
    print(f"   Anomaly Diff maxabs: {ad_mabs:.3f}")

# ====================== 主繪圖函數（所有圖都使用上面固定色階） ======================
def generate_plots(var_name, thresh, dur, output_dir, cmap, R, P, global_ranges):
    data = load_data(var_name, thresh, dur)
    if data is None:
        print(f"Skipping {var_name} {thresh}kt/{dur}h：無資料")
        return
   
    mT1 = data['mT1']
    mT2 = data['mT2']
    mT = data['mT']
    mT1_adjusted = data['mT1_a']
    mT2_adjusted = data['mT2_a']
    mT_adjusted = data['mT_a']
    Nifn1 = data['N1']
    Nifn2 = data['N2']
    var1 = data['var1']
    var2 = data['var2']
    adjusted_var1 = data['adjusted_var1']
    adjusted_var2 = data['adjusted_var2']
   
    if np.all(np.isnan(mT1)) or np.all(np.isnan(mT2)):
        print(f"Skipping {var_name} {thresh}kt/{dur}h：資料全 NaN")
        return
   
    # t-test
    if Nifn1 < 2 or Nifn2 < 2:
        conf = np.full_like(mT, np.nan)
        conf_adjusted = np.full_like(mT_adjusted, np.nan)
    else:
        varT1 = np.nanvar(var1, axis=0, ddof=1)
        varT2 = np.nanvar(var2, axis=0, ddof=1)
        var_pooled = ((Nifn1 - 1) * varT1 + (Nifn2 - 1) * varT2) / (Nifn1 + Nifn2 - 2)
        t_stat = (mT2 - mT1) / np.sqrt(var_pooled * (1/Nifn1 + 1/Nifn2))
        df = Nifn1 + Nifn2 - 2
        p_value = 2 * t.sf(np.abs(t_stat), df)
        conf = 100 * (1 - p_value)
       
        varT1_a = np.nanvar(adjusted_var1, axis=0, ddof=1)
        varT2_a = np.nanvar(adjusted_var2, axis=0, ddof=1)
        var_pooled_a = ((Nifn1 - 1) * varT1_a + (Nifn2 - 1) * varT2_a) / (Nifn1 + Nifn2 - 2)
        t_stat_a = (mT2_adjusted - mT1_adjusted) / np.sqrt(var_pooled_a * (1/Nifn1 + 1/Nifn2))
        p_value_a = 2 * t.sf(np.abs(t_stat_a), df)
        conf_adjusted = 100 * (1 - p_value_a)
   
    g = global_ranges[var_name]
    levels_composite          = np.linspace(g['composite_min'], g['composite_max'], 20)
    levels_composite_adjusted = np.linspace(g['anomaly_min'], g['anomaly_max'], 20)
    levels_diff               = np.linspace(-g['diff_maxabs'], g['diff_maxabs'], 21)
    levels_diff_adjusted      = np.linspace(-g['anomaly_diff_maxabs'], g['anomaly_diff_maxabs'], 21)
   
    unit = 'PVU' if var_name == 'PV' else 'K'
    var_label = f'{var_name} ({unit})'
    diff_label = f'{var_name} Difference ({unit})'
    anomaly_diff_label = f'Anomaly {var_name} Difference ({unit})'
   
    # Plot 1: Pre-RI
    plt.figure(figsize=(8, 6))
    cs1 = plt.contourf(R, P, mT1, levels=levels_composite, cmap=cmap)
    plt.colorbar(cs1, label=var_label)
    plt.gca().invert_yaxis()
    plt.xlabel('Radius from TC center (km)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Composite {var_name} for Pre-RI ({thresh}kt/{dur}h)')
    plt.savefig(os.path.join(output_dir, f'preRI_composite_{var_name}_{thresh}-{dur}.png'))
    plt.close()
   
    # Plot 2: RI
    plt.figure(figsize=(8, 6))
    cs2 = plt.contourf(R, P, mT2, levels=levels_composite, cmap=cmap)
    plt.colorbar(cs2, label=var_label)
    plt.gca().invert_yaxis()
    plt.xlabel('Radius from TC center (km)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Composite {var_name} for RI ({thresh}kt/{dur}h)')
    plt.savefig(os.path.join(output_dir, f'RI_composite_{var_name}_{thresh}-{dur}.png'))
    plt.close()
   
    # Plot 3: Difference
    plt.figure(figsize=(8, 6))
    cs3 = plt.contourf(R, P, mT, levels=levels_diff, cmap='RdBu_r', extend='both')
    plt.colorbar(cs3, label=diff_label)
    plt.gca().invert_yaxis()
    plt.xlabel('Radius from TC center (km)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Difference of {var_name} (RI - Pre-RI) ({thresh}kt/{dur}h)')
    mask = (conf >= 99) & np.isfinite(conf)
    plt.scatter(R[mask], P[mask], color='black', s=10, label='99% Confidence')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'difference_composite_{var_name}_{thresh}-{dur}.png'))
    plt.close()
   
    # Plot 4: Anomaly Pre-RI
    plt.figure(figsize=(8, 6))
    cs4 = plt.contourf(R, P, mT1_adjusted, levels=levels_composite_adjusted, cmap=cmap)
    plt.colorbar(cs4, label=var_label)
    plt.gca().invert_yaxis()
    plt.xlabel('Radius from TC center (km)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Anomaly Composite {var_name} for Pre-RI ({thresh}kt/{dur}h)')
    plt.savefig(os.path.join(output_dir, f'Anomaly_preRI_composite_{var_name}_{thresh}-{dur}.png'))
    plt.close()
   
    # Plot 5: Anomaly RI
    plt.figure(figsize=(8, 6))
    cs5 = plt.contourf(R, P, mT2_adjusted, levels=levels_composite_adjusted, cmap=cmap)
    plt.colorbar(cs5, label=var_label)
    plt.gca().invert_yaxis()
    plt.xlabel('Radius from TC center (km)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Anomaly Composite {var_name} for RI ({thresh}kt/{dur}h)')
    plt.savefig(os.path.join(output_dir, f'Anomaly_RI_composite_{var_name}_{thresh}-{dur}.png'))
    plt.close()
   
    # Plot 6: Anomaly Difference
    plt.figure(figsize=(8, 6))
    cs6 = plt.contourf(R, P, mT_adjusted, levels=levels_diff_adjusted, cmap='RdBu_r', extend='both')
    plt.colorbar(cs6, label=anomaly_diff_label)
    plt.gca().invert_yaxis()
    plt.xlabel('Radius from TC center (km)')
    plt.ylabel('Pressure (hPa)')
    plt.title(f'Difference of Anomaly Composite {var_name} (RI - Pre-RI) ({thresh}kt/{dur}h)')
    mask_a = (conf_adjusted >= 99) & np.isfinite(conf_adjusted)
    plt.scatter(R[mask_a], P[mask_a], color='black', s=10, label='99% Confidence')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'Anomaly_difference_composite_{var_name}_{thresh}-{dur}.png'))
    plt.close()
   
    print(f"✅ 完成 {var_name} {thresh}kt/{dur}h")

# ====================== 執行 ======================
print("\n=== 第二階段：開始繪製所有圖（全部使用 30kt/24h + 65kt/24h 算出的固定色階） ===")
for var_name in ['PV', 'THE']:
    cmap = 'rainbow' if var_name == 'PV' else 'plasma'
    for thresh in ri_thresholds:
        for dur in ri_durations:
            generate_plots(var_name, thresh, dur, output_dir, cmap, R, P, global_ranges)

print("\n🎉 全部完成！")
print("   所有圖已使用「僅來自 30kt/24h 和 65kt/24h 這兩組資料」計算出的固定色階")
print("   → 不再受其他極端 threshold 影響，顏色範圍合理，可直接跨圖比較")
print(f"儲存路徑：{output_dir}")