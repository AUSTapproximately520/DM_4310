import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
import time

# ================= 配置区域 =================

# 1. 路径设置
DATA_FOLDER = './Motor_DM4310/Python/MotorData' 
PARAM_FILE = './Motor_DM4310/Python/MotorResultsData/identification_summary.csv'
OUTPUT_DIR = './Motor_DM4310/Python/Cross_Validation_Results'

# 2. 显示与保存控制
SHOW_PLOTS_ON_SCREEN = False 

# 3. 信号处理参数
SAMPLING_FREQ = 100.0  # Hz
CUTOFF_FREQ_POS = 3  
CUTOFF_FREQ_VEL = 4  
CUTOFF_FREQ_ACC = 2.5  
CUTOFF_FREQ_TAU = 3  

# 4. 物理常量
GRAVITY_PHASE = 1  

# ===========================================

def low_pass_filter(data, cutoff, fs, order=2):
    """零相位低通滤波器"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0: normal_cutoff = 0.99
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
    return signal.filtfilt(b, a, data)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("--- 开始动力学参数交叉验证 (Cross-Validation) ---\n")
    print(f"显示模式: {'[弹窗显示]' if SHOW_PLOTS_ON_SCREEN else '[静默保存]'}")

    # --- 1. 读取参数集 (Sources) ---
    try:
        df_params = pd.read_csv(PARAM_FILE)
        if 'FileName' not in df_params.columns:
            print("错误: 参数表缺少 'FileName' 列")
            return
        df_params = df_params.dropna(subset=['FileName'])
        print(f"【参数加载】共 {len(df_params)} 组待验证参数。")
    except Exception as e:
        print(f"错误: 读取参数文件失败 {e}")
        return

    # --- 2. 读取数据集 (Targets) ---
    csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    if not csv_files:
        print("错误: 未找到任何数据文件。")
        return
    print(f"【数据加载】共 {len(csv_files)} 个原始数据文件待验证。\n")

    summary_results = []
    
    total_tasks = len(df_params) * len(csv_files)
    current_task = 0

    # ==========================================
    #       双重循环: 参数组 x 数据文件
    # ==========================================
    
    # 外层循环: 遍历每一组参数 (Source Params)
    for idx, param_row in df_params.iterrows():
        param_source_name = str(param_row['FileName']).strip()
        
        # --- [关键修改 1] --- 
        # 为当前这组参数创建一个专属文件夹
        # 去掉文件后缀作为文件夹名 (例如: "result_01.csv" -> 文件夹 "result_01")
        safe_param_folder_name = os.path.splitext(param_source_name)[0]
        current_param_dir = os.path.join(OUTPUT_DIR, safe_param_folder_name)
        
        if not os.path.exists(current_param_dir):
            os.makedirs(current_param_dir)
        # --------------------

        # 提取参数
        J = param_row.get('J (Inertia)', 0)
        B = param_row.get('B (Viscous)', 0)
        C = param_row.get('C (Coulomb)', 0)
        G = param_row.get('Gravity_Max (mgl)', 0)
        Bias = param_row.get('Bias_Torque', 0)
        
        print(f"\n>>> 正在测试参数组: [{safe_param_folder_name}] -> 存入文件夹: {current_param_dir}")
        print(f"    (J={J:.4f}, B={B:.4f}, C={C:.4f}, G={G:.4f})")

        # 内层循环: 遍历每一个数据文件 (Target Data)
        for file_path in csv_files:
            target_data_name = os.path.basename(file_path).strip()
            current_task += 1
            
            try:
                # --- 数据预处理 ---
                df = pd.read_csv(file_path)
                dt = np.mean(np.diff(df['time']))
                if np.isnan(dt) or dt == 0: dt = 1/SAMPLING_FREQ

                # 滤波与微分
                pos_filt = low_pass_filter(df['pos_act'].values, CUTOFF_FREQ_POS, SAMPLING_FREQ)
                
                vel_raw = np.gradient(pos_filt, dt)
                vel_calc = low_pass_filter(vel_raw, CUTOFF_FREQ_VEL, SAMPLING_FREQ)
                
                acc_raw = np.gradient(vel_calc, dt)
                acc_calc = low_pass_filter(acc_raw, CUTOFF_FREQ_ACC, SAMPLING_FREQ)
                
                # 实际力矩滤波 (Ground Truth)
                tau_act_filt = low_pass_filter(df['tau_act'].values, CUTOFF_FREQ_TAU, SAMPLING_FREQ)

                # --- 动力学预测 ---
                tau_pred = (J * acc_calc) + (B * vel_calc) + (C * np.sign(vel_calc)) + \
                           (G * np.sin(pos_filt) * GRAVITY_PHASE) + Bias

                # --- 误差计算 ---
                error = tau_act_filt - tau_pred
                rmse = np.sqrt(np.mean(error**2))
                ss_res = np.sum(error**2)
                ss_tot = np.sum((tau_act_filt - np.mean(tau_act_filt))**2)
                r2 = 1 - (ss_res / ss_tot)

                print(f"    [{current_task}/{total_tasks}] 验证 -> {target_data_name} | R²={r2:.4f}")

                # 记录结果 (包含相对路径以便知道图片在哪)
                summary_results.append({
                    'Param_Folder': safe_param_folder_name, # 方便筛选
                    'Param_Source': param_source_name,
                    'Data_Target': target_data_name,
                    'R2': r2,
                    'RMSE': rmse,
                    'J': J, 'B': B, 'C': C, 'G': G
                })

                # ================= 绘图 =================
                fig = plt.figure(figsize=(12, 14))
                plt.subplots_adjust(top=0.84, bottom=0.05, hspace=0.3)

                header_text = (
                    f"CROSS-VALIDATION RESULT\n"
                    f"-----------------------\n"
                    f"PARAMS FROM : {param_source_name}\n"
                    f"TEST DATA   : {target_data_name}\n\n"
                    f"METRICS     : R² = {r2:.4f}  |  RMSE = {rmse:.4f}\n"
                    f"VALUES      : J={J:.4f} B={B:.4f} C={C:.4f} G={G:.4f}"
                )
                
                box_color = "#eaffea" if r2 > 0.9 else ("#fff8ea" if r2 > 0.8 else "#ffeaea")
                edge_color = "green" if r2 > 0.9 else ("orange" if r2 > 0.8 else "red")

                fig.text(0.5, 0.92, header_text, fontsize=11, fontfamily='monospace',
                         ha='center', va='center',
                         bbox=dict(boxstyle="round,pad=0.8", facecolor=box_color, edgecolor=edge_color))

                # 图1: 力矩
                ax1 = plt.subplot(3, 1, 1)
                ax1.plot(df['time'], df['tau_act'], color='lightgray', label='Raw', linewidth=1)
                ax1.plot(df['time'], tau_act_filt, 'k-', label='Act (Filt)', linewidth=1.5)
                ax1.plot(df['time'], tau_pred, 'r--', label='Pred', linewidth=1.5)
                ax1.set_title('Torque Comparison')
                ax1.legend(loc='upper right')
                ax1.grid(True, alpha=0.3)

                # 图2: 运动状态
                ax2 = plt.subplot(3, 1, 2)
                ax2.plot(df['time'], vel_calc, 'b', label='Vel')
                ax2.plot(df['time'], acc_calc, 'm', label='Acc', alpha=0.6)
                ax2.set_title('Motion States')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # 图3: 误差
                ax3 = plt.subplot(3, 1, 3)
                ax3.plot(df['time'], error, 'k-')
                ax3.fill_between(df['time'], error, 0, color='red', alpha=0.1)
                ax3.set_title(f'Error Residual (RMSE={rmse:.3f})')
                ax3.grid(True, alpha=0.3)

                # --- [关键修改 2] ---
                # 保存图片路径更改为当前参数的子文件夹
                # 文件名简化为: VS__目标数据名.png (因为已经在参数专属文件夹里了)
                safe_data_name = os.path.splitext(target_data_name)[0]
                save_filename = f"VS__{safe_data_name}.png"
                
                save_path = os.path.join(current_param_dir, save_filename)
                
                plt.savefig(save_path, dpi=80)
                # --------------------
                
                if SHOW_PLOTS_ON_SCREEN:
                    plt.show() 
                else:
                    plt.close(fig) 

            except Exception as e:
                print(f"    !!! 验证失败 ({target_data_name}): {e}")

    # --- 保存汇总 CSV ---
    # CSV 依然保存在根目录下，作为一个总索引
    if summary_results:
        df_res = pd.DataFrame(summary_results)
        df_res = df_res.sort_values(by='R2', ascending=False)
        
        # 将列顺序调整得好看一些
        cols = ['Param_Folder', 'Data_Target', 'R2', 'RMSE', 'J', 'B', 'C', 'G', 'Param_Source']
        df_res = df_res[cols]

        report_path = os.path.join(OUTPUT_DIR, 'Cross_Validation_Summary.csv')
        df_res.to_csv(report_path, index=False)
        
        print(f"\n==========================================")
        print(f"交叉验证完成！")
        print(f"总计生成图表: {len(summary_results)} 张")
        print(f"文件结构说明: 结果已按 [参数文件名] 分类存入子文件夹")
        print(f"汇总报告路径: {report_path}")
        print(f"==========================================")

if __name__ == "__main__":
    main()