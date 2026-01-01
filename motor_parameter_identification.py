import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# ==============================================================================
#                               1. 全局配置参数
# ==============================================================================

# --- 路径配置 ---
INPUT_DATA_DIR = './Motor_DM4310/Python/MotorData'
OUTPUT_SAVE_DIR = './Motor_DM4310/Python/MotorResultsData'

# --- 采样配置 ---
FS = 100.0       

# --- 裁剪配置 (新增) ---
TRIM_START_SEC = 1.0  # 去掉开头多少秒
TRIM_END_SEC = 1.0    # 去掉结尾多少秒

# --- 滤波器配置 (Hz) ---
FILTER_CONFIG = {
    'pos': 3,   # Position Cutoff
    'vel': 4,   # Velocity Cutoff
    'acc': 2.5,   # Acceleration Cutoff
    'tau': 3    # Torque Cutoff
}

# --- 功能开关 ---
SHOW_PLOTS = False 
SAVE_PLOTS = True

# ==============================================================================
#                               2. 工具函数
# ==============================================================================

def low_pass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False) # type: ignore
    return signal.filtfilt(b, a, data)

# ==============================================================================
#                               3. 主程序逻辑
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(INPUT_DATA_DIR):
        print(f"错误：输入文件夹 '{INPUT_DATA_DIR}' 不存在！")
        exit(1)
        
    if not os.path.exists(OUTPUT_SAVE_DIR):
        try:
            os.makedirs(OUTPUT_SAVE_DIR)
        except OSError as e:
            print(f"错误：无法创建输出文件夹。{e}")
            exit(1)

    csv_files = [f for f in os.listdir(INPUT_DATA_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"在 '{INPUT_DATA_DIR}' 中未找到 CSV 文件！")
        exit(0)

    print(f"找到 {len(csv_files)} 个数据文件，开始参数辨识...\n")
    
    all_results_list = []

    for i, filename in enumerate(csv_files):
        file_path = os.path.join(INPUT_DATA_DIR, filename)
        print(f"[{i+1}/{len(csv_files)}] 正在处理: {filename}")

        try:
            # Step 1: 读取数据
            df = pd.read_csv(file_path)
            
            # ==================================================================
            # Step 1.5: 数据裁剪 (新增核心逻辑)
            # ==================================================================
            # 获取原始数据的时间范围
            t_min = df['time'].iloc[0]
            t_max = df['time'].iloc[-1]
            
            # 计算保留的时间窗口
            valid_start = t_min + TRIM_START_SEC
            valid_end = t_max - TRIM_END_SEC
            
            # 检查剩余时间是否有效 (防止数据太短被切没了)
            if valid_end > valid_start:
                original_len = len(df)
                # 执行裁剪
                df = df[(df['time'] >= valid_start) & (df['time'] <= valid_end)]
                new_len = len(df)
                print(f"    -> 已裁剪数据: {original_len} -> {new_len} 行 (去掉首尾各 {TRIM_START_SEC}s, {TRIM_END_SEC}s)")
            else:
                print(f"    -> [警告] 数据过短 ({t_max - t_min:.2f}s)，无法裁剪首尾 {TRIM_START_SEC + TRIM_END_SEC}s，将使用原始数据。")

            # 重置索引，防止后续处理出错 (虽然主要用values，但是个好习惯)
            df = df.reset_index(drop=True)
            
            # ==================================================================

            pos_act = df['pos_act'].values 
            vel_act = df['vel_act'].values 
            tau_act = df['tau_act'].values 
            time = df['time'].values       

            # Step 2: 数据预处理
            acc_raw = np.gradient(vel_act, time) # type: ignore

            # 应用滤波 (使用字典中的配置)
            pos_filt = low_pass_filter(pos_act, FILTER_CONFIG['pos'], FS)
            vel_filt = low_pass_filter(vel_act, FILTER_CONFIG['vel'], FS)
            acc_filt = low_pass_filter(acc_raw, FILTER_CONFIG['acc'], FS)
            tau_filt = low_pass_filter(tau_act, FILTER_CONFIG['tau'], FS)

            # Step 3: 构建回归矩阵
            Phi = np.vstack([
                acc_filt,               # J
                vel_filt,               # B
                np.sign(vel_filt),      # C
                np.sin(pos_filt),       # Gs
                np.cos(pos_filt),       # Gc
                np.ones_like(vel_filt)  # Bias
            ]).T
            
            Y = tau_filt

            # Step 4: 求解
            theta, residuals, rank, s = np.linalg.lstsq(Phi, Y, rcond=None)

            J_est    = theta[0]
            B_est    = theta[1]
            C_est    = theta[2]
            Gs_est   = theta[3]
            Gc_est   = theta[4]
            Bias_est = theta[5]

            Gravity_Moment = np.sqrt(Gs_est**2 + Gc_est**2)
            tau_gravity_reconstruct = Gs_est * np.sin(pos_filt) + Gc_est * np.cos(pos_filt)
            
            tau_pred = (J_est * acc_filt + B_est * vel_filt + C_est * np.sign(vel_filt) + 
                        tau_gravity_reconstruct + Bias_est)

            ss_res = np.sum((tau_filt - tau_pred) ** 2)
            ss_tot = np.sum((tau_filt - np.mean(tau_filt)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

            # Step 5: 收集结果
            result_data = {
                "FileName": filename,
                "R2_Score": r2,
                "J (Inertia)": J_est,
                "B (Viscous)": B_est,
                "C (Coulomb)": C_est,
                "Gs (Sin_Comp)": Gs_est,
                "Gc (Cos_Comp)": Gc_est,
                "Gravity_Max (mgl)": Gravity_Moment,
                "Bias_Torque": Bias_est
            }
            all_results_list.append(result_data)
            print(f"    -> 拟合结果 R²: {r2:.4f} | J: {J_est:.5f} | mgl: {Gravity_Moment:.5f}")

            # ==================================================================
            # Step 6: 可视化 (绘图)
            # ==================================================================
            if SHOW_PLOTS or SAVE_PLOTS:
                fig, axes = plt.subplots(3, 2, figsize=(16, 10))
                fig.suptitle(f"ID Result: {filename} (R²={r2:.4f}) [Trimmed {TRIM_START_SEC}s/{TRIM_END_SEC}s]", fontsize=14)
                
                # 定义绘图辅助函数
                def plot_signal(ax, time, raw, filt, cutoff_freq, title, ylabel, raw_color='lightgray', filt_color='b'):
                    ax.plot(time, raw, color=raw_color, label='Raw')
                    ax.plot(time, filt, color=filt_color, linewidth=1.5, label=f'Filt (Fc={cutoff_freq}Hz)')
                    ax.set_title(title)
                    ax.set_ylabel(ylabel)
                    ax.legend(loc='upper right', fontsize='small')
                    ax.grid(True, alpha=0.3)

                # 1. Position
                plot_signal(axes[0, 0], time, pos_act, pos_filt, FILTER_CONFIG['pos'], 
                            "1. Position", "Rad", filt_color='b')

                # 2. Velocity
                plot_signal(axes[0, 1], time, vel_act, vel_filt, FILTER_CONFIG['vel'],
                            "2. Velocity", "Rad/s", filt_color='g')

                # 3. Acceleration
                plot_signal(axes[1, 0], time, acc_raw, acc_filt, FILTER_CONFIG['acc'],
                            "3. Acceleration", "Rad/s²", filt_color='m')

                # 4. Torque Fitting
                ax4 = axes[1, 1]
                ax4.plot(time, tau_filt, 'b', label=f'Actual Torque (Fc={FILTER_CONFIG["tau"]}Hz)')
                ax4.plot(time, tau_pred, 'r--', label='Predicted Torque')
                ax4.set_title(f"4. Torque Fitting (R²={r2:.3f})")
                ax4.set_ylabel("Nm")
                ax4.legend(loc='upper right', fontsize='small')
                ax4.grid(True, alpha=0.3)

                # 5. Component Breakdown
                ax5 = axes[2, 0]
                ax5.plot(time, J_est * acc_filt, label='Inertia (J*a)')
                ax5.plot(time, B_est * vel_filt + C_est * np.sign(vel_filt), label='Friction (B+C)')
                ax5.plot(time, tau_gravity_reconstruct, label='Gravity (G)')
                ax5.set_title("5. Torque Components Breakdown")
                ax5.set_ylabel("Nm")
                ax5.legend(loc='upper right', fontsize='small')
                ax5.grid(True, alpha=0.3)

                # 6. Residuals
                ax6 = axes[2, 1]
                error = tau_filt - tau_pred
                ax6.plot(time, error, 'k', linewidth=1)
                ax6.set_title("6. Residual Error")
                ax6.set_ylabel("Error (Nm)")
                ax6.set_xlabel("Time (s)")
                ax6.grid(True, alpha=0.3)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore

                if SAVE_PLOTS:
                    base_name = os.path.splitext(filename)[0]
                    img_path = os.path.join(OUTPUT_SAVE_DIR, f"{base_name}_result.png")
                    plt.savefig(img_path)
                
                if SHOW_PLOTS:
                    plt.show()
                else:
                    plt.close(fig)

        except Exception as e:
            print(f"!!! 处理文件 {filename} 时发生异常: {e}")
            all_results_list.append({"FileName": filename, "R2_Score": 0, "Error": str(e)})

    # --- C. 保存汇总 CSV ---
    if all_results_list:
        output_csv_path = os.path.join(OUTPUT_SAVE_DIR, 'identification_summary.csv')
        df_results = pd.DataFrame(all_results_list)
        cols = ['FileName', 'R2_Score', 'J (Inertia)', 'B (Viscous)', 
                'C (Coulomb)', 'Gs (Sin_Comp)', 'Gc (Cos_Comp)', 
                'Gravity_Max (mgl)', 'Bias_Torque']
        final_cols = [c for c in cols if c in df_results.columns] + \
                     [c for c in df_results.columns if c not in cols]
        df_results = df_results[final_cols]
        df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*50)
        print("所有文件处理完毕！")
        print(f"汇总数据已保存至: {output_csv_path}")
        print("="*50)