import math
from DM_CAN import * 
import serial
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

# ==============================================================================
#                               1. 全局配置参数
# ==============================================================================

# --- 【新增】速度计算模式配置 ---
# 选项: 
# 'SENSOR' : 直接读取电机内部传感器速度 (噪声可能较大，但实时性高)
# 'DIFF'   : 使用位置差分计算速度 (公式: (Pos_now - Pos_last)/dt，曲线较平滑但在低速有量化噪声)
VEL_FEEDBACK_MODE = 'SENSOR'  # <--- 在这里修改 'SENSOR' 或 'DIFF'

# --- 通讯配置 ---
SERIAL_PORT = 'COM10'       
BAUD_RATE = 921600 
MOTOR_TYPE = DM_Motor_Type.DM4310_48V 
MOTOR_SLAVE_ID = 0x04             
MOTOR_MASTER_ID = 0x14

# --- 实验时间配置 ---
RECORD_TIME = 20       # 总时长 (秒)
TIME_STEP = 0.010      # 控制周期 (秒) -> 100Hz
TOTAL_STEP = int(RECORD_TIME / TIME_STEP)

# --- MIT 控制器参数 ---
MIT_Kp = 3.0
MIT_Kd = 0.1

# --- 轨迹参数 (变幅值正弦波) ---
# P_des = (a_des * t / T + 1) * sin(w_des * pi * t)
a_des = 2   # 幅值增长系数
w_des = 2   # 频率系数

# --- 文件保存配置 ---
SAVE_DIR = './Motor_DM4310/Python/MotorData'   # 数据保存文件夹路径
PLOT_SWITCH = True         # 是否画图

# ==============================================================================
#                               2. 初始化与主循环
# ==============================================================================

# 初始化数据容器
data_time = []
data_pos_des = []
data_pos_act = []
data_vel_des = []
data_vel_act = []
data_tau_act = []

# 创建对象
Motor1 = Motor(MOTOR_TYPE, MOTOR_SLAVE_ID, MOTOR_MASTER_ID)
try:
    serial_device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.5)
except Exception as e:
    print(f"【错误】无法打开串口 {SERIAL_PORT}: {e}")
    exit(1)

MotorControl1 = MotorControl(serial_device)
MotorControl1.addMotor(Motor1)

global start_time
start_time = 0.0

print("="*40)
print(f"开始实验: {RECORD_TIME}s, MIT模式")
print(f"当前速度反馈模式: {VEL_FEEDBACK_MODE}")
print("="*40)

try:    
    # --- 电机初始化 ---
    MotorControl1.switchControlMode(Motor1, Control_Type.MIT)
    MotorControl1.save_motor_param(Motor1)
    # MotorControl1.set_zero_position(Motor1) # 设置当前位置为零点，仅在需要时打开
    MotorControl1.enable(Motor1)
    print(">>> 电机已使能")   
    
    # --- 【新增】差分计算所需的变量初始化 ---
    # 先读取一次当前状态作为初始值，防止第一次差分出错
    MotorControl1.refresh_motor_status(Motor1)
    last_pos_act = Motor1.getPosition() 
    
    start_time = time.perf_counter()
    
    # --- 控制循环 ---
    for i in range(TOTAL_STEP):
        loop_start_time = time.perf_counter()
        t = (i) * TIME_STEP  # 当前时间
        
        # 1. 刷新并读取电机状态
        # 注意：这里我们每次都读取，但下面根据模式决定使用哪一个值
        raw_pos_now = Motor1.getPosition()
        raw_vel_sensor = Motor1.getVelocity()
        tau_now = Motor1.getTorque()
        
        # --- 【新增】核心修改：速度源选择逻辑 ---
        if VEL_FEEDBACK_MODE == 'DIFF':
            if i == 0:
                # 第一帧没有前一帧数据，暂时使用传感器数据或设为0
                v_now = raw_vel_sensor
            else:
                # 差分计算: (当前位置 - 上次位置) / 时间步长
                v_now = (raw_pos_now - last_pos_act) / TIME_STEP
        else:
            # 默认模式：使用传感器数据
            v_now = raw_vel_sensor
        
        # 更新用于下次计算的“上次位置”
        last_pos_act = raw_pos_now
        
        # 为了保持变量名统一，位置也就使用当前读取的
        p_now = raw_pos_now

        # 2. 计算期望轨迹
        # 公式: (a * t/T + 1) * sin(w * pi * t)
        p_des = (a_des * t / RECORD_TIME + 1) * math.sin(w_des * math.pi * t)
        
        # 计算期望速度
        v_des = (a_des * t / RECORD_TIME + 1) * w_des * math.pi * math.cos(w_des * math.pi * t)+ (a_des / RECORD_TIME) * math.sin(w_des * math.pi * t)
    
        # 3. 计算误差 (用于显示或调试，使用上面选定的 v_now)
        e = p_now - p_des
        e_dot = v_now - v_des 
        
        # 4. 记录数据 (记录选定模式下的 v_now)
        data_time.append(t)
        data_pos_des.append(p_des)
        data_pos_act.append(p_now)
        data_vel_des.append(v_des)
        data_vel_act.append(v_now) 
        data_tau_act.append(tau_now)
        
        # 5. 安全保护 (力矩过大紧急停止)
        if abs(tau_now) > 3 or abs(e) > 2: 
            print(f"【警告】电机过载 {tau_now:.2f}Nm,位置误差{e:.2f},速度误差{e_dot:.2f}! 紧急停止。")            
            break
            
        # 6. 发送 MIT 控制指令
        # controlMIT(motor, kp, kd, q_des, qd_des, tau_ff)
        MotorControl1.controlMIT(Motor1, MIT_Kp, MIT_Kd, p_des, 0, 0)
        
        # 时间控制
        if (time.perf_counter() - loop_start_time) < TIME_STEP:
            while (time.perf_counter() - loop_start_time) < TIME_STEP:
                pass
        else:
            print(f"第{i}次循环超时,超时时间{(time.perf_counter() - loop_start_time)*1000}ms")
            
        # 每次循环刷新一次状态给下一次用 (放在循环头还是尾取决于库的实现，这里放在循环头获取数据更实时，因此循环尾不需要再 refresh，除非库要求)
        MotorControl1.refresh_motor_status(Motor1)

except KeyboardInterrupt:
    print("\n【用户中断】正在停止...")

except Exception as e:
    print(f"\n【运行时错误】: {e}")

finally:
    # --- 安全退出序列 ---
    duration = time.perf_counter() - start_time
    print(f"实验结束，实际运行时间: {duration:.2f}秒")
    print("正在关闭电机...")

    MotorControl1.controlMIT(Motor1, 2, 0.2, 0, 0, 0)

    # 简单的等待归零逻辑
    stop_wait_start = time.time()
    while abs(Motor1.getPosition()) > 0.05 and (time.time() - stop_wait_start) < 3.0:  
        MotorControl1.refresh_motor_status(Motor1)
        
    MotorControl1.disable(Motor1)
    serial_device.close() 
    print("电机已关闭")


# ==============================================================================
#                               3. 数据保存与可视化
# ==============================================================================

# 生成时间戳
timestamp = time.strftime("%Y%m%d_%H%M%S")

# --- 1. 创建目录 ---
if not os.path.exists(SAVE_DIR):
    try:
        os.makedirs(SAVE_DIR)
        print(f"已创建目录: {SAVE_DIR}")
    except OSError as e:
        print(f"创建目录失败: {e}, 将保存至当前目录")
        SAVE_DIR = '.'

# --- 2. 保存 CSV ---
try:
    csv_filename = f"motor_exp_data_{timestamp}.csv"
    csv_path = os.path.join(SAVE_DIR, csv_filename)
    
    # 保存时记录当前使用的模式，方便后续分析知道是测出来的还是算出来的
    df = pd.DataFrame({
        'time': data_time,
        'pos_des': data_pos_des,
        'pos_act': data_pos_act,
        'vel_des': data_vel_des,
        'vel_act': data_vel_act,
        'tau_act': data_tau_act
    })
    df.to_csv(csv_path, index=False)
    print(f"数据已保存: {csv_path}")
except Exception as e:
    print(f"保存CSV失败: {e}")

# --- 3. 绘图 (优化版) ---
if PLOT_SWITCH and len(data_time) > 0:
    try:
        print("正在生成图表...")
        
        # 设置画布
        fig = plt.figure(figsize=(14, 12)) 
        
        # --- 构造顶部信息栏 ---
        formula_str = fr"({a_des} \cdot \frac{{t}}{{{RECORD_TIME}}} + 1) \cdot \sin({w_des}\pi t)"
        
        vel_source_str = "Position Differentiation" if VEL_FEEDBACK_MODE == 'DIFF' else "Motor Sensor"

        info_text = (
            f"[ Experiment Parameters ] Duration: {RECORD_TIME}s | Step: {TIME_STEP}s | Count: {len(data_time)}\n"
            f"[ Control Gains ] MIT Mode: $K_p = {MIT_Kp}$  |  $K_d = {MIT_Kd}$\n"
            f"[ Velocity Source ] {vel_source_str}\n" 
            f"[ Trajectory ] $P_{{des}} = {formula_str}$"
        )
        
        # 绘制文本框
        plt.figtext(0.5, 0.95, info_text, ha='center', va='top', fontsize=11, family='serif',
                    bbox={"facecolor":"#f8f9fa", "alpha":0.9, "pad":12, "edgecolor":"#ced4da", "boxstyle":"round,pad=0.5"})
        
        # 调整子图布局
        plt.subplots_adjust(top=0.86, hspace=0.35)

        # --- 子图 1: 位置 ---
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(data_time, data_pos_des, 'r--', label='Desired Pos', linewidth=1.5)
        ax1.plot(data_time, data_pos_act, 'b', label='Actual Pos', linewidth=1.0, alpha=0.8)
        ax1.set_title('Position Tracking', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Position (rad)')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # --- 子图 2: 速度 ---
        ax2 = plt.subplot(3, 1, 2)
        ax2.plot(data_time, data_vel_des, 'r--', label='Desired Vel')
        
        # 动态修改 Label 名字
        act_vel_label = f'Actual Vel ({VEL_FEEDBACK_MODE})'
        ax2.plot(data_time, data_vel_act, 'b', label=act_vel_label, alpha=0.8)
        
        ax2.set_title('Velocity Tracking', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.legend(loc='upper right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # --- 子图 3: 力矩 ---
        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(data_time, data_tau_act, 'g', label='Actual Torque', linewidth=1.0)
        ax3.set_title('Actual Torque Response', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time (s)', fontsize=10)
        ax3.set_ylabel('Torque (N.m)')
        ax3.legend(loc='upper right')
        ax3.grid(True, linestyle='--', alpha=0.6)

        # 保存图片
        img_filename = f"motor_exp_plot_{timestamp}.png"
        img_path = os.path.join(SAVE_DIR, img_filename)
        
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {img_path}")
        
        plt.show()

    except Exception as e:
        print(f"绘图失败: {e}")