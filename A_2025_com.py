import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置绘图风格，符合学术论文标准
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

class MCM_Stair_Model:
    def __init__(self, stair_width=1.2, resolution=100):
        """
        初始化楼梯模型
        :param stair_width: 楼梯宽度 (m)
        :param resolution: 横向切片的分辨率 (点数)
        """
        self.width = stair_width
        self.x = np.linspace(-stair_width/2, stair_width/2, resolution)
        self.wear_depth = np.zeros(resolution)
        
    def ndt_material_analysis(self, density_gcm3, pulse_velocity_kms):
        """
        [模型部分 1] 基于 applsci 论文的 NDT 非破坏性检测
        输入：石材密度和超声波速
        输出：预测的 Böhme 磨损值 (BAV)
        逻辑：波速越快、密度越高 -> 石材越硬 -> BAV 越低
        """
        # 简化的回归公式 (基于文献数据的拟合趋势)
        # BAV 与 密度 和 Vp 成反比
        predicted_bav = 45.0 - (8.0 * density_gcm3) - (3.5 * pulse_velocity_kms)
        predicted_bav = max(2.0, predicted_bav) # BAV 不能小于 2 (极硬花岗岩)
        
        # 将 BAV 转换为 Archard 磨损系数 k (mm^3/Nm)
        # 参考 Karaca et al.: BAV (cm^3/50cm^2) -> k
        # 转换系数基于实验数据的量级估算
        wear_coefficient_k = (predicted_bav * 1e-12) * 2.5 
        
        return predicted_bav, wear_coefficient_k

    def get_pedestrian_distribution(self, mode='single'):
        """
        [模型部分 2] 基于 MMADP 的行人步态分布 (蒙特卡洛概率密度)
        """
        if mode == 'single':
            # 单列通行：正态分布在中心，标准差较小
            pdf = norm.pdf(self.x, loc=0, scale=0.15)
        elif mode == 'side_by_side':
            # 并排通行：双峰分布
            pdf = 0.5 * norm.pdf(self.x, loc=-0.3, scale=0.12) + \
                  0.5 * norm.pdf(self.x, loc=0.3, scale=0.12)
        elif mode == 'random':
            # 随机漫步：分布更平坦
            pdf = norm.pdf(self.x, loc=0, scale=0.4)
            
        # 归一化，保证总积分为 1
        pdf /= np.trapz(pdf, self.x)
        return pdf

    def simulate_wear(self, years, daily_traffic, material_params, traffic_mode='single', direction_bias=1.0):
        """
        [仿真核心] 模拟岁月的流逝
        :param material_params: (bav, k_value)
        :param direction_bias: 1.0 = 平衡, >1.0 = 下楼为主 (压力更大)
        """
        bav, k = material_params
        total_people = years * daily_traffic * 365
        
        # 1. 获取横向分布概率
        prob_dist = self.get_pedestrian_distribution(traffic_mode)
        
        # 2. 计算平均单次踩踏压力 (Archard Law Input)
        # 假设平均体重 75kg (750N), 足底面积 0.015 m^2
        # 下楼冲击力大，上楼小。direction_bias 模拟这个加权平均。
        avg_load_n = 750 * direction_bias 
        
        # 3. 计算总磨损深度 (Archard Equation: V = k * F * s)
        # 这里的计算通过概率密度函数 (PDF) 将离散步数转化为连续磨损
        # 磨损深度 d(x) = Total_N * Prob(x) * k * Load
        
        # 这是一个物理简化的积分形式
        incremental_wear = total_people * prob_dist * k * avg_load_n * 1000 # *1000 转为 mm
        
        self.wear_depth += incremental_wear
        return self.wear_depth

# ==========================================
#  执行仿真与绘图 (生成论文所需的图表)
# ==========================================

# 初始化模型
stair = MCM_Stair_Model()

# --- 场景设定 ---
# 场景 A: 古老教堂的石灰岩楼梯 (Limestone)
# NDT 数据: 密度 2.4 g/cm3, 波速 3.5 km/s (较软)
limestone_bav, limestone_k = stair.ndt_material_analysis(2.4, 3.5)

# 场景 B: 城堡的花岗岩楼梯 (Granite)
# NDT 数据: 密度 2.7 g/cm3, 波速 5.5 km/s (很硬)
granite_bav, granite_k = stair.ndt_material_analysis(2.7, 5.5)

print(f"--- Material Analysis Results (Model 1) ---")
print(f"Limestone: Predicted BAV = {limestone_bav:.2f}, k = {limestone_k:.2e}")
print(f"Granite:   Predicted BAV = {granite_bav:.2f}, k = {granite_k:.2e}")

# -------------------------------------------------------
# # 图 1: NDT 材料定性分析图 (对应 applsci 论文)
# # -------------------------------------------------------
# densities = np.linspace(2.2, 3.0, 50)
# velocities = np.linspace(2.0, 6.0, 50)
# D, V = np.meshgrid(densities, velocities)
# BAV_matrix = 45.0 - (8.0 * D) - (3.5 * V) # 使用上面的回归公式
#
# plt.figure(figsize=(10, 6))
# cp = plt.contourf(D, V, BAV_matrix, levels=20, cmap='viridis_r')
# cbar = plt.colorbar(cp)
# cbar.set_label('Predicted Böhme Abrasion Value (BAV)')
# plt.scatter([2.4], [3.5], color='red', s=100, label='Sample A (Limestone)', edgecolors='white')
# plt.scatter([2.7], [5.5], color='blue', s=100, label='Sample B (Granite)', edgecolors='white')
# plt.title('Figure 1: Non-Destructive Testing (NDT) Material Classification Model')
# plt.xlabel('Rock Density (g/cm³)')
# plt.ylabel('Pulse Wave Velocity (km/s)')
# plt.legend()
# plt.tight_layout()
# plt.show()

# -------------------------------------------------------
# 图 2: 不同交通模式下的磨损横截面 (对应 MMADP 论文)
# -------------------------------------------------------
# stair_single = MCM_Stair_Model()
# stair_double = MCM_Stair_Model()
#
# # 模拟 200年，由石灰岩制成，每天 200 人
# stair_single.simulate_wear(200, 200, (limestone_bav, limestone_k), 'single')
# stair_double.simulate_wear(200, 200, (limestone_bav, limestone_k), 'side_by_side')
#
# plt.figure(figsize=(12, 5))
# plt.plot(stair_single.x, -stair_single.wear_depth, label='Single File Traffic', linewidth=2.5)
# plt.plot(stair_double.x, -stair_double.wear_depth, label='Side-by-Side Traffic', linewidth=2.5, linestyle='--')
# plt.fill_between(stair_single.x, -stair_single.wear_depth, 0, alpha=0.1, color='blue')
# plt.fill_between(stair_double.x, -stair_double.wear_depth, 0, alpha=0.1, color='orange')
# plt.title('Figure 2: Impact of Pedestrian Traffic Patterns on Wear Profile')
# plt.xlabel('Stair Width Position (m)')
# plt.ylabel('Wear Depth (mm)')
# plt.legend()
# plt.grid(True, linestyle=':')
# plt.show()

# # -------------------------------------------------------
# # 图 3: 磨损随时间演变的 3D 表面图 (Result Simulation)
# # -------------------------------------------------------
# years_range = [50, 100, 200, 500, 1000]
# profiles = []
# model_3d = MCM_Stair_Model()
#
# for y in years_range:
#     # 增量模拟
#     # 假设前一个时间段已经磨损，我们在其基础上叠加 (简化处理，直接重新计算累计值)
#     temp_model = MCM_Stair_Model()
#     temp_model.simulate_wear(y, 200, (limestone_bav, limestone_k), 'single')
#     profiles.append(temp_model.wear_depth)
#
# profiles = np.array(profiles)
# X_grid, Y_grid = np.meshgrid(model_3d.x, np.array(years_range))
#
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X_grid, Y_grid, -profiles, cmap='copper', edgecolor='none', alpha=0.9)
# ax.set_title('Figure 3: Temporal Evolution of Stair Wear (3D Simulation)')
# ax.set_xlabel('Position on Step (m)')
# ax.set_ylabel('Time (Years)')
# ax.set_zlabel('Wear Depth (mm)')
# ax.view_init(elev=30, azim=-60)
# fig.colorbar(surf, shrink=0.5, aspect=5, label='Depth')
# plt.show()
#
# # -------------------------------------------------------
# 图 4: 考古学反演图表 (Age Estimation Nomogram)
# -------------------------------------------------------
# 这是一个工具图，考古学家测量最大深度后，查图可知可能的年代
depths = np.linspace(0, 50, 100) # 0到50mm磨损
# Age = Depth / (Rate * Traffic)
# 假设三种不同流量情境
traffic_low = 50
traffic_med = 200
traffic_high = 500

# 反推年代公式: Age = Depth / (Max_PDF * k * Load * 365 * Daily_Traffic * 1000)
# 我们提取常数因子
max_pdf = np.max(stair.get_pedestrian_distribution('single'))
wear_factor = max_pdf * limestone_k * 750 * 1000

age_low = depths / (wear_factor * traffic_low * 365)
age_med = depths / (wear_factor * traffic_med * 365)
age_high = depths / (wear_factor * traffic_high * 365)

plt.figure(figsize=(10, 6))
plt.plot(depths, age_low, label='Low Traffic (50/day) - Rural Temple')
plt.plot(depths, age_med, label='Medium Traffic (200/day) - City Church')
plt.plot(depths, age_high, label='High Traffic (500/day) - Public Plaza')

plt.title('Figure 4: Archaeological Dating Nomogram (Inverse Model)')
plt.xlabel('Measured Max Wear Depth (mm)')
plt.ylabel('Estimated Age of Stairs (Years)')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.yscale('log') # 使用对数坐标，因为时间跨度大
plt.show()
