import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
# 粒子群算法的参数
w = 0.5
# 惯性权重
c1 = 1.5
# 个体学习因
c2 = 1.5
# 社会学习因千
num_particles = 30
# 粒子数最
max_iter = 10
# 最大选代次数

# 问题参数(以你的符号定义)↓
num_years = 7
# 从2024到2030年
num_crops = 41
# 作物秘类数
num_plots = 54  # 地块数量
num_seasons = 2  # 委节数最
# 读取附件2.xlsx中的数据)
file_path = 'C:\\Users\\DELL\\Desktop\\vscodePython\\math\\data2.xlsx'
#
file_path_2 = 'C:\\Users\\DELL\\Desktop\\vscodePython\\math\\data1.xlsx'
# 读取销售价格(p)、种植成本(c)、亩产量(q)等数据
data_stats = pd.read_excel(file_path, sheet_name='2023年统计的相关数据')
# 将数据转换为字符申，以防止非字符申类型导致错误!
data_stats['销售单价/(元/斤)'] = data_stats['销售单价/(元/斤)'].astype(str)
# 获取作物的销售价格(p)，使用区间的平均值
p = data_stats['销售单价/(元/斤)'].apply(lambda x: (float(x.split('-')[0]) +
                                              float(x.split('-')[1]))/2 if '-' in x else float(x)).values
# 获取作物的种植成本(c)↓
c = data_stats['种植成本/(元/亩)'].values

# 获取作物的亩产量(g)，将产量从厅转换为千克
q = (data_stats['亩产量/斤'].values/2).astype(float)  # 1斤=05 千兜
# 读取2023年农作物种植情况，假设其为预期销售量(D)
data_crop_situation = pd.read_excel(file_path, sheet_name='2023年的农作物种植情况')

data_stats_2 = pd.read_excel(file_path_2, sheet_name='乡村的现有耕地')
plot_types = data_stats_2['地块类型']


# 计算预期销售量D:使用种植面积(亩:乘以 对应作物的亩产量(q)↓
D = (data_crop_situation['种植面积/亩'].values *
     q[data_crop_situation['作物编号'].values-1])

# 输出目标函数2定义和参数读取部分
print("目标函数2已经定义，销售价格、种植成本、亩产量、预期销售量已从附件中读取。")
# 初始化!
particles = np.random.rand(num_particles, num_crops,
                           num_plots, num_seasons, num_years)  # 子的位置
velocities = np.random.rand(
    num_particles, num_crops, num_plots, num_seasons, num_years)  # 子所速度
p_best = np.copy(particles)  # 每个粒子的最佳位置
g_best = np.copy(particles[0])  # 全局最佳位智
best_fitness_over_time = []

# 确保x是0.l的倍数


def ensure_tenth_multiples(x):
    return np.round(x*10) / 10
# 目标函数八


def objective_functionl(x):
    Z1 = 0
    for t in range(num_years):
        for k in range(num_seasons):
            for i in range(num_crops):
                y_ikt = np.sum(x[i, :, k, t])*q[i]
                Z1 += p[i] * min(y_ikt, D[i])-c[i] * np.sum(x[i, :, k, t])
    return Z1
# ·由于PSO算法是求最大化问题，直接返回Z14
# ·目标函数2的定义


def objective_function2(x):
    Z2 = 0
    for t in range(num_years):
        for k in range(num_seasons):
            for i in range(num_crops):
                y_ikt = np.sum(x[i, :, k, t])*q[i]
                # 计算收益，考虑滞销部分按折扣价出售
                Z2 += p[i] * min(y_ikt, D[i])+(0.5 * p[i]) * \
                    max(y_ikt-D[i], 0)-c[i] * np.sum(x[i, :, k, t])
    return Z2  # 由于PSO导法是求最大化问题，直接返回Z2


A_j = [80, 55, 35, 72, 68, 55, 60, 46, 40, 28, 25, 86, 55, 44, 50, 25, 60, 45, 35, 20, 15, 13, 15, 18, 27, 20, 15, 10, 14,
       6, 10, 12, 22, 20, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]


def apply_constraints(_particles):
    for n in range(num_particles):
        for j in range(num_plots):
            # ·种植面积为0.1的倍数的约束
            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years):
                        _particles[n][i, j, k, t] = ensure_tenth_multiples(
                            _particles[n][i, j, k, t])
            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years):
                        if plot_types[j] == '普通大棚' and k == 1 and i + 1 not in range(38, 42):
                            # ·普通大棚限制
                            _particles[n][i, j, k, t] = 0
                        if plot_types[j] == '普通大棚' and k == 0 and i + 1 not in range(17, 35):
                            # ·普通大棚限制
                            _particles[n][i, j, k, t] = 0
                        if plot_types[j] == '智慧大棚' and i + 1 not in range(17, 35):
                            # ·智慧大棚限制<
                            _particles[n][i, j, k, t] = 0
                        if plot_types[j] == '水浇地' and k == 0 and (i + 1 not in range(16, 35)):
                            _particles[n][i, j, k, t] = 0
                        if plot_types[j] == '水浇地' and k == 1 and i + 1 not in [16, 35, 36, 37]:
                            _particles[n][i, j, k, t] = 0
                        if (plot_types[j] == '平旱地' or plot_types[j] == '梯田' or plot_types[j] == '山坡地') and (i + 1 not in range(1, 16)):
                            _particles[n][i, j, k, t] = 0

            # ·作物种植集中度约束←
            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years):
                        if _particles[n][i, j, k, t] < 0.1 and _particles[n][i, j, k, t] > 0:
                            _particles[n][i, j, k, t] = 0.1

            # ·地块种植面积不超过总面积<
            for k in range(num_seasons):
                for t in range(num_years):
                    if np.sum(_particles[n][:, j, k, t]) > 0:
                        # ·地块类型与作物类型的适应性约束←
                        _particles[n][:, j, k, t] *= A_j[j] / \
                            np.sum(_particles[n][:, j, k, t])

            for i in range(num_crops):
                for k in range(num_seasons):
                    for t in range(num_years-1):
                        if _particles[n][i, j, k, t] > 0:
                            _particles[n][i, j, k, t+1] = 0
    return _particles


def pso():
    global particles  # 声明为全局变量
    global g_best, p_best, best_fitness_over_time

    for iter in range(max_iter):
        for n in range(num_particles):
            fitness = objective_function2(particles[n])
            p_best_fitness = objective_function2(p_best[n])
            g_best_fitness = objective_function2(g_best)

            # 更新个体最佳位置
            if fitness > p_best_fitness:
                p_best[n] = particles[n]

            # 更新全局最佳位置
            if fitness > g_best_fitness:
                g_best = particles[n]

            # 更新粒子的速度和位置
            velocities[n] = w * velocities[n] + c1 * np.random.rand() * (p_best[n] - particles[n]) + \
                c2 * np.random.rand() * (g_best - particles[n])
            particles[n] += velocities[n]

            # 非负性约束和确保粒子位置是0.1的倍数
            particles[n] = np.maximum(particles[n], 0)
            particles[n] = ensure_tenth_multiples(particles[n])

        # 再次应用约束
        particles = apply_constraints(particles)

        # 记录每次迭代后的全局最佳适应值
        best_fitness_over_time.append(g_best_fitness)
        print("Iteration " + str(iter) + " Fitness: " + str(g_best_fitness))

    return g_best


# 运行粒子群算法
best_solution = pso()
best_value = objective_function2(best_solution)
# 计算最优解对应的目标兩数值
# 创建文件夹，命名为当前目期+时间
folder_name = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
os.makedirs(folder_name, exist_ok=True)
# 输出x解空间矩阵到多个csv 文件
for t in range(num_years):
    for k in range(num_seasons):
        df = pd.DataFrame(best_solution[:, :, k, t], columns=[f"p_{
                          j+1}" for j in range(num_plots)], index=[f'C {i+1}' for i in range(num_crops)])
        file_name = f"{folder_name}/{2024+t}_Season_{k+1}.csv"
        df.to_csv(file_name)


# 绘制核心图表
plt.figure(figsize=(10, 6))
plt.plot(range(max_iter), best_fitness_over_time,
         label='Best Fitness over terations')
plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.title('Convergence of PSO')
plt.legend()
plt.grid()
plt.savefig(f"{folder_name}/Convergence PsO.png")
plt.show()

print("最优解对应的总收益:", best_value)
print(f"解空间矩阵已输出到文件夹:{folder_name}")

# ·施加约束的函数
