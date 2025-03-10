import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os
import re
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置绘图风格
sns.set_style("whitegrid")
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
sns.set_palette(sns.color_palette(colors))

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    font_prop = FontProperties(family='sans-serif', size=12)
except:
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    font_prop = FontProperties(family='sans-serif', size=12)

def extract_info(info_str):
    try:
        if isinstance(info_str, str):
            data = ast.literal_eval(info_str)
            if isinstance(data, list):
                return {
                    'cost': data[0],
                    'profit': data[1],
                    'user_count': data[2]
                }
        return info_str
    except:
        return info_str

def calculate_average_bandwidth(bandwidth_matrix):
    """计算带宽矩阵的平均非零带宽值"""
    bandwidth_matrix = ast.literal_eval(bandwidth_matrix)
    non_zero_values = [val for row in bandwidth_matrix for val in row if val > 0]
    return np.mean(non_zero_values) if non_zero_values else 0

def calculate_average_compute(computation_capacity):
    """计算平均计算能力"""
    capacity_list = ast.literal_eval(computation_capacity)
    return np.mean(capacity_list)

def analyze_data():
    print("Analyzing Results...")
    
    # 检查文件是否存在且不为空
    test_data_path = '../../data/test/test_data.csv'
    results_path = '../../data/analysis/table/results.csv'
    
    try:
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"File not found: {test_data_path}")
        if os.path.getsize(test_data_path) == 0:
            raise ValueError(f"File is empty: {test_data_path}")
        test_data = pd.read_csv(test_data_path)
    except Exception as e:
        print(f"Error reading test_data.csv: {str(e)}")
        return
    
    try:
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"File not found: {results_path}")
        if os.path.getsize(results_path) == 0:
            raise ValueError(f"File is empty: {results_path}")
        results_df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading results.csv: {str(e)}")
        return
    
    # 创建输出目录
    os.makedirs('../../data/analysis/figures', exist_ok=True)
    os.makedirs('../../data/analysis/tables', exist_ok=True)
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    # 计算平均带宽和计算能力
    test_data['avg_bandwidth'] = test_data['bandwidth_matrix'].apply(calculate_average_bandwidth)
    test_data['avg_compute'] = test_data['computation_capacity'].apply(lambda x: calculate_average_compute(x))
    
    # 合并数据集
    data = pd.merge(test_data, results_df, on='test_data_id')
    print(f"Original rows: {len(data)}")
    
    # 删除含 NA 的行
    data = data.dropna()
    print(f"Rows after dropping NA: {len(data)}")
    
    # 使用更丰富的颜色方案
    colors = plt.cm.tab10.colors
    
    # 定义要分析的算法
    algorithms = {
        'multi_func_profit': 'multi_func_profit_info',
        'multi_func_worst_profit': 'multi_func_worst_profit_info',
        'multi_func_min_cost': 'multi_func_min_cost_info',
        'multi_func_max_users': 'multi_func_max_users_info',
        'single_func': 'single_func_info',
        'random_deploy': 'random_deploy_info'
    }
    
    # 提取所有信息并转换为数据框
    for alg_name, alg_col in algorithms.items():
        data[f'{alg_name}_cost'] = data[alg_col].apply(lambda x: extract_info(x)['cost'] if isinstance(extract_info(x), dict) else None)
        data[f'{alg_name}_profit'] = data[alg_col].apply(lambda x: extract_info(x)['profit'] if isinstance(extract_info(x), dict) else None)
        data[f'{alg_name}_users'] = data[alg_col].apply(lambda x: extract_info(x)['user_count'] if isinstance(extract_info(x), dict) else None)
    
    # 分析各种因素对性能指标的影响
    analyze_by_numeric_factor(data, 'profit_per_user', algorithms, 'Per User Profit Value', os.path.join('../../data/analysis/tables', 'profit_per_user_analysis.csv'))
    analyze_by_numeric_factor(data, 'avg_compute', algorithms, 'Average Compute', os.path.join('../../data/analysis/tables', 'avg_compute_analysis.csv'))
    analyze_by_numeric_factor(data, 'avg_bandwidth', algorithms, 'Average Bandwidth', os.path.join('../../data/analysis/tables', 'avg_bandwidth_analysis.csv'))
    analyze_by_categorical_factor(data, 'topology_degree', algorithms, 'Topology Degree', os.path.join('../../data/analysis/tables', 'topology_degree_analysis.csv'))
    analyze_by_categorical_factor(data, 'model_size', algorithms, 'Model Size', os.path.join('../../data/analysis/tables', 'model_size_analysis.csv'))
    analyze_by_categorical_factor(data, 'module_count', algorithms, 'Module Count', os.path.join('../../data/analysis/tables', 'module_count_analysis.csv'))
    
    # 特殊分析：per_user_profit 对利润、成本和用户数的影响
    plot_per_user_profit_impact(data, algorithms)
    
    print("Analysis completed. Results saved to data/analysis/")

def analyze_by_numeric_factor(data, factor, algorithms, factor_label, output_file):
    """按数值因子分析并生成图表和表格"""
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    # 组图（3合1）
    plt.figure(figsize=(15, 18))
    factor_values = sorted(data[factor].unique())
    
    # 将数值分为5个区间
    min_val = min(factor_values)
    max_val = max(factor_values)
    bin_width = (max_val - min_val) / 5
    bins = [min_val + i * bin_width for i in range(6)]
    
    # 创建结果数据框，x 轴标签为区间
    result_df = pd.DataFrame({factor: [f'[{bins[i]:.2f}, {bins[i+1]:.2f})' for i in range(5)]})
    
    # 分析成本
    plt.subplot(3, 1, 1)
    bar_width = 0.8 / len(algorithms)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        bin_means = []
        for j in range(5):
            bin_data = data[(data[factor] >= bins[j]) & (data[factor] < bins[j+1])]
            cost_values = bin_data[f'{alg_name}_cost'].dropna()
            bin_means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        
        x = np.arange(len(result_df[factor]))
        plt.bar(x + i * bar_width, bin_means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
        result_df[f'{alg_name}_cost'] = bin_means
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(f'{factor_label} (Interval)')
    plt.ylabel('Average Cost')
    plt.title(f'Impact of {factor_label} on Cost')
    plt.xticks(np.arange(len(result_df[factor])) + bar_width * len(algorithms) / 2 - bar_width/2, result_df[factor])
    plt.legend()
    
    # 分析利润
    plt.subplot(3, 1, 2)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        bin_means = []
        for j in range(5):
            bin_data = data[(data[factor] >= bins[j]) & (data[factor] < bins[j+1])]
            profit_values = bin_data[f'{alg_name}_profit'].dropna()
            bin_means.append(profit_values.mean() if len(profit_values) > 0 else 0)
        
        x = np.arange(len(result_df[factor]))
        plt.bar(x + i * bar_width, bin_means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
        result_df[f'{alg_name}_profit'] = bin_means
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(f'{factor_label} (Interval)')
    plt.ylabel('Average Profit')
    plt.title(f'Impact of {factor_label} on Profit')
    plt.xticks(np.arange(len(result_df[factor])) + bar_width * len(algorithms) / 2 - bar_width/2, result_df[factor])
    plt.legend()
    
    # 分析用户数
    plt.subplot(3, 1, 3)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        bin_means = []
        for j in range(5):
            bin_data = data[(data[factor] >= bins[j]) & (data[factor] < bins[j+1])]
            user_values = bin_data[f'{alg_name}_users'].dropna()
            bin_means.append(user_values.mean() if len(user_values) > 0 else 0)
        
        x = np.arange(len(result_df[factor]))
        plt.bar(x + i * bar_width, bin_means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
        result_df[f'{alg_name}_users'] = bin_means
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(f'{factor_label} (Interval)')
    plt.ylabel('Average User Count')
    plt.title(f'Impact of {factor_label} on User Count')
    plt.xticks(np.arange(len(result_df[factor])) + bar_width * len(algorithms) / 2 - bar_width/2, result_df[factor])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'../../data/analysis/figures/{factor}_analysis.png')
    plt.close()
    
    # 单独图表
    # 1. 成本分析
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        bin_means = []
        for j in range(5):
            bin_data = data[(data[factor] >= bins[j]) & (data[factor] < bins[j+1])]
            cost_values = bin_data[f'{alg_name}_cost'].dropna()
            bin_means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        x = np.arange(len(result_df[factor]))
        plt.bar(x + i * bar_width, bin_means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(f'{factor_label} (Interval)')
    plt.ylabel('Average Cost')
    plt.title(f'Impact of {factor_label} on Cost')
    plt.xticks(np.arange(len(result_df[factor])) + bar_width * len(algorithms) / 2 - bar_width/2, result_df[factor])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../../data/analysis/figure/{factor}_cost_continuous.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. 利润分析
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        bin_means = []
        for j in range(5):
            bin_data = data[(data[factor] >= bins[j]) & (data[factor] < bins[j+1])]
            profit_values = bin_data[f'{alg_name}_profit'].dropna()
            bin_means.append(profit_values.mean() if len(profit_values) > 0 else 0)
        x = np.arange(len(result_df[factor]))
        plt.bar(x + i * bar_width, bin_means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(f'{factor_label} (Interval)')
    plt.ylabel('Average Profit')
    plt.title(f'Impact of {factor_label} on Profit')
    plt.xticks(np.arange(len(result_df[factor])) + bar_width * len(algorithms) / 2 - bar_width/2, result_df[factor])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../../data/analysis/figure/{factor}_profit_continuous.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. 用户数分析
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        bin_means = []
        for j in range(5):
            bin_data = data[(data[factor] >= bins[j]) & (data[factor] < bins[j+1])]
            user_values = bin_data[f'{alg_name}_users'].dropna()
            bin_means.append(user_values.mean() if len(user_values) > 0 else 0)
        x = np.arange(len(result_df[factor]))
        plt.bar(x + i * bar_width, bin_means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(f'{factor_label} (Interval)')
    plt.ylabel('Average User Count')
    plt.title(f'Impact of {factor_label} on User Count')
    plt.xticks(np.arange(len(result_df[factor])) + bar_width * len(algorithms) / 2 - bar_width/2, result_df[factor])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'../../data/analysis/figure/{factor}_users_continuous.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存表格数据
    result_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

def analyze_by_categorical_factor(data, factor, algorithms, factor_label, output_file):
    """按分类因子分析并生成图表和表格"""
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    # 组图（3合1）
    plt.figure(figsize=(15, 18))
    factor_values = sorted(data[factor].unique())  # 确保升序排列
    
    # 创建结果数据框
    result_df = pd.DataFrame({factor: factor_values})
    
    # 分析成本
    plt.subplot(3, 1, 1)
    bar_width = 0.8 / len(algorithms)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            cost_values = subset[f'{alg_name}_cost'].dropna()
            means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        
        x = np.arange(len(factor_values))
        plt.bar(x + i * bar_width, means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
        result_df[f'{alg_name}_cost'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(factor_label)
    plt.ylabel('Average Cost')
    plt.title(f'Impact of {factor_label} on Cost')
    plt.xticks(np.arange(len(factor_values)) + bar_width * len(algorithms) / 2 - bar_width/2, factor_values)
    plt.legend()
    
    # 分析利润
    plt.subplot(3, 1, 2)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            profit_values = subset[f'{alg_name}_profit'].dropna()
            means.append(profit_values.mean() if len(profit_values) > 0 else 0)
        
        x = np.arange(len(factor_values))
        plt.bar(x + i * bar_width, means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
        result_df[f'{alg_name}_profit'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(factor_label)
    plt.ylabel('Average Profit')
    plt.title(f'Impact of {factor_label} on Profit')
    plt.xticks(np.arange(len(factor_values)) + bar_width * len(algorithms) / 2 - bar_width/2, factor_values)
    plt.legend()
    
    # 分析用户数
    plt.subplot(3, 1, 3)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            user_values = subset[f'{alg_name}_users'].dropna()
            means.append(user_values.mean() if len(user_values) > 0 else 0)
        
        x = np.arange(len(factor_values))
        plt.bar(x + i * bar_width, means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
        result_df[f'{alg_name}_users'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(factor_label)
    plt.ylabel('Average User Count')
    plt.title(f'Impact of {factor_label} on User Count')
    plt.xticks(np.arange(len(factor_values)) + bar_width * len(algorithms) / 2 - bar_width/2, factor_values)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figures', f'{factor}_analysis.png'), dpi=300)
    plt.close()
    
    # 单独图表
    # 1. 成本分析
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            cost_values = subset[f'{alg_name}_cost'].dropna()
            means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        x = np.arange(len(factor_values))
        plt.bar(x + i * bar_width, means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(factor_label)
    plt.ylabel('Average Cost')
    plt.title(f'Impact of {factor_label} on Cost')
    plt.xticks(np.arange(len(factor_values)) + bar_width * len(algorithms) / 2 - bar_width/2, factor_values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_cost_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. 利润分析
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            profit_values = subset[f'{alg_name}_profit'].dropna()
            means.append(profit_values.mean() if len(profit_values) > 0 else 0)
        x = np.arange(len(factor_values))
        plt.bar(x + i * bar_width, means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(factor_label)
    plt.ylabel('Average Profit')
    plt.title(f'Impact of {factor_label} on Profit')
    plt.xticks(np.arange(len(factor_values)) + bar_width * len(algorithms) / 2 - bar_width/2, factor_values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_profit_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. 用户数分析
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            user_values = subset[f'{alg_name}_users'].dropna()
            means.append(user_values.mean() if len(user_values) > 0 else 0)
        x = np.arange(len(factor_values))
        plt.bar(x + i * bar_width, means, width=bar_width, label=alg_name, color=plt.cm.tab10.colors[i])
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel(factor_label)
    plt.ylabel('Average User Count')
    plt.title(f'Impact of {factor_label} on User Count')
    plt.xticks(np.arange(len(factor_values)) + bar_width * len(algorithms) / 2 - bar_width/2, factor_values)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_users_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 保存表格数据
    result_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

def plot_per_user_profit_impact(data, algorithms):
    """特殊分析：分析 per_user_profit 对利润、成本和用户数的影响，仅保留数据处理和表格保存"""
    # 获取唯一的 profit_per_user 值并排序
    profit_values = sorted(data['profit_per_user'].unique())
    
    # 创建结果数据框
    result_df = pd.DataFrame({'profit_per_user': profit_values})
    
    # 计算每个算法在不同 profit_per_user 下的平均值
    for alg_name, _ in algorithms.items():
        profit_means = []
        cost_means = []
        user_means = []
        for val in profit_values:
            subset = data[data['profit_per_user'] == val]
            profit_values_subset = subset[f'{alg_name}_profit'].dropna()
            cost_values = subset[f'{alg_name}_cost'].dropna()
            user_values = subset[f'{alg_name}_users'].dropna()
            profit_means.append(profit_values_subset.mean() if len(profit_values_subset) > 0 else 0)
            cost_means.append(cost_values.mean() if len(cost_values) > 0 else 0)
            user_means.append(user_values.mean() if len(user_values) > 0 else 0)
        result_df[f'{alg_name}_profit'] = profit_means
        result_df[f'{alg_name}_cost'] = cost_means
        result_df[f'{alg_name}_users'] = user_means
    
    # 保存结果到 CSV 文件
    result_df.to_csv('../../data/analysis/tables/per_user_profit_impact.csv', index=False)
    print("Per user profit impact analysis results saved")

if __name__ == "__main__":
    analyze_data()