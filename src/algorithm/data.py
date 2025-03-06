import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import os

# Set the style using seaborn
sns.set_style("whitegrid")
colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5']
sns.set_palette(sns.color_palette(colors))

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
    
    # Check if files exist and are not empty
    test_data_path = '../../data/test/test_data.csv'
    results_path = '../../data/analysis/table/results.csv'
    
    # Check test_data.csv
    try:
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"File not found: {test_data_path}")
        if os.path.getsize(test_data_path) == 0:
            raise ValueError(f"File is empty: {test_data_path}")
        test_data = pd.read_csv(test_data_path)
    except Exception as e:
        print(f"Error reading test_data.csv: {str(e)}")
        return
    
    # Check results.csv
    try:
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"File not found: {results_path}")
        if os.path.getsize(results_path) == 0:
            raise ValueError(f"File is empty: {results_path}")
        results_df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading results.csv: {str(e)}")
        return
    
    # Create output directories
    os.makedirs('../../data/analysis/figures', exist_ok=True)
    os.makedirs('../../data/analysis/tables', exist_ok=True)
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    # Calculate actual bandwidth and compute values
    test_data['avg_bandwidth'] = test_data['bandwidth_matrix'].apply(calculate_average_bandwidth)
    test_data['avg_compute'] = test_data['computation_capacity'].apply(lambda x: calculate_average_compute(x))
    
    # Merge datasets
    data = pd.merge(test_data, results_df, on='test_data_id')
    print(f"Original rows: {len(data)}")
    
    # Drop rows with NA values
    data = data.dropna()
    print(f"Rows after dropping NA: {len(data)}")
    
    # Use richer color scheme
    colors = plt.cm.tab10.colors  # Use matplotlib's tab10 color scheme
    
    # Define algorithms to analyze
    algorithms = {
        'multi_func_profit': 'multi_func_profit_info',
        'multi_func_worst_profit': 'multi_func_worst_profit_info',
        'multi_func_min_cost': 'multi_func_min_cost_info',
        'multi_func_max_users': 'multi_func_max_users_info',
        'single_func': 'single_func_info',
        'random_deploy': 'random_deploy_info'
    }
    
    # Extract all information and convert to dataframe
    for alg_name, alg_col in algorithms.items():
        data[f'{alg_name}_cost'] = data[alg_col].apply(lambda x: extract_info(x)['cost'] if isinstance(extract_info(x), dict) else None)
        data[f'{alg_name}_profit'] = data[alg_col].apply(lambda x: extract_info(x)['profit'] if isinstance(extract_info(x), dict) else None)
        data[f'{alg_name}_users'] = data[alg_col].apply(lambda x: extract_info(x)['user_count'] if isinstance(extract_info(x), dict) else None)
    
    # Analyze various factors' impact on performance metrics
    analyze_by_numeric_factor(data, 'profit_per_user', algorithms, 'Per User Profit Value', os.path.join('../../data/analysis/tables', 'profit_per_user_analysis.csv'))
    analyze_by_numeric_factor(data, 'avg_compute', algorithms, 'Average Compute', os.path.join('../../data/analysis/tables', 'avg_compute_analysis.csv'))
    analyze_by_numeric_factor(data, 'avg_bandwidth', algorithms, 'Average Bandwidth', os.path.join('../../data/analysis/tables', 'avg_bandwidth_analysis.csv'))
    analyze_by_categorical_factor(data, 'topology_degree', algorithms, 'Topology Degree', os.path.join('../../data/analysis/tables', 'topology_degree_analysis.csv'))
    analyze_by_categorical_factor(data, 'model_size', algorithms, 'Model Size', os.path.join('../../data/analysis/tables', 'model_size_analysis.csv'))
    analyze_by_categorical_factor(data, 'module_count', algorithms, 'Module Count', os.path.join('../../data/analysis/tables', 'module_count_analysis.csv'))
    
    # Special analysis of per_user_profit's impact on profit, cost, and user count
    plot_per_user_profit_impact(data, algorithms)
    
    print("Analysis completed. Results saved to data/analysis/")

def analyze_by_numeric_factor(data, factor, algorithms, factor_label, output_file):
    """Analyze by numeric factor and generate charts and tables"""
    # Ensure output directories exist
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    # Combined chart (3 in 1)
    plt.figure(figsize=(15, 18))
    factor_values = sorted(data[factor].unique())
    
    # Create result dataframe
    result_df = pd.DataFrame({factor: factor_values})
    
    # Analyze cost
    plt.subplot(3, 1, 1)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            cost_values = subset[f'{alg_name}_cost'].dropna()
            means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        plt.plot(factor_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
        result_df[f'{alg_name}_cost'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(factor_label)
    plt.ylabel('Average Cost')
    plt.title(f'Impact of {factor_label} on Cost')
    plt.legend()
    
    # Analyze profit
    plt.subplot(3, 1, 2)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            profit_values = subset[f'{alg_name}_profit'].dropna()
            means.append(profit_values.mean() if len(profit_values) > 0 else 0)
        plt.plot(factor_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
        result_df[f'{alg_name}_profit'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(factor_label)
    plt.ylabel('Average Profit')
    plt.title(f'Impact of {factor_label} on Profit')
    plt.legend()
    
    # Analyze user count
    plt.subplot(3, 1, 3)
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            user_values = subset[f'{alg_name}_users'].dropna()
            means.append(user_values.mean() if len(user_values) > 0 else 0)
        plt.plot(factor_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
        result_df[f'{alg_name}_users'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(factor_label)
    plt.ylabel('Average User Count')
    plt.title(f'Impact of {factor_label} on User Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figures', f'{factor}_analysis.png'), dpi=300)
    plt.close()
    
    # Separate charts for each metric
    # 1. Cost analysis
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            cost_values = subset[f'{alg_name}_cost'].dropna()
            means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        plt.plot(factor_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(factor_label)
    plt.ylabel('Average Cost')
    plt.title(f'Impact of {factor_label} on Cost')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_cost_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Profit analysis
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            profit_values = subset[f'{alg_name}_profit'].dropna()
            means.append(profit_values.mean() if len(profit_values) > 0 else 0)
        plt.plot(factor_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(factor_label)
    plt.ylabel('Average Profit')
    plt.title(f'Impact of {factor_label} on Profit')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_profit_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. User count analysis
    plt.figure(figsize=(12, 6))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in factor_values:
            subset = data[data[factor] == val]
            user_values = subset[f'{alg_name}_users'].dropna()
            means.append(user_values.mean() if len(user_values) > 0 else 0)
        plt.plot(factor_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel(factor_label)
    plt.ylabel('Average User Count')
    plt.title(f'Impact of {factor_label} on User Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_user_count_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save table data
    result_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

def analyze_by_categorical_factor(data, factor, algorithms, factor_label, output_file):
    """Analyze by categorical factor and generate charts and tables"""
    # Ensure output directories exist
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    # Combined chart (3 in 1)
    plt.figure(figsize=(15, 18))
    factor_values = sorted(data[factor].unique())
    
    # Create result dataframe
    result_df = pd.DataFrame({factor: factor_values})
    
    # Analyze cost
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
    
    # Analyze profit
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
    
    # Analyze user count
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
    
    # Separate charts for each metric
    # 1. Cost analysis
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
    
    # 2. Profit analysis
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
    
    # 3. User count analysis
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
    plt.savefig(os.path.join('../../data/analysis/figure', f'{factor}_user_count_continuous.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save table data
    result_df.to_csv(output_file, index=False)
    print(f"Analysis results saved to {output_file}")

def plot_per_user_profit_impact(data, algorithms):
    """Special analysis of the impact of per_user_profit on profit, cost, and user count"""
    # Ensure output directories exist
    os.makedirs('../../data/analysis/figure', exist_ok=True)
    
    profit_values = sorted(data['profit_per_user'].unique())
    
    # Create a dataframe to save the results
    result_df = pd.DataFrame({'profit_per_user': profit_values})
    
    # Draw three separate charts, one for each metric
    # 1. Analysis of impact on total profit
    plt.figure(figsize=(12, 7))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in profit_values:
            subset = data[data['profit_per_user'] == val]
            profit_values_subset = subset[f'{alg_name}_profit'].dropna()
            means.append(profit_values_subset.mean() if len(profit_values_subset) > 0 else 0)
        plt.plot(profit_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
        result_df[f'{alg_name}_profit'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Per User Profit Value')
    plt.ylabel('Average Total Profit')
    plt.title('Impact of Per User Profit on Total Profit')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../data/analysis/figures/per_user_profit_on_total_profit.png', dpi=300)
    plt.savefig('../../data/analysis/figure/per_user_profit_profit_continuous.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Analysis of impact on cost
    plt.figure(figsize=(12, 7))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in profit_values:
            subset = data[data['profit_per_user'] == val]
            cost_values = subset[f'{alg_name}_cost'].dropna()
            means.append(cost_values.mean() if len(cost_values) > 0 else 0)
        plt.plot(profit_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
        result_df[f'{alg_name}_cost'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Per User Profit Value')
    plt.ylabel('Average Cost')
    plt.title('Impact of Per User Profit on Total Cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../data/analysis/figures/per_user_profit_on_cost.png', dpi=300)
    plt.savefig('../../data/analysis/figure/per_user_profit_cost_continuous.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Analysis of impact on user count
    plt.figure(figsize=(12, 7))
    for i, (alg_name, _) in enumerate(algorithms.items()):
        means = []
        for val in profit_values:
            subset = data[data['profit_per_user'] == val]
            user_values = subset[f'{alg_name}_users'].dropna()
            means.append(user_values.mean() if len(user_values) > 0 else 0)
        plt.plot(profit_values, means, 'o-', label=alg_name, color=plt.cm.tab10.colors[i], linewidth=2)
        result_df[f'{alg_name}_users'] = means
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Per User Profit Value')
    plt.ylabel('Average User Count')
    plt.title('Impact of Per User Profit on User Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../../data/analysis/figures/per_user_profit_on_users.png', dpi=300)
    plt.savefig('../../data/analysis/figure/per_user_profit_user_count_continuous.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save table data
    result_df.to_csv('../../data/analysis/tables/per_user_profit_impact.csv', index=False)
    print("Per user profit impact analysis results saved")

if __name__ == "__main__":
    analyze_data()