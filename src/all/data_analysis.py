import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def extract_info(info_str):
    if info_str is None or not isinstance(info_str, str) or info_str == 'None':
        return None
    try:
        info = eval(info_str)
        return {'cost': info[0], 'profit': info[1], 'U_max': info[2], 'plan': info[3]}
    except (SyntaxError, ValueError, TypeError) as e:
        print(f"警告: 无法解析 info_str: {info_str}, 错误: {e}")
        return None


def analyze_data():
    # 读取结果和输入数据
    results = pd.read_csv('results.csv')
    input_data = pd.read_csv('../test/enhanced_connectivity_data.csv')  # 调整路径根据实际情况
    algorithms = ['multi_func_profit', 'compute_first', 'memory_first', 'single_func', 'random_deploy']

    # 合并数据，按 test_data_id 对齐
    merged_data = pd.merge(results, input_data[['test_data_id', 'node_count', 'function_count']],
                           on='test_data_id', how='left')

    # 打印调试信息
    print("results.csv 列名:", merged_data.columns.tolist())
    print("results.csv 前几行完整数据:\n", merged_data.head().to_string())

    # 1. Profit Comparison Across All Algorithms
    profits = {algo: [] for algo in algorithms}
    for _, row in merged_data.iterrows():
        for algo in algorithms:
            info = extract_info(row[f'{algo}_info'])
            profits[algo].append(info['profit'] if info else np.nan)

    df_profits = pd.DataFrame(profits, index=merged_data['test_data_id'])
    df_profits.plot(kind='bar', figsize=(12, 6))
    plt.title('Profit Comparison Across All Algorithms')
    plt.xlabel('Test Case ID')
    plt.ylabel('Profit ($)')
    plt.legend(title='Algorithms')
    plt.tight_layout()
    plt.savefig('profit_comparison.png')
    plt.close()

    # 2. Multi Func vs Single Func Profit Comparison
    multi_profits = []
    single_profits = []
    same_profit_count = 0
    for _, row in merged_data.iterrows():
        multi_info = extract_info(row['multi_func_profit_info'])
        single_info = extract_info(row['single_func_info'])
        if multi_info and single_info:
            multi_profit = multi_info['profit']
            single_profit = single_info['profit']
            multi_profits.append(multi_profit)
            single_profits.append(single_profit)
            if abs(multi_profit - single_profit) < 1e-2:
                same_profit_count += 1

    plt.figure(figsize=(10, 5))
    plt.plot(multi_profits, label='Multi Func Profit', marker='o')
    plt.plot(single_profits, label='Single Func Profit', marker='x')
    plt.title('Multi Func vs Single Func Profit')
    plt.xlabel('Test Case Index')
    plt.ylabel('Profit ($)')
    plt.legend()
    plt.savefig('multi_vs_single_profit.png')
    plt.close()

    total_cases = len(multi_profits)
    same_profit_pct = (same_profit_count / total_cases * 100) if total_cases > 0 else 0

    # 3. Multi Func vs Single Func Time Comparison
    multi_times = merged_data['multi_func_time'].dropna().tolist()
    single_times = merged_data['single_func_time'].dropna().tolist()
    min_len = min(len(multi_times), len(single_times))
    df_times = pd.DataFrame({
        'Multi Func Time': multi_times[:min_len],
        'Single Func Time': single_times[:min_len]
    }, index=merged_data['test_data_id'][:min_len])
    df_times.plot(kind='bar', figsize=(12, 6))
    plt.title('Multi Func vs Single Func Execution Time')
    plt.xlabel('Test Case ID')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('time_comparison.png')
    plt.close()

    time_savings = [
        (m - s) / m * 100 for m, s in zip(multi_times, single_times)
        if m > 0 and s is not None
    ]
    avg_time_saving = np.mean(time_savings) if time_savings else 0

    # 4. U_max Comparison
    u_maxes = {algo: [] for algo in algorithms}
    for _, row in merged_data.iterrows():
        for algo in algorithms:
            info = extract_info(row[f'{algo}_info'])
            u_maxes[algo].append(info['U_max'] if info else np.nan)

    df_umax = pd.DataFrame(u_maxes, index=merged_data['test_data_id'])
    df_umax.plot(kind='bar', figsize=(12, 6))
    plt.title('U_max Comparison Across Algorithms')
    plt.xlabel('Test Case ID')
    plt.ylabel('Maximum Users (U_max)')
    plt.legend(title='Algorithms')
    plt.tight_layout()
    plt.savefig('umax_comparison.png')
    plt.close()

    # 5. Node Count vs Profit (Line Plot)
    node_counts = sorted(merged_data['node_count'].unique())
    node_profit_table = pd.DataFrame(index=node_counts, columns=algorithms)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        node_groups = merged_data.groupby('node_count')[f'{algo}_info'].apply(
            lambda x: np.nanmean([extract_info(i)['profit'] if extract_info(i) else np.nan for i in x])
            if any(extract_info(i) is not None for i in x) else np.nan
        )
        node_profit_table[algo] = [node_groups.get(nc, np.nan) for nc in node_counts]
        plt.plot(node_counts, node_profit_table[algo], label=algo, marker='o')
    plt.title('Profit vs Node Count Across Algorithms')
    plt.xlabel('Node Count')
    plt.ylabel('Average Profit ($)')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('profit_vs_node_count_line.png')
    plt.close()

    # 保存节点数量利润表为 CSV
    node_profit_table.to_csv('node_count_profit_table.csv', index_label='Node Count')

    # 6. Function Count vs Profit (Line Plot)
    function_counts = sorted(merged_data['function_count'].unique())
    func_profit_table = pd.DataFrame(index=function_counts, columns=algorithms)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        func_groups = merged_data.groupby('function_count')[f'{algo}_info'].apply(
            lambda x: np.nanmean([extract_info(i)['profit'] if extract_info(i) else np.nan for i in x])
            if any(extract_info(i) is not None for i in x) else np.nan
        )
        func_profit_table[algo] = [func_groups.get(fc, np.nan) for fc in function_counts]
        plt.plot(function_counts, func_profit_table[algo], label=algo, marker='o')
    plt.title('Profit vs Function Count Across Algorithms')
    plt.xlabel('Function Count')
    plt.ylabel('Average Profit ($)')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('profit_vs_function_count_line.png')
    plt.close()

    # 保存功能数量利润表为 CSV
    func_profit_table.to_csv('function_count_profit_table.csv', index_label='Function Count')

    # 7. Node Count vs U_max (Line Plot)
    node_umax_table = pd.DataFrame(index=algorithms, columns=node_counts)  # 转置表格
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        node_groups = merged_data.groupby('node_count')[f'{algo}_info'].apply(
            lambda x: np.nanmean([extract_info(i)['U_max'] if extract_info(i) else np.nan for i in x])
            if any(extract_info(i) is not None for i in x) else np.nan
        )
        node_umax_table.loc[algo] = [node_groups.get(nc, np.nan) for nc in node_counts]
        plt.plot(node_counts, node_umax_table.loc[algo], label=algo, marker='o')
    plt.title('U_max vs Node Count Across Algorithms')
    plt.xlabel('Node Count')
    plt.ylabel('Average U_max')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('umax_vs_node_count_line.png')
    plt.close()

    # 保存节点数量 U_max 表为 CSV
    node_umax_table.to_csv('node_count_umax_table.csv', index_label='Algorithm')

    # 8. Function Count vs U_max (Line Plot)
    func_umax_table = pd.DataFrame(index=algorithms, columns=function_counts)  # 转置表格
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        func_groups = merged_data.groupby('function_count')[f'{algo}_info'].apply(
            lambda x: np.nanmean([extract_info(i)['U_max'] if extract_info(i) else np.nan for i in x])
            if any(extract_info(i) is not None for i in x) else np.nan
        )
        func_umax_table.loc[algo] = [func_groups.get(fc, np.nan) for fc in function_counts]
        plt.plot(function_counts, func_umax_table.loc[algo], label=algo, marker='o')
    plt.title('U_max vs Function Count Across Algorithms')
    plt.xlabel('Function Count')
    plt.ylabel('Average U_max')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('umax_vs_function_count_line.png')
    plt.close()

    # 保存功能数量 U_max 表为 CSV
    func_umax_table.to_csv('function_count_umax_table.csv', index_label='Algorithm')

    # Generate Summary Report
    with open('analysis_summary.txt', 'w') as f:
        f.write("=== Deployment Optimization Analysis Summary ===\n\n")

        # 1. Profit Comparison Across All Algorithms
        f.write("1. Profit Comparison Across All Algorithms\n")
        f.write("   - See 'profit_comparison.png' for visual comparison.\n")
        f.write(
            f"   - Average Profits: {', '.join(f'{algo}: {np.nanmean(profits[algo]):.2f}$' for algo in algorithms)}\n\n")

        # 2. Multi Func vs Single Func Profit Comparison
        f.write("2. Multi Func vs Single Func Profit Comparison\n")
        f.write(f"   - Cases where Single Func equals Multi Func Profit: {same_profit_count} out of {total_cases}\n")
        f.write(f"   - Percentage: {same_profit_pct:.2f}%\n")
        f.write("   - See 'multi_vs_single_profit.png' for details.\n\n")

        # 3. Multi Func vs Single Func Time Comparison
        f.write("3. Multi Func vs Single Func Time Comparison\n")
        f.write(f"   - Average Time Saving by Single Func: {avg_time_saving:.2f}%\n")
        f.write("   - See 'time_comparison.png' for details.\n\n")

        # 4. U_max Comparison
        f.write("4. U_max Comparison\n")
        f.write("   - See 'umax_comparison.png' for visual comparison.\n")
        f.write(
            f"   - Average U_max: {', '.join(f'{algo}: {np.nanmean(u_maxes[algo]):.2f}' for algo in algorithms)}\n\n")

        # 5. Profit vs Node Count Analysis
        f.write("5. Profit vs Node Count Analysis\n")
        f.write("   - See 'profit_vs_node_count_line.png' for line plot.\n")
        f.write("   - Average Profit by Node Count Table saved to 'node_count_profit_table.csv'\n")
        f.write("   - Average Profit by Node Count:\n")
        f.write(node_profit_table.to_string() + "\n\n")

        # 6. Profit vs Function Count Analysis
        f.write("6. Profit vs Function Count Analysis\n")
        f.write("   - See 'profit_vs_function_count_line.png' for line plot.\n")
        f.write("   - Average Profit by Function Count Table saved to 'function_count_profit_table.csv'\n")
        f.write("   - Average Profit by Function Count:\n")
        f.write(func_profit_table.to_string() + "\n\n")

        # 7. U_max vs Node Count Analysis
        f.write("7. U_max vs Node Count Analysis\n")
        f.write("   - See 'umax_vs_node_count_line.png' for line plot.\n")
        f.write("   - Average U_max by Node Count Table saved to 'node_count_umax_table.csv'\n")
        f.write("   - Average U_max by Node Count:\n")
        f.write(node_umax_table.to_string() + "\n\n")

        # 8. U_max vs Function Count Analysis
        f.write("8. U_max vs Function Count Analysis\n")
        f.write("   - See 'umax_vs_function_count_line.png' for line plot.\n")
        f.write("   - Average U_max by Function Count Table saved to 'function_count_umax_table.csv'\n")
        f.write("   - Average U_max by Function Count:\n")
        f.write(func_umax_table.to_string() + "\n")

    print(f"Analysis complete. See 'analysis_summary.txt', CSV tables, and generated plots.")


if __name__ == "__main__":
    analyze_data()