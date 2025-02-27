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
    results = pd.read_csv('../../data/analysis/table/results.csv')
    input_data = pd.read_csv('../test/enhanced_connectivity_data.csv')
    algorithms = ['multi_func_profit', 'compute_first', 'memory_first', 'single_func', 'random_deploy', 'min_profit']

    # 合并数据，按 test_data_id 对齐
    merged_data = pd.merge(results, input_data[['test_data_id', 'node_count', 'function_count']],
                           on='test_data_id', how='left')

    # 打印调试信息
    print("results.csv 列名:", merged_data.columns.tolist())
    print("results.csv 前几行完整数据:\n", merged_data.head().to_string())

    # 定义需要检查的列
    info_columns = [f'{algo}_info' for algo in algorithms[:-1]] + ['multi_func_worst_profit_info']

    # 筛选出所有算法都有数据的行（即所有 info 列都不为 None）
    valid_rows = merged_data.dropna(subset=info_columns)

    # 打印筛选前后的数据行数
    print(f"原始数据行数: {len(merged_data)}, 筛选后数据行数: {len(valid_rows)}")

    # 基于筛选后的数据进行分析
    merged_data = valid_rows

    # 1. Profit Comparison Across All Algorithms (条形图)
    profits = {algo: [] for algo in algorithms}
    for _, row in merged_data.iterrows():
        for algo in algorithms:
            if algo == 'min_profit':
                info = extract_info(row['multi_func_worst_profit_info'])
            else:
                info = extract_info(row[f'{algo}_info'])
            profits[algo].append(info['profit'] if info else np.nan)

    df_profits = pd.DataFrame(profits, index=merged_data['test_data_id'])
    df_profits.plot(kind='bar', figsize=(12, 6))
    plt.title('Profit Comparison Across All Algorithms (Complete Cases Only)')
    plt.xlabel('Test Case ID')
    plt.ylabel('Profit ($)')
    plt.legend(title='Algorithms')
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/profit_comparison_complete_cases.png')
    plt.close()

    # 2. Multi Func vs Single Func Profit Comparison (折线图)
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
    plt.title('Multi Func vs Single Func Profit (Complete Cases Only)')
    plt.xlabel('Test Case Index')
    plt.ylabel('Profit ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/multi_vs_single_profit_complete_cases.png')
    plt.close()

    total_cases = len(multi_profits)
    same_profit_pct = (same_profit_count / total_cases * 100) if total_cases > 0 else 0

    # 3. Multi Func vs Single Func Time Comparison (条形图)
    multi_times = merged_data['multi_func_time'].dropna().tolist()
    single_times = merged_data['single_func_time'].dropna().tolist()
    min_len = min(len(multi_times), len(single_times))
    df_times = pd.DataFrame({
        'Multi Func Time': multi_times[:min_len],
        'Single Func Time': single_times[:min_len]
    }, index=merged_data['test_data_id'][:min_len])
    df_times.plot(kind='bar', figsize=(12, 6))
    plt.title('Multi Func vs Single Func Execution Time (Complete Cases Only)')
    plt.xlabel('Test Case ID')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/time_comparison_complete_cases.png')
    plt.close()

    time_savings = [
        (m - s) / m * 100 for m, s in zip(multi_times, single_times)
        if m > 0 and s is not None
    ]
    avg_time_saving = np.mean(time_savings) if time_savings else 0

    # 4. U_max Comparison (条形图)
    u_maxes = {algo: [] for algo in algorithms}
    for _, row in merged_data.iterrows():
        for algo in algorithms:
            if algo == 'min_profit':
                info = extract_info(row['multi_func_worst_profit_info'])
            else:
                info = extract_info(row[f'{algo}_info'])
            u_maxes[algo].append(info['U_max'] if info else np.nan)

    df_umax = pd.DataFrame(u_maxes, index=merged_data['test_data_id'])
    df_umax.plot(kind='bar', figsize=(12, 6))
    plt.title('U_max Comparison Across All Algorithms (Complete Cases Only)')
    plt.xlabel('Test Case ID')
    plt.ylabel('Maximum Users (U_max)')
    plt.legend(title='Algorithms')
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/umax_comparison_complete_cases.png')
    plt.close()

    # 5. Node Count vs Profit (折线图)
    node_counts = sorted(merged_data['node_count'].unique())
    node_profit_table = pd.DataFrame(index=node_counts, columns=algorithms)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        if algo == 'min_profit':
            node_groups = merged_data.groupby('node_count')['multi_func_worst_profit_info'].apply(
                lambda x: np.mean([extract_info(i)['profit'] for i in x if extract_info(i) is not None])
            )
        else:
            node_groups = merged_data.groupby('node_count')[f'{algo}_info'].apply(
                lambda x: np.mean([extract_info(i)['profit'] for i in x if extract_info(i) is not None])
            )
        node_profit_table[algo] = [node_groups.get(nc, np.nan) for nc in node_counts]
        plt.plot(node_counts, node_profit_table[algo], label=algo, marker='o')
    plt.title('Profit vs Node Count Across Algorithms (Complete Cases Only)')
    plt.xlabel('Node Count')
    plt.ylabel('Average Profit ($)')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/profit_vs_node_count_line_complete_cases.png')
    plt.close()

    node_profit_table.to_csv('../../data/analysis/table/node_count_profit_table_complete_cases.csv', index_label='Node Count')

    # 6. Function Count vs Profit (折线图)
    function_counts = sorted(merged_data['function_count'].unique())
    func_profit_table = pd.DataFrame(index=function_counts, columns=algorithms)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        if algo == 'min_profit':
            func_groups = merged_data.groupby('function_count')['multi_func_worst_profit_info'].apply(
                lambda x: np.mean([extract_info(i)['profit'] for i in x if extract_info(i) is not None])
            )
        else:
            func_groups = merged_data.groupby('function_count')[f'{algo}_info'].apply(
                lambda x: np.mean([extract_info(i)['profit'] for i in x if extract_info(i) is not None])
            )
        func_profit_table[algo] = [func_groups.get(fc, np.nan) for fc in function_counts]
        plt.plot(function_counts, func_profit_table[algo], label=algo, marker='o')
    plt.title('Profit vs Function Count Across Algorithms (Complete Cases Only)')
    plt.xlabel('Function Count')
    plt.ylabel('Average Profit ($)')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/profit_vs_function_count_line_complete_cases.png')
    plt.close()

    func_profit_table.to_csv('../../data/analysis/table/function_count_profit_table_complete_cases.csv', index_label='Function Count')

    # 7. Node Count vs U_max (折线图)
    node_umax_table = pd.DataFrame(index=algorithms, columns=node_counts)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        if algo == 'min_profit':
            node_groups = merged_data.groupby('node_count')['multi_func_worst_profit_info'].apply(
                lambda x: np.mean([extract_info(i)['U_max'] for i in x if extract_info(i) is not None])
            )
        else:
            node_groups = merged_data.groupby('node_count')[f'{algo}_info'].apply(
                lambda x: np.mean([extract_info(i)['U_max'] for i in x if extract_info(i) is not None])
            )
        node_umax_table.loc[algo] = [node_groups.get(nc, np.nan) for nc in node_counts]
        plt.plot(node_counts, node_umax_table.loc[algo], label=algo, marker='o')
    plt.title('U_max vs Node Count Across Algorithms (Complete Cases Only)')
    plt.xlabel('Node Count')
    plt.ylabel('Average U_max')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/umax_vs_node_count_line_complete_cases.png')
    plt.close()

    node_umax_table.to_csv('../../data/analysis/table/node_count_umax_table_complete_cases.csv', index_label='Algorithm')

    # 8. Function Count vs U_max (折线图)
    func_umax_table = pd.DataFrame(index=algorithms, columns=function_counts)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        if algo == 'min_profit':
            func_groups = merged_data.groupby('function_count')['multi_func_worst_profit_info'].apply(
                lambda x: np.mean([extract_info(i)['U_max'] for i in x if extract_info(i) is not None])
            )
        else:
            func_groups = merged_data.groupby('function_count')[f'{algo}_info'].apply(
                lambda x: np.mean([extract_info(i)['U_max'] for i in x if extract_info(i) is not None])
            )
        func_umax_table.loc[algo] = [func_groups.get(fc, np.nan) for fc in function_counts]
        plt.plot(function_counts, func_umax_table.loc[algo], label=algo, marker='o')
    plt.title('U_max vs Function Count Across Algorithms (Complete Cases Only)')
    plt.xlabel('Function Count')
    plt.ylabel('Average U_max')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/umax_vs_function_count_line_complete_cases.png')
    plt.close()

    func_umax_table.to_csv('../../data/analysis/table/function_count_umax_table_complete_cases.csv', index_label='Algorithm')

    # 9. Node Count vs Cost (折线图)
    node_cost_table = pd.DataFrame(index=node_counts, columns=algorithms)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        if algo == 'min_profit':
            node_groups = merged_data.groupby('node_count')['multi_func_worst_profit_info'].apply(
                lambda x: np.mean([extract_info(i)['cost'] for i in x if extract_info(i) is not None])
            )
        else:
            node_groups = merged_data.groupby('node_count')[f'{algo}_info'].apply(
                lambda x: np.mean([extract_info(i)['cost'] for i in x if extract_info(i) is not None])
            )
        node_cost_table[algo] = [node_groups.get(nc, np.nan) for nc in node_counts]
        plt.plot(node_counts, node_cost_table[algo], label=algo, marker='o')
    plt.title('Cost vs Node Count Across Algorithms (Complete Cases Only)')
    plt.xlabel('Node Count')
    plt.ylabel('Average Cost ($)')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/cost_vs_node_count_line_complete_cases.png')
    plt.close()

    node_cost_table.to_csv('../../data/analysis/table/node_count_cost_table_complete_cases.csv', index_label='Node Count')

    # 10. Function Count vs Cost (折线图)
    func_cost_table = pd.DataFrame(index=function_counts, columns=algorithms)
    plt.figure(figsize=(12, 6))
    for algo in algorithms:
        if algo == 'min_profit':
            func_groups = merged_data.groupby('function_count')['multi_func_worst_profit_info'].apply(
                lambda x: np.mean([extract_info(i)['cost'] for i in x if extract_info(i) is not None])
            )
        else:
            func_groups = merged_data.groupby('function_count')[f'{algo}_info'].apply(
                lambda x: np.mean([extract_info(i)['cost'] for i in x if extract_info(i) is not None])
            )
        func_cost_table[algo] = [func_groups.get(fc, np.nan) for fc in function_counts]
        plt.plot(function_counts, func_cost_table[algo], label=algo, marker='o')
    plt.title('Cost vs Function Count Across Algorithms (Complete Cases Only)')
    plt.xlabel('Function Count')
    plt.ylabel('Average Cost ($)')
    plt.legend(title='Algorithms')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../../data/analysis/visual/cost_vs_function_count_line_complete_cases.png')
    plt.close()

    func_cost_table.to_csv('../../data/analysis/table/function_count_cost_table_complete_cases.csv', index_label='Function Count')

    # 更新分析报告
    with open('../../data/analysis/analysis_summary_complete_cases.txt', 'w') as f:
        f.write("=== Deployment Optimization Analysis Summary (Complete Cases Only) ===\n\n")

        f.write("1. Profit Comparison Across All Algorithms (Complete Cases Only)\n")
        f.write("   - See 'profit_comparison_complete_cases.png' for visual comparison.\n")
        f.write(
            f"   - Average Profits: {', '.join(f'{algo}: {np.nanmean(profits[algo]):.2f}$' for algo in algorithms)}\n\n")

        f.write("2. Multi Func vs Single Func Profit Comparison (Complete Cases Only)\n")
        f.write(f"   - Cases where Single Func equals Multi Func Profit: {same_profit_count} out of {total_cases}\n")
        f.write(f"   - Percentage: {same_profit_pct:.2f}%\n")
        f.write("   - See 'multi_vs_single_profit_complete_cases.png' for details.\n\n")

        f.write("3. Multi Func vs Single Func Time Comparison (Complete Cases Only)\n")
        f.write(f"   - Average Time Saving by Single Func: {avg_time_saving:.2f}%\n")
        f.write("   - See 'time_comparison_complete_cases.png' for details.\n\n")

        f.write("4. U_max Comparison (Complete Cases Only)\n")
        f.write("   - See 'umax_comparison_complete_cases.png' for visual comparison.\n")
        f.write(
            f"   - Average U_max: {', '.join(f'{algo}: {np.nanmean(u_maxes[algo]):.2f}' for algo in algorithms)}\n\n")

        f.write("5. Profit vs Node Count Analysis (Complete Cases Only)\n")
        f.write("   - See 'profit_vs_node_count_line_complete_cases.png' for line plot.\n")
        f.write("   - Average Profit by Node Count Table saved to 'node_count_profit_table_complete_cases.csv'\n")
        f.write("   - Average Profit by Node Count:\n")
        f.write(node_profit_table.to_string() + "\n\n")

        f.write("6. Profit vs Function Count Analysis (Complete Cases Only)\n")
        f.write("   - See 'profit_vs_function_count_line_complete_cases.png' for line plot.\n")
        f.write("   - Average Profit by Function Count Table saved to 'function_count_profit_table_complete_cases.csv'\n")
        f.write("   - Average Profit by Function Count:\n")
        f.write(func_profit_table.to_string() + "\n\n")

        f.write("7. U_max vs Node Count Analysis (Complete Cases Only)\n")
        f.write("   - See 'umax_vs_node_count_line_complete_cases.png' for line plot.\n")
        f.write("   - Average U_max by Node Count Table saved to 'node_count_umax_table_complete_cases.csv'\n")
        f.write("   - Average U_max by Node Count:\n")
        f.write(node_umax_table.to_string() + "\n\n")

        f.write("8. U_max vs Function Count Analysis (Complete Cases Only)\n")
        f.write("   - See 'umax_vs_function_count_line_complete_cases.png' for line plot.\n")
        f.write("   - Average U_max by Function Count Table saved to 'function_count_umax_table_complete_cases.csv'\n")
        f.write("   - Average U_max by Function Count:\n")
        f.write(func_umax_table.to_string() + "\n\n")

        f.write("9. Cost vs Node Count Analysis (Complete Cases Only)\n")
        f.write("   - See 'cost_vs_node_count_line_complete_cases.png' for line plot.\n")
        f.write("   - Average Cost by Node Count Table saved to 'node_count_cost_table_complete_cases.csv'\n")
        f.write("   - Average Cost by Node Count:\n")
        f.write(node_cost_table.to_string() + "\n\n")

        f.write("10. Cost vs Function Count Analysis (Complete Cases Only)\n")
        f.write("   - See 'cost_vs_function_count_line_complete_cases.png' for line plot.\n")
        f.write("   - Average Cost by Function Count Table saved to 'function_count_cost_table_complete_cases.csv'\n")
        f.write("   - Average Cost by Function Count:\n")
        f.write(func_cost_table.to_string() + "\n")

    print("分析完成。查看 'analysis_summary_complete_cases.txt'、CSV表格和生成的图表。")

if __name__ == "__main__":
    analyze_data()