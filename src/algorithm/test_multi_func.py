from multi_func_profit import run_multi_func_profit
import pandas as pd

def main():
    print("Running Multi Func Profit...")
    result_df = run_multi_func_profit()
    
    # 检查结果
    if 'multi_func_min_cost_info' in result_df.columns and 'multi_func_max_users_info' in result_df.columns:
        print("测试成功: 找到最小成本和最大用户量列")
    else:
        print("测试失败: 缺少最小成本或最大用户量列")
        print("实际列: ", result_df.columns.tolist())
    
    # 查看第一行数据
    if not result_df.empty:
        print("\n第一行数据:")
        first_row = result_df.iloc[0]
        print("test_data_id:", first_row['test_data_id'])
        print("最大利润方案:", first_row['multi_func_profit_info'])
        print("最小利润方案:", first_row['multi_func_worst_profit_info'])
        print("最小成本方案:", first_row['multi_func_min_cost_info'])
        print("最大用户量方案:", first_row['multi_func_max_users_info'])

if __name__ == "__main__":
    main() 