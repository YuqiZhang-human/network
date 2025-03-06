from multi_func_profit import run_multi_func_profit
from single_func import run_single_func
from random_deploy import run_random_deploy
from data import analyze_data


def main():
    print("Running Multi Func Profit...")
    result_df = run_multi_func_profit()  # 第一个算法，覆盖写入

    # 保存包含最小成本和最大用户量的完整结果
    complete_result_df = result_df.copy()

    print("Running Single Func...")
    result_df = run_single_func(result_df)  # 追加并合并

    print("Running Random Deploy...")
    result_df = run_random_deploy(result_df)  # 追加并合并

    # 确保保留最小成本和最大用户量列
    if 'multi_func_min_cost_info' in complete_result_df.columns:
        result_df['multi_func_min_cost_info'] = complete_result_df['multi_func_min_cost_info']
    if 'multi_func_max_users_info' in complete_result_df.columns:
        result_df['multi_func_max_users_info'] = complete_result_df['multi_func_max_users_info']

    # 写入最终结果
    result_df.to_csv('../../data/analysis/table/results.csv', index=False)

    print("Analyzing Results...")
    analyze_data()
    print("Execution completed.")


if __name__ == "__main__":
    main()