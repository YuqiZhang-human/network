from multi_func_profit import run_multi_func_profit
from compute_first import run_compute_first
from memory_first import run_memory_first
from single_func import run_single_func
from random_deploy import run_random_deploy
from data_analysis import analyze_data


def main():
    print("Running Multi Func Profit...")
    result_df = run_multi_func_profit()  # 第一个算法，覆盖写入

    print("Running Compute First...")
    result_df = run_compute_first(result_df)  # 追加并合并

    print("Running Memory First...")
    result_df = run_memory_first(result_df)  # 追加并合并

    print("Running Single Func...")
    result_df = run_single_func(result_df)  # 追加并合并

    print("Running Random Deploy...")
    result_df = run_random_deploy(result_df)  # 追加并合并

    # 写入最终结果
    result_df.to_csv('results.csv', index=False)

    print("Analyzing Results...")
    analyze_data()
    print("Execution completed.")


if __name__ == "__main__":
    main()