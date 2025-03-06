import pandas as pd
import time
from optimizers import ComputeFirstDeploymentOptimizer


def run_compute_first(existing_df=None):
    test_data = pd.read_csv("../../data/test/test_data.csv")
    results = []
    for _, row in test_data.iterrows():
        test_id = row['test_data_id']
        start_time = time.time()
        optimizer = ComputeFirstDeploymentOptimizer(config_data=row.to_dict())
        plan = optimizer.compute_first_deployment()
        elapsed_time = time.time() - start_time
        info = (
            [
                round(plan['total_cost'], 2),
                round(plan['profit'], 2),
                plan['U_max'],
                plan['deployment_plan']
            ]
            if plan
            else None
        )
        result = {
            'test_data_id': test_id,
            'compute_first_info': str(info) if info else 'None',
            'compute_first_time': round(elapsed_time, 4)
        }
        results.append(result)

    df = pd.DataFrame(results)
    if existing_df is None:
        return df  # 如果是第一个算法，直接返回 DataFrame
    else:
        # 合并现有 DataFrame 和新结果，按 test_data_id 对齐
        merged_df = pd.merge(existing_df, df, on='test_data_id', how='outer')
        return merged_df