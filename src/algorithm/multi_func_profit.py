import pandas as pd
import time
from optimizers import EnhancedDeploymentOptimizer

def run_multi_func_profit():
    test_data = pd.read_csv("../test/enhanced_connectivity_data.csv")
    results = []
    for _, row in test_data.iterrows():
        test_id = row['test_data_id']
        start_time = time.time()
        optimizer = EnhancedDeploymentOptimizer(config_data=row.to_dict())
        optimizer.build_optimization_tree()
        elapsed_time = time.time() - start_time
        info = (
            [
                round(optimizer.best_node.total_cost, 2),
                round(optimizer.best_node.final_profit, 2),
                optimizer.best_node.U_max,
                optimizer.best_node.deployment_plan
            ]
            if optimizer.best_node
            else None
        )
        result = {
            'test_data_id': test_id,
            'multi_func_profit_info': str(info) if info else 'None',
            'multi_func_time': round(elapsed_time, 4)
        }
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv('results.csv', mode='w', index=False, header=True)  # 覆盖写入，带列名
    return df
