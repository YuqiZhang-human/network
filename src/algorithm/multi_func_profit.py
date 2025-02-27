import pandas as pd
import time
from optimizers import EnhancedDeploymentOptimizer

import pandas as pd
import time

def run_multi_func_profit():
    test_data = pd.read_csv("../test/enhanced_connectivity_data.csv")
    results = []
    for _, row in test_data.iterrows():
        test_id = row['test_data_id']
        start_time = time.time()
        optimizer = EnhancedDeploymentOptimizer(config_data=row.to_dict())
        optimizer.build_optimization_tree()
        elapsed_time = time.time() - start_time
        best_info = (
            [
                round(optimizer.best_node.total_cost, 2),
                round(optimizer.best_node.final_profit, 2),
                optimizer.best_node.U_max,
                optimizer.best_node.deployment_plan
            ]
            if optimizer.best_node
            else None
        )
        worst_info = (
            [
                round(optimizer.min_profit_node.total_cost, 2),
                round(optimizer.min_profit_node.final_profit, 2),
                optimizer.min_profit_node.U_max,
                optimizer.min_profit_node.deployment_plan
            ]
            if optimizer.min_profit_node
            else None
        )
        result = {
            'test_data_id': test_id,
            'multi_func_profit_info': str(best_info) if best_info else 'None',
            'multi_func_worst_profit_info': str(worst_info) if worst_info else 'None',
            'multi_func_time': round(elapsed_time, 4)
        }
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv('../../data/analysis/table/results.csv', mode='w', index=False, header=True)
    return df
