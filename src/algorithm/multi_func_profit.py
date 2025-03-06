import pandas as pd
import time
from optimizers import EnhancedDeploymentOptimizer


def run_multi_func_profit():
    test_data = pd.read_csv("../../data/test/test_data.csv")
    results = []
    for _, row in test_data.iterrows():
        test_id = row['test_data_id']
        start_time = time.time()
        optimizer = EnhancedDeploymentOptimizer(config_data=row.to_dict())

        print(test_id)

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
        
        # 添加最小成本方案信息
        min_cost_info = (
            [
                round(optimizer.min_cost_node.total_cost, 2),
                round(optimizer.min_cost_node.final_profit, 2),
                optimizer.min_cost_node.U_max,
                optimizer.min_cost_node.deployment_plan
            ]
            if optimizer.min_cost_node
            else None
        )
        
        # 添加最大用户量方案信息
        max_users_info = (
            [
                round(optimizer.max_users_node.total_cost, 2),
                round(optimizer.max_users_node.final_profit, 2),
                optimizer.max_users_node.U_max,
                optimizer.max_users_node.deployment_plan
            ]
            if optimizer.max_users_node
            else None
        )
        
        result = {
            'test_data_id': test_id,
            'multi_func_profit_info': str(best_info) if best_info else 'None',
            'multi_func_worst_profit_info': str(worst_info) if worst_info else 'None',
            'multi_func_min_cost_info': str(min_cost_info) if min_cost_info else 'None',
            'multi_func_max_users_info': str(max_users_info) if max_users_info else 'None',
            'multi_func_time': round(elapsed_time, 4)
        }
        results.append(result)
    df = pd.DataFrame(results)
    df.to_csv('../../data/analysis/table/results.csv', mode='w', index=False, header=True)
    return df
