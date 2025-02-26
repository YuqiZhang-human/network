import pandas as pd
import time
from optimizers import ConfigLoader


class SingleFuncOptimizer:
    def __init__(self, config_data):
        config_loader = ConfigLoader(config_data)
        self.config = config_loader.load()
        self.physical_nodes = list(range(1, self.config["node_count"] + 1))
        self.total_resources = {n: self.config["computation_capacity"][n - 1] for n in self.physical_nodes}
        self.function_demands = self.config["resource_demands"]
        self.data_sizes = self.config["data_sizes"]
        self.bandwidth_matrix = self.config["bandwidth_matrix"]
        self.cost_params = {
            "gpu_cost": self.config["gpu_cost"],
            "memory_cost": self.config["memory_cost"],
            "bandwidth_cost": self.config["bandwidth_cost"],
            "profit_per_user": self.config["profit_per_user"]
        }
        self.total_gpu_demand = sum(gpu for gpu, _ in self.function_demands)
        self.total_mem_demand = sum(mem for _, mem in self.function_demands)

    def single_func_deployment(self):
        for node_id in self.physical_nodes:
            total_gpu, total_mem = self.total_resources[node_id]
            if self.total_gpu_demand <= total_gpu and self.total_mem_demand <= total_mem:
                plan = [(i, node_id) for i in range(len(self.function_demands))]
                U_max = self.calculate_u_max(plan)
                if U_max >= 1:
                    cost = self.calculate_total_cost(plan, U_max)
                    profit = U_max * self.cost_params["profit_per_user"] - cost
                    return {'deployment_plan': plan, 'U_max': U_max, 'total_cost': cost, 'profit': profit}
        return None

    def calculate_u_max(self, deployment_plan):
        node_id = deployment_plan[0][1]
        total_gpu, total_mem = self.total_resources[node_id]
        u_gpu = total_gpu // self.total_gpu_demand
        u_mem = total_mem // self.total_mem_demand
        return min(u_gpu, u_mem)

    def calculate_total_cost(self, deployment_plan, U_max):
        node_id = deployment_plan[0][1]
        gpu_cost = self.total_gpu_demand * U_max * self.cost_params["gpu_cost"]
        mem_cost = self.total_mem_demand * U_max * self.cost_params["memory_cost"]
        return gpu_cost + mem_cost


def run_single_func(existing_df=None):
    test_data = pd.read_csv("../test/enhanced_connectivity_data.csv")
    results = []
    for _, row in test_data.iterrows():
        test_id = row['test_data_id']
        optimizer = SingleFuncOptimizer(config_data=row.to_dict())
        start_time = time.time()
        plan = optimizer.single_func_deployment()
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
            'single_func_info': str(info) if info else 'None',
            'single_func_time': round(elapsed_time, 4)
        }
        results.append(result)

    df = pd.DataFrame(results)
    if existing_df is None:
        return df
    else:
        merged_df = pd.merge(existing_df, df, on='test_data_id', how='outer')
        return merged_df