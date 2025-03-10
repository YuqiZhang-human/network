import pandas as pd
import time
from optimizers import ConfigLoader
from collections import defaultdict

class SingleFuncOptimizer:
    def __init__(self, config_data):
        config_loader = ConfigLoader(config_data)
        self.config = config_loader.load()
        self.physical_nodes = list(range(1, self.config["node_count"] + 1))
        self.total_resources = {n: self.config["computation_capacity"][n - 1] for n in self.physical_nodes}
        self.function_demands = self.config["resource_demands"]
        self.data_sizes = self.config["data_sizes"]
        self.bandwidth_matrix = self.config["bandwidth_matrix"]
        self.link_weights = self.config["link_weights"]
        self.cost_params = {
            "gpu_cost": self.config["gpu_cost"],
            "memory_cost": self.config["memory_cost"],
            "bandwidth_cost": self.config["bandwidth_cost"],
            "profit_per_user": self.config["profit_per_user"]
        }

    def single_func_deployment(self):
        # 确保节点数 >= 功能数
        if len(self.physical_nodes) < len(self.function_demands):
            return None

        # 选择一组节点，每个节点部署一个功能
        selected_nodes = self.physical_nodes[:len(self.function_demands)]
        plan = [(i, selected_nodes[i]) for i in range(len(self.function_demands))]

        # 检查资源是否足够
        for func_idx, node_id in plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            total_gpu, total_mem = self.total_resources[node_id]
            if req_gpu > total_gpu or req_mem > total_mem:
                return None

        U_max = self.calculate_u_max(plan)
        if U_max < 1:
            return None

        cost = self.calculate_total_cost(plan, U_max)
        profit = U_max * self.cost_params["profit_per_user"] - cost
        return {'deployment_plan': plan, 'U_max': U_max, 'total_cost': cost, 'profit': profit}

    def calculate_u_max(self, deployment_plan):
        comp_limits = []
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            total_gpu, total_mem = self.total_resources[node_id]
            u_gpu = total_gpu // req_gpu if req_gpu != 0 else float('inf')
            u_mem = total_mem // req_mem if req_mem != 0 else float('inf')
            comp_limits.append(min(u_gpu, u_mem))

        bw_limits = []
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
                if bw <= 0 or (data_size > 0 and bw < data_size):
                    return 0
                bw_limit = bw // data_size if data_size != 0 else float('inf')
                bw_limits.append(bw_limit)

        u_max = min(comp_limits) if comp_limits else 0
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_total_cost(self, deployment_plan, U_max):
        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        comp_cost = sum(
            used_gpu * self.cost_params["gpu_cost"] + used_mem * self.cost_params["memory_cost"]
            for used_gpu, used_mem in node_usage.values()
        )

        comm_cost = 0
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                link_weight = self.link_weights[from_node - 1][to_node - 1]
                link_connectivity = 1 if self.bandwidth_matrix[from_node - 1][to_node - 1] > 0 else 0
                comm_cost += data_size * link_weight * link_connectivity * self.cost_params["bandwidth_cost"] * U_max

        return comp_cost + comm_cost

def run_single_func(existing_df=None):
    test_data = pd.read_csv("../../data/test/test_data.csv")
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