import pandas as pd
import random
import time
from collections import defaultdict
from optimizers import ConfigLoader


class RandomDeploymentOptimizer:
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

    def random_deployment(self, num_trials=1000):
        for _ in range(num_trials):
            plan = [(i, random.choice(self.physical_nodes)) for i in range(len(self.function_demands))]
            u_max = self.calculate_u_max(plan)
            if u_max >= 1 and self.is_connected(plan):
                cost = self.calculate_total_cost(plan, u_max)
                profit = u_max * self.cost_params["profit_per_user"] - cost
                return {'deployment_plan': plan, 'U_max': u_max, 'total_cost': cost, 'profit': profit}  # 统一使用 'U_max'
        return None

    def calculate_u_max(self, deployment_plan):
        node_demands = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_demands[node_id][0] += req_gpu
            node_demands[node_id][1] += req_mem

        comp_limits = []
        for node_id, (total_gpu, total_mem) in self.total_resources.items():
            demand_gpu, demand_mem = node_demands.get(node_id, [0, 0])
            if demand_gpu > total_gpu or demand_mem > total_mem:
                return 0
            u_gpu = total_gpu // demand_gpu if demand_gpu else float('inf')
            u_mem = total_mem // demand_mem if demand_mem else float('inf')
            comp_limits.append(min(u_gpu, u_mem))

        bw_limits = []
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
                bw_limit = bw // data_size if data_size else float('inf')
                bw_limits.append(bw_limit)

        u_max = min(comp_limits)
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max

    def calculate_total_cost(self, deployment_plan, U_max):
        """Calculate the total cost of resource usage and bandwidth."""
        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        # 处理成本参数，确保使用正确的值类型
        gpu_cost = self.cost_params["gpu_cost"]
        if isinstance(gpu_cost, list) and len(gpu_cost) > 0:
            gpu_cost = gpu_cost[0]
            
        memory_cost = self.cost_params["memory_cost"]
        if isinstance(memory_cost, list) and len(memory_cost) > 0:
            memory_cost = memory_cost[0]
            
        bandwidth_cost = self.cost_params["bandwidth_cost"]
        if isinstance(bandwidth_cost, list) and len(bandwidth_cost) > 0:
            bandwidth_cost = bandwidth_cost[0]

        comp_cost = sum(
            used_gpu * gpu_cost + used_mem * memory_cost
            for used_gpu, used_mem in node_usage.values()
        )

        comm_cost = 0
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                # 考虑链路权重：带宽开销 = 带宽需求 * 链路权重 * 链路连通性 * 带宽价格
                data_size = self.data_sizes[i - 1]
                link_weight = self.link_weights[from_node - 1][to_node - 1]
                link_connectivity = 1 if self.bandwidth_matrix[from_node - 1][to_node - 1] > 0 else 0
                comm_cost += data_size * link_weight * link_connectivity * bandwidth_cost * U_max

        return comm_cost + comp_cost

    def is_connected(self, plan):
        for i in range(1, len(plan)):
            from_node = plan[i - 1][1]
            to_node = plan[i][1]
            if from_node != to_node and self.bandwidth_matrix[from_node - 1][to_node - 1] == 0:
                return False
        return True


def run_random_deploy(existing_df=None):
    test_data = pd.read_csv("../../data/test/test_data.csv")
    results = []
    for _, row in test_data.iterrows():
        test_id = row['test_data_id']
        optimizer = RandomDeploymentOptimizer(config_data=row.to_dict())
        start_time = time.time()
        plan = optimizer.random_deployment()
        elapsed_time = time.time() - start_time
        info = (
            [
                round(plan['total_cost'], 2),
                round(plan['profit'], 2),
                plan['U_max'],  # 使用 'U_max' 与返回字典一致
                plan['deployment_plan']
            ]
            if plan
            else None
        )
        result = {
            'test_data_id': test_id,
            'random_deploy_info': str(info) if info else 'None',
            'random_deploy_time': round(elapsed_time, 4)
        }
        results.append(result)

    df = pd.DataFrame(results)
    if existing_df is None:
        return df
    else:
        merged_df = pd.merge(existing_df, df, on='test_data_id', how='outer')
        return merged_df