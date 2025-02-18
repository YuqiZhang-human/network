import json
import random
from collections import defaultdict


class RandomDeploymentOptimizer:
    def __init__(self, config_file):
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)
        self.physical_nodes = list(range(1, self.config["node_settings"]["node_count"] + 1))
        self.total_resources = {n: self.config["node_settings"]["computation_capacity"][n - 1] for n in
                                self.physical_nodes}
        self.function_demands = self.config["function_settings"]["resource_demands"]
        self.data_sizes = self.config["function_settings"]["data_sizes"]
        self.bandwidth_matrix = self.config["network_settings"]["bandwidth_matrix"]
        self.cost_params = self.config["cost_settings"]

    def calculate_u_max(self, deployment_plan):
        """基于初始资源计算最大用户量"""
        node_demands = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_demands[node_id][0] += req_gpu
            node_demands[node_id][1] += req_mem

        comp_limits = []
        for node_id, (total_gpu, total_mem) in self.total_resources.items():
            demand_gpu, demand_mem = node_demands.get(node_id, [0, 0])
            u_gpu = total_gpu // demand_gpu if demand_gpu != 0 else float('inf')
            u_mem = total_mem // demand_mem if demand_mem != 0 else float('inf')
            comp_limits.append(min(u_gpu, u_mem))

        # 计算带宽限制
        bw_limits = []
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
                if data_size == 0:
                    bw_limit = float('inf')
                else:
                    bw_limit = bw // data_size
                bw_limits.append(bw_limit)

        u_max = min(comp_limits) if comp_limits else 0
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max

    def calculate_total_cost(self, deployment_plan, U_max):
        """计算总成本"""
        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        comp_cost = 0
        for node_id, (used_gpu, used_mem) in node_usage.items():
            comp_cost += used_gpu * self.cost_params["gpu_cost"] + used_mem * self.cost_params["memory_cost"]

        comm_cost = 0
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data = self.data_sizes[i - 1]
                comm_cost += data * self.cost_params["bandwidth_cost"] * U_max

        return comp_cost + comm_cost

    def random_deployment(self, num_trials=1000):
        """随机部署算法，找到第一个有效的可行方案"""
        for _ in range(num_trials):
            # 随机选择每个功能部署的节点
            plan = [(i, random.choice(self.physical_nodes)) for i in range(len(self.function_demands))]

            # 计算最大用户量
            U_max = self.calculate_u_max(plan)
            if U_max < 1:
                continue  # 如果该方案不可行，跳过

            # 计算总成本
            cost = self.calculate_total_cost(plan, U_max)

            # 计算利润
            profit = U_max * self.cost_params["profit_per_user"] - cost

            return {'deployment_plan': plan, 'U_max': U_max, 'total_cost': cost, 'profit': profit}

        return None  # 如果没有找到有效方案

    def print_plan(self, plan):
        """打印部署方案"""
        print(f"部署路径: {plan['deployment_plan']}")
        print(f"最大用户量: {plan['U_max']}")
        print(f"总成本: {plan['total_cost']:.2f}$")
        print(f"最终利润: {plan['profit']:.2f}$")


if __name__ == "__main__":
    optimizer = RandomDeploymentOptimizer("deployment_config6.json")

    print("===== 随机部署算法 =====")
    random_plan = optimizer.random_deployment(num_trials=1000)
    if random_plan:
        optimizer.print_plan(random_plan)
    else:
        print("未找到有效方案")
