import json
from collections import defaultdict


class ComputeFirstDeploymentOptimizer:
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

    def compute_first_deployment(self):
        """算力优先部署算法"""
        gpu_usage = defaultdict(int)
        plan = []
        print("===== 算力优先部署算法 =====")

        for func_idx in range(len(self.function_demands)):
            req_gpu, req_mem = self.function_demands[func_idx]

            candidates = [(node_id, self.total_resources[node_id][0] - gpu_usage[node_id]) for node_id in
                          self.physical_nodes]
            candidates = [(node_id, remaining_gpu) for node_id, remaining_gpu in candidates if remaining_gpu >= req_gpu]

            if not candidates:
                print(f"功能 {func_idx} 无法找到合适的物理节点部署！")
                return None  # 若没有足够算力的节点，则没有可行方案

            selected = max(candidates, key=lambda x: x[1])[0]
            plan.append((func_idx, selected))
            gpu_usage[selected] += req_gpu

            print(
                f"功能 {func_idx} 部署到节点 {selected} (剩余算力: {self.total_resources[selected][0] - gpu_usage[selected]}GPU)")

        # 计算最大用户量、总成本、利润
        U_max = self.calculate_u_max(plan)
        print(f"\n计算出的最大用户量 U_max: {U_max}")

        if U_max < 1:
            print("此部署方案不可行，最大用户量不足！")
            return None

        cost = self.calculate_total_cost(plan, U_max)
        print(f"总成本: {cost:.2f}$")

        profit = U_max * self.cost_params["profit_per_user"] - cost
        print(f"最终利润: {profit:.2f}$")

        return {
            'deployment_plan': plan,
            'U_max': U_max,
            'total_cost': cost,
            'profit': profit
        }

    def print_plan(self, plan):
        """打印部署方案"""
        print(f"\n部署路径: {plan['deployment_plan']}")
        print(f"最大用户量: {plan['U_max']}")
        print(f"总成本: {plan['total_cost']:.2f}$")
        print(f"最终利润: {plan['profit']:.2f}$")


if __name__ == "__main__":
    optimizer = ComputeFirstDeploymentOptimizer("deployment_config.json")
    compute_plan = optimizer.compute_first_deployment()
    if compute_plan:
        optimizer.print_plan(compute_plan)
    else:
        print("未找到有效方案")
