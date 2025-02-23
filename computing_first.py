import ast
from collections import defaultdict
import pandas as pd

class ConfigLoader:
    def __init__(self, config_data):
        self.config_data = config_data
        self.config = {}

    def load(self):
        """解析并加载配置文件"""
        try:
            config = {
                "node_count": int(self.config_data.get("node_count", 0)),
                "computation_capacity": ast.literal_eval(self.config_data.get("computation_capacity", "[]")),
                "resource_demands": ast.literal_eval(self.config_data.get("resource_demands", "[]")),
                "bandwidth_matrix": ast.literal_eval(self.config_data.get("bandwidth_matrix", "[]")),
                "data_sizes": list(map(float, ast.literal_eval(self.config_data.get("data_sizes", "[]")))),
                "gpu_cost": float(self.config_data.get("gpu_cost", 0)),
                "memory_cost": float(self.config_data.get("memory_cost", 0)),
                "bandwidth_cost": float(self.config_data.get("bandwidth_cost", 0)),
                "profit_per_user": float(self.config_data.get("profit_per_user", 0))
            }
        except Exception as e:
            raise ValueError(f"配置文件解析失败: {e}")

        return config

class ComputeFirstDeploymentOptimizer:
    def __init__(self, config_data):
        config_loader = ConfigLoader(config_data)
        self.config = config_loader.load()  # 使用加载的配置
        self.cost_params = None
        self.bandwidth_matrix = None
        self.data_sizes = None
        self.function_demands = None
        self.total_resources = None
        self.physical_nodes = None
        self.node_counter = 1
        self.best_profit = -float('inf')
        self.best_node = None
        self.max_user_node = None
        self.min_cost_node = None
        self.all_solutions = []
        self.test_result_id = config_data.get('test_data_id', 1)
        self.load_config()  # 依然调用本类的load_config进行其它初始化

    def load_config(self):
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
        print(self.function_demands)

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

            candidates = [(node_id, self.total_resources[node_id][1] - gpu_usage[node_id]) for node_id in
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

    def update_computing_first_node(self):
        print(4)
        compute_plan = self.compute_first_deployment()
        print(5)
        print(compute_plan)
        node_computing_first_info = []
        if compute_plan:
            optimizer.print_plan(compute_plan)
            # 构造目标格式的列表并存储
            node_computing_first_info = [
                compute_plan['total_cost'],
                compute_plan['profit'],
                compute_plan['U_max'],
                compute_plan['deployment_plan']
            ]
            print("node_computing_first_info:", node_computing_first_info)
        else:
            print("未找到有效方案")
        return node_computing_first_info


if __name__ == "__main__":
    test_data = pd.read_csv("test_data.csv")
    for _, row in test_data.iterrows():
        try:
            print(f"正在处理测试用例 {row['test_data_id']}")
            print(1)
            optimizer = ComputeFirstDeploymentOptimizer(config_data=row.to_dict())
            print(optimizer.function_demands)
            print(2)
            optimizer.update_computing_first_node()
            print(3)
            print(f"测试用例 {row['test_data_id']} 处理成功\n")
        except Exception as e:
            print(f"用例 {row['test_data_id']} 处理失败: {str(e)}\n")