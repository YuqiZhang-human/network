import ast
from collections import deque, defaultdict

class ConfigLoader:
    def __init__(self, config_data):
        self.config_data = config_data

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

    def calculate_u_max(self, deployment_plan):
        """基于初始资源计算最大用户量"""
        node_demands = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_demands[node_id][0] += req_gpu
            node_demands[node_id][1] += req_mem

        comp_limits = []
        for node_id in self.physical_nodes:
            total_gpu, total_mem = self.total_resources[node_id]
            demand_gpu, demand_mem = node_demands.get(node_id, [0, 0])
            if demand_gpu > total_gpu or demand_mem > total_mem:
                return 0
            u_gpu = total_gpu // demand_gpu if demand_gpu != 0 else float('inf')
            u_mem = total_mem // demand_mem if demand_mem != 0 else float('inf')
            comp_limits.append(min(u_gpu, u_mem))

        bw_limits = []
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
                if bw <= 0:
                    return 0
                bw_limit = float('inf') if data_size == 0 else bw // data_size
                bw_limits.append(bw_limit)

        u_max = min(comp_limits) if comp_limits else 0
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_total_cost(self, deployment_plan, U_max):
        """计算总成本"""
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
                data = self.data_sizes[i - 1]
                comm_cost += data * self.cost_params["bandwidth_cost"] * U_max

        return comp_cost + comm_cost

    def compute_first_deployment(self):
        """算力优先部署算法"""
        gpu_usage = defaultdict(int)
        plan = []
        for func_idx in range(len(self.function_demands)):
            req_gpu, req_mem = self.function_demands[func_idx]
            candidates = []
            for node_id in self.physical_nodes:
                remaining_gpu = self.total_resources[node_id][0] - gpu_usage[node_id]
                if remaining_gpu < req_gpu or self.total_resources[node_id][1] < req_mem:
                    continue
                if func_idx == 0 or self.bandwidth_matrix[plan[-1][1] - 1][node_id - 1] > 0:
                    candidates.append((node_id, remaining_gpu))
            if not candidates:
                return None
            selected_node = max(candidates, key=lambda x: x[1])[0]
            plan.append((func_idx, selected_node))
            gpu_usage[selected_node] += req_gpu

        U_max = self.calculate_u_max(plan)
        if U_max < 1:
            return None

        cost = self.calculate_total_cost(plan, U_max)
        profit = U_max * self.cost_params["profit_per_user"] - cost

        return {
            'deployment_plan': plan,
            'U_max': U_max,
            'total_cost': cost,
            'profit': profit
        }

class MemoryFirstDeploymentOptimizer:
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

    def calculate_u_max(self, deployment_plan):
        """复用 ComputeFirstDeploymentOptimizer 的方法"""
        return ComputeFirstDeploymentOptimizer.calculate_u_max(self, deployment_plan)

    def calculate_total_cost(self, deployment_plan, U_max):
        """复用 ComputeFirstDeploymentOptimizer 的方法"""
        return ComputeFirstDeploymentOptimizer.calculate_total_cost(self, deployment_plan, U_max)

    def memory_first_deployment(self):
        """存储优先部署算法"""
        mem_usage = defaultdict(int)
        plan = []
        for func_idx in range(len(self.function_demands)):
            req_gpu, req_mem = self.function_demands[func_idx]
            candidates = []
            for node_id in self.physical_nodes:
                remaining_mem = self.total_resources[node_id][1] - mem_usage[node_id]
                if remaining_mem < req_mem or self.total_resources[node_id][0] < req_gpu:
                    continue
                if func_idx == 0 or self.bandwidth_matrix[plan[-1][1] - 1][node_id - 1] > 0:
                    candidates.append((node_id, remaining_mem))
            if not candidates:
                return None
            selected_node = max(candidates, key=lambda x: x[1])[0]
            plan.append((func_idx, selected_node))
            mem_usage[selected_node] += req_mem

        U_max = self.calculate_u_max(plan)
        if U_max < 1:
            return None

        cost = self.calculate_total_cost(plan, U_max)
        profit = U_max * self.cost_params["profit_per_user"] - cost

        return {
            'deployment_plan': plan,
            'U_max': U_max,
            'total_cost': cost,
            'profit': profit
        }

class EnhancedTreeNode:
    def __init__(self, node_id, level, config, parent=None):
        self.node_id = node_id
        self.level = level
        self.config = config
        self.parent = parent
        self.deployment_plan = parent.deployment_plan.copy() if parent else []
        self.U_max = 0
        self.total_cost = 0
        self.final_profit = 0
        self.is_valid = True
        if level >= 0:
            node_value = None  # 将在构建时动态赋值
            self.deployment_plan.append((level, node_value))

class EnhancedDeploymentOptimizer:
    def __init__(self, config_data):
        config_loader = ConfigLoader(config_data)
        self.config = config_loader.load()
        self.best_profit = -float('inf')  # 最大利润初始化为负无穷
        self.best_node = None  # 最大利润对应的节点
        self.min_profit = float('inf')  # 最小利润初始化为正无穷
        self.min_profit_node = None
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
        self.best_profit = -float('inf')
        self.best_node = None
        self.node_counter = 1

    def calculate_u_max(self, deployment_plan):
        """复用 ComputeFirstDeploymentOptimizer 的方法"""
        return ComputeFirstDeploymentOptimizer.calculate_u_max(self, deployment_plan)

    def calculate_total_cost(self, deployment_plan, U_max):
        """复用 ComputeFirstDeploymentOptimizer 的方法"""
        return ComputeFirstDeploymentOptimizer.calculate_total_cost(self, deployment_plan, U_max)

    def build_optimization_tree(self):
        """构建优化树，动态赋值节点"""
        root = EnhancedTreeNode(node_id=0, level=-1, config=self.config)
        queue = deque([root])
        while queue:
            current_node = queue.popleft()
            if current_node.level == len(self.function_demands) - 1:
                current_node.U_max = self.calculate_u_max(current_node.deployment_plan)
                current_node.is_valid = current_node.U_max >= 1
                if current_node.is_valid:
                    self.evaluate_final_plan(current_node)
                continue
            next_level = current_node.level + 1
            req_gpu, req_mem = self.function_demands[next_level]
            for node_id in self.physical_nodes:
                total_gpu, total_mem = self.total_resources[node_id]
                used_gpu = sum(self.function_demands[func_idx][0] for func_idx, n_id in current_node.deployment_plan if n_id == node_id)
                used_mem = sum(self.function_demands[func_idx][1] for func_idx, n_id in current_node.deployment_plan if n_id == node_id)
                if used_gpu + req_gpu > total_gpu or used_mem + req_mem > total_mem:
                    continue  # 提前剪枝
                child = EnhancedTreeNode(node_id=self.node_counter, level=next_level, config=self.config, parent=current_node)
                self.node_counter += 1
                child.deployment_plan[-1] = (next_level, node_id)  # 更新节点值
                if next_level > 0 and self.bandwidth_matrix[current_node.deployment_plan[-1][1] - 1][node_id - 1] <= 0:
                    child.is_valid = False
                queue.append(child)

    def evaluate_final_plan(self, node):
        """评估最终部署方案"""
        if not node.is_valid:
            return
        node.total_cost = self.calculate_total_cost(node.deployment_plan, node.U_max)
        node.final_profit = node.U_max * self.cost_params["profit_per_user"] - node.total_cost

        # 更新最大利润
        if node.final_profit > self.best_profit:
            self.best_profit = node.final_profit
            self.best_node = node

        # 更新最小利润
        if node.final_profit < self.min_profit:
            self.min_profit = node.final_profit
            self.min_profit_node = node