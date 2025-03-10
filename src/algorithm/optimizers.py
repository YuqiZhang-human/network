import ast
from collections import defaultdict

class ConfigLoader:
    def __init__(self, config_data):
        self.config_data = config_data

    def load(self):
        """Parse and load configuration data."""
        try:
            config = {
                "node_count": int(self.config_data.get("node_count", 0)),
                "computation_capacity": ast.literal_eval(self.config_data.get("computation_capacity", "[]")),
                "resource_demands": ast.literal_eval(self.config_data.get("resource_demands", "[]")),
                "bandwidth_matrix": ast.literal_eval(self.config_data.get("bandwidth_matrix", "[]")),
                "data_sizes": list(map(float, ast.literal_eval(self.config_data.get("data_sizes", "[]")))),
            }
            
            # 添加链路权重参数，默认都为1.0（如果没有提供）
            link_weights = self.config_data.get("link_weights", None)
            if link_weights and isinstance(link_weights, str):
                config["link_weights"] = ast.literal_eval(link_weights)
            else:
                # 如果没有提供链路权重，则根据bandwidth_matrix创建默认值1.0的权重矩阵
                bandwidth_matrix = config["bandwidth_matrix"]
                default_weights = []
                for i in range(len(bandwidth_matrix)):
                    row = []
                    for j in range(len(bandwidth_matrix[i])):
                        # 如果有带宽连接，则权重为1.0，否则为0
                        weight = 1.0 if bandwidth_matrix[i][j] > 0 else 0.0
                        row.append(weight)
                    default_weights.append(row)
                config["link_weights"] = default_weights
            
            # 处理gpu_cost - 可能是列表或单一值
            gpu_cost = self.config_data.get("gpu_cost", 0)
            if isinstance(gpu_cost, str) and gpu_cost and (gpu_cost.startswith('[') and gpu_cost.endswith(']')):
                config["gpu_cost"] = ast.literal_eval(gpu_cost)
            else:
                config["gpu_cost"] = float(gpu_cost)
                
            # 处理memory_cost - 可能是列表或单一值
            memory_cost = self.config_data.get("memory_cost", 0)
            if isinstance(memory_cost, str) and memory_cost and (memory_cost.startswith('[') and memory_cost.endswith(']')):
                config["memory_cost"] = ast.literal_eval(memory_cost)
            else:
                config["memory_cost"] = float(memory_cost)
                
            # 处理bandwidth_cost - 可能是列表或单一值
            bandwidth_cost = self.config_data.get("bandwidth_cost", 0)
            if isinstance(bandwidth_cost, str) and bandwidth_cost and (bandwidth_cost.startswith('[') and bandwidth_cost.endswith(']')):
                config["bandwidth_cost"] = ast.literal_eval(bandwidth_cost)
            else:
                config["bandwidth_cost"] = float(bandwidth_cost)
                
            # 处理profit_per_user - 可能是列表或单一值
            profit_per_user = self.config_data.get("profit_per_user", 0)
            if isinstance(profit_per_user, str) and profit_per_user and (profit_per_user.startswith('[') and profit_per_user.endswith(']')):
                config["profit_per_user"] = ast.literal_eval(profit_per_user)
            else:
                config["profit_per_user"] = float(profit_per_user)
        except Exception as e:
            raise ValueError(f"Configuration parsing failed: {e}")
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
        self.link_weights = self.config["link_weights"]  # 添加链路权重参数
        self.cost_params = {
            "gpu_cost": self.config["gpu_cost"],
            "memory_cost": self.config["memory_cost"],
            "bandwidth_cost": self.config["bandwidth_cost"],
            "profit_per_user": self.config["profit_per_user"]
        }

    def calculate_u_max(self, deployment_plan):
        """Calculate the maximum number of users based on resources and bandwidth."""
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
                if bw <= 0 or (data_size > 0 and bw < data_size):
                    return 0
                bw_limit = float('inf') if data_size == 0 else bw // data_size
                bw_limits.append(bw_limit)

        u_max = min(comp_limits) if comp_limits else 0
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_total_cost(self, deployment_plan, U_max):
        """Calculate the total cost of resource usage and bandwidth."""
        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        # 处理成本参数，如果是列表则取第一个值
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

    def compute_first_deployment(self):
        """Compute deployment prioritizing GPU usage."""
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

        total_cost = self.calculate_total_cost(plan, U_max)
        
        # 处理profit_per_user，如果是列表则取第一个值
        profit_per_user = self.cost_params["profit_per_user"]
        if isinstance(profit_per_user, list) and len(profit_per_user) > 0:
            profit_per_user = profit_per_user[0]
            
        profit = U_max * profit_per_user - total_cost

        return {
            'deployment_plan': plan,
            'U_max': U_max,
            'total_cost': total_cost,
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
        self.link_weights = self.config["link_weights"]  # 添加链路权重参数
        self.cost_params = {
            "gpu_cost": self.config["gpu_cost"],
            "memory_cost": self.config["memory_cost"],
            "bandwidth_cost": self.config["bandwidth_cost"],
            "profit_per_user": self.config["profit_per_user"]
        }

    def calculate_u_max(self, deployment_plan):
        """Reuse ComputeFirstDeploymentOptimizer's method."""
        return ComputeFirstDeploymentOptimizer.calculate_u_max(self, deployment_plan)

    def calculate_total_cost(self, deployment_plan, U_max):
        """Calculate the total cost of resource usage and bandwidth."""
        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        # 处理成本参数，如果是列表则取第一个值
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

    def memory_first_deployment(self):
        """Compute deployment prioritizing memory usage."""
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

        total_cost = self.calculate_total_cost(plan, U_max)
        
        # 处理profit_per_user，如果是列表则取第一个值
        profit_per_user = self.cost_params["profit_per_user"]
        if isinstance(profit_per_user, list) and len(profit_per_user) > 0:
            profit_per_user = profit_per_user[0]
            
        profit = U_max * profit_per_user - total_cost

        return {
            'deployment_plan': plan,
            'U_max': U_max,
            'total_cost': total_cost,
            'profit': profit
        }

class EnhancedTreeNode:
    def __init__(self, deployment_plan, final_profit, total_cost, U_max):
        self.deployment_plan = deployment_plan
        self.final_profit = final_profit
        self.total_cost = total_cost
        self.U_max = U_max

class EnhancedDeploymentOptimizer:
    def __init__(self, config_data):
        config_loader = ConfigLoader(config_data)
        self.config = config_loader.load()
        self.physical_nodes = list(range(1, self.config["node_count"] + 1))
        self.total_resources = {n: self.config["computation_capacity"][n - 1] for n in self.physical_nodes}
        self.function_demands = self.config["resource_demands"]
        self.data_sizes = self.config["data_sizes"]
        self.bandwidth_matrix = self.config["bandwidth_matrix"]
        self.link_weights = self.config["link_weights"]  # 添加链路权重参数
        self.cost_params = {
            "gpu_cost": self.config["gpu_cost"],
            "memory_cost": self.config["memory_cost"],
            "bandwidth_cost": self.config["bandwidth_cost"],
            "profit_per_user": self.config["profit_per_user"]
        }
        self.best_profit = -float('inf')
        self.best_node = None
        self.min_profit = float('inf')
        self.min_profit_node = None
        self.min_cost = float('inf')
        self.min_cost_node = None
        self.max_users = -float('inf')
        self.max_users_node = None

    def calculate_u_max(self, deployment_plan):
        """Calculate U_max for a deployment plan."""
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
                if bw <= 0 or (data_size > 0 and bw < data_size):
                    return 0
                bw_limit = bw // data_size if data_size != 0 else float('inf')
                bw_limits.append(bw_limit)

        u_max = min(comp_limits) if comp_limits else 0
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_total_cost(self, deployment_plan, U_max):
        """Calculate the total cost of resource usage and bandwidth."""
        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        # 处理成本参数，如果是列表则取第一个值
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

    def is_feasible(self, partial_plan):
        """Check if a partial deployment plan is feasible."""
        node_demands = defaultdict(lambda: [0, 0])
        for func_idx, node_id in partial_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            node_demands[node_id][0] += req_gpu
            node_demands[node_id][1] += req_mem

        for node_id, (req_gpu, req_mem) in node_demands.items():
            total_gpu, total_mem = self.total_resources[node_id]
            if req_gpu > total_gpu or req_mem > total_mem:
                return False

        for i in range(1, len(partial_plan)):
            from_node = partial_plan[i - 1][1]
            to_node = partial_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
                if bw <= 0 or (data_size > 0 and bw < data_size):
                    return False

        return True

    def build_optimization_tree(self):
        """Build an optimization tree by exhaustively evaluating all feasible deployment plans."""
        num_functions = len(self.function_demands)
        stack = [[]]  # Start with an empty plan

        while stack:
            current_plan = stack.pop()

            if len(current_plan) == num_functions:
                U_max = self.calculate_u_max(current_plan)
                if U_max >= 1:
                    total_cost = self.calculate_total_cost(current_plan, U_max)
                    
                    # 处理profit_per_user，如果是列表则取第一个值
                    profit_per_user = self.cost_params["profit_per_user"]
                    if isinstance(profit_per_user, list) and len(profit_per_user) > 0:
                        profit_per_user = profit_per_user[0]
                        
                    final_profit = U_max * profit_per_user - total_cost
                    
                    if final_profit > self.best_profit:
                        self.best_profit = final_profit
                        self.best_node = EnhancedTreeNode(current_plan, final_profit, total_cost, U_max)
                    if final_profit < self.min_profit:
                        self.min_profit = final_profit
                        self.min_profit_node = EnhancedTreeNode(current_plan, final_profit, total_cost, U_max)
                    if total_cost < self.min_cost:
                        self.min_cost = total_cost
                        self.min_cost_node = EnhancedTreeNode(current_plan, final_profit, total_cost, U_max)
                    if U_max > self.max_users:
                        self.max_users = U_max
                        self.max_users_node = EnhancedTreeNode(current_plan, final_profit, total_cost, U_max)
                continue

            for node_id in self.physical_nodes:
                new_plan = current_plan + [(len(current_plan), node_id)]
                if self.is_feasible(new_plan):
                    stack.append(new_plan)

    def get_best_deployment(self):
        """Return the best deployment plan."""
        return self.best_node.deployment_plan if self.best_node else None

    def get_worst_deployment(self):
        """Return the worst deployment plan."""
        return self.min_profit_node.deployment_plan if self.min_profit_node else None

    def get_min_cost_deployment(self):
        """Return the minimum cost deployment plan."""
        return self.min_cost_node.deployment_plan if self.min_cost_node else None

    def get_max_users_deployment(self):
        """Return the maximum users deployment plan."""
        return self.max_users_node.deployment_plan if self.max_users_node else None

if __name__ == "__main__":
    config_data = {
        "node_count": 4,
        "computation_capacity": "[[10, 20], [15, 25], [20, 30], [25, 35]]",
        "resource_demands": "[[2, 4], [3, 6], [4, 8]]",
        "bandwidth_matrix": "[[0, 10, 5, 2], [10, 0, 8, 3], [5, 8, 0, 6], [2, 3, 6, 0]]",
        "data_sizes": "[1.0, 2.0]",
        "gpu_cost": 1.0,
        "memory_cost": 0.5,
        "bandwidth_cost": 0.1,
        "profit_per_user": 10.0
    }
    optimizer = EnhancedDeploymentOptimizer(config_data)
    optimizer.build_optimization_tree()
    best_plan = optimizer.get_best_deployment()
    worst_plan = optimizer.get_worst_deployment()
    min_cost_plan = optimizer.get_min_cost_deployment()
    max_users_plan = optimizer.get_max_users_deployment()
    print(f"Best deployment plan: {best_plan}")
    print(f"Worst deployment plan: {worst_plan}")
    print(f"Minimum cost deployment plan: {min_cost_plan}")
    print(f"Maximum users deployment plan: {max_users_plan}")