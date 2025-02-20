from collections import deque, defaultdict


class EnhancedTreeNode:
    def __init__(self, value, node_id, level, config, parent=None):
        self.value = value
        self.node_id = node_id
        self.level = level
        self.parent = parent
        self.config = config
        self.deployment_plan = parent.deployment_plan.copy() if parent else []
        if value is not None and level >= 0:
            self.deployment_plan.append((level, value))
        self.children = []
        self.is_valid = True
        self.U_max = 0
        self.total_cost = 0
        self.final_profit = 0

    def get_actual_resources(self, U_max):
        res = {n: [c[0], c[1]] for n, c in enumerate(self.config["computation"], 1)}
        for func, node in self.deployment_plan:
            req = self.config["demands"][func]
            res[node][0] -= req[0] * U_max
            res[node][1] -= req[1] * U_max
        return res


class MultiFuncOptimizer:
    def __init__(self, config):
        self.config = {
            "node_count": int(config["node_count"]),
            "computation": [
                list(map(int, row.split(',')))
                for row in config["computation_matrix"].split(';')
            ],
            "function_count": int(config["function_count"]),
            "demands": [
                list(map(int, row.split(',')))
                for row in config["demand_matrix"].split(';')
            ],
            "data_sizes": list(map(int, config["data_sizes"].split(','))),
            "bandwidth": [
                list(map(int, row.split(',')))
                for row in config["bandwidth_matrix"].split(';')
            ],
            "cost_params": {
                "gpu": float(config["gpu_cost"]),
                "memory": float(config["memory_cost"]),
                "bandwidth": float(config["bandwidth_cost"]),
                "profit": int(config["profit_per_user"])
            }
        }
        self.physical_nodes = list(range(1, self.config["node_count"] + 1))
        self.node_counter = 1
        self.best_profit = -float('inf')
        self.best_node = None
        self.feasible_nodes = []

    def calculate_u_max(self, plan):
        node_demands = defaultdict(lambda: [0, 0])
        for f, n in plan:
            req = self.config["demands"][f]
            node_demands[n][0] += req[0]
            node_demands[n][1] += req[1]

        limits = []
        for n in self.physical_nodes:
            total = self.config["computation"][n - 1]
            used = node_demands[n]
            u_gpu = total[0] // used[0] if used[0] else float('inf')
            u_mem = total[1] // used[1] if used[1] else float('inf')
            limits.append(min(u_gpu, u_mem))

        bw_limits = []
        for i in range(1, len(plan)):
            from_n = plan[i - 1][1]
            to_n = plan[i][1]
            if from_n != to_n:
                data = self.config["data_sizes"][i - 1]
                bw = self.config["bandwidth"][from_n - 1][to_n - 1]
                bw_limits.append(bw // data if data else float('inf'))

        u_max = min(limits)
        if bw_limits: u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_total_cost(self, plan, U_max):
        node_usage = defaultdict(lambda: [0, 0])
        for f, n in plan:
            req = self.config["demands"][f]
            node_usage[n][0] += req[0] * U_max
            node_usage[n][1] += req[1] * U_max

        comp_cost = sum(
            u[0] * self.config["cost_params"]["gpu"] + u[1] * self.config["cost_params"]["memory"]
            for u in node_usage.values()
        )

        comm_cost = 0
        for i in range(1, len(plan)):
            from_n = plan[i - 1][1]
            to_n = plan[i][1]
            if from_n != to_n:
                comm_cost += self.config["data_sizes"][i - 1] * self.config["cost_params"]["bandwidth"] * U_max

        return comp_cost + comm_cost

    def build_optimization_tree(self):
        root = EnhancedTreeNode(None, 0, -1, self.config)
        queue = deque([root])

        while queue:
            current = queue.popleft()

            if current.level == self.config["function_count"] - 1:
                current.U_max = self.calculate_u_max(current.deployment_plan)
                current.is_valid = current.U_max >= 1
                if current.is_valid:
                    self.evaluate_final_plan(current)
                continue

            next_level = current.level + 1
            req = self.config["demands"][next_level]

            for n in self.physical_nodes:
                used_gpu = sum(self.config["demands"][f][0] for f, node in current.deployment_plan if node == n)
                used_mem = sum(self.config["demands"][f][1] for f, node in current.deployment_plan if node == n)
                new_gpu = used_gpu + req[0]
                new_mem = used_mem + req[1]

                total = self.config["computation"][n - 1]
                valid = new_gpu <= total[0] and new_mem <= total[1]

                child = EnhancedTreeNode(n, self.node_counter, next_level, self.config, current)
                self.node_counter += 1
                child.is_valid = valid
                if valid: queue.append(child)
                current.children.append(child)

        return self.best_node

    def evaluate_final_plan(self, node):
        node.total_cost = self.calculate_total_cost(node.deployment_plan, node.U_max)
        node.final_profit = node.U_max * self.config["cost_params"]["profit"] - node.total_cost
        self.feasible_nodes.append(node)

        if node.final_profit > self.best_profit:
            self.best_profit = node.final_profit
            self.best_node = node
