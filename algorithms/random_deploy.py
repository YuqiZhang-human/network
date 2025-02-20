import random
from collections import defaultdict


class RandomDeploymentOptimizer:
    def __init__(self, config):
        self.node_count = int(config["node_count"])
        self.computation = [
            list(map(int, row.split(',')))
            for row in config["computation_matrix"].split(';')
        ]
        self.function_count = int(config["function_count"])
        self.demands = [
            list(map(int, row.split(',')))
            for row in config["demand_matrix"].split(';')
        ]
        self.data_sizes = list(map(int, config["data_sizes"].split(',')))
        self.bandwidth = [
            list(map(int, row.split(',')))
            for row in config["bandwidth_matrix"].split(';')
        ]
        self.cost_params = {
            "gpu": float(config["gpu_cost"]),
            "memory": float(config["memory_cost"]),
            "bandwidth": float(config["bandwidth_cost"]),
            "profit": int(config["profit_per_user"])
        }
        self.physical_nodes = list(range(1, self.node_count + 1))

    def calculate_u_max(self, plan):
        node_demands = defaultdict(lambda: [0, 0])
        for f, n in plan:
            req = self.demands[f]
            node_demands[n][0] += req[0]
            node_demands[n][1] += req[1]

        limits = []
        for n in self.physical_nodes:
            total = self.computation[n - 1]
            used = node_demands[n]
            u_gpu = total[0] // used[0] if used[0] else float('inf')
            u_mem = total[1] // used[1] if used[1] else float('inf')
            limits.append(min(u_gpu, u_mem))

        bw_limits = []
        for i in range(1, len(plan)):
            from_n = plan[i - 1][1]
            to_n = plan[i][1]
            if from_n != to_n:
                data = self.data_sizes[i - 1]
                bw = self.bandwidth[from_n - 1][to_n - 1]
                bw_limits.append(bw // data if data else float('inf'))

        u_max = min(limits)
        if bw_limits: u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_cost(self, plan, u_max):
        if u_max < 1: return float('inf')

        node_usage = defaultdict(lambda: [0, 0])
        for f, n in plan:
            req = self.demands[f]
            node_usage[n][0] += req[0] * u_max
            node_usage[n][1] += req[1] * u_max

        comp_cost = sum(
            u[0] * self.cost_params["gpu"] + u[1] * self.cost_params["memory"]
            for u in node_usage.values()
        )

        comm_cost = 0
        for i in range(1, len(plan)):
            from_n = plan[i - 1][1]
            to_n = plan[i][1]
            if from_n != to_n:
                comm_cost += self.data_sizes[i - 1] * self.cost_params["bandwidth"] * u_max

        return comp_cost + comm_cost

    def optimize(self, max_trials=1000):
        for _ in range(max_trials):
            plan = [(f, random.choice(self.physical_nodes)) for f in range(self.function_count)]
            u_max = self.calculate_u_max(plan)
            if u_max >= 1:
                cost = self.calculate_cost(plan, u_max)
                return {
                    "deployment": plan,
                    "u_max": u_max,
                    "cost": cost,
                    "profit": u_max * self.cost_params["profit"] - cost
                }
        return None
