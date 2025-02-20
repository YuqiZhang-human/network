from collections import defaultdict


class MemoryFirstOptimizer:
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
        for func_idx, node_id in plan:
            req_gpu, req_mem = self.demands[func_idx]
            node_demands[node_id][0] += req_gpu
            node_demands[node_id][1] += req_mem

        comp_limits = []
        for node_id in self.physical_nodes:
            total_gpu, total_mem = self.computation[node_id - 1]
            d_gpu, d_mem = node_demands[node_id]
            u_gpu = total_gpu // d_gpu if d_gpu else float('inf')
            u_mem = total_mem // d_mem if d_mem else float('inf')
            comp_limits.append(min(u_gpu, u_mem))

        bw_limits = []
        for i in range(1, len(plan)):
            from_node = plan[i - 1][1]
            to_node = plan[i][1]
            if from_node != to_node:
                data = self.data_sizes[i - 1]
                bw = self.bandwidth[from_node - 1][to_node - 1]
                bw_limits.append(bw // data if data else float('inf'))

        u_max = min(comp_limits)
        if bw_limits: u_max = min(u_max, min(bw_limits))
        return u_max if u_max >= 1 else 0

    def calculate_cost(self, plan, u_max):
        if u_max < 1: return float('inf')

        node_usage = defaultdict(lambda: [0, 0])
        for func_idx, node_id in plan:
            req_gpu, req_mem = self.demands[func_idx]
            node_usage[node_id][0] += req_gpu * u_max
            node_usage[node_id][1] += req_mem * u_max

        comp_cost = sum(
            used_gpu * self.cost_params["gpu"] + used_mem * self.cost_params["memory"]
            for used_gpu, used_mem in node_usage.values()
        )

        comm_cost = 0
        for i in range(1, len(plan)):
            from_node = plan[i - 1][1]
            to_node = plan[i][1]
            if from_node != to_node:
                comm_cost += self.data_sizes[i - 1] * self.cost_params["bandwidth"] * u_max

        return comp_cost + comm_cost

    def optimize(self):
        mem_usage = defaultdict(int)
        plan = []

        for func_idx in range(self.function_count):
            req_gpu, req_mem = self.demands[func_idx]

            candidates = []
            for node_id in self.physical_nodes:
                remaining = self.computation[node_id - 1][1] - mem_usage[node_id]
                if remaining >= req_mem:
                    candidates.append((node_id, remaining))

            if not candidates: return None

            selected_node = max(candidates, key=lambda x: x[1])[0]
            plan.append((func_idx, selected_node))
            mem_usage[selected_node] += req_mem

        u_max = self.calculate_u_max(plan)
        if u_max < 1: return None

        cost = self.calculate_cost(plan, u_max)
        profit = u_max * self.cost_params["profit"] - cost

        return {
            "deployment": plan,
            "u_max": u_max,
            "cost": cost,
            "profit": profit
        }
