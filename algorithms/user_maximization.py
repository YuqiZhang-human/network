from algorithms.multi_func import MultiFuncOptimizer


class UserMaximizationOptimizer(MultiFuncOptimizer):
    def __init__(self, config):
        super().__init__(config)
        self.feasible_nodes = []

    def evaluate_final_plan(self, node):
        super().evaluate_final_plan(node)
        self.feasible_nodes.append(node)

    def get_optimal(self):
        if not self.feasible_nodes:
            return None

        # 找到最大用户量
        max_u = max(n.U_max for n in self.feasible_nodes)
        candidates = [n for n in self.feasible_nodes if n.U_max == max_u]

        # 在相同用户量中选利润最高的
        best = max(candidates, key=lambda x: x.final_profit)

        return {
            "deployment": best.deployment_plan,
            "u_max": best.U_max,
            "cost": best.total_cost,
            "profit": best.final_profit
        }
