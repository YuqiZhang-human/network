from algorithms.multi_func import MultiFuncOptimizer


class CostMinimizationOptimizer(MultiFuncOptimizer):
    def __init__(self, config):
        super().__init__(config)
        self.feasible_nodes = []

    def evaluate_final_plan(self, node):
        super().evaluate_final_plan(node)
        self.feasible_nodes.append(node)

    def get_optimal(self):
        if not self.feasible_nodes:
            return None
        # 按成本升序，利润降序排序
        sorted_nodes = sorted(self.feasible_nodes,
                              key=lambda x: (x.total_cost, -x.final_profit))
        best = sorted_nodes[0]
        return {
            "deployment": best.deployment_plan,
            "u_max": best.U_max,
            "cost": best.total_cost,
            "profit": best.final_profit
        }
