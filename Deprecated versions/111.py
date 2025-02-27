def validate_path_connectivity(self, plan):
    """验证完整路径的连通性"""
    for i in range(1, len(plan)):
        from_node = plan[i - 1][1]
        to_node = plan[i][1]
        if from_node == to_node:
            continue
        if self.bandwidth_matrix[from_node - 1][to_node - 1] <= 0:
            print(f"路径中断：节点 {from_node} 无法连接到 {to_node}")
            return False
    return True