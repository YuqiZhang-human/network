import json
from pyvis.network import Network
from collections import deque, defaultdict
from copy import deepcopy
from colorama import init

# 初始化颜色输出
init(autoreset=True)


class EnhancedTreeNode:
    def __init__(self, value, node_id, level, config, parent=None):
        """
        :param value: 当前层部署的物理节点ID
        :param node_id: 节点唯一标识
        :param level: 当前功能索引(0-based)
        :param config: 配置文件数据
        :param parent: 父节点
        """
        self.value = value
        self.node_id = node_id
        self.level = level
        self.parent = parent
        self.config = config

        # 部署记录 [(func_idx, node_id)]
        self.deployment_plan = parent.deployment_plan.copy() if parent else []
        if value is not None and level >= 0:
            self.deployment_plan.append((level, value))

        # 子节点列表
        self.children = []

        # 指标数据
        self.is_valid = True
        self.U_max = 0
        self.total_cost = 0
        self.final_profit = 0

    def get_actual_resources(self, U_max):
        """根据U_max计算实际资源消耗"""
        actual_resources = {}
        node_count = self.config["node_settings"]["node_count"]
        for node_id in range(1, node_count + 1):
            total_gpu, total_mem = self.config["node_settings"]["computation_capacity"][node_id - 1]
            actual_resources[node_id] = [total_gpu, total_mem]

        for func_idx, node_id in self.deployment_plan:
            req_gpu, req_mem = self.config["function_settings"]["resource_demands"][func_idx]
            actual_resources[node_id][0] -= req_gpu * U_max
            actual_resources[node_id][1] -= req_mem * U_max
        return actual_resources


class EnhancedDeploymentOptimizer:
    def __init__(self, config_file, visualize=True):
        self.load_config(config_file)
        self.node_counter = 1
        self.best_profit = -float('inf')
        self.best_node = None
        self.visualize = visualize  # 控制是否显示可视化

    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)
        self.physical_nodes = list(range(1, self.config["node_settings"]["node_count"] + 1))
        self.total_resources = {n: self.config["node_settings"]["computation_capacity"][n - 1] for n in
                                self.physical_nodes}
        self.function_demands = self.config["function_settings"]["resource_demands"]
        self.data_sizes = self.config["function_settings"]["data_sizes"]
        self.bw_matrix = self.config["network_settings"]["bandwidth_matrix"]
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
            if demand_gpu == 0 and demand_mem == 0:
                continue
            if demand_gpu == 0:
                u_gpu = float('inf')
            else:
                u_gpu = total_gpu // demand_gpu
            if demand_mem == 0:
                u_mem = float('inf')
            else:
                u_mem = total_mem // demand_mem
            comp_limits.append(min(u_gpu, u_mem))

        # 计算通信资源限制
        bw_limits = []
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data_size = self.data_sizes[i - 1]
                bw = self.bw_matrix[from_node - 1][to_node - 1]
                if data_size == 0:
                    bw_limit = float('inf')
                else:
                    bw_limit = bw // data_size
                bw_limits.append(bw_limit)

        u_max = min(comp_limits) if comp_limits else 0
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max

    def build_optimization_tree(self):
        """构建优化树（修正后的逻辑）"""
        root = EnhancedTreeNode(value=None, node_id=0, level=-1, config=self.config)
        queue = deque([root])  # 队列只需要节点，不需要传递U_max

        while queue:
            current_node = queue.popleft()

            # 终止条件：到达最后一个功能层时计算U_max和利润
            if current_node.level == len(self.function_demands) - 1:
                # 计算实际U_max（基于初始资源）
                current_node.U_max = self.calculate_u_max(current_node.deployment_plan)
                current_node.is_valid = current_node.U_max >= 1
                if current_node.is_valid:
                    self.evaluate_final_plan(current_node)
                continue

            # 处理下一层功能
            next_level = current_node.level + 1
            req_gpu, req_mem = self.function_demands[next_level]

            for node_id in self.physical_nodes:
                # 检查是否满足用户量1的资源需求（不实际扣除资源）
                total_gpu = self.total_resources[node_id][0]
                total_mem = self.total_resources[node_id][1]

                # 统计该节点已部署功能的总需求
                used_gpu = sum(self.function_demands[func_idx][0] for func_idx, n_id in current_node.deployment_plan if
                               n_id == node_id)
                used_mem = sum(self.function_demands[func_idx][1] for func_idx, n_id in current_node.deployment_plan if
                               n_id == node_id)

                # 新增当前功能的资源需求（用户量1）
                new_used_gpu = used_gpu + req_gpu * 1
                new_used_mem = used_mem + req_mem * 1

                # 检查是否超过初始资源
                if new_used_gpu > total_gpu or new_used_mem > total_mem:
                    # 资源不足，创建无效节点（灰色）
                    child = EnhancedTreeNode(
                        value=node_id,
                        node_id=self.node_counter,
                        level=next_level,
                        config=self.config,
                        parent=current_node
                    )
                    child.is_valid = False
                    self.node_counter += 1
                    current_node.children.append(child)
                    continue

                # 资源足够，创建有效节点
                child = EnhancedTreeNode(
                    value=node_id,
                    node_id=self.node_counter,
                    level=next_level,
                    config=self.config,
                    parent=current_node
                )
                self.node_counter += 1
                child.is_valid = True
                queue.append(child)
                current_node.children.append(child)

        # 输出最优方案
        if self.best_node:
            print("\n===== 最优部署方案 =====")
            self.print_optimal_plan(self.best_node)
        else:
            print("没有找到可行方案")

        # 根据 visualize 参数控制是否生成可视化
        if self.visualize:
            self.visualize_tree(root)

    def evaluate_final_plan(self, node):
        """评估完整方案"""
        if not node.is_valid:
            return

        # 输出当前节点的部署方案
        print(f"方案 {node.node_id - 20} 的部署方案: {node.deployment_plan}")

        # 计算最终成本
        node.total_cost = self.calculate_total_cost(node.deployment_plan, node.U_max)
        node.final_profit = node.U_max * self.cost_params["profit_per_user"] - node.total_cost

        # 更新最优解
        if node.final_profit > self.best_profit:
            self.best_profit = node.final_profit
            self.best_node = node

    def calculate_total_cost(self, deployment_plan, U_max):
        """精确成本计算"""
        # 计算资源成本
        node_usage = {}
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            if node_id not in node_usage:
                node_usage[node_id] = [0, 0]
            node_usage[node_id][0] += req_gpu * U_max
            node_usage[node_id][1] += req_mem * U_max

        comp_cost = 0
        for node_id, (used_gpu, used_mem) in node_usage.items():
            comp_cost += used_gpu * self.cost_params["gpu_cost"] + used_mem * self.cost_params["memory_cost"]

        # 通信成本
        comm_cost = 0
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:
                data = self.data_sizes[i - 1]
                comm_cost += data * self.cost_params["bandwidth_cost"] * U_max

        return comp_cost + comm_cost

    def print_optimal_plan(self, node):
        """打印方案详情"""
        print("部署路径:", node.deployment_plan)
        print(f"最大用户量: {node.U_max}")
        print(f"总成本: {node.total_cost:.2f}$")
        print(f"最终利润: {node.final_profit:.2f}$\n")

        print("资源消耗详情:")
        actual_resources = node.get_actual_resources(node.U_max)
        for node_id in actual_resources:
            total_gpu, total_mem = self.total_resources[node_id]
            used_gpu = total_gpu - actual_resources[node_id][0]
            used_mem = total_mem - actual_resources[node_id][1]
            print(f"节点{node_id}: {used_gpu}/{total_gpu} GPU | {used_mem}/{total_mem} MEM")

    def visualize_tree(self, root):
        """可视化树结构"""
        net = Network(directed=True, height="800px", width="100%")

        def add_nodes(node):
            label = [
                f"NodeID: {node.node_id}",
                f"PhysNode: {node.value}",
                f"Valid: {node.is_valid}",
                f"U_max: {node.U_max if node.level >= 0 else 'N/A'}"
            ]
            if node.level == len(self.function_demands) - 1:
                label.append(f"Profit: {node.final_profit:.2f}$")

            color = "#CCCCCC" if not node.is_valid else \
                "#00FF00" if node == self.best_node else "#1f78b4"

            net.add_node(node.node_id, label="\n".join(label), color=color)

            if node.is_valid:
                for child in node.children:
                    edge_label = f"Deploy F{child.level}→N{child.value}"
                    add_nodes(child)
                    net.add_edge(node.node_id, child.node_id, label=edge_label)

        add_nodes(root)
        net.show("enhanced_tree.html", notebook=False)


# 主程序
if __name__ == "__main__":
    # 控制是否开启可视化
    optimizer = EnhancedDeploymentOptimizer("deployment_config6.json", visualize=False)
    optimizer.build_optimization_tree()
