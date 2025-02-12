import json
from pyvis.network import Network
from collections import deque
from copy import deepcopy
from colorama import Fore, Style, init

# 初始化颜色输出
init(autoreset=True)


class EnhancedTreeNode:
    def __init__(self, value, node_id, level, parent=None, remaining_resources=None):
        """
        :param value: 当前层要部署的物理节点ID
        :param node_id: 节点唯一标识
        :param level: 当前功能索引(0-based)
        :param parent: 父节点
        :param remaining_resources: 各节点剩余资源 {node_id: [gpu, mem], ...}
        """
        self.value = value  # 当前部署的物理节点ID
        self.node_id = node_id
        self.level = level
        self.parent = parent

        # 剩余资源管理（深拷贝）
        self.remaining_resources = deepcopy(remaining_resources) if remaining_resources else {}
        if value is not None:
            # 初始化所有节点资源（如果是根节点）
            if parent is None:
                for node_id, (total_gpu, total_mem) in enumerate(config["node_settings"]["computation_capacity"], 1):
                    self.remaining_resources[node_id] = [total_gpu, total_mem]
            else:
                # 更新当前节点的资源消耗（假设先按1用户计算）
                req_gpu, req_mem = config["function_settings"]["resource_demands"][level]
                self.remaining_resources[value][0] -= req_gpu
                self.remaining_resources[value][1] -= req_mem

        # 部署记录 [(func_idx, node_id)]
        self.deployment_plan = parent.deployment_plan.copy() if parent else []
        if value is not None and level >= 0:
            self.deployment_plan.append((level, value))

        # 子节点列表
        self.children = []

        # 指标数据
        self.is_valid = True  # 路径是否有效
        self.U_max = 0  # 当前路径的最大用户数
        self.total_cost = 0  # 总成本
        self.final_profit = 0  # 最终利润


class EnhancedDeploymentOptimizer:
    def __init__(self, config_file):
        self.load_config(config_file)
        self.node_counter = 1
        self.best_profit = -float('inf')
        self.best_node = None

    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = json.load(f)

        # 节点初始化数据
        self.physical_nodes = list(range(1, self.config["node_settings"]["node_count"] + 1))
        self.total_resources = {n: self.config["node_settings"]["computation_capacity"][n - 1] for n in
                                self.physical_nodes}

        # 功能需求
        self.function_demands = self.config["function_settings"]["resource_demands"]
        self.data_sizes = self.config["function_settings"]["data_sizes"]

        # 带宽矩阵
        self.bw_matrix = self.config["network_settings"]["bandwidth_matrix"]

        # 经济参数
        self.cost_params = self.config["cost_settings"]

    def calculate_u_max(self, deployment_plan):
        """计算最大用户并发量"""
        # ----------------------------
        # 1. 计算资源限制
        # ----------------------------
        node_usage = {}
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            if node_id not in node_usage:
                node_usage[node_id] = [0, 0]
            node_usage[node_id][0] += req_gpu
            node_usage[node_id][1] += req_mem

        comp_limits = []
        for node_id, (used_gpu, used_mem) in node_usage.items():
            total_gpu, total_mem = self.total_resources[node_id]
            comp_limits.append(total_gpu // used_gpu)
            comp_limits.append(total_mem // used_mem)

        # ----------------------------
        # 2. 通信资源限制
        # ----------------------------
        bw_limits = []
        for i in range(1, len(deployment_plan)):
            from_node = deployment_plan[i - 1][1]
            to_node = deployment_plan[i][1]
            if from_node != to_node:  # 只有跨节点需要计算
                data = self.data_sizes[i - 1]
                bw = self.bw_matrix[from_node - 1][to_node - 1]
                bw_limits.append(bw // data)

        # ----------------------------
        # 3. 综合计算
        # ----------------------------
        u_max = min(comp_limits)
        if bw_limits:
            u_max = min(u_max, min(bw_limits))
        return u_max

    def build_optimization_tree(self):
        """构建允许节点复用的优化树"""
        root = EnhancedTreeNode(value=None, node_id=0, level=-1)
        queue = deque([root])

        while queue:
            current_node = queue.popleft()

            # 终止条件：完成所有功能部署
            if current_node.level == len(self.function_demands) - 1:
                self.evaluate_final_plan(current_node)
                continue

            # 遍历所有物理节点
            next_level = current_node.level + 1
            req_gpu, req_mem = self.function_demands[next_level]

            for node_id in self.physical_nodes:
                # 检查剩余资源是否足够支撑至少1个用户
                remaining_gpu = current_node.remaining_resources[node_id][0]
                remaining_mem = current_node.remaining_resources[node_id][1]

                if remaining_gpu >= req_gpu and remaining_mem >= req_mem:
                    # 创建子节点
                    child = EnhancedTreeNode(
                        value=node_id,
                        node_id=self.node_counter,
                        level=next_level,
                        parent=current_node,
                        remaining_resources=current_node.remaining_resources
                    )
                    self.node_counter += 1

                    # 更新剩余资源（按1用户计算）
                    child.remaining_resources[node_id][0] -= req_gpu
                    child.remaining_resources[node_id][1] -= req_mem

                    # 计算当前路径的U_max
                    try:
                        child.U_max = self.calculate_u_max(child.deployment_plan)
                    except:
                        child.is_valid = False

                    # 有效性检查
                    if child.U_max < 1:
                        child.is_valid = False

                    current_node.children.append(child)
                    queue.append(child)

        # 输出最优方案
        if self.best_node:
            print("\n===== 最优部署方案 =====")
            self.print_optimal_plan(self.best_node)
        else:
            print("没有找到可行方案")

        # 可视化
        self.visualize_tree(root)

    def evaluate_final_plan(self, node):
        """评估完整方案"""
        try:
            # 计算最终指标
            node.U_max = self.calculate_u_max(node.deployment_plan)
            node.total_cost = self.calculate_total_cost(node.deployment_plan, node.U_max)
            node.final_profit = node.U_max * self.cost_params["profit_per_user"] - node.total_cost

            # 更新最优解
            if node.final_profit > self.best_profit:
                self.best_profit = node.final_profit
                self.best_node = node
        except:
            node.is_valid = False

    def calculate_total_cost(self, deployment_plan, U_max):
        """计算总成本"""
        # 计算成本
        comp_cost = 0
        node_usage = {}
        for func_idx, node_id in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            if node_id not in node_usage:
                node_usage[node_id] = [0, 0]
            node_usage[node_id][0] += req_gpu
            node_usage[node_id][1] += req_mem

        for node_id, (total_gpu, total_mem) in node_usage.items():
            comp_cost += U_max * (
                        total_gpu * self.cost_params["gpu_cost"] + total_mem * self.cost_params["memory_cost"])

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
        """打印最优方案详情"""
        print("部署路径:", node.deployment_plan)
        print(f"最大用户量: {node.U_max}")
        print(f"总成本: {node.total_cost:.2f}$")
        print(f"最终利润: {node.final_profit:.2f}$\n")

        print("各节点资源消耗:")
        node_usage = {}
        for func_idx, node_id in node.deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            if node_id not in node_usage:
                node_usage[node_id] = [0, 0]
            node_usage[node_id][0] += req_gpu
            node_usage[node_id][1] += req_mem

        for node_id, (gpu, mem) in node_usage.items():
            print(
                f"节点{node_id}: {gpu * node.U_max}/{self.total_resources[node_id][0]} GPU | {mem * node.U_max}/{self.total_resources[node_id][1]} MEM")

    def visualize_tree(self, root):
        """可视化树结构"""
        net = Network(directed=True, height="800px", width="100%")

        def add_nodes(node):
            # 节点标签
            label = [
                f"NodeID: {node.node_id}",
                f"PhysNode: {node.value}",
                f"Valid: {node.is_valid}",
                f"U_max: {node.U_max if node.level >= 0 else 'N/A'}"
            ]
            if node.level == len(self.function_demands) - 1:
                label.append(f"Profit: {node.final_profit:.2f}$")

            # 节点颜色
            if not node.is_valid:
                color = "#ff0000"  # 红色表示无效
            elif node == self.best_node:
                color = "#00ff00"  # 绿色表示最优
            else:
                color = "#1f78b4"  # 蓝色表示普通

            net.add_node(node.node_id, label="\n".join(label), color=color)

            for child in node.children:
                # 边标签显示部署决策
                edge_label = f"Deploy F{child.level}→N{child.value}"
                add_nodes(child)
                net.add_edge(node.node_id, child.node_id, label=edge_label)

        add_nodes(root)
        net.show("enhanced_tree.html")


# 主程序
if __name__ == "__main__":
    optimizer = EnhancedDeploymentOptimizer("config.json")
    optimizer.build_optimization_tree()