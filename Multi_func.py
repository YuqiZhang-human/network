import json
import ast
import pandas as pd
from pyvis.network import Network
from collections import deque, defaultdict
from copy import deepcopy
from colorama import init
import os

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

    def get_actual_resources(self, u_max):
        """根据U_max计算实际资源消耗"""
        actual_resources = {}
        node_count = self.config["node_count"]
        for node_id in range(1, node_count + 1):
            total_gpu, total_mem = self.config["computation_capacity"][node_id - 1]
            actual_resources[node_id] = [total_gpu, total_mem]

        for func_idx, node_id in self.deployment_plan:
            req_gpu, req_mem = self.config["resource_demands"][func_idx]
            actual_resources[node_id][0] -= req_gpu * u_max
            actual_resources[node_id][1] -= req_mem * u_max
        return actual_resources


class EnhancedDeploymentOptimizer:
    def __init__(self, config_data, visualize=True):
        self.cost_params = None
        self.bw_matrix = None
        self.data_sizes = None
        self.function_demands = None
        self.total_resources = None
        self.physical_nodes = None
        self.config = None
        self.load_config(config_data)
        self.node_counter = 1
        self.best_profit = -float('inf')
        self.best_node = None
        self.max_user_node=None
        self.min_cost_node=None
        self.all_solutions = []
        self.visualize = visualize  # 控制是否显示可视化
        self.test_result_id = config_data.get('test_data_id', 1)  # 存储测试ID

    def load_config(self, config_data):
        # # 读取配置文件
        # df = pd.read_csv(config_data)
        # config = df.iloc[0].to_dict()  # 假设配置数据在第一行
        #
        # # 将字符串表示的列表转换为实际的列表
        # config["computation_capacity"] = ast.literal_eval(config["computation_capacity"])
        # config["resource_demands"] = ast.literal_eval(config["resource_demands"])
        # config["bandwidth_matrix"] = ast.literal_eval(config["bandwidth_matrix"])
        #
        # # 强制转换数据类型：确保每个字段的数据类型正确
        # config["node_count"] = int(config["node_count"])  # 转换节点数量为整数
        # config["computation_capacity"] = [list(map(int, item)) for item in
        #                                   config["computation_capacity"]]  # 确保每个元素为整数列表
        # config["resource_demands"] = [list(map(int, item)) for item in config["resource_demands"]]  # 确保每个元素为整数列表
        # config["bandwidth_matrix"] = [list(map(int, item)) for item in config["bandwidth_matrix"]]  # 确保带宽矩阵为整数列表
        #
        # # 使用 ast.literal_eval 安全地转换字符串形式的列表
        # config["data_sizes"] = ast.literal_eval(config["data_sizes"])  # 转换为列表
        # config["data_sizes"] = [float(item) for item in config["data_sizes"]]  # 确保每个元素为浮动类型
        #
        # config["gpu_cost"] = float(config["gpu_cost"])  # 转换 GPU 成本为浮动类型
        # config["memory_cost"] = float(config["memory_cost"])  # 转换内存成本为浮动类型
        # config["bandwidth_cost"] = float(config["bandwidth_cost"])  # 转换带宽成本为浮动类型
        # config["profit_per_user"] = float(config["profit_per_user"])  # 转换每个用户利润为浮动类型
        #
        # # 将配置数据赋值给实例变量
        # self.config = config
        # self.physical_nodes = list(range(1, config["node_count"] + 1))
        # self.total_resources = {n: config["computation_capacity"][n - 1] for n in self.physical_nodes}
        # self.function_demands = config["resource_demands"]
        # self.data_sizes = config["data_sizes"]
        # self.bw_matrix = config["bandwidth_matrix"]
        # self.cost_params = {
        #     "gpu_cost": config["gpu_cost"],
        #     "memory_cost": config["memory_cost"],
        #     "bandwidth_cost": config["bandwidth_cost"],
        #     "profit_per_user": config["profit_per_user"]
        # }
        try:
            config = {
                "node_count": int(config_data.get("node_count", 0)),
                "computation_capacity": ast.literal_eval(config_data.get("computation_capacity", "[]")),
                "resource_demands": ast.literal_eval(config_data.get("resource_demands", "[]")),
                "bandwidth_matrix": ast.literal_eval(config_data.get("bandwidth_matrix", "[]")),
                "data_sizes": list(map(float, ast.literal_eval(config_data.get("data_sizes", "[]")))),
                "gpu_cost": float(config_data.get("gpu_cost", 0)),
                "memory_cost": float(config_data.get("memory_cost", 0)),
                "bandwidth_cost": float(config_data.get("bandwidth_cost", 0)),
                "profit_per_user": float(config_data.get("profit_per_user", 0))
            }
        except Exception as e:
            raise ValueError(f"配置文件解析失败: {e}")

        self.config = config
        self.physical_nodes = list(range(1, config["node_count"] + 1))
        self.total_resources = {n: config["computation_capacity"][n - 1] for n in self.physical_nodes}
        self.function_demands = config["resource_demands"]
        self.data_sizes = config["data_sizes"]
        self.bw_matrix = config["bandwidth_matrix"]
        self.cost_params = {
            "gpu_cost": config["gpu_cost"],
            "memory_cost": config["memory_cost"],
            "bandwidth_cost": config["bandwidth_cost"],
            "profit_per_user": config["profit_per_user"]
        }

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
                #更新用户量最大的可用节点
                self.update_max_user_node(current_node)
                #更新成本最低的可用节点
                self.update_min_cost_node(current_node)
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
        # 计算最终成本
        node.total_cost = self.calculate_total_cost(node.deployment_plan, node.U_max)
        node.final_profit = node.U_max * self.cost_params["profit_per_user"] - node.total_cost

        #输出多功能部署的完整方案  将所有的部署方案合并 表格填充形式
        node_multi_func_all_solve_info = [node.total_cost, node.final_profit, node.U_max, node.deployment_plan]
        self.all_solutions.append(node_multi_func_all_solve_info)
        print(f"node_multi_func_all_solve_info: {node_multi_func_all_solve_info}")

        # 更新最优解
        if node.final_profit > self.best_profit:
            self.best_profit = node.final_profit
            self.best_node = node

        # 更新最大用户量方案
        self.update_max_user_node(node)

    def update_max_user_node(self, node):
        """更新最大用户量的节点"""
        if self.max_user_node is None or node.U_max > self.max_user_node.U_max:
            self.max_user_node = node

    def update_min_cost_node(self, node):
        """更新成本最低的节点"""
        if self.min_cost_node is None or node.total_cost < self.min_cost_node.total_cost:
            self.min_cost_node = node

    def print_max_user_plan(self):
        """打印最大用户量的部署方案"""
        if self.max_user_node:
            print("\n===== 用户量最大部署方案 =====")
            print(f"部署方案: {self.max_user_node.deployment_plan}")
            print(f"最大用户量: {self.max_user_node.U_max}")
            print(f"总成本: {self.max_user_node.total_cost:.2f}$")
            print(f"最终利润: {self.max_user_node.final_profit:.2f}$\n")

            # 输出最大用户量的部署方案和相关信息 表格填充形式
            node_multi_func_max_user_info = [
                round(self.max_user_node.total_cost, 2),  # 将 total_cost 四舍五入为两位小数
                round(self.max_user_node.final_profit, 2),  # 将 final_profit 四舍五入为两位小数
                self.max_user_node.U_max,
                self.max_user_node.deployment_plan
            ]
            print(f"node_multi_func_max_user_info: {node_multi_func_max_user_info}")
            # 输出资源消耗详情
            print("资源消耗详情:")
            actual_resources = self.max_user_node.get_actual_resources(self.max_user_node.U_max)
            for node_id in actual_resources:
                total_gpu, total_mem = self.total_resources[node_id]
                used_gpu = total_gpu - actual_resources[node_id][0]
                used_mem = total_mem - actual_resources[node_id][1]
                print(f"节点{node_id}: {used_gpu}/{total_gpu} GPU | {used_mem}/{total_mem} MEM")
        else:
            print("没有找到最大用户量的部署方案")

    def print_min_cost_plan(self):
        """打印成本最低的部署方案"""
        if self.min_cost_node:
            print("\n===== 成本最低部署方案 =====")
            print(f"部署方案: {self.min_cost_node.deployment_plan}")
            print(f"最大用户量: {self.min_cost_node.U_max}")
            print(f"总成本: {self.min_cost_node.total_cost:.2f}$")
            print(f"最终利润: {self.min_cost_node.final_profit:.2f}$\n")

            # 输出最小成本的部署方案和相关信息 表格填充形式
            node_multi_func_min_cost_info = [
                round(self.min_cost_node.total_cost, 2),  # 将 total_cost 四舍五入为两位小数
                round(self.min_cost_node.final_profit, 2),  # 将 final_profit 四舍五入为两位小数
                self.min_cost_node.U_max,
                self.min_cost_node.deployment_plan
            ]
            print(f"node_multi_func_min_cost_info: {node_multi_func_min_cost_info}")

            # 输出资源消耗详情
            print("资源消耗详情:")
            actual_resources = self.min_cost_node.get_actual_resources(self.min_cost_node.U_max)
            for node_id in actual_resources:
                total_gpu, total_mem = self.total_resources[node_id]
                used_gpu = total_gpu - actual_resources[node_id][0]
                used_mem = total_mem - actual_resources[node_id][1]
                print(f"节点{node_id}: {used_gpu}/{total_gpu} GPU | {used_mem}/{total_mem} MEM")
        else:
            print("没有找到成本最低的部署方案")

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
        # 输出最大利润的部署方案和相关信息 表格填充形式
        node_multi_func_max_profit_info = [
            round(node.total_cost, 2),  # 将 total_cost 四舍五入为两位小数
            round(node.final_profit, 2),  # 将 final_profit 四舍五入为两位小数
            node.U_max,
            node.deployment_plan
        ]
        print(f"node_multi_func_max_profit_info: {node_multi_func_max_profit_info}")

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

    def print_all_solutions(self):
        """打印所有收集到的解"""
        print("\n===== 所有方案 =====")
        print(self.all_solutions)

    def _format_solutions(self, solutions):
        """格式化所有解决方案为元组列表"""
        return [
            [
                round(sol[0], 2),  # 总成本
                round(sol[1], 2),  # 最终利润
                sol[2],  # U_max
                sol[3]  # 部署方案
            ] for sol in solutions
        ]

    def _format_node_info(self, node):
        """将字典格式转换为无标签元组格式"""
        if node is None:
            return None
        # 返回顺序：总成本、最终利润、最大用户量、部署方案
        return [
            round(node.total_cost, 2),
            round(node.final_profit, 2),
            node.U_max,
            node.deployment_plan
        ]

    def export_results(self):
        """改进的导出方法"""
        results = {
            'test_result_id': self.test_result_id,
            'multi_func_all_solve': self._safe_serialize(
                self._format_solutions(self.all_solutions)
            ),
            'multi_func_max_profit': self._safe_serialize(
                self._format_node_info(self.best_node)
            ),
            'multi_func_max_user': self._safe_serialize(
                self._format_node_info(self.max_user_node)
            ),
            'multi_func_min_cost': self._safe_serialize(
                self._format_node_info(self.min_cost_node)
            )
        }

        # 通过自定义序列化生成指定格式
        output_file = 'test_solutions.csv'
        df = pd.DataFrame([results])

        # 将数据转换为目标字符串格式
        str_columns = ['multi_func_all_solve', 'multi_func_max_profit',
                       'multi_func_max_user', 'multi_func_min_cost']

        for col in str_columns:
            df[col] = df[col].apply(
                lambda x: "[ " + x[1:-1].replace('"', '') + " ]"
                if isinstance(x, str) and x.startswith('[')
                else x
            )

        # 文件写入逻辑（与之前相同）
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            existing_df = existing_df[existing_df['test_result_id'] != self.test_result_id]
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(output_file, index=False)

    def _safe_serialize(self, data):
        """安全的序列化方法"""
        if data is None:
            return None
        try:
            return json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
        except TypeError:
            return str(data)

# 主程序
if __name__ == "__main__":
    # # 控制是否开启可视化
    # optimizer = EnhancedDeploymentOptimizer("test_data.csv", visualize=True)
    # optimizer.build_optimization_tree()
    # optimizer.print_max_user_plan()
    # optimizer.print_min_cost_plan()
    # optimizer.print_all_solutions()

    # 从test_data.csv读取
    test_data = pd.read_csv("test_data.csv")

    # 为每一组测试数据生成结果
    for _, row in test_data.iterrows():
        try:
            print(f"正在处理测试用例 {row['test_data_id']}")
            optimizer = EnhancedDeploymentOptimizer(config_data=row.to_dict(), visualize=False)
            optimizer.build_optimization_tree()
            optimizer.print_max_user_plan()
            optimizer.print_min_cost_plan()
            optimizer.print_all_solutions()
            optimizer.export_results()
            print(f"测试用例 {row['test_data_id']} 处理成功\n")
        except Exception as e:
            print(f"用例 {row['test_data_id']} 处理失败: {str(e)}\n")