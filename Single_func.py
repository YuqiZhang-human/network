import json
from pyvis.network import Network
from collections import deque
from colorama import Fore, Style, init

# 初始化颜色输出
init(autoreset=True)


class TreeNode:
    def __init__(self, value, node_id, level, parent=None, used_nodes=None):
        """
        :param value: 当前层要部署的物理节点ID (本功能放在哪个物理节点)
        :param node_id: 当前节点在可视化中的唯一ID
        :param level: 当前功能索引(0-based)
        :param parent: 父节点
        :param used_nodes: 已使用的物理节点集合（若只允许一个功能对应一个物理节点，则需要）
        """
        self.value = value
        self.node_id = node_id
        self.level = level
        self.parent = parent

        # 若需要防止重复使用节点，就在此更新 used_nodes
        self.used_nodes = set(used_nodes) if used_nodes else set()
        if value is not None:
            self.used_nodes.add(value)

        # 部署链: [(func_idx, node_id), ...]
        if parent is not None:
            self.deployment_plan = parent.deployment_plan.copy()
        else:
            self.deployment_plan = []
        if value is not None and level >= 0:
            self.deployment_plan.append((level, value))

        # 当前节点的子节点列表
        self.children = []

        # ---- 部分/最终指标 ----
        # partial U_max (只到当前为止的资源+带宽限制), 方便可视化
        self.U_max = 0

        # 只有在完整部署时(最后一层)才会被赋值
        self.final_cost = None
        self.final_profit = None
        self.final_Umax = None


class DeploymentOptimizer:
    def __init__(self, config_file,visualize=False):
        self.load_config(config_file)
        self.node_counter = 1  # 给TreeNode分配ID
        self.visualize = visualize
        # 全局最优
        self.best_profit = -float('inf')
        self.best_node = None

    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        # 物理节点信息
        self.physical_nodes = list(range(1, config["node_settings"]["node_count"] + 1))
        self.computation_capacity = config["node_settings"]["computation_capacity"]

        # 功能需求
        self.function_demands = config["function_settings"]["resource_demands"]  # [(gpu,mem), (gpu,mem), ...]
        self.data_sizes = config["function_settings"]["data_sizes"]              # [data_0, data_1, ...]

        # 带宽矩阵
        self.bandwidth_matrix = config["network_settings"]["bandwidth_matrix"]

        # 成本参数
        self.cost_params = config["cost_settings"]

    # -------------------------------------------------------------------------
    #  1) 计算“部分”最大用户量，用于中间节点 (部署不完整时)
    # -------------------------------------------------------------------------
    def calc_partial_u_max(self, deployment_plan):
        """
        给定一个尚未完成的部署计划 (部分功能)，计算基于资源和已用链路的当前U_max
        仅用于可视化或剪枝中的“参考值”，不计算成本和利润
        """
        # 1) 计算资源限制
        #    统计每个物理节点上，当前部署的功能资源之和
        usage_per_node = {}
        for (func_idx, phys_node) in deployment_plan:
            req_gpu, req_mem = self.function_demands[func_idx]
            if phys_node not in usage_per_node:
                usage_per_node[phys_node] = [0, 0]
            usage_per_node[phys_node][0] += req_gpu
            usage_per_node[phys_node][1] += req_mem

        comp_limits = []
        for node_id, (used_gpu, used_mem) in usage_per_node.items():
            total_gpu, total_mem = self.computation_capacity[node_id - 1]
            if used_gpu > total_gpu or used_mem > total_mem:
                # 已经超出该节点资源，视为0
                return 0
            else:
                comp_limits.append(total_gpu // used_gpu if used_gpu else float('inf'))
                comp_limits.append(total_mem // used_mem if used_mem else float('inf'))

        # 2) 计算带宽限制(只考虑已部署功能间的链路)
        bw_limits = []
        # 先按功能索引排序
        plan_sorted = sorted(deployment_plan, key=lambda x: x[0])
        for i in range(1, len(plan_sorted)):
            prev_func_idx, prev_node = plan_sorted[i - 1]
            curr_func_idx, curr_node = plan_sorted[i]
            bw = self.bandwidth_matrix[prev_node - 1][curr_node - 1]
            data = self.data_sizes[prev_func_idx]
            if data > 0:
                bw_limits.append(bw // data)
            else:
                bw_limits.append(float('inf'))

        if not comp_limits:
            # 如果还没有功能放置，返回∞
            return float('inf')

        partial_u_max = min(comp_limits)
        if bw_limits:
            partial_u_max = min(partial_u_max, min(bw_limits))

        return partial_u_max

    # -------------------------------------------------------------------------
    #  2) 当一个部署方案完整时，计算最终U_max、成本和利润并更新全局最优
    # -------------------------------------------------------------------------
    def evaluate_final_plan(self, node):
        """
        node.deployment_plan 已包含所有功能 [(func_idx, node_id), ...]
        这里计算最终U_max、总成本、总利润；打印日志；若优于全局最优，则更新
        """
        plan_sorted = sorted(node.deployment_plan, key=lambda x: x[0])
        plan_nodes = [phys_node for (_, phys_node) in plan_sorted]

        # ---- 打印标题 ----
        print(Fore.CYAN + "\n" + "="*60)
        print(Fore.YELLOW + "[完整方案评估]")
        print(Fore.CYAN + "="*60 + Style.RESET_ALL)

        # ---- 1) 计算资源限制U_max ----
        final_comp_limit = self.log_computation_limits(plan_nodes)
        # ---- 2) 计算带宽限制U_max ----
        final_comm_limit = self.log_communication_limits(plan_nodes)

        final_u_max = min(final_comp_limit, final_comm_limit)

        # ---- 3) 计算成本 ----
        total_cost = self.log_cost_calculation(plan_nodes, final_u_max)

        # ---- 4) 计算最终利润 ----
        final_profit = final_u_max * self.cost_params["profit_per_user"] - total_cost
        print(Fore.GREEN + f"\n【综合结果】 U_max = {final_u_max}, 总成本 = {total_cost:.2f}$, 最终利润 = {final_profit:.2f}$\n")

        # ---- 存入 node，用于可视化显示 ----
        node.final_Umax = final_u_max
        node.final_cost = total_cost
        node.final_profit = final_profit

        # ---- 若优于全局最优，则更新 ----
        if final_profit > self.best_profit:
            self.best_profit = final_profit
            self.best_node = node

    # -------------------------------------------------------------------------
    #  3) 日志函数 (只在完整方案时调用)
    # -------------------------------------------------------------------------
    def log_computation_limits(self, plan):
        """
        打印并返回 基于计算资源的用户上限
        plan: [node_for_func0, node_for_func1, ...]
        """
        print(Fore.BLUE + "\n[计算资源限制分析]")
        limits = []
        # 逐节点统计
        used_dict = {}
        for func_idx, node_id in enumerate(plan):
            req_gpu, req_mem = self.function_demands[func_idx]
            if node_id not in used_dict:
                used_dict[node_id] = [0, 0]
            used_dict[node_id][0] += req_gpu
            used_dict[node_id][1] += req_mem

        for node_id, (used_gpu, used_mem) in used_dict.items():
            cap_gpu, cap_mem = self.computation_capacity[node_id - 1]
            gpu_limit = cap_gpu // used_gpu if used_gpu else float('inf')
            mem_limit = cap_mem // used_mem if used_mem else float('inf')

            print(f"  节点{node_id}: GPU需求 {used_gpu}/{cap_gpu} => 用户上限 {gpu_limit}")
            print(f"          内存需求 {used_mem}/{cap_mem} => 用户上限 {mem_limit}")
            limits.append(gpu_limit)
            limits.append(mem_limit)

        return min(limits) if limits else float('inf')

    def log_communication_limits(self, plan):
        """
        打印并返回 基于带宽的用户上限
        plan: [node_for_func0, node_for_func1, ...]
        """
        print(Fore.BLUE + "\n[通信资源限制分析]")
        limits = []
        for i in range(1, len(plan)):
            from_node = plan[i - 1]
            to_node = plan[i]
            bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
            data = self.data_sizes[i - 1]  # 功能 i-1 处理后要传给功能 i
            user_limit = bw // data if data else float('inf')
            print(f"  链路 {from_node}->{to_node}: 带宽 {bw}MBps / 数据 {data}MB => 用户上限 {user_limit}")
            limits.append(user_limit)

        return min(limits) if limits else float('inf')

    def log_cost_calculation(self, plan, U_max):
        """
        打印并返回 总成本 (计算+通信)
        plan: [node_for_func0, node_for_func1, ...]
        U_max: 并发用户数
        """
        print(Fore.BLUE + "\n[成本计算过程]")
        comp_cost = 0
        print(Fore.GREEN + "计算成本:")
        used_dict = {}
        for func_idx, node_id in enumerate(plan):
            req_gpu, req_mem = self.function_demands[func_idx]
            if node_id not in used_dict:
                used_dict[node_id] = [0, 0]
            used_dict[node_id][0] += req_gpu
            used_dict[node_id][1] += req_mem

        for node_id, (gpu_used, mem_used) in used_dict.items():
            node_cost = U_max * gpu_used * self.cost_params["gpu_cost"] + \
                        U_max * mem_used * self.cost_params["memory_cost"]
            print(f"  节点{node_id}: {U_max}用户 * {gpu_used}GPU × {self.cost_params['gpu_cost']} + "
                  f"{U_max}用户 * {mem_used}MB × {self.cost_params['memory_cost']} = {node_cost:.2f}$")
            comp_cost += node_cost

        comm_cost = 0
        print(Fore.GREEN + "\n通信成本:")
        for i in range(1, len(plan)):
            data = self.data_sizes[i - 1]
            c = data * self.cost_params["bandwidth_cost"] * U_max
            print(f"  功能{i-1}->{i}: {data}MB × {self.cost_params['bandwidth_cost']} × {U_max} = {c:.2f}$")
            comm_cost += c

        total_cost = comp_cost + comm_cost
        print(Fore.GREEN + f"\n  总成本: {total_cost:.2f}$")
        return total_cost

    # -------------------------------------------------------------------------
    #  4) BFS 构建搜索树，只有最后一层节点才调用 evaluate_final_plan
    # -------------------------------------------------------------------------
    def build_optimization_tree(self):
        """
        使用 BFS 构建搜索树。中间节点只做基本资源可行性判断和partial U_max计算；
        当某节点已完成所有功能部署时，evaluate_final_plan做最终评估。
        """
        root = TreeNode(value=None, node_id=0, level=-1, parent=None, used_nodes=set())
        root.U_max = float('inf')  # 根节点尚未部署任何功能，可视为∞
        queue = deque([(root, set())])

        while queue:
            parent_node, used_nodes = queue.popleft()

            next_level = parent_node.level + 1
            if next_level >= len(self.function_demands):
                # 已完成全部功能部署 => 整个方案
                # 这里评估完整方案
                self.evaluate_final_plan(parent_node)
                continue

            # 还需部署下一个功能
            req_gpu, req_mem = self.function_demands[next_level]

            for node_id in self.physical_nodes:
                if node_id in used_nodes:
                    # 不允许重复用物理节点
                    continue

                # 基本判断: 该节点至少能放下这一功能
                cap_gpu, cap_mem = self.computation_capacity[node_id - 1]
                if req_gpu > cap_gpu or req_mem > cap_mem:
                    continue

                # 创建子节点
                child = TreeNode(value=node_id,
                                 node_id=self.node_counter,
                                 level=next_level,
                                 parent=parent_node,
                                 used_nodes=used_nodes)
                self.node_counter += 1
                parent_node.children.append(child)

                # 计算到此为止的 partial U_max
                partial_umax = self.calc_partial_u_max(child.deployment_plan)
                child.U_max = partial_umax

                # 入队
                new_used = set(used_nodes)
                new_used.add(node_id)
                queue.append((child, new_used))

        # BFS结束后，若有最优节点，则打印
        if self.best_node:
            print("\n===== 最优部署方案 =====")
            self.print_optimal_plan(self.best_node)
        else:
            print("No valid deployment found")

        # 最后可视化
        # 根据 visualize 参数控制是否生成可视化
        if self.visualize:
            self.visualize_tree(root)

    # -------------------------------------------------------------------------
    #  5) 可视化
    # -------------------------------------------------------------------------
    def visualize_tree(self, root, best_node=None):
        """
        每个节点上方显示：
         - Node ID
         - 当前物理节点编号
         - partial U_max
         - 若是完整方案节点，则额外显示 final_Umax, final_cost, final_profit
        连线上显示：该链路可支撑的最大用户数(仅限上一功能->下一功能)
        """
        net = Network(directed=True, height="750px", width="100%", notebook=False)

        def add_nodes(curr):
            # 基本信息
            label_lines = [
                f"NodeID: {curr.node_id}",
                f"PhysNode: {curr.value}",
                f"Partial Umax: {curr.U_max}"
            ]
            # 若是完整方案（最后一层）
            if curr.final_Umax is not None:
                label_lines.append(f"Final Umax: {curr.final_Umax}")
                label_lines.append(f"Cost: {curr.final_cost:.2f}")
                label_lines.append(f"Profit: {curr.final_profit:.2f}")

            label_text = "\n".join(str(x) for x in label_lines)

            # 如果是最优方案叶子节点，则标成绿色，否则蓝色
            color = "green" if curr == best_node else "#1f78b4"
            net.add_node(curr.node_id, label=label_text, color=color)

            # 递归处理子节点
            for child in curr.children:
                # 给连线加label: 该链路支持的最大用户量
                link_label = ""
                # child.level表示要部署第 child.level 个功能
                if curr.value is not None and child.value is not None and curr.level >= 0:
                    # parent是功能 curr.level, child是功能 child.level
                    # 对应 data_sizes[curr.level], 带宽
                    data_size = self.data_sizes[curr.level]
                    bw = self.bandwidth_matrix[curr.value - 1][child.value - 1]
                    link_user_limit = float('inf')
                    if data_size > 0:
                        link_user_limit = bw // data_size
                    link_label = f"Link Limit: {link_user_limit}"

                add_nodes(child)
                net.add_edge(curr.node_id, child.node_id, label=link_label)

        add_nodes(root)
        net.show("optimization_tree.html", notebook=False)

    # -------------------------------------------------------------------------
    #  6) 打印最优方案
    # -------------------------------------------------------------------------
    def print_optimal_plan(self, node):
        plan_sorted = sorted(node.deployment_plan, key=lambda x: x[0])
        plan_nodes = [n for (_, n) in plan_sorted]

        print("最优部署方案:")
        for func_idx, phys_node in plan_sorted:
            req_gpu, req_mem = self.function_demands[func_idx]
            print(f"  功能 {func_idx} -> 节点 {phys_node} (GPU={req_gpu}, MEM={req_mem})")

        print("\n通信链路:")
        for i in range(len(plan_nodes) - 1):
            from_node = plan_nodes[i]
            to_node = plan_nodes[i + 1]
            data_size = self.data_sizes[i]
            bw = self.bandwidth_matrix[from_node - 1][to_node - 1]
            link_limit = bw // data_size if data_size else float('inf')
            print(f"  {from_node}->{to_node} : 带宽 {bw}MBps, 数据 {data_size}MB, 链路用户上限 {link_limit}")

        print(f"\n最终U_max: {node.final_Umax}")
        print(f"总成本: {node.final_cost:.2f}$")
        print(f"最终利润: {node.final_profit:.2f}$")


# ================ 主程序入口 =================
if __name__ == "__main__":
    optimizer = DeploymentOptimizer("data/deployment_config.json", visualize=True)
    optimizer.build_optimization_tree()
