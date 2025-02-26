import csv
import random
import numpy as np


def generate_test_data(n_samples: int, output_file: str):
    """
    生成增强连通性且节点数量更均衡的测试数据
    """
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'test_data_id', 'node_count', 'computation_capacity',
            'function_count', 'resource_demands', 'data_sizes',
            'bandwidth_matrix', 'gpu_cost', 'memory_cost',
            'bandwidth_cost', 'profit_per_user'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for data_id in range(1, n_samples + 1):
            # =======================
            # 节点数量生成（增强小规模样本）
            # =======================
            function_count = random.randint(2, 5)
            node_options = [
                function_count + random.randint(1, 3),  # 小型组网
                function_count * 2 + random.randint(2, 5)  # 大型组网
            ]
            node_count = random.choices(
                node_options,
                weights=[75, 25],  # 3:1比例偏向小规模
                k=1
            )[0]
            node_count = max(node_count, function_count + 1, 4)

            # =======================
            # 增强的带宽矩阵生成
            # =======================
            bandwidth_matrix = np.zeros((node_count, node_count), dtype=int)

            # 阶段1：构造基础环形拓扑
            for i in range(node_count):
                j = (i + 1) % node_count  # 保证环路闭合
                bw = random.choice([200, 500, 1000])  # 环形主干使用较高带宽
                bandwidth_matrix[i][j] = bw
                bandwidth_matrix[j][i] = bw

            # 阶段2：补充随机全连接
            connection_prob = 0.65 + 0.15 * (node_count < 8)  # 小规模网络更高概率
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    if bandwidth_matrix[i][j] == 0 and random.random() < connection_prob:
                        bw = random.choice([50, 100, 200, 500, 1000])
                        bandwidth_matrix[i][j] = bw
                        bandwidth_matrix[j][i] = bw

            # =======================
            # 计算资源配置（保持原有逻辑增强可读性）
            # =======================
            computation_capacity = []
            for node_idx in range(node_count):
                node_type = random.choice(['edge', 'cloud', 'high-end'])
                # 各类型节点资源配置参数
                if node_type == 'edge':
                    cpu = random.randint(50, 100)  # 边缘节点算力范围 (TFLOPS)
                    mem_ratio = random.randint(3, 6)
                elif node_type == 'cloud':
                    cpu = random.randint(100, 250)
                    mem_ratio = random.randint(5, 8)
                else:  # high-end
                    cpu = random.randint(250, 600)
                    mem_ratio = random.randint(2, 3)
                computation_capacity.append([cpu, cpu * mem_ratio])

            # =======================
            # 价格参数生成（添加注释说明）
            # =======================
            gpu_cost = round(random.uniform(0.1, 0.6), 3)  # GPU算力成本 ($/TFLOPS-hour)
            memory_cost = round(random.uniform(0.05, 0.15), 2)  # 内存成本 ($/GB-hour)
            bandwidth_cost = round(random.uniform(0.01, 0.08), 2)  # 带宽传输成本 ($/GB)
            profit_per_user = random.randint(200, 500)  # 用户价值参数 ($)

            # =======================
            # 函数需求生成（优化指数增长公式）
            # =======================
            resource_demands = []
            base_compute = random.uniform(5, 15)
            growth_factor = 1.2  # 层级计算需求增长率
            for layer in range(function_count):
                compute_demand = base_compute * (growth_factor ** layer)
                memory_demand = compute_demand * random.uniform(10, 20)
                resource_demands.append([
                    max(3.0, round(compute_demand, 1)),  # 最小计算需求3 TFLOPS
                    max(40, int(memory_demand))  # 最小内存需求40 GB
                ])

            # =======================
            # 数据序列化与写入
            # =======================
            writer.writerow({
                'test_data_id': data_id,
                'node_count': node_count,
                'computation_capacity': str([[c[0], c[1]] for c in computation_capacity]).replace(" ", ""),
                'function_count': function_count,
                'resource_demands': str([[round(d[0], 1), d[1]] for d in resource_demands]),
                'data_sizes': str([random.randint(20, 100) for _ in range(function_count - 1)]),
                'bandwidth_matrix': str([list(map(int, row)) for row in bandwidth_matrix]),
                'gpu_cost': gpu_cost,
                'memory_cost': memory_cost,
                'bandwidth_cost': bandwidth_cost,
                'profit_per_user': profit_per_user
            })


if __name__ == "__main__":
    generate_test_data(100, 'enhanced_connectivity_data.csv')
