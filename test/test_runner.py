import csv
import random
import numpy as np


def generate_test_data(n_samples: int, output_file: str):
    """
    生成完全价格独立的测试数据
    所有成本参数均与节点/资源参数无关
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
            # 基础参数生成（保持原有逻辑）
            # =======================
            function_count = random.randint(4, 7)
            node_count = random.choice([
                function_count + random.randint(1, 3),
                function_count * 2 + random.randint(2, 5)
            ])
            node_count = max(node_count, function_count + 1, 4)

            # 节点资源（示例数据调整为合理规模）
            computation_capacity = []
            for _ in range(node_count):
                node_type = random.choice(['edge', 'cloud', 'high-end'])
                if node_type == 'edge':
                    cpu = random.randint(50, 100)  # TFLOPS
                    mem = cpu * random.randint(3, 6)  # GB
                elif node_type == 'cloud':
                    cpu = random.randint(100, 250)
                    mem = cpu * random.randint(5, 8)
                else:
                    cpu = random.randint(250, 600)
                    mem = cpu * random.randint(2, 3)
                computation_capacity.append([cpu, mem])

            # =======================
            # 随机价格参数生成（完全独立）
            # =======================
            # GPU成本（$/TFLOPS-hour）
            gpu_cost = round(random.uniform(0.1, 0.6), 3)
            # 参照Azure NCv3系列价格波动范围

            # 内存成本（$/GB-hour）
            memory_cost = round(random.uniform(0.05, 0.15), 2)
            # 参照AWS r5d实例内存定价范围

            # 带宽成本（$/GB-transfer）
            bandwidth_cost = round(random.uniform(0.01, 0.08), 2)
            # 参照Google Cloud CDN价格区间

            # 用户利润（$）
            profit_per_user = random.randint(200, 500)
            # 模拟不同的用户商业模型

            # =======================
            # 其余参数生成（保持原有逻辑）
            # =======================
            resource_demands = []
            base_compute = random.uniform(5, 15)
            for layer in range(function_count):
                compute = base_compute * (1.2 ** layer)
                memory = compute * random.uniform(10, 20)
                resource_demands.append([
                    max(3, round(compute, 1)),
                    max(40, int(memory))
                ])

            # 带宽矩阵（带随机断连）
            bandwidth_matrix = np.zeros((node_count, node_count), dtype=int)
            for i in range(node_count):
                for j in range(i + 1, node_count):
                    if random.random() < 0.7:
                        bw = random.choice([50, 100, 200, 500, 1000])
                        bandwidth_matrix[i][j] = bw
                        bandwidth_matrix[j][i] = bw

            # =======================
            # 数据写入
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
    # 生成测试数据示例
    generate_test_data(1000, '../indepent_pricing_data.csv')
