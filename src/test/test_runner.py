import csv
import random
import numpy as np
import math

# NY20网络的连通性矩阵（度数>=2, >=3, >=4, >=5）
ny20_topologies = {
    2: [
        [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
        [0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1],
        [0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0],
    ],
    3: [
        [0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
        [0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
        [1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1],
        [0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0],
        [0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1],
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0],
    ],
    4: [
        [0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0],
        [0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0],
        [0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
        [1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1],
        [0,0,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,1,1],
        [0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0],
        [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0],
        [0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0],
        [0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,1],
        [0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1],
        [0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0],
    ],
    5: [
        [0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0],
        [0,1,0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0],
        [0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
        [0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0],
        [0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0],
        [0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0],
        [1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1,0],
        [1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
        [1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1],
        [1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,0,0,0,1,1],
        [0,0,0,1,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0],
        [0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0],
        [0,1,1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0],
        [0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,1],
        [0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,1],
        [0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,1],
        [0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0],
    ],
}

# 模型的算力和内存需求
MODEL_SPECS = {
    '7B': {'compute': 50, 'memory': 5000, 'hidden_size': 4096},
    '13B': {'compute': 100, 'memory': 10000, 'hidden_size': 5120},
    '70B': {'compute': 500, 'memory': 50000, 'hidden_size': 8192},
    '110B': {'compute': 800, 'memory': 80000, 'hidden_size': 12288},
    '180B': {'compute': 1200, 'memory': 120000, 'hidden_size': 14336},
    '310B': {'compute': 2000, 'memory': 200000, 'hidden_size': 17408},
    '540B': {'compute': 3500, 'memory': 350000, 'hidden_size': 20480},
    '1T': {'compute': 6000, 'memory': 600000, 'hidden_size': 24576}
}

def generate_module_split(module_count):
    """生成模块切割比例，确保分布合理"""
    if module_count == 1:
        return [1.0]
    ratios = np.random.dirichlet(np.ones(module_count) * 3.0)
    min_ratio = 0.05
    while min(ratios) < min_ratio:
        small_idx = np.argmin(ratios)
        deficit = min_ratio - ratios[small_idx]
        ratios[small_idx] = min_ratio
        others = [i for i in range(len(ratios)) if i != small_idx]
        total_others = sum(ratios[i] for i in others)
        for i in others:
            ratios[i] -= deficit * (ratios[i] / total_others)
        ratios = ratios / sum(ratios)
    return ratios.tolist()

def generate_model_partition_schema(model_size, num_partitions):
    """生成模型分区方案"""
    base_module_count = 8
    base_module_split = generate_module_split(base_module_count)
    
    base_compute = MODEL_SPECS[model_size]['compute']
    base_memory = MODEL_SPECS[model_size]['memory']
    hidden_size = MODEL_SPECS[model_size]['hidden_size']
    
    modules = []
    for i in range(base_module_count):
        ratio = base_module_split[i]
        modules.append({
            'module_id': i,
            'compute': base_compute * ratio,
            'memory': base_memory * ratio,
        })
    
    data_sizes = []
    for i in range(base_module_count - 1):
        base_size = hidden_size * 0.2
        position_factor = (i + 1) / base_module_count
        variation = random.uniform(0.5, 1.0)
        transfer_size = base_size * position_factor * variation
        data_sizes.append(transfer_size)
    
    if num_partitions >= base_module_count:
        partitions = [([i], modules[i]['compute'], modules[i]['memory']) for i in range(base_module_count)]
        partition_data_sizes = data_sizes
    else:
        partition_sizes = []
        remaining_modules = base_module_count
        remaining_partitions = num_partitions
        for i in range(num_partitions):
            size = remaining_modules // remaining_partitions
            partition_sizes.append(size)
            remaining_modules -= size
            remaining_partitions -= 1
        
        partitions = []
        module_index = 0
        partition_data_sizes = []
        for size in partition_sizes:
            module_ids = list(range(module_index, module_index + size))
            total_compute = sum(modules[i]['compute'] for i in module_ids)
            total_memory = sum(modules[i]['memory'] for i in module_ids)
            partitions.append((module_ids, total_compute, total_memory))
            if module_index + size - 1 < base_module_count - 1:
                partition_data_sizes.append(data_sizes[module_index + size - 1])
            module_index += size
    
    return {
        'partitions': partitions,
        'data_sizes': partition_data_sizes,
        'all_modules': modules,
        'all_data_sizes': data_sizes
    }

def generate_test_data(n_samples: int, output_file: str):
    """生成测试数据，使用连续均匀采样而非离散级别"""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'test_data_id', 'node_count', 'computation_capacity',
            'function_count', 'resource_demands', 'data_sizes',
            'bandwidth_matrix', 'link_weights', 'gpu_cost', 'memory_cost',
            'bandwidth_cost', 'profit_per_user', 'topology_degree',
            'gpu_compute_range', 'bandwidth_range', 'model_size',
            'module_count'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        model_sizes = list(MODEL_SPECS.keys())
        
        # 重新定义更符合实际的参数范围
        # 计算能力范围 - 根据实际GPU性能调整，避免过大数值
        compute_capacity_range = {
            '7B': (100, 300),     # 小模型需要较少计算能力
            '13B': (200, 400),    # 中小模型
            '70B': (300, 600),    # 中型模型
            '110B': (400, 700),   # 中大型模型
            '180B': (500, 800),   # 大型模型
            '310B': (600, 900),   # 超大模型
            '540B': (700, 1000),  # 巨型模型
            '1T': (900, 1200)     # 超级巨型模型
        }
        
        # 带宽范围 - 根据实际网络条件调整（单位Mbps），避免过大数值
        bandwidth_range = (100, 2000)  # 从100Mbps到2Gbps的范围
        
        # 成本范围 - 更贴近实际云服务定价
        gpu_cost_range = (0.2, 2.0)               # 每小时GPU成本(美元)
        memory_cost_range = (0.00005, 0.0005)     # 每GB内存成本(美元)
        bandwidth_cost_range = (0.05, 0.12)       # 每GB带宽成本(美元)
        
        # 用户利润范围 - 基于实际业务模型
        user_profit_range = (50, 300)             # 每用户利润(美元)
        
        # 创建更有意义的参数相关性
        # 为不同规模的模型分配不同的潜在用户群体大小
        max_users_by_model = {
            '7B': (10, 50),      # 小模型可能有更多用户
            '13B': (8, 40),
            '70B': (5, 30),
            '110B': (4, 25),
            '180B': (3, 20),
            '310B': (2, 15),
            '540B': (1, 10),
            '1T': (1, 5)         # 超大模型用户较少
        }
        
        # 创建均匀分布的参数组合
        topology_degrees = [2, 3, 4, 5]
        module_counts = [2, 3, 4, 5, 6]
        
        # 创建测试数据ID的列表
        data_ids = list(range(1, n_samples + 1))
        
        # 为主要离散参数创建均匀分布
        # 首先确保所有拓扑度和模块数量的组合都能被覆盖
        topo_module_combinations = []
        for topo in topology_degrees:
            for module in module_counts:
                topo_module_combinations.append((topo, module))
        
        # 确保模型大小的分布更均匀
        model_distributions = []
        # 按照实际使用频率分配不同模型的权重
        model_weights = {
            '7B': 20,      # 小模型使用最频繁
            '13B': 18,
            '70B': 16,
            '110B': 14,
            '180B': 12,
            '310B': 10,
            '540B': 6,
            '1T': 4        # 最大模型使用最少
        }
        
        # 根据权重创建模型分布
        for model, weight in model_weights.items():
            count = int(n_samples * weight / sum(model_weights.values()))
            model_distributions.extend([model] * count)
        
        # 如果样本数量与分布不匹配，随机补充
        while len(model_distributions) < n_samples:
            # 按权重随机选择
            weights = list(model_weights.values())
            models = list(model_weights.keys())
            model = random.choices(models, weights=weights, k=1)[0]
            model_distributions.append(model)
            
        # 确保随机性
        random.shuffle(model_distributions)
        
        # 为每个测试数据ID分配参数值
        for data_id in data_ids:
            # 分配拓扑度和模块数
            topo_module_idx = (data_id - 1) % len(topo_module_combinations)
            topology_degree, module_count = topo_module_combinations[topo_module_idx]
            
            # 分配模型大小
            model_size = model_distributions[data_id - 1]
            
            topology_matrix = ny20_topologies[topology_degree]
            node_count = len(topology_matrix)
            
            partition_schema = generate_model_partition_schema(model_size, module_count)
            
            try:
                # 基于模型大小选择合适的计算能力范围
                compute_min, compute_max = compute_capacity_range[model_size]
                
                # 均匀分布的计算能力 - 确保覆盖整个范围
                # 使用data_id确保均匀性
                segment_size = (compute_max - compute_min) / n_samples
                base_compute = compute_min + (data_id % n_samples) * segment_size
                # 添加少量随机扰动以避免完全规则的数据
                base_compute += random.uniform(-segment_size/4, segment_size/4)
                # 确保在有效范围内
                base_compute = max(compute_min, min(compute_max, base_compute))
                
                # 带宽 - 同样使用均匀分布
                band_min = bandwidth_range[0]
                band_max = bandwidth_range[1]
                band_segment = (band_max - band_min) / n_samples
                # 使用不同的偏移确保计算能力和带宽之间没有强相关性
                bandwidth_id = (data_id + 73) % n_samples
                base_bandwidth = band_min + bandwidth_id * band_segment
                # 添加少量随机扰动
                base_bandwidth += random.uniform(-band_segment/4, band_segment/4)
                # 确保在有效范围内
                base_bandwidth = max(band_min, min(band_max, base_bandwidth))
                
                # 成本参数 - 均匀分布
                # GPU成本
                gpu_segment = (gpu_cost_range[1] - gpu_cost_range[0]) / n_samples
                gpu_id = (data_id + 37) % n_samples
                gpu_cost = gpu_cost_range[0] + gpu_id * gpu_segment
                gpu_cost += random.uniform(-gpu_segment/4, gpu_segment/4)
                gpu_cost = max(gpu_cost_range[0], min(gpu_cost_range[1], gpu_cost))
                
                # 内存成本
                mem_segment = (memory_cost_range[1] - memory_cost_range[0]) / n_samples
                mem_id = (data_id + 19) % n_samples
                memory_cost = memory_cost_range[0] + mem_id * mem_segment
                memory_cost += random.uniform(-mem_segment/4, mem_segment/4)
                memory_cost = max(memory_cost_range[0], min(memory_cost_range[1], memory_cost))
                
                # 带宽成本
                bw_segment = (bandwidth_cost_range[1] - bandwidth_cost_range[0]) / n_samples
                bw_id = (data_id + 53) % n_samples
                bandwidth_cost = bandwidth_cost_range[0] + bw_id * bw_segment
                bandwidth_cost += random.uniform(-bw_segment/4, bw_segment/4)
                bandwidth_cost = max(bandwidth_cost_range[0], min(bandwidth_cost_range[1], bandwidth_cost))
                
                # 用户利润
                profit_segment = (user_profit_range[1] - user_profit_range[0]) / n_samples
                profit_id = (data_id + 91) % n_samples
                profit_per_user = user_profit_range[0] + profit_id * profit_segment
                profit_per_user += random.uniform(-profit_segment/4, profit_segment/4)
                profit_per_user = max(user_profit_range[0], min(user_profit_range[1], profit_per_user))
                
            except Exception as e:
                print(f"均匀采样失败，使用简单随机采样: {e}")
                base_compute = random.uniform(compute_capacity_range['7B'][0], compute_capacity_range['1T'][1])
                base_bandwidth = random.uniform(bandwidth_range[0], bandwidth_range[1])
                gpu_cost = random.uniform(gpu_cost_range[0], gpu_cost_range[1])
                memory_cost = random.uniform(memory_cost_range[0], memory_cost_range[1])
                bandwidth_cost = random.uniform(bandwidth_cost_range[0], bandwidth_cost_range[1])
                profit_per_user = random.uniform(user_profit_range[0], user_profit_range[1])
            
            # 生成更真实的节点计算能力分布
            computation_capacity = []
            # 创建合理的计算资源分布 - 更均匀，避免极端值
            node_capacity_factors = []
            for i in range(node_count):
                # 确保节点计算能力分布在基准值的±30%范围内
                factor = random.uniform(0.7, 1.3)
                node_capacity_factors.append(factor)
            
            # 根据因子生成计算能力
            for i in range(node_count):
                gpu = base_compute * node_capacity_factors[i]
                # 内存与GPU计算能力近似线性相关，有小幅波动
                memory_factor = random.uniform(0.9, 1.1)
                # 更合理的内存比例，GPUx100是合理内存大小
                memory = gpu * random.uniform(80, 120) * memory_factor
                computation_capacity.append([round(gpu, 2), round(memory, 2)])
            
            # 生成更合理的带宽矩阵 - 考虑地理距离和网络拓扑
            bandwidth_matrix = []
            # 计算拓扑距离矩阵
            distances = []
            for i in range(node_count):
                dist_row = []
                for j in range(node_count):
                    if i == j:
                        dist_row.append(0)  # 自己到自己的距离为0
                    elif topology_matrix[i][j] == 1:
                        dist_row.append(1)  # 直接相连的距离为1
                    else:
                        # 使用一个大值表示非直接相连
                        dist_row.append(999)
                distances.append(dist_row)
            
            # 使用Floyd-Warshall算法计算所有节点对之间的最短距离
            for k in range(node_count):
                for i in range(node_count):
                    for j in range(node_count):
                        if distances[i][k] + distances[k][j] < distances[i][j]:
                            distances[i][j] = distances[i][k] + distances[k][j]
            
            # 带宽与距离负相关 - 距离越远，带宽越低
            for i in range(node_count):
                row = []
                for j in range(node_count):
                    if i == j:
                        row.append(0)  # 自己到自己没有带宽需求
                    elif topology_matrix[i][j] == 1:
                        # 直接相连的节点带宽较高 - 基准值的±20%范围内
                        variation = random.uniform(0.8, 1.2)
                        row.append(round(base_bandwidth * variation, 2))
                    else:
                        row.append(0)  # 非直接相连的节点之间没有带宽
                bandwidth_matrix.append(row)
            
            # 生成链路权重矩阵 - 基于距离和带宽计算
            link_weights = []
            for i in range(node_count):
                weight_row = []
                for j in range(node_count):
                    if i == j:
                        weight_row.append(0)  # 自己到自己的权重为0
                    elif topology_matrix[i][j] == 1:
                        # 链路权重受距离影响 - 距离越远权重越大(1.0-2.0)
                        # 同时考虑带宽影响 - 带宽越高权重越小
                        # 这里简化为：权重 = 1.0 + 0.1 * 距离 - 0.1 * (带宽/最大带宽)
                        base_weight = 1.0 + 0.1 * distances[i][j]
                        bandwidth_factor = 0.1 * (bandwidth_matrix[i][j] / base_bandwidth)
                        weight = max(0.5, min(2.0, base_weight - bandwidth_factor))
                        weight_row.append(round(weight, 2))
                    else:
                        weight_row.append(0)  # 非直接相连的链路权重为0
                link_weights.append(weight_row)
            
            # 提取资源需求和数据大小
            resource_demands = [[p[1], p[2]] for p in partition_schema['partitions']]
            data_sizes = partition_schema['data_sizes']

            # 保持级别描述的兼容性 - 用于分析
            def get_level(value, min_val, max_val):
                norm_val = (value - min_val) / (max_val - min_val)
                if norm_val < 0.33:
                    return "Level 1"  # 低
                elif norm_val < 0.66:
                    return "Level 2"  # 中
                else:
                    return "Level 3"  # 高
            
            # 计算能力级别描述
            model_min, model_max = compute_capacity_range[model_size]
            gpu_compute_range = get_level(base_compute, model_min, model_max)
            
            # 带宽级别描述
            bandwidth_range_str = get_level(base_bandwidth, bandwidth_range[0], bandwidth_range[1])
            
            # 写入CSV
            writer.writerow({
                'test_data_id': data_id,
                'node_count': node_count,
                'computation_capacity': str(computation_capacity),  # 确保是字符串形式
                'function_count': len(resource_demands),
                'resource_demands': str(resource_demands),  # 确保是字符串形式
                'data_sizes': str(data_sizes),  # 确保是字符串形式
                'bandwidth_matrix': str(bandwidth_matrix),  # 确保是字符串形式
                'link_weights': str(link_weights),  # 添加链路权重数据
                'gpu_cost': round(gpu_cost, 2),
                'memory_cost': round(memory_cost, 6),
                'bandwidth_cost': round(bandwidth_cost, 3),
                'profit_per_user': round(profit_per_user, 2),
                'topology_degree': topology_degree,
                'gpu_compute_range': gpu_compute_range,
                'bandwidth_range': bandwidth_range_str,
                'model_size': model_size,
                'module_count': module_count
            })

if __name__ == '__main__':
    generate_test_data(200, '../../data/test/test_data.csv')