import csv
import random
import numpy as np

# NY-20网络的连通性矩阵（度数>=2, >=3, >=4, >=5）
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
        [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
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
ny20_topologies = {
    2: [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    3: [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    4: [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
    5: [
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    ],
}

# 各种规模模型的算力和内存需求（基于真实模型，单位按照相对比例估算）
# 降低资源需求，使其与计算能力范围更加匹配
MODEL_SPECS = {
    '7B': {
        'compute': 70,        # 计算需求基准值（降低一半）
        'memory': 7000,       # 存储需求基准值 (单位MB)（降低一半）
        'hidden_size': 4096   # 隐藏层大小，用于计算参数传输
    },
    '13B': {
        'compute': 130,
        'memory': 13000,
        'hidden_size': 5120
    },
    '70B': {
        'compute': 700,
        'memory': 70000,
        'hidden_size': 8192
    },
    '110B': {
        'compute': 1100,
        'memory': 110000,
        'hidden_size': 12288
    },
    '180B': {
        'compute': 1800,
        'memory': 180000,
        'hidden_size': 14336
    },
    '310B': {
        'compute': 3100,
        'memory': 310000,
        'hidden_size': 17408
    },
    '540B': {
        'compute': 5400,
        'memory': 540000,
        'hidden_size': 20480
    },
    '1T': {
        'compute': 10000,
        'memory': 1000000,
        'hidden_size': 24576
    }
}

def generate_module_split(module_count):
    """生成模块的切割比例，确保总和为1，且分布更加合理"""
    if module_count == 1:
        return [1.0]
    
    # 生成随机但总和为1的切割比例
    # 使用Dirichlet分布生成模块切割比例，以获得更加合理的分布
    ratios = np.random.dirichlet(np.ones(module_count) * 2.0)
    
    # 确保没有特别小的比例（最小不低于3%）
    min_ratio = 0.03
    while min(ratios) < min_ratio:
        # 重新分配比例
        small_idx = np.argmin(ratios)
        deficit = min_ratio - ratios[small_idx]
        ratios[small_idx] = min_ratio
        
        # 从其他比例中按比例减去
        others = [i for i in range(len(ratios)) if i != small_idx]
        total_others = sum(ratios[i] for i in others)
        for i in others:
            ratios[i] -= deficit * (ratios[i] / total_others)
        
        # 归一化确保总和为1
        ratios = ratios / sum(ratios)
    
    return ratios.tolist()

def generate_model_partition_schema(model_size, num_partitions):
    """生成模型分区方案，使用连续的模块组合"""
    # 总是先生成8个基础模块
    base_module_count = 8
    base_module_split = generate_module_split(base_module_count)
    
    # 基于模型信息获取基础资源需求
    base_compute = MODEL_SPECS[model_size]['compute']
    base_memory = MODEL_SPECS[model_size]['memory']
    hidden_size = MODEL_SPECS[model_size]['hidden_size']
    
    # 计算每个模块的资源需求
    modules = []
    for i in range(base_module_count):
        ratio = base_module_split[i]
        modules.append({
            'module_id': i,
            'compute': base_compute * ratio,
            'memory': base_memory * ratio,
        })
    
    # 生成模块间的数据传输量 - 大幅降低传输数据量
    data_sizes = []
    for i in range(base_module_count - 1):
        # 随机生成参数传输量，与模型尺寸和隐藏层大小成正比
        # 使用阶梯增长，但大幅降低数据量
        base_size = hidden_size * 0.5  # 基础传输量，降低为原来的1/4
        position_factor = (i + 1) / base_module_count  # 位置因子，表示在模型中的相对位置
        variation = random.uniform(0.4, 0.7)  # 随机变化因子，进一步降低
        
        # 随位置递增的数据传输量
        transfer_size = base_size * position_factor * variation
        data_sizes.append(transfer_size)
    
    # 现在根据num_partitions将8个模块组合成num_partitions个连续的分区
    if num_partitions >= base_module_count:
        # 如果分区数大于等于模块数，每个模块单独作为一个分区
        partitions = [([i], modules[i]['compute'], modules[i]['memory']) for i in range(base_module_count)]
        partition_data_sizes = data_sizes
    else:
        # 将8个模块合并为num_partitions个连续分区
        # 先确定每个分区包含多少个基础模块
        partition_sizes = []
        remaining_modules = base_module_count
        remaining_partitions = num_partitions
        
        for i in range(num_partitions):
            # 为每个分区分配大致相等数量的模块
            size = remaining_modules // remaining_partitions
            partition_sizes.append(size)
            remaining_modules -= size
            remaining_partitions -= 1
        
        # 创建分区
        partitions = []
        module_index = 0
        partition_data_sizes = []
        
        for size in partition_sizes:
            module_ids = list(range(module_index, module_index + size))
            total_compute = sum(modules[i]['compute'] for i in module_ids)
            total_memory = sum(modules[i]['memory'] for i in module_ids)
            
            partitions.append((module_ids, total_compute, total_memory))
            
            # 如果不是最后一个分区，添加数据传输量
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
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'test_data_id', 'node_count', 'computation_capacity',
            'function_count', 'resource_demands', 'data_sizes',
            'bandwidth_matrix', 'gpu_cost', 'memory_cost',
            'bandwidth_cost', 'profit_per_user', 'topology_degree',
            'gpu_compute_range', 'bandwidth_range', 'model_size',
            'module_count'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 所有可用的模型尺寸
        model_sizes = list(MODEL_SPECS.keys())
        
        # 定义阶梯增长的计算能力范围 - 简化为3个级别
        compute_capacity_ranges = [
            (300, 700),    # 低级计算能力
            (900, 1500),   # 中级计算能力
            (2000, 4000)   # 高级计算能力
        ]
        
        # 定义阶梯增长的带宽范围（单位Mbps）- 简化为3个级别
        bandwidth_ranges = [
            (2000, 8000),     # 低带宽
            (10000, 20000),   # 中带宽
            (30000, 50000)    # 高带宽
        ]
        
        # GPU成本范围 - 简化为3个级别
        gpu_cost_ranges = [
            (0.5, 1.5),      # 低成本
            (2.0, 3.0),      # 中成本
            (3.5, 4.5)       # 高成本
        ]
        
        # 内存成本范围 - 简化为3个级别
        memory_cost_ranges = [
            (0.0005, 0.0015),  # 低成本
            (0.0020, 0.0030),  # 中成本
            (0.0035, 0.0045)   # 高成本
        ]
        
        # 带宽成本范围 - 简化为3个级别
        bandwidth_cost_ranges = [
            (1, 4),    # 低成本
            (5, 7),    # 中成本
            (8, 11)    # 高成本
        ]
        
        # 用户利润范围 - 简化为3个级别
        user_profit_ranges = [
            (80, 180),     # 低利润
            (230, 380),    # 中利润
            (430, 600)     # 高利润
        ]

        for data_id in range(1, n_samples + 1):
            # 网络拓扑度数（2-5）
            topology_degree = random.randint(2, 5)
            topology_matrix = ny20_topologies[topology_degree]
            node_count = len(topology_matrix)

            # 选择模型尺寸
            model_size = random.choice(model_sizes)
            
            # 模块数量（2-6）
            module_count = random.randint(2, 6)
            
            # 生成模型分区方案
            partition_schema = generate_model_partition_schema(model_size, module_count)
            
            # 选择计算能力级别并生成节点计算能力
            compute_level = random.randint(0, 2)  # 0-2共3个级别
            compute_range = compute_capacity_ranges[compute_level]
            
            # 为每个节点生成计算能力 (gpu, memory)
            computation_capacity = []
            for _ in range(node_count):
                gpu = random.uniform(compute_range[0], compute_range[1])
                # 内存与GPU计算能力成正比，但有随机变化
                memory_ratio = random.uniform(80, 120)  # 内存/GPU比例变化因子
                memory = gpu * memory_ratio
                computation_capacity.append([round(gpu, 2), round(memory, 2)])
            
            # 选择带宽级别并生成基础带宽值
            bandwidth_level = random.randint(0, 2)  # 0-2共3个级别
            bandwidth_range = bandwidth_ranges[bandwidth_level]
            base_bandwidth = random.uniform(bandwidth_range[0], bandwidth_range[1])
            
            # 生成带宽矩阵（基础带宽 × 连通性矩阵）
            bandwidth_matrix = []
            for i in range(node_count):
                row = []
                for j in range(node_count):
                    if i == j:
                        row.append(0)  # 对角线为0
                    elif topology_matrix[i][j] == 1:
                        # 为每条连接生成略有变化的带宽
                        variation = random.uniform(0.9, 1.1)
                        row.append(round(base_bandwidth * variation, 2))
                    else:
                        row.append(0)  # 不相连的节点带宽为0
                bandwidth_matrix.append(row)
            
            # 选择价格级别并生成价格
            gpu_cost_level = random.randint(0, 2)
            memory_cost_level = random.randint(0, 2)
            bandwidth_cost_level = random.randint(0, 2)
            profit_level = random.randint(0, 2)
            
            gpu_cost = random.uniform(gpu_cost_ranges[gpu_cost_level][0], gpu_cost_ranges[gpu_cost_level][1])
            memory_cost = random.uniform(memory_cost_ranges[memory_cost_level][0], memory_cost_ranges[memory_cost_level][1])
            bandwidth_cost = random.uniform(bandwidth_cost_ranges[bandwidth_cost_level][0], bandwidth_cost_ranges[bandwidth_cost_level][1])
            profit_per_user = random.uniform(user_profit_ranges[profit_level][0], user_profit_ranges[profit_level][1])
            
            # 资源需求和数据传输大小
            resource_demands = [[p[1], p[2]] for p in partition_schema['partitions']]
            data_sizes = partition_schema['data_sizes']

            # 为了数据分析，记录选择的等级
            gpu_compute_range = f"Level {compute_level + 1}"
            bandwidth_range_str = f"Level {bandwidth_level + 1}"
            
            # 写入测试数据
            writer.writerow({
                'test_data_id': data_id,
                'node_count': node_count,
                'computation_capacity': computation_capacity,
                'function_count': len(resource_demands),
                'resource_demands': resource_demands,
                'data_sizes': data_sizes,
                'bandwidth_matrix': bandwidth_matrix,
                'gpu_cost': round(gpu_cost, 2),
                'memory_cost': round(memory_cost, 4),
                'bandwidth_cost': round(bandwidth_cost, 2),
                'profit_per_user': round(profit_per_user, 2),
                'topology_degree': topology_degree,
                'gpu_compute_range': gpu_compute_range,
                'bandwidth_range': bandwidth_range_str,
                'model_size': model_size,
                'module_count': module_count
            })

if __name__ == '__main__':
    # 生成100个测试样例
    generate_test_data(100, '../../data/test/test_data.csv')