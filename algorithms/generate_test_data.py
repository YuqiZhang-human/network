import csv
import random
import numpy as np


def generate_test_data(num_cases, output_file):
    headers = [
        "case_id", "node_count", "computation_matrix", "function_count",
        "demand_matrix", "data_sizes", "bandwidth_matrix", "gpu_cost",
        "memory_cost", "bandwidth_cost", "profit_per_user"
    ]

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for case_id in range(1, num_cases + 1):
            function_count = random.randint(2, 5)
            node_count = random.randint(function_count, function_count + 3)

            computation_matrix = [
                [random.randint(500, 2000), random.randint(1000, 4000)]
                for _ in range(node_count)
            ]

            demand_matrix = [
                [random.randint(50, 300), random.randint(100, 600)]
                for _ in range(function_count)
            ]

            data_sizes = [random.randint(10, 100) for _ in range(function_count - 1)]

            bandwidth_matrix = (np.random.randint(100, 500, (node_count, node_count)) * np.triu(
                np.ones((node_count, node_count)), 1)).tolist()
            for i in range(node_count):
                bandwidth_matrix[i][i] = 0

            writer.writerow({
                "case_id": case_id,
                "node_count": node_count,
                "computation_matrix": ";".join([",".join(map(str, row)) for row in computation_matrix]),
                "function_count": function_count,
                "demand_matrix": ";".join([",".join(map(str, row)) for row in demand_matrix]),
                "data_sizes": ",".join(map(str, data_sizes)),
                "bandwidth_matrix": ";".join([",".join(map(str, row)) for row in bandwidth_matrix]),
                "gpu_cost": round(random.uniform(0.1, 0.3), 2),
                "memory_cost": round(random.uniform(0.05, 0.15), 2),
                "bandwidth_cost": round(random.uniform(0.02, 0.1), 2),
                "profit_per_user": random.randint(150, 250)
            })


if __name__ == "__main__":
    generate_test_data(100, "test_cases.csv")
    print("测试数据已生成到 test_cases.csv")
