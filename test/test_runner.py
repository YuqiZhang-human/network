import csv
import sys
from collections import defaultdict

from algorithms.computing_first import ComputeFirstOptimizer
from algorithms.memory_first import MemoryFirstOptimizer
from algorithms.multi_func import MultiFuncOptimizer
from algorithms.random_deploy import RandomDeploymentOptimizer
from algorithms.cost_minimization import CostMinimizationOptimizer
from algorithms.user_maximization import UserMaximizationOptimizer

def parse_config(row):
    return {
        "node_count": row["node_count"],
        "computation_matrix": row["computation_matrix"],
        "function_count": row["function_count"],
        "demand_matrix": row["demand_matrix"],
        "data_sizes": row["data_sizes"],
        "bandwidth_matrix": row["bandwidth_matrix"],
        "gpu_cost": row["gpu_cost"],
        "memory_cost": row["memory_cost"],
        "bandwidth_cost": row["bandwidth_cost"],
        "profit_per_user": row["profit_per_user"]
    }


def format_multi_func_results(mf_optimizer):
    """处理MF类算法的多解结果"""
    solutions = []
    for node in getattr(mf_optimizer, 'feasible_nodes', []):
        solutions.append({
            "deployment": node.deployment_plan,
            "u_max": node.U_max,
            "cost": node.total_cost,
            "profit": node.final_profit
        })

    if not solutions:
        return ["NA"], ["0"], ["NA"], ["NA"], ["NA"]

    deployments = [str(sol["deployment"]) for sol in solutions]
    users = [str(sol["u_max"]) for sol in solutions]
    costs = [f"{sol['cost']:.2f}" if sol['cost'] is not None else "NA" for sol in solutions]
    profits = [f"{sol['profit']:.2f}" if sol['profit'] is not None else "NA" for sol in solutions]

    optimal = max(solutions, key=lambda x: x["profit"]) if solutions else None
    optimal_str = str([
        optimal["deployment"],
        f"{optimal['cost']:.2f}",
        optimal["u_max"],
        f"{optimal['profit']:.2f}"
    ]) if optimal else "NA"

    return deployments, users, costs, profits, optimal_str


def run_single_case(config):
    results = defaultdict(dict)

    try:
        # Multi Func
        mf = MultiFuncOptimizer(config)
        mf.build_optimization_tree()
        mf_results = format_multi_func_results(mf)
        results['multi_func'] = mf_results
    except Exception as e:
        results['multi_func'] = (["NA"], ["0"], ["NA"], ["NA"], "NA")

    # Random
    try:
        rd = RandomDeploymentOptimizer(config)
        random_result = rd.optimize()
        results['random'] = [
            str(random_result["deployment"]) if random_result else "NA",
            f"{random_result['cost']:.2f}" if random_result else "NA",
            random_result["u_max"] if random_result else "0",
            f"{random_result['profit']:.2f}" if random_result else "NA"
        ]
    except:
        results['random'] = ["NA"] * 4

    # Memory First
    try:
        mem = MemoryFirstOptimizer(config)
        mem_result = mem.optimize()
        results['memory'] = [
            str(mem_result["deployment"]) if mem_result else "NA",
            f"{mem_result['cost']:.2f}" if mem_result else "NA",
            mem_result["u_max"] if mem_result else "0",
            f"{mem_result['profit']:.2f}" if mem_result else "NA"
        ]
    except:
        results['memory'] = ["NA"] * 4

    # Compute First
    try:
        comp = ComputeFirstOptimizer(config)
        comp_result = comp.optimize()
        results['compute'] = [
            str(comp_result["deployment"]) if comp_result else "NA",
            f"{comp_result['cost']:.2f}" if comp_result else "NA",
            comp_result["u_max"] if comp_result else "0",
            f"{comp_result['profit']:.2f}" if comp_result else "NA"
        ]
    except:
        results['compute'] = ["NA"] * 4

    # Cost Minimization
    try:
        cost_opt = CostMinimizationOptimizer(config)
        cost_opt.build_optimization_tree()
        cost_node = cost_opt.get_optimal()
        results['cost'] = [
            str(cost_node["deployment"]) if cost_node else "NA",
            f"{cost_node['cost']:.2f}" if cost_node else "NA",
            cost_node["u_max"] if cost_node else "0",
            f"{cost_node['profit']:.2f}" if cost_node else "NA"
        ]
    except:
        results['cost'] = ["NA"] * 4

    # User Maximization
    try:
        user_opt = UserMaximizationOptimizer(config)
        user_opt.build_optimization_tree()
        user_node = user_opt.get_optimal()
        results['user'] = [
            str(user_node["deployment"]) if user_node else "NA",
            f"{user_node['cost']:.2f}" if user_node else "NA",
            user_node["u_max"] if user_node else "0",
            f"{user_node['profit']:.2f}" if user_node else "NA"
        ]
    except:
        results['user'] = ["NA"] * 4

    return results


def main(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)

        headers = [
            "CaseID",
            "MF_Deployments", "MF_Users", "MF_Costs", "MF_Profits", "MF_Optimal",
            "Random_Deploy", "Random_Cost", "Random_Users", "Random_Profit",
            "Memory_Deploy", "Memory_Cost", "Memory_Users", "Memory_Profit",
            "Compute_Deploy", "Compute_Cost", "Compute_Users", "Compute_Profit",
            "CostOpt_Deploy", "CostOpt_Cost", "CostOpt_Users", "CostOpt_Profit",
            "UserOpt_Deploy", "UserOpt_Cost", "UserOpt_Users", "UserOpt_Profit"
        ]
        writer.writerow(headers)

        for idx, row in enumerate(reader, 1):
            try:
                config = parse_config(row)
                results = run_single_case(config)
            except Exception as e:
                print(f"Error processing case {idx}: {str(e)}")
                continue

            # 整理多函数结果
            mf_dep, mf_user, mf_cost, mf_profit, mf_opt = results['multi_func']

            output_row = [
                idx,  # CaseID
                "|".join(mf_dep), "|".join(mf_user), "|".join(mf_cost), "|".join(mf_profit), mf_opt,
                *results['random'],
                *results['memory'],
                *results['compute'],
                *results['cost'],
                *results['user']
            ]

            writer.writerow(output_row)
            print(f"Processed case {idx}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_runner.py <input_file> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
