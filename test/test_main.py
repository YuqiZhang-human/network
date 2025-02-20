from algorithms.generate_test_data import generate_test_data
from test_runner import main as run_tests
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    # 生成测试数据
    print("Generating 111 data...")
    generate_test_data(100, "test_cases.csv")

    # 运行测试
    print("\nRunning tests...")
    run_tests("test_cases.csv", "results.csv")

    print("\nTest completed! Results saved to results.csv")
