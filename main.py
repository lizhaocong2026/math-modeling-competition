"""
主程序入口
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("数学建模竞赛代码库 - 欢迎使用")
    print("=" * 60)
    print("\n请选择运行示例：")
    print("1. 优化问题示例")
    print("2. 预测与评价示例")
    print("3. 运行测试")
    print("0. 退出")
    
    choice = input("\n请输入选项: ")
    
    if choice == '1':
        os.system('python examples/example_optimization.py')
    elif choice == '2':
        os.system('python examples/example_prediction_evaluation.py')
    elif choice == '3':
        os.system('python -m pytest tests/')
    elif choice == '0':
        print("再见！")
    else:
        print("无效选项")


if __name__ == '__main__':
    main()
