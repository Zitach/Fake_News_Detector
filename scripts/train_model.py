"""
模型训练主脚本
"""

from train import main as train_main
import sys

if __name__ == "__main__":
    # 调用根目录的train.py（需确保脚本在根目录下运行）
    sys.exit(train_main())
