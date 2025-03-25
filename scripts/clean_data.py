"""
清理临时文件或未使用的中间数据
"""

import os
import shutil
import argparse

def clean():
    dirs_to_clean = [
        "data/processed/tmp/",
        "outputs/checkpoints/"
    ]
    for d in dirs_to_clean:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"已清理：{d}")

def main():
    parser = argparse.ArgumentParser(description="清理临时文件")
    args = parser.parse_args()
    clean()

if __name__ == "__main__":
    main()
