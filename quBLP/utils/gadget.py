import sys
import numpy as np
import csv
import psutil
import os

main_module = sys.modules['__main__']
should_print = getattr(main_module, 'should_print', False)

def iprint(*args, **kwargs):
    if should_print:
        print(*args, **kwargs)

def get_rss_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / (1024 * 1024)  # 转换为 MB
    iprint(f"rss_usage: {rss_mb:.2f} MB")
    return rss_mb

# 设置numpy输出格式
def set_print_form(suppress=True, precision=4, linewidth=300):
    # 不要截断 是否使用科学计数法 输出浮点数位数 宽度
    np.set_printoptions(threshold=np.inf, suppress=suppress, precision=precision,  linewidth=linewidth)

def read_last_row(csv_file_path):
    last_row = None
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            last_row = row
    return last_row

def get_main_file_info():
    main_module = sys.modules['__main__']
    main_file = main_module.__file__
    main_dir = os.path.dirname(main_file)
    return main_dir, os.path.basename(main_file)

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)