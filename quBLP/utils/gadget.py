import sys
import numpy as np
import psutil

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