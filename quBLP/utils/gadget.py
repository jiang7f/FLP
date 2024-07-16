import sys
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
