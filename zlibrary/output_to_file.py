import os
import sys
import datetime
def output_to_file():
    current_time = datetime.datetime.now().strftime("%d%H%M%S")
    output_dir = './==output==/'
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(__file__).replace(".py", f"_{current_time}") if '__file__' in globals() else "interactive_mode"
    # 定义输出函数
    def stdout_to_file(output):
        with open(output_dir + file_name + '.out', "a") as file:
            file.write(output)
    # 重定向输出
    sys.stdout.write = stdout_to_file
    print(f'startup time: {datetime.datetime.now()}')
    print(f'pid: {os.getpid()}')
    
if __name__ == '__main__':
    output_to_file()
    print("stdout has been redirected")