import os
import sys
import datetime
import inspect
default_stdout_write = sys.stdout.write

def output_message(state):
    print(f'| pid: {os.getpid()}')
    print(f'| {state} time: {datetime.datetime.now()}')

def output_to_file_init(state=None):
    current_time = datetime.datetime.now().strftime("%d%H%M%S")
    output_dir = './==output==/'
    os.makedirs(output_dir, exist_ok=True)
    caller_frame = inspect.stack()[1]
    file_name = os.path.basename(caller_frame.filename).replace(".py", f"_{current_time}_{os.getpid()}") if '__file__' in globals() else "interactive_mode"
    # 定义输出函数
    def stdout_to_file(output):
        with open(output_dir + file_name + '.out', "a") as file:
            file.write(output)
    # 重定向输出
    sys.stdout.write = stdout_to_file
    if state != None:
        output_message(state)

def output_to_file_reset(state=None):
    sys.stdout.write = default_stdout_write
    if state != None:
        output_message(state)
    
if __name__ == '__main__':
    output_to_file_init("start")
    output_to_file_reset("end o.O?")