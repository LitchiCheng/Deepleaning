import threading
import subprocess
import time
import pandas as pd


# 用于存储中断频率数据的列表
interrupt_frequency_data = []


def get_interrupt_frequency():
    global interrupt_frequency_data
    while True:
        try:
            print(f"ssss:")
            # 执行sar命令获取中断信息（这里示例是获取汇总的中断信息，每秒收集一次，可根据实际调整）
            result = subprocess.check_output(['sar', '-I', 'SUM', '1', '1'])
            result_str = result.decode('utf-8')
            # print(result_str)
            # 解析出中断频率相关的值（这里假设输出的第二行第二列是我们要的中断频率值，需根据实际输出格式调整）
            lines = result_str.split('\n')
            if len(lines) > 1:
                split_line = lines[3].split()
                # print(split_line)
                if len(split_line) > 1:
                    frequency_value = float(split_line[2])
                    interrupt_frequency_data.append(frequency_value)
                    print(frequency_value)
        except Exception as e:
            print(f"获取中断频率时出错: {e}")

        time.sleep(1)  # 每隔1秒获取一次

def save_to_csv():
    global interrupt_frequency_data
    while True:
        if interrupt_frequency_data:
            df = pd.DataFrame({
                'Interrupt Frequency': interrupt_frequency_data
            })
            df.to_csv('/home/dar/DL/Deeplearning/lstm/interrupt_frequency_records.csv', mode='a', header=not pd.io.common.file_exists('/home/dar/DL/Deeplearning/lstm/interrupt_frequency_records.csv'),
                      index=False)
            interrupt_frequency_data = []  # 清空已保存的数据列表
        time.sleep(5)  # 每隔5秒保存一次到CSV文件


if __name__ == "__main__":
    # 创建获取中断频率的线程
    interrupt_thread = threading.Thread(target=get_interrupt_frequency)
    # 创建保存到CSV文件的线程
    save_thread = threading.Thread(target=save_to_csv)

    interrupt_thread.start()
    save_thread.start()

    interrupt_thread.join()
    save_thread.join()