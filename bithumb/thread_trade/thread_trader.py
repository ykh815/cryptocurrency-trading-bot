#----------------------------------------------------------------------------------------------------------------------
# PyStock
# Larry Williams Volatility Breakout Strategy + Moving Average
#----------------------------------------------------------------------------------------------------------------------
# A version
#
#----------------------------------------------------------------------------------------------------------------------
from btrader_split24 import PyBithumb
import threading
import time

def run_trade(debug_flag, thread_cnt, total_thread_cnt):
    runner = PyBithumb(debug_flag, thread_cnt, total_thread_cnt)
    runner.run_trade()

if __name__ == "__main__":
    total_thread_cnt = 24
    thread_list = []
    debug_flag = True                      # True: 매매 API 호출 안됨, False: 실제로 매매 API 호출
    
    for thread_cnt in range(total_thread_cnt):
        thread = threading.Thread(target=run_trade, args=(debug_flag, thread_cnt + 1, total_thread_cnt))
        thread.daemon = True
        thread_list.append(thread)
        thread.start()

        time.sleep(60)

    while True:
        with open("flag/trade.txt") as f:
            lines = f.readlines()
            flag = lines[0].strip()
            if (flag != 'Y'):
                break

        time.sleep(600)
