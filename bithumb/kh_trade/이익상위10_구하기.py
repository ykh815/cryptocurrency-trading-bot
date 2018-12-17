import pybithumb
import numpy as np


def run_volatility_breakout(ticker):
    df = pybithumb.get_ohlcv(ticker)
    df = df['2018']                                                 # 일봉 중 2018년도

    df['ma5'] = df['close'].rolling(window=5).mean()                # 5일 이동평균 컬럼
    df['ma5_shift1'] = df['ma5'].shift(1)                           # 5일 이동평균 컬럼에서 하나씩 내림
    df['bull'] = df['open'] > df['ma5_shift1']                      # 상승장/하락장 판단

    df['volatility'] = (df['high'] - df['low'])*0.5                 # 변동성 * 0.5
    df['target'] = df['open'] + df['volatility'].shift(1)           # 목표가 = 시가 + 전일 변동성 * 0.5
    df['er'] = np.where(df['bull'] & (df['high'] >= df['target']),  # 매수 조건
                        df['close'] / df['target'],                 # 매수시 수익률 (매도가/매수가)
                        1)                                          # 홀드시 수익률
    df['cumprod'] = df['er'].cumprod()
    return df['cumprod'][-2]


tickers = pybithumb.get_tickers()
result = []

for ticker in tickers:
    earning_rate = run_volatility_breakout(ticker)
    result.append((ticker, earning_rate))


# 정렬
ranked_data = sorted(result, key=lambda x:x[1], reverse=True)
result_list = []
for data in ranked_data[:10]:
    result_list.append(data[0])
print(result_list)
