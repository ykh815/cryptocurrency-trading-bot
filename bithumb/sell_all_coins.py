#----------------------------------------------------------------------------------------------------------------------
# PyStock
# Larry Williams Volatility Breakout Strategy + Moving Average
#----------------------------------------------------------------------------------------------------------------------
# A version
#
#----------------------------------------------------------------------------------------------------------------------
import pybithumb
import time

MIN_ORDERS = {
        "BTC": 0.001, "ETH": 0.01, "DASH": 0.01, "LTC": 0.01, "ETC": 0.1, "XRP": 10, "BCH": 0.001,
        "XMR": 0.01, "ZEC": 0.01, "QTUM": 0.1, "BTG": 0.1, "EOS": 0.1, "ICX": 1, "VEN": 1, "TRX": 100,
        "ELF": 10, "MITH": 10, "MCO": 10, "OMG": 0.1, "KNC": 1, "GNT": 10, "HSR": 1, "ZIL": 100,
        "ETHOS": 1, "PAY": 1, "WAX": 10, "POWR": 10, "LRC": 10, "GTO": 10, "STEEM": 10, "STRAT": 1,
        "ZRX": 1, "REP": 0.1, "AE": 1, "XEM": 10, "SNT": 10, "ADA": 10, "PPT": 1, "CTXC": 10,
        "CMT": 10, "THETA": 10, "WTC": 1, "ITC": 10
}

#----------------------------------------------------------------------------------------------------------------------
# 아래의 값을 적당히 수정해서 사용하세요.
#----------------------------------------------------------------------------------------------------------------------
INTERVAL = 1                                        # 매수/매도 시도 interval (1초 기본)    

# Load account
with open("bithumb.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)

def retry_sell(ticker, unit, retry_cnt=10):
    '''
    retry count 만큼 매도 시도
    :param ticker: 티커
    :param unit: 매도 수량
    :param retry_cnt: 최대 매수 시도 횟수
    :return:
    '''
    try:
        ret = None
        while ret is None and retry_cnt > 0:
            ret = bithumb.sell_market_order(ticker, unit)
            time.sleep(INTERVAL)
    
            retry_cnt = retry_cnt - 1
    except:
        pass

def try_sell(tickers):
    '''
    보유하고 있는 모든 코인에 대해 전량 매도
    :param tickers: 빗썸에서 지원하는 암호화폐의 티커 목록
    :return:
    '''
    try:
        for ticker in tickers:
            unit = bithumb.get_balance(ticker)[0]
            min_order = MIN_ORDERS.get(ticker, 0.001)

            if unit >= min_order:
                ret = bithumb.sell_market_order(ticker, unit)
                time.sleep(INTERVAL)
                if ret is None:
                    retry_sell(ticker, unit, 10)
    except:
        pass
        
#----------------------------------------------------------------------------------------------------------------------
# 매매 알고리즘 시작
#---------------------------------------------------------------------------------------------------------------------       
tickers = pybithumb.get_tickers()                                       # 티커 리스트 얻기

try:
    tickers.remove('date')
except:
    pass

try:
    tickers.remove('TRUE')
except:
    pass

try:
    tickers.remove('FALSE')
except:
    pass
        
try_sell(tickers)                                                    # 각 가상화폐에 대해 매도 시도

