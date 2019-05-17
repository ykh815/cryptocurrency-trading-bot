#!/usr/lib/python3.6
#----------------------------------------------------------------------------------------------------------------------
# PyStock
# Larry Williams Volatility Breakout Strategy + Moving Average
#----------------------------------------------------------------------------------------------------------------------
# A version
# V1.0 : 헤이비트 변동성 돌파전략 Advanced 1.0 적용 (타임프레임 분산은 미적용)
#----------------------------------------------------------------------------------------------------------------------
import pybithumb
import time
import datetime
import logging
import logging.handlers
import os
import sys
import telepot
from telepot.loop import MessageLoop
import numpy as np
from dateutil.relativedelta import relativedelta

token = "720861567:AAGxAJ463C5hrHMAA0RzYvjbOjdwFf1jCzM"

MIN_ORDERS = {"BTC": 0.001, "ETH": 0.01, "DASH": 0.1, "LTC": 0.1, "ETC": 0.1, "XRP": 10, "BCH": 0.001,
              "XMR": 0.1, "ZEC": 0.1, "QTUM": 0.1, "BTG": 0.1, "EOS": 1, "ICX": 1, "VEN": 1, "TRX": 100,
              "ELF": 10, "MITH": 10, "MCO": 10, "OMG": 0.1, "KNC": 1, "GNT": 10, "HSR": 1, "ZIL": 100,
              "ETHOS": 1, "PAY": 1, "WAX": 10, "POWR": 10, "LRC": 10, "GTO": 10, "STEEM": 10, "STRAT": 10,
              "ZRX": 10, "REP": 0.1, "AE": 1, "XEM": 10, "SNT": 10, "ADA": 10, "PPT": 1, "CTXC": 10,
              "CMT": 10, "THETA": 10, "WTC": 1, "ITC": 10, "WAVES": 1, "ARN": 10, "INS": 10, "PST": 100, 
              "TRUE": 10, "BCD": 10, "XLM": 100, "POLY": 100, "MTL": 10, "BSV": 0.1}

#----------------------------------------------------------------------------------------------------------------------
# 아래의 값을 적당히 수정해서 사용하세요.
#----------------------------------------------------------------------------------------------------------------------
INTERVAL = 1                                        # 매수 시도 interval (1초 기본)
DEBUG = False                                       # True: 매매 API 호출 안됨, False: 실제로 매매 API 호출

COIN_NUMS = 20                                      # 분산 투자 코인 개수 (자산/COIN_NUMS를 각 코인에 투자)
MOVING_AVERAGE_DURATION = 20                        # 이동평균 기간

TARGET_VOLATILITY = 2                               # 타겟 변동성 (%)

BALANCE = 0.8                                       # 자산의 투자비율 (0.0~1.0)
TRAILLING_STOP_MIN_PROOFIT = 0.4                    # 최소 30% 이상 수익이 발생한 경우에 Traillig Stop 동작
TRAILLING_STOP_GAP = 0.15                           # 최고점 대비 10% 하락시 매도

#----------------------------------------------------------------------------------------------------------------------
# Logging
#----------------------------------------------------------------------------------------------------------------------
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)

bot = telepot.Bot(token)

# Load account
with open("flag/bithumb.txt") as f:
    lines = f.readlines()
    key = lines[0].strip()
    secret = lines[1].strip()
    bithumb = pybithumb.Bithumb(key, secret)


def run_volatility_breakout(ticker):
    df = pybithumb.get_ohlcv(ticker)
    df = df.sort_index()
    df = df[-50:]

    df['ma'] = df['close'].rolling(window=MOVING_AVERAGE_DURATION).mean()              # 20일 이동평균 컬럼
    df['ma_shift'] = df['ma'].shift(1)                         # 20일 이동평균 컬럼에서 하나씩 내림
    df['bull'] = df['open'] > df['ma_shift']                     # 상승장/하락장 판단

    df['volatility'] = (df['high'] - df['low']) * 0.5               # 변동성 * 0.5
    df['target'] = df['open'] + df['volatility'].shift(1)           # 목표가 = 시가 + 전일 변동성 * 0.5
    df['er'] = np.where(df['bull'] & (df['high'] >= df['target']),  # 매수 조건
                        df['close'] / df['target'],                 # 매수시 수익률 (매도가/매수가)
                        1)                                          # 홀드시 수익률
    df['cumprod'] = df['er'].cumprod()
    return df['cumprod'][-2]


def make_sell_times(now):
    '''
    당일 23:50:00 시각과 23:50:10초를 만드는 함수
    :param now: DateTime
    :return:
    '''
    sell_time = datetime.datetime(year=now.year,
                                  month=now.month,
                                  day=now.day,
                                  hour=23,
                                  minute=50,
                                  second=0)
    sell_time_after_10secs = sell_time + datetime.timedelta(seconds=10)
    return sell_time, sell_time_after_10secs


def inquiry_cur_prices(tickers):
    '''
    모든 가상화폐에 대한 현재가 조회
    :param tickers: 티커 목록, ['BTC', 'XRP', ... ]
    :return: 현재가, {'BTC': 7200000, 'XRP': 500, ...}
    '''
    try:
        all = pybithumb.get_current_price("ALL")
        cur_prices = {ticker: float(all[ticker]['closing_price']) for ticker in tickers}
        return cur_prices
    except Exception as e:
        logger.info("inquiry_cur_prices error: {}".format(e))
        return None


def set_tickers_to_trade():
    global COIN_NUMS
    global BALANCE
    global MOVING_AVERAGE_DURATION

    result_list = []
    try:
        with open("flag/coinnum.txt") as c:
            coins = c.readlines()[0].strip()
            COIN_NUMS = int(coins)

        with open("flag/balance.txt") as b:
            rate = b.readlines()[0].strip()
            BALANCE = float(rate)

        with open("flag/moving_average_duration.txt") as b:
            rate = b.readlines()[0].strip()
            MOVING_AVERAGE_DURATION = int(rate)

        # 거래 코인수 읽어오기 - 동적변화를 위함
        # 추후 자산량 변화에 따라 자동으로 변화하도록 로직 구현
        # (자산이 많아지면 코인종류를 늘려 코인별 비중 축소)
        tickers = get_tickers()
        result = []

        for ticker in tickers:
            try:
                earning_rate = run_volatility_breakout(ticker)
                result.append((ticker, earning_rate))
            except:
                pass

        # 정렬
        ranked_data = sorted(result, key=lambda x:x[1], reverse=True)
        leng = min(len(ranked_data), int(COIN_NUMS * 2))
        with open("flag/ticker_list.txt", "w") as f:
            for data in ranked_data[:leng]:
                if float(data[1]) >= 1.0:
                    result_list.append(data[0])
                    f.write("{}\n".format(data[0]))

    except Exception as e:
        logger.info("Set Ticker Error : {}".format(e))

    return result_list


def get_tickers():
    tickers = pybithumb.get_tickers()
    buy_ticker_list = []

    try:
        tickers.remove('date')
    except:
        pass

#    all_tickers = []

#    try:
#        with open('flag/ticker_list.txt') as f:
#            lines = f.readlines()
#            for line in lines:
#                buy_tickers.append(line.strip('\n'))
#    except:
#        pass

    try:
        for ticker in tickers:
#            all_tickers.append(ticker)
            df = pybithumb.get_ohlcv(ticker)
            if (len(df) < 2):
                tickers.remove(ticker)
#            elif ticker not in buy_ticker_list:
#                tickers.remove(ticker)
    except Exception as e:
        logger.info("ticker error : {}".format(e))

    return tickers


def cal_noise(tickers, window=20):
    '''
    모든 가상화폐에 대한 최근 20일 noise의 평균을 계산
    :param tickers: 티커 리스트
    :param window: 평균을 위한 윈도우 길이
    :return:
    '''
    try:
        noise_dict = {}

        for ticker in tickers:
            df = pybithumb.get_ohlcv(ticker)
            noise = 1 - abs(df['open'] - df['close']) / (df['high'] - df['low'])
            average_noise = noise.rolling(window=window).mean()
            idx = -1
            if (len(df) > 1):
                idx = -2
            noise_dict[ticker] = average_noise[idx]

        return noise_dict
    except Exception as e:
        logger.info("cal_noise error : {}".format(e))
        return None


def cal_target(ticker, noises):
    '''
    각 코인에 대한 목표가 계산
    :param ticker: 코인에 대한 티커
    :return:
    '''
    try:
        df = pybithumb.get_ohlcv(ticker)
        yesterday = df.iloc[-2]
        today = df.iloc[-1]
        today_open = today['open']
        yesterday_high = yesterday['high']
        yesterday_low = yesterday['low']
        diff = yesterday_high - yesterday_low
        target = today_open + diff * noises[ticker]
        return target, diff
    except:
        logger.info("cal_target error {}".format(ticker))
        return None, None


def inquiry_high_prices(tickers):
    try:
        high_prices = {}
        for ticker in tickers:
            df = pybithumb.get_ohlcv(ticker)
            today = df.iloc[-1]
            today_high = today['high']
            high_prices[ticker] = today_high
        return high_prices
    except:
        logger.info("inquiry_high_prices error")
        return  {ticker:0 for ticker in tickers}


def inquiry_targets(tickers, noises):
    '''
    모든 코인에 대한 목표가 계산
    :param tickers: 코인에 대한 티커 리스트
    :return:
    '''
    targets = {}
    yesterday_diff = {}

    for ticker in tickers:
        targets[ticker], yesterday_diff[ticker] = cal_target(ticker, noises)
    return targets, yesterday_diff


# 최근 상승비율에 따른 매수비율 계산 (현재 미사용)
def cal_buy_ratio(ticker, target, yesterday_diff, sell_price):
    try:
        score = 0
        for day in range(3, 21):
            try:
                if (target >= cal_moving_average(ticker, day)):
                    score += 1
            except:
                pass

        yesterday_volatility = yesterday_diff * 100.0 / float(sell_price)

        ratio = (score / 18) * (TARGET_VOLATILITY / yesterday_volatility)
        return ratio
    except Exception as e:
        logger.info("cal_buy_ratio error : {}".format(e))
        return 0


def cal_moving_average(ticker="BTC", window=5):
    '''
    5일 이동평균을 계산
    :param ticker:
    :param window:
    :return:
    '''
    try:
        df = pybithumb.get_ohlcv(ticker)
        close = df['close']
        ma_series = close.rolling(window=window).mean()
        yesterday_ma = ma_series[-2]
        return yesterday_ma
    except:
        logger.info("cal_moving_average error")
        return None


def inquiry_moving_average(tickers):
    '''
    모든 코인에 대해 5일 이동평균값을 계산
    :param tickers: 티커 리스트
    :return:
    '''
    mas = {}
    for ticker in tickers:
        ma = cal_moving_average(ticker, MOVING_AVERAGE_DURATION)
        mas[ticker] = ma
    return mas


def try_buy(tickers, prices, targets, noises, mas, budget_per_coin, holdings, high_prices, yesterday_diff, now):
    '''
    모든 가상화폐에 대해 매수 조건 확인 후 매수 시도
    :param tickers: 티커 리스트
    :param prices: 현재가 리스트
    :param targets: 목표가 리스트
    :param noises: noise 리스트
    :param mas: 이동평균 리스트
    :param budget_per_coin: 코인 당 투자 금액
    :param holdings: 보유 여부 리스트
    :param high_prices: 당일 고가 리스트
    :return:
    '''
    tmp = ''
    try:
        for ticker in tickers:
            tmp = ticker
            price = prices[ticker]              # 현재가
            target = targets[ticker]            # 목표가
            ma = mas[ticker]                    # N일 이동평균
            high = high_prices[ticker]          # 당일 고가

            # 매수 조건
            # 1) 현재가가 목표가 이상이고
            # 2) 당일 고가가 목표가 대비 2% 이상 오르지 않았으며 (프로그램을 장중에 실행했을 때 고점찍고 하락중인 종목을 사지 않기 위해)
            # 3) 현재가가 5일 이동평균 이상이고
            # 4) 해당 코인을 보유하지 않았을 때
            # 5) 현재가가 100원 이상
            if holdings[ticker] is False:
                if price >= 100 and price >= target and target >= ma and high <= target * 1.02:
                    orderbook = pybithumb.get_orderbook(ticker)
                    asks = orderbook['asks']
                    sell_price = asks[0]['price']
					buy_ratio = 1
                    # 최근 21일간 상승비율만큼 매수
                    buy_ratio = cal_buy_ratio(ticker, targets[ticker], yesterday_diff[ticker], sell_price)
                    unit = (budget_per_coin / float(sell_price)) * buy_ratio
                    min_order = MIN_ORDERS.get(ticker, 0.001)

                    if unit >= min_order:
                        buy_ret = 'check'
                        if DEBUG is False:
                            logger.info("BUY [{}] {} {} - MIN : {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), ticker, unit, min_order))
                            buy_ret = bithumb.buy_market_order(ticker, unit)
                            logger.info("BUY Result : {}".format(buy_ret))
                        else:
                            logger.info("BUY API CALLED {} {}".format(ticker, unit))
                        time.sleep(INTERVAL)
                        if buy_ret != None:
                            if buy_ret != 'None':
                                holdings[ticker] = True

                                try:
                                    bot = telepot.Bot(token)
                                    ret = bot.sendMessage(348034499, "Buy {} {}".format(ticker, unit))
                                    logger.info("Telegram sent - {}".format(ticker))
                                except Exception as e:
                                    logger.info("Telegram send Error : {} {}".format(ticker, e))
                                    pass
    except Exception as e:
        logger.info("try buy error : {} - {}".format(tmp, e))
        pass


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
            if DEBUG is False:
                ret = bithumb.sell_market_order(ticker, unit)
                time.sleep(INTERVAL)
            else:
                logger.info("SELL API CALLED {} {}".format(ticker, unit))

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
            ret = None

            if unit >= min_order:
                if DEBUG is False:
                    try:
                        logger.info("Sell Coin: {} - {}".format(ticker, unit))
                        ret = bithumb.sell_market_order(ticker, unit)
                        logger.info(ret)
                        time.sleep(INTERVAL)
                        if ret is None:
                            retry_sell(ticker, unit, 10)
                    except Exception as e:
                        logger.info("try_sell error : {}".format(ticker))
                        logger.info(ret)
                        pass
                else:
                    logger.info("SELL API CALLED {} {}".format(ticker, unit))
    except Exception as e:
        logger.info("try_sell error : {}".format(e))
        pass


def try_profit_cut(tickers, prices, targets, holdings, high_prices, now):
    '''
    trailling stop
    :param tickers: 티커 리스트
    :param prices: 현재가 리스트
    :param targets: 목표가 리스트
    :param holdings: 보유 여부 리스트
    :return:
    '''
    tmp = ''
    try:
        for ticker in tickers:
            tmp = ticker
            price = prices[ticker]                          # 현재가
            target = targets[ticker]                        # 매수가
            high_price = high_prices[ticker]                # 당일 최고가
            gain = (price - target) / target                # 이익률: (매도가-매수가)/매수가

            if holdings[ticker] is True:
                if (gain >= TRAILLING_STOP_MIN_PROOFIT) or (gain <= -TRAILLING_STOP_GAP):
                    unit = bithumb.get_balance(ticker)[0]
                    min_order = MIN_ORDERS.get(ticker, 0.001)
                    if gain >= TRAILLING_STOP_MIN_PROOFIT:
                        unit = unit / 2

                    if unit >= min_order:
                        if DEBUG is False:
                            ret = bithumb.sell_market_order(ticker, unit)
                            time.sleep(INTERVAL)
                            if ret is None:
                                retry_sell(ticker, unit, 10)
                            else:
                                holdings[ticker] = False
                        else:
                            logger.info("Trailing Stop {} {}".format(ticker, unit))

    except Exception as e:
        logger.info("try_trailing_stop error : {} - {}".format(tmp, e))
        pass


def cal_budget(new_day_flag):
    '''
    한 코인에 대해 투자할 투자 금액 계산
    :return: 원화잔고/투자 코인 수
    '''
    try:
        now = datetime.datetime.now()
        if new_day_flag:
            now = datetime.datetime.now() + datetime.timedelta(1)

        krw_balance = bithumb.get_balance("BTC")[2]
        krw_balance = krw_balance * BALANCE
        budget_per_coin = int(krw_balance / COIN_NUMS)

        try:
            if not (os.path.isdir('balance_logs/')):
                os.makedirs(os.path.join('balance_logs'))
            if not (os.path.isdir('balance_logs/{}'.format(now.strftime('%Y')))):
                os.makedirs(os.path.join('balance_logs/{}'.format(now.strftime('%Y'))))

            with open("balance_logs/{}/balance_{}.txt".format(now.strftime('%Y'),
                                                              now.strftime('%m')),
                      "a") as fname:
                fname.write("{}\t{}\n".format(now.strftime('%Y%m%d'), krw_balance))
            bot = telepot.Bot(token)
            ret = bot.sendMessage(348034499, "Balance : {}".format(krw_balance))
        except Exception:
            pass

        return budget_per_coin
    except:
        return 0


def update_high_prices(tickers, high_prices, cur_prices):
    '''
    모든 코인에 대해서 당일 고가를 갱신하여 저장
    :param tickers: 티커 목록 리스트
    :param high_prices: 당일 고가
    :param cur_prices: 현재가
    :return:
    '''
    try:
        for ticker in tickers:
            cur_price = cur_prices[ticker]
            high_price = high_prices[ticker]
            if cur_price > high_price:
                high_prices[ticker] = cur_price
    except:
        pass


def print_status(now, tickers, targets, holdings):
    '''
    현재 상태를 출력
    :param now: 현재 시간
    :param tickers: 티커 리스트
    :param prices: 현재가 리스트
    :param targets: 목표가 리스트
    :param noises: noise 리스트
    :param mas: moving average 리스트
    :param high_prices: 당일 고가 리스트
    :return:
    '''
    ticker_list = tickers
    hold_list = holdings
    try:
        cnt = 0
        nameList = ''

        for ticker in tickers:
            if holdings[ticker] == True:
                cnt += 1
                nameList = nameList + ticker + ','

        print("[{}] 보유수 : {:2} / 보유리스트 : {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), cnt, nameList[:-1]))
    except Exception as e:
        logger.info("print_status error: {}".format(e))
        pass


def set_trade(new_day_flag):
    noises = None
    while noises is None:
        noises = cal_noise(tickers)
    targets = None
    yesterday_diff = None
    while targets is None:
        targets, yesterday_diff = inquiry_targets(tickers, noises)          # 코인별 목표가 계산
    mas = inquiry_moving_average(tickers)                                   # 코인별로 5일 이동평균 계산
    budget_per_coin = cal_budget(new_day_flag)                              # 코인별 최대 배팅 금액 계산

    holdings = {ticker:False for ticker in tickers}                         # 보유 상태 초기화

    return noises, targets, yesterday_diff, mas, budget_per_coin, holdings


#----------------------------------------------------------------------------------------------------------------------
# 매매 알고리즘 시작
#---------------------------------------------------------------------------------------------------------------------
ticker_list = {}
hold_list = {}

now = datetime.datetime.now()                                           # 현재 시간 조회
tomorrow = now + datetime.timedelta(1)


#----------------------------------------------------------------------------------------------------------------------
# Logging Start
#----------------------------------------------------------------------------------------------------------------------
try:
    if not(os.path.isdir('logs')):
        os.makedirs(os.path.join('logs'))
except Exception:
    pass

file_handler = logging.handlers.TimedRotatingFileHandler("logs/coin_log.log", when='midnight', interval=1, encoding='utf-8')
file_handler.suffix = "%Y%m%d"
stream_handler = logging.StreamHandler()
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

sell_time1, sell_time2 = make_sell_times(now)                           # 초기 매도 시간 설정

tickers = get_tickers()                                       # 티커 리스트 얻기
try_sell(tickers)                                               # 최초 실행시 보유 코인 매도
set_tickers_to_trade()

noises, targets, yesterday_diff, mas, budget_per_coin, holdings = set_trade(False)
high_prices = inquiry_high_prices(tickers)                              # 코인별 당일 고가 저장

while True:
    now = datetime.datetime.now()

    # 당일 청산 (23:50:00 ~ 23:50:10)
    if sell_time1 < now < sell_time2:
        logger.info("===== Sell Ticker : {} =====".format(now))

        holdings = {ticker:True for ticker in tickers}                         # 당일에는 더 이상 매수되지 않도록
        try_sell(tickers)                                                      # 각 가상화폐에 대해 매도 시도

        sys.exit()

    else:
        # 현재가 조회
        prices = inquiry_cur_prices(tickers)
        update_high_prices(tickers, high_prices, prices)
        print_status(now, tickers, targets, holdings)

        if prices is not None:
            # 매수
            try_buy(tickers, prices, targets, noises, mas, budget_per_coin, holdings, high_prices, yesterday_diff, now)
            # 손절/익절
            try_profit_cut(tickers, prices, targets, holdings, high_prices, now)

    time.sleep(INTERVAL)
