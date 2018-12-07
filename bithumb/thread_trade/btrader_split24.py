#----------------------------------------------------------------------------------------------------------------------
# PyStock
# Larry Williams Volatility Breakout Strategy + Moving Average
#----------------------------------------------------------------------------------------------------------------------
# A version
#
#----------------------------------------------------------------------------------------------------------------------
import pybithumb
import time
import datetime
import logging
import logging.handlers
import os

class PyBithumb:
    _MIN_ORDERS = {
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
    _INTERVAL = 1                                        # 매수 시도 interval (1초 기본)    
    _COIN_NUMS = 15                                      # 분산 투자 코인 개수 (자산/_COIN_NUMS를 각 코인에 투자)
    _GAIN = 0.3                                          # 30% 이상 이익시 50% 물량 익절
    _TARGET_VOLATILITY = 2                               # 타겟 변동성 (%)
    
    def __init__(self, execute_mode=True, thread_cnt=1, total_thread_cnt=1):
        self.DEBUG = execute_mode                               # True: 매매 API 호출 안됨, False: 실제로 매매 API 호출
        self.start_hour = thread_cnt - 1
        self.total_thread_cnt = total_thread_cnt
        self.thread_cnt = thread_cnt
        #----------------------------------------------------------------------------------------------------------------------
        # Logging
        #----------------------------------------------------------------------------------------------------------------------
        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)

        try:
            if not(os.path.isdir('logs')):
                os.makedirs(os.path.join('logs'))
        except:
            pass
        
        log_file = "logs/log_thread{:1}.log".format(thread_cnt)        
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=100 * 1000000, backupCount=5)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        # Load account
        with open("flag/bithumb.txt") as f:
            lines = f.readlines()
            key = lines[0].strip()
            secret = lines[1].strip()
            self.bithumb = pybithumb.Bithumb(key, secret)

    def _make_sell_times(self, now):
        '''
        (기준시간 한시간전):50:00 시각과 (기준시간 한시간전):50:10초를 만드는 함수
        :param now: DateTime
        :return:
        '''
        sell_time = datetime.datetime(year=now.year,
                                      month=now.month,
                                      day=now.day,
                                      hour=(self.start_hour+23)%24,
                                      minute=50,
                                      second=0)
        sell_time_after_10secs = sell_time + datetime.timedelta(seconds=10)
        return sell_time, sell_time_after_10secs
    
    def _make_setup_times(self, now):
        '''
        다음날 (기준시간):01:00 시각과 (기준시간):01:10초를 만드는 함수
        :param now:
        :return:
        '''
        tomorrow = now + datetime.timedelta(1)
        setup_time = datetime.datetime(year=tomorrow.year,
                                       month=tomorrow.month,
                                       day=tomorrow.day,
                                       hour=self.start_hour,
                                       minute=1,
                                       second=0)
        setup_time_after_10secs = setup_time + datetime.timedelta(seconds=10)
        return setup_time, setup_time_after_10secs
    
    def inquiry_cur_prices(self, tickers):
        '''
        모든 가상화폐에 대한 현재가 조회
        :param tickers: 티커 목록, ['BTC', 'XRP', ... ]
        :return: 현재가, {'BTC': 7200000, 'XRP': 500, ...}
        '''
        try:
            all = pybithumb.get_current_price("ALL")
            cur_prices = {ticker: float(all[ticker]['closing_price']) for ticker in tickers}
            return cur_prices
        except:
            self.logger.info("inquiry_cur_prices error")
            return None
        
    def cal_noise(self, tickers, window=20):
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
                noise_dict[ticker] = average_noise[-2]
    
            return noise_dict
        except:
            self.logger.info("cal_noise error")
            return None
        
    def cal_target(self, ticker, noises):
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
            target = today_open + (yesterday_high - yesterday_low) * noises[ticker]
            return target, (yesterday_high - yesterday_low)
        except:
            self.logger.info("cal_target error {}".format(ticker))
            return None, None
        
    def inquiry_high_prices(self, tickers):
        try:
            high_prices = {}
            for ticker in tickers:
                df = pybithumb.get_ohlcv(ticker)
                today = df.iloc[-1]
                today_high = today['high']
                high_prices[ticker] = today_high
            return high_prices
        except:
            self.logger.info("inquiry_high_prices error")
            return  {ticker:0 for ticker in tickers}

    def inquiry_targets(self, tickers, noises):
        '''
        모든 코인에 대한 목표가 계산
        :param tickers: 코인에 대한 티커 리스트
        :return:
        '''
        targets = {}
        yesterday_diff = {}
        for ticker in tickers:
            targets[ticker], yesterday_diff[ticker] =self.cal_target(ticker, noises)
        return targets, yesterday_diff
    
    def cal_buy_ratio(self, ticker, target, yesterday_diff, sell_price):
        try:
            score = 0
            for day in range(3, 21):
                try:
                    if (target >= self.cal_moving_average(ticker, day)):
                        score += 1
                except:
                    pass
                
            yesterday_volatility = yesterday_diff * 100.0 / float(sell_price)
        
            ratio = (score / 18) * (self._TARGET_VOLATILITY / yesterday_volatility)
            return ratio
        except Exception as e:
            self.logger.info("cal_buy_ratio error : {}".format(e))
            return 0
    
    def cal_moving_average(self, ticker="BTC", window=5):
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
            self.logger.info("cal_moving_average error")
            return None
        
    def inquiry_moving_average(self, tickers):
        '''
        모든 코인에 대해 5일 이동평균값을 계산
        :param tickers: 티커 리스트
        :return:
        '''
        mas = {}
        for ticker in tickers:
            ma = self.cal_moving_average(ticker)
            mas[ticker] = ma
        return mas

    def try_buy(self, tickers, prices, targets, noises, mas, budget_per_coin, holdings, high_prices, hold_amount, yesterday_diff):
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
        try:
            for ticker in tickers:
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
                        buy_ratio = self.cal_buy_ratio(ticker, targets[ticker], yesterday_diff[ticker], sell_price)
                        unit = (budget_per_coin / float(sell_price)) * buy_ratio
    
                        if self.DEBUG is False:
                            self.bithumb.buy_market_order(ticker, unit)
                            self.logger.info("Try Buy {} {}".format(ticker, unit))
                        else:
                            self.logger.info("BUY API CALLED {} {}".format(ticker, unit))
                        time.sleep(self._INTERVAL)
                        holdings[ticker] = True
                        hold_amount[ticker] += unit
        except Exception as e:
            self.logger.info("try buy error : {}".format(e))
            pass

    def retry_sell(self, ticker, unit, hold_amount, retry_cnt=10):
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
                if self.DEBUG is False:
                    ret = self.bithumb.sell_market_order(ticker, unit)
                    time.sleep(self._INTERVAL)
                else:
                    self.logger.info("SELL API CALLED {} {}".format(ticker, unit))
    
                retry_cnt = retry_cnt - 1
            hold_amount[ticker] -= unit
        except:
            pass

    def try_sell(self, tickers, hold_amount):
        '''
        보유하고 있는 모든 코인에 대해 전량 매도
        :param tickers: 빗썸에서 지원하는 암호화폐의 티커 목록
        :return:
        '''
        try:
            for ticker in tickers:
#                unit = bithumb.get_balance(ticker)[0]
                unit = hold_amount[ticker]
                min_order = self._MIN_ORDERS.get(ticker, 0.001)
    
                if unit >= min_order:
                    if self.DEBUG is False:
                        ret = self.bithumb.sell_market_order(ticker, unit)
                        time.sleep(self._INTERVAL)
                        if ret is None:
                            self.retry_sell(ticker, unit, hold_amount, 10)
                        else:
                            hold_amount[ticker] -=  unit
                        self.logger.info("Try sell {} {}".format(ticker, unit))
                    else:
                        self.logger.info("SELL API CALLED {} {}".format(ticker, unit))
        except Exception as e:
            self.logger.info("try_sell error : {}".format(e))
            pass

    def try_profit_cut(self, tickers, prices, targets, holdings, hold_amount):
        '''
        trailling stop
        :param tickers: 티커 리스트
        :param prices: 현재가 리스트
        :param targets: 목표가 리스트
        :param holdings: 보유 여부 리스트
        :return:
        '''
        try:
            for ticker in tickers:
                price = prices[ticker]                          # 현재가
                target = targets[ticker]                        # 매수가
                gain = (price - target) / target                # 이익률: (매도가-매수가)/매수가
    
                if holdings[ticker] is True:
                    if gain >= self._GAIN:
                        unit = hold_amount[ticker] / 2 # 50% 물량 매도
#                        unit = self.bithumb.get_balance(ticker)[0] / 2   # 50% 물량 매도
                        min_order = self._MIN_ORDERS.get(ticker, 0.001)
    
                        if unit >= min_order:
                            if self.DEBUG is False:
                                ret = self.bithumb.sell_market_order(ticker, unit)
                                time.sleep(self._INTERVAL)
                                if ret is None:
                                    self.retry_sell(ticker, unit, hold_amount, 10)
                                else:
                                    holdings[ticker] = False
                                    hold_amount[ticker] -= unit
                            else:
                                self.logger.info("Trailing Stop {} {}".format(ticker, unit))
        except Exception as e:
            self.logger.info("try_trailing_stop error : {}".format(e))
            pass
    
    def cal_budget(self, thread=1):
        '''
        한 코인에 대해 투자할 투자 금액 계산
        :return: 원화잔고/투자 코인 수
        '''
        try:
            krw_balance = self.bithumb.get_balance("BTC")[2]
            budget_per_coin = int(krw_balance / self._COIN_NUMS / thread)
            return budget_per_coin
        except:
            return 0
    
    def update_high_prices(self, tickers, high_prices, cur_prices):
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

    def print_status(self, now, tickers, prices, targets, noises, mas, high_prices, holdings, hold_amount):
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
        try:
            cnt = 0
            nameList = ''
            coinAmount = ''

            for ticker in tickers:
                if holdings[ticker] == True:
                    cnt += 1
                    nameList = nameList + ticker + ','
                    coinAmount = coinAmount + str(hold_amount[ticker]) + ','
    
            print("Thread {:2} - 보유수 : {:2} | 보유리스트 : {}/{} | 시간 : {}".format(self.thread_cnt, cnt, nameList[:-1], coinAmount[:-1], now.strftime("%Y-%m-%d %H:%M:%S")))
        except Exception as e:
            self.logger.info("print_status error: {}".format(e))
            pass
        
    def run_trade(self):
        #----------------------------------------------------------------------------------------------------------------------
        # 매매 알고리즘 시작
        #---------------------------------------------------------------------------------------------------------------------
        now = datetime.datetime.now()                                           # 현재 시간 조회
        sell_time1, sell_time2 = self._make_sell_times(now)                           # 초기 매도 시간 설정
        setup_time1, setup_time2 = self._make_setup_times(now)                        # 초기 셋업 시간 설정
        
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
        
        noises = None
        while noises is None:
            noises = self.cal_noise(tickers)
        targets = None
        while targets is None:
            targets, yesterday_diff = self.inquiry_targets(tickers, noises)                                      # 코인별 목표가 계산
        mas = self.inquiry_moving_average(tickers)                                   # 코인별로 5일 이동평균 계산
        budget_per_coin = self.cal_budget(self.total_thread_cnt)                                          # 코인별 최대 배팅 금액 계산
        
        holdings = {ticker:False for ticker in tickers}                       # 보유 상태 초기화
        hold_amount = {ticker:0 for ticker in tickers}                        # 보유 수량 초기화
        high_prices = self.inquiry_high_prices(tickers)                              # 코인별 당일 고가 저장
        
        while True:
            now = datetime.datetime.now()
        
            # 당일 청산 (23:50:00 ~ 23:50:10)
            if sell_time1 < now < sell_time2:
                self.try_sell(tickers, hold_amount)                                                    # 각 가상화폐에 대해 매도 시도
                holdings = {ticker:True for ticker in tickers}                     # 당일에는 더 이상 매수되지 않도록
                time.sleep(10)
        
            # 새로운 거래일에 대한 데이터 셋업 (00:01:00 ~ 00:01:10)
            if setup_time1 < now < setup_time2:
                tickers = pybithumb.get_tickers()                                   # 티커 목록 갱신
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
                self.try_sell(tickers, hold_amount)                                                   # 매도 되지 않은 코인에 대해서 한 번 더 매도 시도
        
                noises = None
                while noises is None:
                    noises = self.cal_noise(tickers)
                targets = None
                while targets is None:
                    targets = self.inquiry_targets(tickers, noises)                                  # 목표가 갱신
                mas = self.inquiry_moving_average(tickers)                               # 이동평균 갱신
                budget_per_coin = self.cal_budget()                                      # 코인별 최대 배팅 금액 계산
        
                sell_time1, sell_time2 = self._make_sell_times(now)                       # 당일 매도 시간 갱신
                setup_time1, setup_time2 = self._make_setup_times(now)                    # 다음 거래일 셋업 시간 갱신
        
                holdings = {ticker:False for ticker in tickers}                   # 모든 코인에 대한 보유 상태 초기화
                high_prices = {ticker: 0 for ticker in tickers}                    # 코인별 당일 고가 초기화
                time.sleep(10)
        
            # 현재가 조회
            prices = self.inquiry_cur_prices(tickers)
            self.update_high_prices(tickers, high_prices, prices)
            self.print_status(now, tickers, prices, targets, noises, mas, high_prices, holdings, hold_amount)
        
            # 매수
            if prices is not None:
                self.try_buy(tickers, prices, targets, noises, mas, budget_per_coin, holdings, high_prices, hold_amount, yesterday_diff)
        
            # 익절
            self.try_profit_cut(tickers, prices, targets, holdings, hold_amount)
        
            time.sleep(self._INTERVAL)


if __name__ == "__main__":
    thread_cnt = 1
    total_thread_cnt = 1
    debug_flag = False                      # True: 매매 API 호출 안됨, False: 실제로 매매 API 호출
    
    runner = PyBithumb(debug_flag, thread_cnt, total_thread_cnt)
    runner.run_trade()

