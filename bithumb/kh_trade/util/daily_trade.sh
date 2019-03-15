#!/bin/bash

if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

if [ -f ~/tmp/daily.out ]; then
	mv ~/tmp/daily.out ~/tmp/daily_$(date +%y%m%d -d '-1days').out
fi

(
	cd ~/cryptocurrency-trading-bot/bithumb/kh_trade/
	nohup python3 trade_oneday.py &
)

