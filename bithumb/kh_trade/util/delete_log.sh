#!/bin/bash

if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

	cd ~/tmp
	find -maxdepth 1 -mtime +5 -delete
)

