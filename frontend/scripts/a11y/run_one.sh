#!/bin/bash
# usage: one.sh width mode path
pkill -f "remote-debugging-port=9333" 2>/dev/null; sleep 2
("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --remote-debugging-port=9333 --user-data-dir=/tmp/chrome-o$RANDOM --no-first-run about:blank > /dev/null 2>&1 &)
sleep 10
/Users/sid/Claude/swarmmind/.venv/bin/python /tmp/kbd_axe.py "$1" "$2" "$3" 2>&1 | grep -v loaded | tail -4 | cut -c1-400
