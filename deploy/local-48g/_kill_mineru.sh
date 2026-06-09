#!/usr/bin/env bash
pkill -9 -f mineru_server.py 2>/dev/null
pkill -9 -f resource_tracker 2>/dev/null
pkill -9 -f spawn_main 2>/dev/null
sleep 2
echo "stray mineru procs remaining:"; pgrep -af mineru_server.py | head || echo "  none"
