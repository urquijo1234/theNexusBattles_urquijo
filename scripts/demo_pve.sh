#!/bin/bash
# Demo PvE: levanta server, IA y AIclient2 contra Autoclient1
set -e
# Este script es solo ilustrativo
nohup npm run aiclient2 >/tmp/aiclient2.log 2>&1 &
echo "Demo PvE iniciada (ver logs en /tmp/aiclient2.log)"
