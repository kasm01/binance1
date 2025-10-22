#!/bin/bash

# Supervisor servisini başlat
sudo service supervisor start

# Supervisor konfigürasyonu kopyala
sudo cp config/supervisor.conf /etc/supervisor/conf.d/binance1-pro-bot.conf

# Supervisor ile botu başlat
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start binance1-pro-bot
