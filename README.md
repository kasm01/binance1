binance1-pro/
│
├── README.md
├── requirements.txt
├── .env.example
├── main.py
│
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── credentials.py
│   ├── redis_config.py
│   └── supervisor.conf
│
├── core/
│   ├── __init__.py
│   ├── logger.py
│   ├── utils.py
│   ├── exceptions.py
│   ├── notifier.py
│   └── cache_manager.py
│
├── data/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py
│   ├── online_learning.py
│   └── batch_learning.py
│
├── models/
│   ├── __init__.py
│   ├── ensemble_model.py
│   ├── lstm_model.py
│   ├── lightgbm_model.py
│   ├── catboost_model.py
│   ├── hyperparameter_tuner.py
│   └── fallback_model.py
│
├── trading/
│   ├── __init__.py
│   ├── position_manager.py
│   ├── capital_manager.py
│   ├── risk_manager.py
│   ├── strategy_engine.py
│   ├── trade_executor.py
│   ├── multi_trade_engine.py
│   └── fallback_trade.py
│
├── websocket/
│   ├── __init__.py
│   ├── stream_handler.py
│   ├── binance_ws.py
│   └── reconnect_manager.py
│
├── monitoring/
│   ├── __init__.py
│   ├── performance_tracker.py
│   ├── system_health.py
│   ├── trade_logger.py
│   └── alert_system.py
│
├── telegram/
│   ├── __init__.py
│   ├── telegram_bot.py
│   ├── commands.py
│   └── message_formatter.py
│
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_trading.py
│   ├── test_api.py
│   ├── test_redis.py
│   └── test_notifier.py
│
├── deployment/
│   ├── __init__.py
│   ├── dockerfile
│   ├── cloud_run.yaml
│   ├── setup_google_cloud.sh
│   └── startup_supervisor.sh
│
└── logs/
    ├── trades.log
    ├── errors.log
    └── system.log
