#!/usr/bin/env python3
"""SHORT1 parity with LONG1 price-risk thresholds. No real Binance writes."""
from decimal import Decimal
from dotenv import dotenv_values
from unittest.mock import patch
from core.trade_executor_short import TradeExecutorShort
from scripts.short1_native_stop_reconciler import expected_native_stop

config=dotenv_values('.env')
assert config['DRY_RUN']=='1' and config['ARMED']=='0'
assert config['FIXED_ROI_TP_ENABLE']=='1'
assert abs(float(config['FIXED_ROI_TP_PCT'])-.0125)<1e-12
assert float(config['FIXED_ROI_TP_GIVEBACK'])==0.0
assert abs(float(config['ROI_HARD_STOP_PCT'])+0.0897)<1e-12
assert abs(float(config['SHORT_DYNAMIC_STOP_CATASTROPHIC_PCT'])-.035)<1e-12
assert abs(float(config['SHORT_DYNAMIC_STOP_CONFIRM_SEC'])-30)<1e-12
assert config['SHORT_NATIVE_STOP_ENABLE']=='1'

class D:
  dry_run=True
  logger=None
  short_native_stop_enable=True
  short_native_stop_working_type='MARK_PRICE'
  short_native_stop_price_protect=False
  hedge_mode_enabled=True
  roi_hard_stop_pct=float(config['ROI_HARD_STOP_PCT'])
  short_dynamic_stop_enable=True
  short_dynamic_stop_confirm_sec=30
  short_dynamic_stop_catastrophic_pct=.035
  fixed_roi_tp_enable=True
  fixed_roi_tp_pct=.0125
  fixed_roi_tp_giveback=0
  _set_position=lambda self,s,p:None
  _get_symbol_info=lambda self,s:{'pricePrecision':3,'filters':[{'filterType':'PRICE_FILTER','tickSize':'0.001'}]}
  _normalize_short_stop_price=TradeExecutorShort._normalize_short_stop_price
  _floor_to_step=staticmethod(TradeExecutorShort._floor_to_step)

d=D()
# 15.000 short, 3x: 15.000*(1+.0897/3)=15.4485 -> floor 15.448 (not wider)
stop=expected_native_stop('15','3','0.001',config['ROI_HARD_STOP_PCT'])
assert stop==Decimal('15.448'),stop
res=TradeExecutorShort._place_native_short_stop(d,symbol='INJUSDT',entry_price=15.0,leverage=3)
assert res['status']=='dry_run' and abs(res['stop_price']-15.448)<1e-8,res
assert res['payload']['side']=='BUY' and res['payload']['positionSide']=='SHORT'
assert res['payload']['workingType']=='MARK_PRICE'
assert abs(TradeExecutorShort._calc_short_roi(d,{'entry_price':15,'leverage':3},15.4485)+.0897)<1e-10
assert abs(TradeExecutorShort._calc_short_roi(d,{'entry_price':15,'leverage':3},15.525)+.105)<1e-10
position={'entry_price':15.0,'dynamic_stop_pending_ts':0,'leverage':3}
with patch('core.trade_executor_short.time.time',return_value=1000):
 state=TradeExecutorShort._evaluate_short_dynamic_stop_state(d,symbol='INJUSDT',position=position,price=15.4485,roi=-.0897)
assert state['reason']=='dynamic_stop_pending',state
with patch('core.trade_executor_short.time.time',return_value=1031):
 state=TradeExecutorShort._evaluate_short_dynamic_stop_state(d,symbol='INJUSDT',position=position,price=15.455,roi=-.091)
assert state['reason']=='dynamic_stop_confirmed',state
with patch('core.trade_executor_short.time.time',return_value=1001):
 state=TradeExecutorShort._evaluate_short_dynamic_stop_state(d,symbol='INJUSDT',position=dict(position,dynamic_stop_pending_ts=1000),price=15.525,roi=-.105)
assert state['reason']=='dynamic_stop_catastrophic',state
tp=TradeExecutorShort._resolve_short_fixed_roi_tp_state(d,old_best_roi=.01,best_roi=.0125)
assert tp['armed_now'] and tp['giveback']==0 and tp['trigger_roi']==.0125,tp
assert TradeExecutorShort._resolve_short_fixed_roi_tp_state(d,old_best_roi=0,best_roi=.0124) is None
print('PASS SHORT1 LONG1 parity: native stop 2.99% price, 30s app confirm, 3.5% catastrophe, 1.25% ROI TP, 0 giveback')
print('NOTE: Exchange native stop triggers immediately at mark price; 30s applies only app-managed soft stop.')
