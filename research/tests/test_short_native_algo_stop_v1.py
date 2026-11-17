#!/usr/bin/env python3
"""Mock Binance-only SHORT1 stop tests. DRY_RUN and no external requests."""
from unittest.mock import patch
import core.short_native_algo_stop as m
from core.trade_executor_short import TradeExecutorShort

def algo(id,client,price='15.15'):
    return dict(symbol='INJUSDT',side='BUY',positionSide='SHORT',orderType='STOP_MARKET',
                closePosition=True,algoId=id,clientAlgoId=client,triggerPrice=price,algoStatus='NEW')

def run():
    cid=m.PREFIX+'123'
    with patch('core.short_native_algo_stop.get_open',side_effect=[[],[algo(12,cid)]]),         patch('core.short_native_algo_stop._signed',return_value={'algoId':12,'clientAlgoId':cid}) as write,         patch('core.short_native_algo_stop.time.time',return_value=.123):
        out=m.place('INJUSDT',15.15)
        assert out['order']['algoId']==12 and write.call_count==1
        args=write.call_args.args
        assert args[:2]==('POST','/fapi/v1/algoOrder')
        payload=args[2]
        assert (payload['side'],payload['positionSide'],payload['type'],payload['closePosition'],payload['triggerPrice'])==('BUY','SHORT','STOP_MARKET','true','15.15')
        print('TEST_PASS placed short algo STOP_MARKET close-all')
    with patch('core.short_native_algo_stop.get_open',return_value=[algo(13,'foreign')]),         patch('core.short_native_algo_stop._signed') as write:
        try:m.place('INJUSDT',15.15)
        except RuntimeError as e:assert 'existing_short_stop' in str(e)
        else:raise AssertionError('preexisting stop must block')
        assert not write.called
        print('TEST_PASS existing/foreign stop never overwritten')
    with patch('core.short_native_algo_stop.get_open',side_effect=[[algo(12,cid)],[]]),         patch('core.short_native_algo_stop._signed',return_value={'algoId':12}) as write:
        assert m.cancel_owned('INJUSDT',12)['status']=='canceled'
        assert write.call_args.args[0:2]==('DELETE','/fapi/v1/algoOrder')
        print('TEST_PASS own tagged stop canceled')
    with patch('core.short_native_algo_stop.get_open',return_value=[algo(12,'foreign')]),         patch('core.short_native_algo_stop._signed') as write:
        assert m.cancel_owned('INJUSDT',12)['status']=='not_owned_or_not_open'
        assert not write.called
        print('TEST_PASS foreign stop remains untouched')
    class Dummy:
        dry_run=True;short_native_stop_enable=True;roi_hard_stop_pct=-.12
        short_native_stop_working_type='MARK_PRICE';short_native_stop_price_protect=False
        hedge_mode_enabled=True
        _normalize_short_stop_price=lambda self,s,p:round(p,3)
    with patch('core.short_native_algo_stop._signed') as write:
        out=TradeExecutorShort._place_native_short_stop(Dummy(),symbol='INJUSDT',entry_price=15.0,leverage=3)
        assert out['status']=='dry_run' and abs(out['stop_price']-15.6)<1e-10 and not write.called
        print('TEST_PASS SHORT1 dry-run stop computes +4% underlying price at ROI -12% / 3x and no POST')
    print('ALL_SHORT1_ALGO_TESTS_OK')

if __name__=='__main__':run()
