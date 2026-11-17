"""Read-only research labels — NEVER feed result into production order authorization."""
from __future__ import annotations
from datetime import datetime
from bisect import bisect_right
from pathlib import Path
import json
DATA_PATH=Path(__file__).with_name('BINANCE1_LONG_SHORT_WINDOWS_20261111_15_V1.json')
def load_reference():
    obj=json.loads(DATA_PATH.read_text())
    assert obj['research_only'] is True and obj['production_order_authority'] is False
    return obj
def lookup(timestamp_tr, *, dataset=None):
    """Turkey UTC+3 timestamp, half-open [start,end); returns UNLABELLED outside reference."""
    if not isinstance(timestamp_tr,datetime) or timestamp_tr.tzinfo is None:
        raise ValueError('timezone-aware datetime required')
    if timestamp_tr.utcoffset().total_seconds()!=3*3600:
        raise ValueError('TR timezone offset +03:00 required')
    data=dataset if dataset is not None else load_reference()
    rows=data['derived_disjoint_segments']
    t=timestamp_tr.isoformat()
    starts=[z['start_tr'] for z in rows]
    ix=bisect_right(starts,t)-1
    if ix>=0 and starts[ix]<=t<rows[ix]['end_tr']:
        labels=rows[ix]['labels']
        return dict(long=labels[0],short=labels[1],production_authority=False)
    return dict(long='UNLABELLED',short='UNLABELLED',production_authority=False)
