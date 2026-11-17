from datetime import datetime
from research.shared_reference.window_reference_reader_v1 import lookup,load_reference
def check(date,clock,longwant,shortwant):
    v=lookup(datetime.fromisoformat(date+'T'+clock+'+03:00'))
    assert (v['long'],v['short'])==(longwant,shortwant),(date,clock,v)
    assert v['production_authority'] is False
D=load_reference()
assert D['long_wanted_count']==44 and D['long_avoid_count']==26
check('2026-11-11','13:00','LONG_TARGET','SHORT_AVOID')
check('2026-11-12','01:00','LONG_AVOID','SHORT_RESEARCH_CANDIDATE')
check('2026-11-12','12:16','CONFLICT_NO_TRADE','CONFLICT_NO_TRADE')
check('2026-11-13','18:37','CONFLICT_NO_TRADE','CONFLICT_NO_TRADE')
check('2026-11-14','00:30','LONG_AVOID','SHORT_RESEARCH_CANDIDATE')
check('2026-11-14','23:30','LONG_AVOID','SHORT_RESEARCH_CANDIDATE')
check('2026-11-15','00:10','LONG_AVOID','SHORT_RESEARCH_CANDIDATE')
check('2026-11-15','06:45','CONFLICT_NO_TRADE','CONFLICT_NO_TRADE')
check('2026-11-15','00:30','LONG_TARGET','SHORT_AVOID')
check('2026-11-11','02:00','UNLABELLED','UNLABELLED')
try:
    lookup(datetime.fromisoformat('2026-11-15T00:30:00'))
    raise AssertionError('naive timestamp was accepted')
except ValueError: pass
print('PASS 12 common-reference invariants: interval boundaries, overlapping conflicts, short inverse research, TZ and never-live')
