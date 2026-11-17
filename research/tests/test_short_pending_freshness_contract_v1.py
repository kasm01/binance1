from core.short_pending_freshness_contract import check_independent_model_revalidation as check
a=dict(intent_id='origin1',source_signal_event_id='sig-orig',model_provider='short_v6',completed_close_ms=600000)
b=dict(intent_id='new2',source_signal_event_id='sig-new',model_provider='short_v6',completed_close_ms=780000,
       model_score=0.58,model_score_gate_threshold=0.555,model_score_gate_passed=True)
assert check(a,b,now_ms=790000)[0]
assert not check(a,dict(b,intent_id='origin1'),now_ms=790000)[0]
assert not check(a,dict(b,source_signal_event_id='sig-orig'),now_ms=790000)[0]
assert not check(a,dict(b,completed_close_ms=600000),now_ms=790000)[0]
assert not check(a,dict(b,model_score=0.50),now_ms=790000)[0]
assert not check(a,dict(b,model_provider='other'),now_ms=790000)[0]
assert not check(a,dict(b,model_score_gate_passed=False),now_ms=790000)[0]
assert not check(a,dict(b,completed_close_ms=780000),now_ms=1400000)[0]
assert not check(a,dict(b,completed_close_ms=None),now_ms=790000)[0]
print('PASS: independent SHORT model intent and completed-bar freshness contract (8 cases)')
