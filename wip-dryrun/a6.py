import json, collections, sys
from llm_ports.evals import decompose as d
rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()]
rows=[r for r in rows if r.get('replayable')]
stat=collections.Counter(); bad=[]
for r in rows:
    fk=r.get('field_kinds') or {}
    lv=d.leaves(r['output'])
    bypath={l.path:l for l in lv}
    for c in r['claims']:
        f=c.get('field')
        leaf=bypath.get(f) if f else None
        if leaf is None: stat['claim_no_leaf']+=1; continue
        kind=fk.get(leaf.field) or fk.get(f) or 'free'
        if kind not in d.KINDS: kind='free'
        if leaf.field.endswith('ticker') or leaf.field in ('tickers','tickers_affected'): kind='free'
        pm=d.prematch(leaf,kind,input_text=r['input'])
        stat[f'matched_{kind}']+=1
        if pm.ruled:
            stat['settled']+=1
            if c['label'] not in ('supported','derived'): bad.append((r['case_id'],f,c['label'],leaf.value[:40]))
    stat['leaves']+=len(lv)
print(stat); print('OVERTURNS',bad)
