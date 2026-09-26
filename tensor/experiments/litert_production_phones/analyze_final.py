# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from pathlib import Path
import statistics as st

ROOT=Path('/data/bt/tmp/gemma4-litert-production-phones-20260924')
historical=json.loads((ROOT/'historical-native.json').read_text())
result={'configuration':{'prompt_tokens':1024,'forced_decode_tokens':64,'threads':4,'kv_capacity':8448,'prefill_chunk_rows':128,'kv_alignment':32,'warmup_sessions_per_process':1,'measured_sessions_per_process':3,'process_order':['production','native','native','production'],'decode_timer':'Includes runner invocation, logits access, finite validation, and greedy selection; excludes result logging and disk dumps.'},'devices':{}}
for name in ['tecno','pixel8','samsung']:
    root=ROOT/name
    if not (root/'completed-final.json').exists():
        print(name,'pending')
        continue
    summaries={}
    blocks={}
    peaks={}
    telemetry=[]
    for variant in ['production','native']:
        rows=[]
        block_values=[]
        rss=[]
        for block in [0,1]:
            d=root/f'final-{variant}-b{block}'
            r=json.loads((d/'summary.json').read_text())
            assert len(r)==3
            assert json.loads((d/'status.json').read_text())['returncode']==0
            assert json.loads((d/'command.json').read_text())['timings_valid_for_benchmark']
            for session in r:
                assert len(session['argmax'])==65
                assert session['argmax']==historical[name]['argmax'],(name,variant,block,session['argmax'])
            if variant=='production':
                rss += [s['peak_rss_mib'] for s in r]
            else:
                rss += [json.loads((d/'capture/run.json').read_text())['peak_process_rss_kib']/1024]
            rows+=r
            block_values.append({'decode_ms':st.median(s['decode_ms'] for s in r),'first_token_ms':st.median(s['first_logits_ms'] for s in r)})
            telemetry+=json.loads((d/'telemetry.json').read_text())
        summaries[variant]={'sessions':6,'decode_ms':st.median(r['decode_ms'] for r in rows),'decode_session_range_ms':[min(r['decode_ms'] for r in rows),max(r['decode_ms'] for r in rows)],'first_token_ms':st.median(r['first_logits_ms'] for r in rows),'request_1024_plus_64_ms':st.median(r['first_logits_ms']+64*r['decode_ms'] for r in rows),'peak_rss_mib':max(rss),'blocks':block_values}
        if variant=='production':
            summaries[variant]['model_ms']=st.median(r['model_ms'] for r in rows)
            summaries[variant]['validation_selection_ms']=st.median(r['decode_ms']-r['model_ms'] for r in rows)
    old=historical[name]
    a,b=summaries['native'],summaries['production']
    info={'historical_native':{k:v for k,v in old.items() if k not in ['argmax','files']},**summaries,'production_vs_historical_decode_percent':100*(b['decode_ms']/old['decode_ms']-1),'production_vs_fresh_native_decode_percent':100*(b['decode_ms']/a['decode_ms']-1),'production_vs_fresh_native_first_token_percent':100*(b['first_token_ms']/a['first_token_ms']-1),'production_vs_fresh_native_request_percent':100*(b['request_1024_plus_64_ms']/a['request_1024_plus_64_ms']-1),'argmax_predictions_checked':65*12,'all_argmax_match_historical':True,'max_sampled_cpu_c':max(t['cpu_c'] for t in telemetry),'max_sampled_skin_c':max(t['skin_c'] for t in telemetry),'observed_cpu_cooling':any(any(t['cpu_cooling_states'].values()) for t in telemetry)}
    result['devices'][name]=info
    print(name, 'historical',round(old['decode_ms'],3),'fresh native',round(a['decode_ms'],3),'production',round(b['decode_ms'],3),'delta %',round(info['production_vs_fresh_native_decode_percent'],2),'first token',round(a['first_token_ms'],1),'->',round(b['first_token_ms'],1),'peak RSS',round(a['peak_rss_mib'],1),'->',round(b['peak_rss_mib'],1))
(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
