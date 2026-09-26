#!/usr/bin/env python3
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

"""Read-only model accounting and a diagnostic RSS run of an existing runner."""
import collections
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('GEMMA4_MEMORY_OUTPUT_DIR',
                          '/data/bt/tmp/gemma4-e2b-memory-explanation-20260925'))
sys.path.insert(0, str(REPO / 'tensor/examples/gemma4/litert/tools'))
from ai_edge_litert import schema_py_generated as ts
from schema import litertlm_header_schema_py_generated as ls

SOURCE = Path('/data/bt/models/hub/models--litert-community--gemma-4-E2B-it-litert-lm/blobs/ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42')
PROD = Path('/data/bt/tmp/gemma4-litert-production-20260924')
FIXTURE = REPO / 'local_docs/gemma4_phone_reproduction/fixtures/performance_1024_64.tsv'
MIB = 2**20


def _items(obj):
    """Decode scalar container metadata for the accounting report."""
    result = {}
    for index in range(obj.ItemsLength()):
        item = obj.Items(index)
        for name in ("UInt8", "Int8", "UInt16", "Int16", "UInt32", "Int32",
                     "UInt64", "Int64", "Float32", "Double", "Bool", "StringValue"):
            if item.ValueType() == getattr(ls.VData, name):
                raw = item.Value()
                value = getattr(ls, name)()
                value.Init(raw.Bytes, raw.Pos)
                value = value.Value()
                result[item.Key().decode("utf-8")] = (
                    value.decode("utf-8") if isinstance(value, bytes) else value)
                break
    return result


def header(data):
    end = int.from_bytes(data[24:32], 'little')
    return ls.LiteRTLMMetaData.GetRootAs(data[32:end], 0)


def sections(h):
    out = []
    for i in range(h.SectionMetadata().ObjectsLength()):
        o = h.SectionMetadata().Objects(i)
        out.append(dict(index=i, begin=o.BeginOffset(), end=o.EndOffset(),
                        bytes=o.EndOffset()-o.BeginOffset(), metadata=_items(o)))
    return out


def pages_for_rows(base, row_bytes, ids, page_size):
    pages = set()
    for token in ids:
        start = base + token * row_bytes
        pages.update(range(start // page_size, (start + row_bytes - 1) // page_size + 1))
    return pages


def static():
    report = {'units': 'bytes', 'audited_source': str(SOURCE)}
    url = 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm'
    req = urllib.request.Request(url, headers={'Range': 'bytes=0-65535'})
    with urllib.request.urlopen(req, timeout=30) as response:
        data = response.read(65536)
        report['published_content_range'] = response.headers.get('Content-Range')
    (OUT / 'published-header.bin').write_bytes(data)
    report['published_sections'] = sections(header(data))
    report['published_total_bytes'] = int(report['published_content_range'].split('/')[-1])
    fields = FIXTURE.read_text().strip().split('\t')
    prompt, decode = [list(map(int, x.split(','))) for x in fields[1:]]
    tokens = prompt + decode
    report['fixture'] = {'prompt_tokens': len(prompt), 'decode_tokens': len(decode),
                         'prompt_unique': len(set(prompt)), 'all_unique': len(set(tokens)),
                         'all_unique_including_padding': len(set(tokens + [0]))}
    ids = set(tokens + [0])
    f = SOURCE.open('rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    types = {v:k for k,v in vars(ts.TensorType).items() if isinstance(v,int)}
    report['audited_sections'] = sections(header(mm))
    report['section_accounting'] = []
    ple_tables = []
    for sec in report['audited_sections']:
        kind = sec['metadata'].get('model_type', '')
        if kind not in ('tf_lite_embedder', 'tf_lite_per_layer_embedder', 'tf_lite_prefill_decode'):
            continue
        model = ts.Model.GetRootAsModel(mm, sec['begin'])
        buffers = {}
        for i in range(model.BuffersLength()):
            b = model.Buffers(i)
            size = int(b.Size() or b.DataLength())
            if not size:
                continue
            at = sec['begin'] + int(b.Offset()) if b.Size() else b._tab.Vector(b._tab.Offset(4))
            buffers[i] = (at, size)
        code_ranges = {}
        q_ranges = {'scales': {}, 'zero_points': {}}
        fc_weight_ranges = {}
        for gi in range(model.SubgraphsLength()):
            g = model.Subgraphs(gi)
            for ti in range(g.TensorsLength()):
                t = g.Tensors(ti)
                if t.Buffer() in buffers:
                    code_ranges[buffers[t.Buffer()]] = types[t.Type()]
                q = t.Quantization()
                if q:
                    for name, offset, count, width in (
                        ('scales', 8, q.ScaleLength(), 4),
                        ('zero_points', 10, q.ZeroPointLength(), 8)):
                        if count:
                            at = q._tab.Vector(q._tab.Offset(offset))
                            q_ranges[name][(at, count*width)] = True
                if kind == 'tf_lite_per_layer_embedder' and gi == 0 and t.Type() == ts.TensorType.INT4:
                    assert [t.Shape(j) for j in range(t.ShapeLength())] == [262144, 256]
                    at, size = buffers[t.Buffer()]
                    ple_tables.append({'begin': at, 'bytes': size, 'row_bytes': 128})
            for oi in range(g.OperatorsLength()):
                op = g.Operators(oi)
                if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() == ts.BuiltinOperator.FULLY_CONNECTED:
                    weight = g.Tensors(op.Inputs(1))
                    if weight.Buffer() in buffers:
                        fc_weight_ranges[buffers[weight.Buffer()]] = types[weight.Type()]
        by_dtype = collections.Counter()
        for (_, size), dtype in code_ranges.items(): by_dtype[dtype] += size
        fc_by_dtype = collections.Counter()
        for (_, size), dtype in fc_weight_ranges.items(): fc_by_dtype[dtype] += size
        report['section_accounting'].append({
            'model_type': kind, 'section_bytes': sec['bytes'],
            'distinct_tensor_buffer_payload_by_dtype': dict(by_dtype),
            'distinct_fc_weight_payload_by_dtype': dict(fc_by_dtype),
            'distinct_scale_vector_bytes': sum(n for _, n in q_ranges['scales']),
            'distinct_zero_point_vector_bytes': sum(n for _, n in q_ranges['zero_points']),
        })
    assert len(ple_tables) == 35
    report['ple_tables'] = ple_tables
    report['ple_page_footprint'] = []
    for page_size in (4096, 16384):
        original = set()
        for t in ple_tables:
            original.update(pages_for_rows(t['begin'], 128, ids, page_size))
        token_major = pages_for_rows(0, 4480, ids, page_size)
        report['ple_page_footprint'].append({
            'page_size': page_size, 'unique_tokens_including_padding': len(ids),
            'raw_row_bytes': len(ids)*4480,
            'original_layer_major_required_page_bytes': len(original)*page_size,
            'runner_token_major_required_page_bytes': len(token_major)*page_size,
            'note': 'Distinct pages intersecting lookup rows only; excludes metadata, read-ahead, and retained pages.'
        })
    (OUT / 'static-accounting.json').write_text(json.dumps(report, indent=2)+'\n')
    for a in report['section_accounting']:
        print(a['model_type'], {k: v/MIB for k,v in a.items() if k.endswith('_bytes')},
              'FC MiB', {k:v/MIB for k,v in a['distinct_fc_weight_payload_by_dtype'].items()})
    print('fixture', report['fixture'])
    print('PLE pages', report['ple_page_footprint'])


def read_smaps(pid):
    entries = []
    current = None
    for line in Path(f'/proc/{pid}/smaps').read_text().splitlines():
        if re.match(r'^[0-9a-f]+-[0-9a-f]+ ', line):
            parts = line.split(None, 5)
            current = {'address': parts[0], 'path': parts[5] if len(parts)>5 else ''}
            entries.append(current)
        elif current and ':' in line:
            key, val = line.split(':',1)
            if re.match(r'^\s*\d+ kB$',val): current[key] = int(val.split()[0])*1024
    return entries


def classify(path):
    if path.endswith('.xnnpack_cache'): return 'packed_weight_cache'
    if path.endswith('/model.embed_tokens_per_layer.weight.i4'): return 'per_layer_embedding_codes'
    if path.endswith('/model.embed_tokens.weight.i4'): return 'main_embedding_codes'
    if '/scales/' in path: return 'mapped_scale_sources'
    if path.endswith('/gemma4-validated.tflite'): return 'exported_model'
    if not path or path.startswith('['): return 'anonymous_or_special'
    return 'other_files'


def resident():
    root = OUT / 'production-residency'
    root.mkdir(exist_ok=False)
    argv = ['taskset','-c','0,2,4,6', str(PROD/'runner-final'),
            '--bundle_dir='+str(PROD/'prepared/weights'),
            '--model_path='+str(PROD/'gemma4-validated.tflite'),
            '--weight_cache_path='+str(PROD/'gemma4-validated.tflite.xnnpack_cache'),
            '--cases_file='+str(FIXTURE), '--output_dir='+str(root/'capture'),
            '--num_threads=4','--cache_capacity=8448','--warmup_runs=0','--measured_runs=1']
    (root/'command.json').write_text(json.dumps({'argv':argv,'timings_valid':False},indent=2))
    snapshots = []
    with (root/'process.log').open('wb') as log:
        p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while p.poll() is None:
            try:
                entries = read_smaps(p.pid)
                groups = {}
                for e in entries:
                    g = groups.setdefault(classify(e['path']), collections.Counter())
                    for key in ('Size','Rss','Pss','Anonymous','Swap','Private_Dirty'):
                        g[key] += e.get(key,0)
                groups = {k:dict(v) for k,v in groups.items()}
                status = Path(f'/proc/{p.pid}/status').read_text()
                fields = {}
                for key in ('VmRSS','VmHWM','VmSwap'):
                    m = re.search(r'^'+key+r':\s+(\d+) kB$',status,re.M)
                    if m: fields[key] = int(m.group(1))*1024
                snapshots.append({'elapsed_s':time.monotonic()-started,'groups':groups,'status':fields})
            except (FileNotFoundError, ProcessLookupError): pass
            time.sleep(.05)
        exit_code = p.wait()
    report = {'exit_code':exit_code,'timings_valid':False,'samples':snapshots}
    (root/'residency.json').write_text(json.dumps(report,indent=2)+'\n')
    assert exit_code == 0, (root/'process.log').read_text()
    peak = max(snapshots,key=lambda s:sum(g['Rss'] for g in s['groups'].values()))
    last = next(s for s in reversed(snapshots) if s['groups'].get('packed_weight_cache',{}).get('Rss',0)>700*MIB)
    print('Sampled peak MiB',sum(g['Rss'] for g in peak['groups'].values())/MIB)
    print('Last execution snapshot', {k:round(v['Rss']/MIB,3) for k,v in last['groups'].items()})
    print('Last anonymous and swap MiB', sum(v['Anonymous'] for v in last['groups'].values())/MIB,
          sum(v['Swap'] for v in last['groups'].values())/MIB)


def lookup():
    # Separate processes for the two layouts. Read every code byte in the same
    # token rows; this is a memory diagnostic, not an inference/timing test.
    layout = sys.argv[2]
    report = json.loads((OUT/'static-accounting.json').read_text())
    fields = FIXTURE.read_text().strip().split('\t')
    ids = set([0] + [int(v) for field in fields[1:] for v in field.split(',')])
    source = SOURCE if layout == 'layer' else PROD/'prepared/weights/tensors/model.embed_tokens_per_layer.weight.i4'
    f = source.open('rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    regions = report['ple_tables'] if layout == 'layer' else [{'begin':0,'row_bytes':4480}]
    def mapped_rss():
        return sum(e.get('Rss',0) for e in read_smaps(os.getpid()) if e['path']==str(source))
    before = mapped_rss()
    checksum = 0
    digest = hashlib.sha256()
    for token in sorted(ids):
        for region in regions:
            start = region['begin'] + token*region['row_bytes']
            row = mm[start:start+region['row_bytes']]
            checksum += sum(row)
            digest.update(row)
    after = mapped_rss()
    result = {'layout':layout, 'tokens':len(ids), 'checksum':checksum, 'lookup_sha256':digest.hexdigest(),
              'mapped_rss_before_bytes':before, 'mapped_rss_after_bytes':after,
              'delta_bytes':after-before, 'page_size':os.sysconf('SC_PAGESIZE'),
              'scope':'PLE code reads only; default mmap, no cache flushing or forced pageout; not an inference benchmark'}
    (OUT/f'lookup-{layout}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    {'static':static, 'resident':resident, 'lookup':lookup}[sys.argv[1]]()
