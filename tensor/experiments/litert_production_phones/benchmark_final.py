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

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time

ROOT = Path('/data/bt/tmp/gemma4-litert-production-phones-20260924')
REPO = Path(__file__).resolve().parents[3]
REMOTE = '/data/local/tmp/gemma4-litert-production-phones-20260924'
OLD = '/data/local/tmp/gemma4-qat-q4_0-perf-20260914'
MODEL = Path('/data/bt/tmp/gemma4-litert-production-20260924/gemma4-validated.tflite')
DEVICES = {
    'tecno': ('144512553T000099', 'f0', '/data/local/tmp/gemma4-native-phones-20260923/baseline', '875026200a50fae7e504b1ef3cc77256eb543e651e566e4b0bcf9d1d6da43906'),
    'pixel8': ('3A111FDJH005A0', '1e0', '/data/local/tmp/gemma4-native-phones-20260923/baseline', '875026200a50fae7e504b1ef3cc77256eb543e651e566e4b0bcf9d1d6da43906'),
    'samsung': ('R5CY51BZC1W', 'f0', OLD + '/native.android', 'fbe9897ce1e6bf05ca00536077dbad255b57204d4bf9d5ce666bb38064df6f74'),
}
EXPECTED_MANIFEST = '3cdd41d803c05a4b29729a9892451122987675867ae1841709b2b480ddd826bf'
EXPECTED_FIXTURE = '01009f2da16a6c32e7288323029d019b49e1507022fd711f471c8001824b4eef'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

class Phone:
    def __init__(self, name):
        self.name = name
        self.serial, self.mask, self.native, self.native_hash = DEVICES[name]
        self.root = ROOT / name
        self.root.mkdir(exist_ok=True)
        self.last_end = 0

    def adb(self, *args, **kwargs):
        return subprocess.run(['adb', '-s', self.serial, *map(str,args)], **kwargs)

    def shell(self, argv, timeout=60):
        command = argv if isinstance(argv,str) else shlex.join(map(str,argv))
        return self.adb('shell', command, capture_output=True, text=True, check=True, timeout=timeout).stdout

    def prepare(self):
        self.shell(['mkdir','-p',REMOTE])
        for source,name in [(ROOT/'runner-timed.android','runner-timed.android'), (MODEL,'gemma4.tflite'), (ROOT/'bundle.expected.sha256','bundle.expected.sha256'), (REPO/'tensor/examples/gemma4/litert/fixtures/smoke.tsv','smoke.tsv')]:
            if name == 'gemma4.tflite':
                check = self.adb('shell', 'sha256sum '+shlex.quote(REMOTE+'/'+name), capture_output=True, text=True)
                if check.returncode == 0 and check.stdout.split()[0] == MODEL_HASH:
                    continue
            log=self.adb('push','-Z',source,REMOTE+'/'+name,capture_output=True,text=True)
            (self.root/('push-'+name+'.log')).write_text(log.stdout+log.stderr)
            log.check_returncode()
        self.shell(['chmod','755',REMOTE+'/runner-timed.android'])
        files={REMOTE+'/runner-timed.android':sha(ROOT/'runner-timed.android'), REMOTE+'/gemma4.tflite':MODEL_HASH, self.native:self.native_hash, OLD+'/matched-bundle/manifest.json':EXPECTED_MANIFEST, OLD+'/fixtures/performance_1024_64.tsv':EXPECTED_FIXTURE}
        actual={}
        for line in self.shell(['sha256sum',*files],timeout=180).splitlines():
            h,p=line.split(maxsplit=1)
            actual[p]=h
        assert actual==files,(self.name,actual,files)
        checked=self.shell('cd '+shlex.quote(OLD+'/matched-bundle')+' && sha256sum -c '+shlex.quote(REMOTE+'/bundle.expected.sha256'),timeout=240)
        (self.root/'bundle-validation.log').write_text(checked)
        assert len(checked.splitlines())==1394 and all(line.endswith(': OK') for line in checked.splitlines())
        save(self.root/'identity-final.json',{'serial':self.serial,'verified_sha256':actual,'coefficient_files_checked':1394,'mask':self.mask,'threads':4})
        save(self.root/'device.json',{k:self.shell(cmd) for k,cmd in {
            'fingerprint':['getprop','ro.build.fingerprint'], 'model':['getprop','ro.product.model'], 'soc':['getprop','ro.soc.model'], 'cpuinfo':['cat','/proc/cpuinfo'], 'policies':'cat /sys/devices/system/cpu/cpufreq/policy*/related_cpus /sys/devices/system/cpu/cpufreq/policy*/cpuinfo_max_freq', 'meminfo':['cat','/proc/meminfo']}.items()})
        print(self.name,'prepared and all hashes verified',flush=True)

    def thermal(self):
        raw=self.shell(['dumpsys','thermalservice'])
        live=raw.split('Current temperatures from HAL:')[1].split('Current cooling devices from HAL:')[0]
        values=[(float(v),int(t),n) for v,t,n in re.findall(r'Temperature\{mValue=([-\d.]+), mType=([-\d]+), mName=([^,]+), mStatus=\d+\}',live)]
        cpu=max(v for v,t,n in values if t==0)
        skin=max(v for v,t,n in values if t==3)
        soc=next((v for v,t,n in values if t==13 or n=='soc_therm'),None)
        cooling={n:int(v) for v,n in re.findall(r'CoolingDevice\{mValue=(\d+), mType=2, mName=([^}]+)\}',raw)}
        eligible=cpu<=45 and skin<=34 and (soc is None or soc<=39) and not any(cooling.values())
        return {'epoch_seconds':time.time(),'cpu_c':cpu,'skin_c':skin,'soc_c':soc,'cpu_cooling_states':cooling,'eligible':eligible,'raw':raw,'policy_current_khz':self.shell('cat /sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq').split()}

    def gate(self,directory):
        readings=[]
        consecutive=0
        start=time.monotonic()
        while time.monotonic()-start<600:
            r=self.thermal()
            readings.append(r)
            save(directory/'start-gate.json',readings)
            consecutive=consecutive+1 if r['eligible'] else 0
            if consecutive>=2 and time.monotonic()-self.last_end>=15:
                return
            if len(readings)%6==0:
                print(self.name,'cooling',round(r['cpu_c'],1),round(r['skin_c'],1),flush=True)
            time.sleep(5)
        raise RuntimeError(self.name+': start temperature gate not reached')

    def capture(self,name,variant='production',smoke=False,dump=False,repeats=3):
        directory=self.root/name
        directory.mkdir()
        remote=REMOTE+'/'+name
        self.shell(['test','!','-e',remote])
        self.gate(directory)
        fixture=REMOTE+'/smoke.tsv' if smoke else OLD+'/fixtures/performance_1024_64.tsv'
        binary=REMOTE+'/runner-timed.android' if variant=='production' else self.native
        argv=['taskset',self.mask,binary,'--bundle_dir='+OLD+'/matched-bundle','--cases_file='+fixture,'--output_dir='+remote,'--num_threads=4','--cache_capacity=8448','--warmup_runs='+('0' if smoke or dump else '1'),'--measured_runs='+str(repeats)]
        if variant=='production':
            argv+=['--model_path='+REMOTE+'/gemma4.tflite']
            if dump: argv+=['--dump_outputs']
        else:
            argv+=['--prefill_chunk_rows=128','--kv_alignment=32','--reuse_runtimes=true','--share_workspace=true','--preserve_static_int2=true','--memory_report=false','--dump_full_logits='+str(dump).lower()]
        save(directory/'command.json',{'serial':self.serial,'argv':argv,'timings_valid_for_benchmark':not(smoke or dump)})
        samples=[self.thermal()]
        (directory/'meminfo-before.txt').write_text(self.shell(['cat','/proc/meminfo']))
        print(self.name,'START',name,flush=True)
        start=time.monotonic()
        with (directory/'process.log').open('w') as log:
            proc=subprocess.Popen(['adb','-s',self.serial,'shell',shlex.join(argv)],stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.monotonic()-start>600:
                    raise RuntimeError(self.name+' timeout: '+name)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    samples.append(self.thermal())
                    save(directory/'telemetry.json',samples)
        self.last_end=time.monotonic()
        samples.append(self.thermal())
        save(directory/'telemetry.json',samples)
        save(directory/'status.json',{'returncode':proc.returncode,'wall_seconds':time.monotonic()-start})
        (directory/'meminfo-after.txt').write_text(self.shell(['cat','/proc/meminfo']))
        assert proc.returncode==0,(directory/'process.log').read_text()[-8000:]
        if dump:
            hashes=self.shell('sha256sum '+shlex.quote(remote)+'/*.f32',timeout=180)
            (directory/'logits.sha256').write_text(hashes)
            target=remote+'/timings.jsonl' if variant=='production' else remote+'/performance_1024_64.run_000.json'
            self.adb('pull',target,directory,check=True,capture_output=True)
        else:
            self.adb('pull',remote+'/.',directory/'capture',check=True,capture_output=True)
        if not smoke and not dump:
            if variant=='production':
                rows=[json.loads(s) for s in (directory/'capture/timings.jsonl').read_text().splitlines()]
                summaries=[]
                for rep in range(repeats):
                    rs=[r for r in rows if r['repetition']==rep]
                    assert len(rs)==65 and [r['context'] for r in rs]==list(range(1024,1089))
                    summaries.append({'decode_ms':statistics.mean(r['token_ms'] for r in rs if r['step']>0),'first_logits_ms':rs[0]['prefix_ms']+rs[0]['token_ms'],'argmax':[r['argmax'] for r in rs],'model_ms':statistics.mean(r['decode_ms'] for r in rs if r['step']>0),'peak_rss_mib':max(r['peak_rss_kib'] for r in rs)/1024})
            else:
                summaries=[]
                for p in sorted((directory/'capture').glob('*.run_*.json')):
                    r=json.loads(p.read_text())
                    assert len(r['passes'])==65
                    summaries.append({'decode_ms':statistics.mean(v['elapsed_ms'] for v in r['passes'][1:]),'first_logits_ms':r['passes'][0]['elapsed_ms'],'argmax':[v['argmax_id'] for v in r['passes']]})
            assert len(summaries)==repeats
            save(directory/'summary.json',summaries)
            print(self.name,'DONE',name,'decode_ms',round(statistics.median(s['decode_ms'] for s in summaries),3),'first_logits_ms',round(statistics.median(s['first_logits_ms'] for s in summaries),1),flush=True)
        else:
            print(self.name,'DONE',name,flush=True)

    def run(self):
        self.prepare()
        self.capture('smoke-prime-v3',smoke=True,repeats=1)
        for name,variant in [('final-production-b0','production'),('final-native-b0','native'),('final-native-b1','native'),('final-production-b1','production')]:
            self.capture(name,variant)
        save(self.root/'completed-final.json',{'complete':True})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--devices',nargs='+',default=list(DEVICES),choices=list(DEVICES))
    args=parser.parse_args()
    if not (ROOT/'runner-timed.android').exists():
        shutil.copy2(REPO/'bazel-bin/tensor/examples/gemma4/litert/runner',ROOT/'runner-timed.android')
    else:
        assert sha(ROOT/'runner-timed.android') == sha(REPO/'bazel-bin/tensor/examples/gemma4/litert/runner')
    MODEL_HASH=sha(MODEL)
    manifest_path=Path('/data/bt/tmp/gemma4-linux-xnn-memory-20260923/matched-bundle/manifest.json')
    assert sha(manifest_path)==EXPECTED_MANIFEST
    manifest=json.loads(manifest_path.read_text())
    files={}
    for record in manifest['tensors']+manifest['constants']:
        files[record['file']]=record['sha256']
        q=record.get('quantization',{})
        if 'scales_file' in q: files[q['scales_file']]=q['scales_sha256']
    files["manifest.json"] = EXPECTED_MANIFEST
    assert len(files)==1394,len(files)
    (ROOT/'bundle.expected.sha256').write_text(''.join(h+'  '+p+'\n' for p,h in sorted(files.items())))
    save(ROOT/'build-identity-final.json',{'branch_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'model_sha256':MODEL_HASH,'runner_sha256':sha(ROOT/'runner-timed.android'),'ndk':'28.2.13676358','android_api':35,'build_file_sha256':sha(REPO/'tensor/examples/gemma4/litert/BUILD'),'main_source_sha256':sha(REPO/'tensor/examples/gemma4/litert/main.cc'),'coefficient_hash_manifest_sha256':sha(ROOT/'bundle.expected.sha256')})
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        futures={pool.submit(Phone(name).run):name for name in args.devices}
        for future in concurrent.futures.as_completed(futures):
            name=futures[future]
            try: future.result()
            except Exception as e:
                save(ROOT/name/'failure.json',{'error':repr(e)})
                print(name,'FAILED',repr(e),flush=True)
                raise
