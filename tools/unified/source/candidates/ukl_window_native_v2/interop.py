"""Real WebGraph compression/offline decoding -> existing Python stream/CSC contract."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
from candidates.ukl_sage_compact_v1.stream import encode_header,Reader
from candidates.ukl_sage_compact_v1.compact import build,CSC
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'results/ukl_window_native_20260930_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 folder=OUT/'checks'/('interop_'+str(len(list((OUT/'checks').glob('interop*')))));folder.mkdir(exist_ok=False)
 jars=OUT/'tools/jars'
 lock=json.loads((HERE/'decoder/dependencies.lock.json').read_text())
 assert {p.name for p in jars.glob('*.jar')}=={v['file'] for v in lock['jars']}
 for item in lock['jars']:assert sha(jars/item['file'])==item['sha256'],'Dependency changed: '+item['file']
 classes=folder/'classes';classes.mkdir();classpath=os.pathsep.join(str(p) for p in sorted(jars.glob('*.jar')))
 command=['/usr/bin/java','-XX:ActiveProcessorCount=1','-Xmx512m','-jar',str(jars/'ecj-3.26.0.jar'),'-11','-cp',classpath,'-d',str(classes),*[str(p) for p in (HERE/'decoder').glob('*.java')]]
 with (folder/'compile.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 n=67;pairs=sorted({(0,v) for v in range(1,n)}|{(u,(17*u+5)%n) for u in range(1,n) if u%4}|{(4,4),(4,9),(9,4)})
 source=folder/'edges.tsv';source.write_text(''.join('%d\t%d\n'%x for x in pairs))
 base=['/usr/bin/java','-XX:ActiveProcessorCount=1','-Xmx512m','-cp',str(classes)+os.pathsep+classpath]
 with (folder/'generate.log').open('w') as log:subprocess.run(base+['Fixture',str(n),str(source),str(folder/'outgoing'),str(folder/'incoming')],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=60)
 binding=dict(schema='digit-ukl-incoming-stream-v1',nodes=n,edges=len(pairs),orientation='incoming',source_sha256=sha(folder/'incoming.graph'),node_map_sha256=hashlib.sha256(np.arange(n,dtype='<i4').tobytes()).hexdigest())
 header=folder/'header.bin';header.write_bytes(encode_header(binding));stream=folder/'incoming.stream'
 # Offline decoding must work even without the random-access offsets file.
 (folder/'incoming.offsets').rename(folder/'incoming.offsets.saved')
 with stream.open('wb') as f,(folder/'decode.log').open('w') as log:
  subprocess.run(base+['UKLStream',str(folder/'incoming'),str(header),str(n),str(len(pairs))],stdout=f,stderr=log,check=True,timeout=60)
 expected=[sorted(u for u,v in pairs if v==owner) for owner in range(n)]
 with stream.open('rb') as f:
  r=Reader(f,binding,fixture=True,chunk_items=7)
  for owner in range(n):
   actual,degree,chunks=r.next_owner();values=[int(x) for c in chunks for x in c]
   assert actual==owner and degree==len(expected[owner]) and values==expected[owner]
  stream_sha=r.finish()
 build(stream,folder/'csc',binding,fixture=True,checkpoint_owners=9);g=CSC.open(folder/'csc',fixture=True)
 for owner,row in enumerate(expected):np.testing.assert_array_equal(g.row(owner),[x for x in row if x!=owner]+[owner])
 # Rejection paths on actual Java-produced bytes, not only a Python mock stream.
 rejected=[]
 for label,data in [('truncated',stream.read_bytes()[:-5]),('trailing',stream.read_bytes()+b'x')]:
  bad=folder/(label+'.stream');bad.write_bytes(data)
  try:build(bad,folder/(label+'_csc'),binding,fixture=True)
  except ValueError:rejected.append(label)
  else:raise AssertionError('Accepted '+label)
  assert not (folder/(label+'_csc')).exists()
 try:
  with stream.open('rb') as f:Reader(f,dict(binding,orientation='outgoing'),fixture=True)
 except ValueError:rejected.append('direction')
 else:raise AssertionError('Direction relabel accepted')
 # Actual producer must reject a false extent and return nonzero.
 p=subprocess.run(base+['UKLStream',str(folder/'incoming'),str(header),str(n),str(len(pairs)+1)],capture_output=True,timeout=60)
 assert p.returncode!=0;rejected.append('producer_extent')
 result=dict(passed=True,real_webgraph_roundtrip=True,nodes=n,edges=len(pairs),normalized_edges=g.edges,offline_without_offsets=True,stream_sha256=stream_sha,rejections=rejected,full_ukl_decoded=False,transpose_from_same_pairs=True,source_files_sha256={name:sha(folder/name) for name in ('edges.tsv','outgoing.graph','incoming.graph')},
  compiler='ECJ 3.26.0 -11 source/target',webgraph='3.6.12',java=subprocess.check_output(['/usr/bin/java','-version'],stderr=subprocess.STDOUT,text=True),
  dependency_sha256={p.name:sha(p) for p in sorted(jars.glob('*.jar'))},class_sha256={p.name:sha(p) for p in classes.glob('*.class')})
 (folder/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('dependency_sha256','class_sha256')},indent=2))
if __name__=='__main__':run()
