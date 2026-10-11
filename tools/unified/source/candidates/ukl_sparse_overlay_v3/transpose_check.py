import json,struct,subprocess,tempfile
from pathlib import Path
import numpy as np
from .transpose import OUT,OLD,BIN,compile_java,pipeline

def main():
 cp=compile_java();fixture=OLD/'checks/interop_5'
 # Select only validated, final v2 interop evidence.
 choices=[p.parent for p in (OLD/'checks').glob('interop*/receipt.json') if json.loads(p.read_text()).get('transpose_from_same_pairs')]
 fixture=choices[-1];pairs=[tuple(map(int,x.split())) for x in (fixture/'edges.tsv').read_text().splitlines()];n=67
 folder=OUT/'checks/transpose';value=pipeline(folder,fixture/'outgoing',n,len(pairs),n,8,'512m',cp)
 ptr=np.fromfile(folder/'indptr.i64',np.int64);idx=np.fromfile(folder/'indices.i32',np.int32)
 for v in range(n):np.testing.assert_array_equal(idx[ptr[v]:ptr[v+1]],sorted(u for u,w in pairs if w==v and u!=v)+[v])
 # Duplicate edges, self normalization, skew, empty owners, cross-bucket destinations.
 rows=[[0,4,4,6],[],[6]*10003,[1,0],[4],[0,3],[]];data=struct.pack('>IIQI',0x554b4c33,7,sum(map(len,rows)),7)
 for u,row in enumerate(rows):data+=struct.pack('>II',u,len(row))+b''.join(struct.pack('>I',v) for v in row)
 data+=struct.pack('>IQ',0x454e4433,sum(map(len,rows)))
 good=OUT/'checks/transpose_duplicates';good.mkdir()
 subprocess.run([str(BIN),'scatter',str(good),'4'],input=data,check=True)
 normalized=sum(sum(v!=u for v in row) for u,row in enumerate(rows))+7
 subprocess.run([str(BIN),'gather',str(good),'7','4',str(normalized),str(2**20)],check=True)
 p=np.fromfile(good/'indptr.i64',np.int64);x=np.fromfile(good/'indices.i32',np.int32)
 for v in range(7):np.testing.assert_array_equal(x[p[v]:p[v+1]],[u for u,row in enumerate(rows) for w in row if w==v and u!=v]+[v])
 rejected=[]
 for name,bad in [('truncated',data[:-1]),('trailing',data+b'x'),('bad_magic',b'xxxx'+data[4:]),('bad_footer',data[:-8]+struct.pack('>Q',2))]:
  f=OUT/'checks'/('transpose_'+name);f.mkdir();r=subprocess.run([str(BIN),'scatter',str(f),'4'],input=bad,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  assert r.returncode and not (f/'scatter.json').exists();rejected.append(name)
 cap=OUT/'checks/transpose_cap';cap.mkdir()
 for b in range(4):(cap/('bucket_%d.u32'%b)).symlink_to(good/('bucket_%d.u32'%b))
 r=subprocess.run([str(BIN),'gather',str(cap),'7','4',str(normalized),'1'],capture_output=True);assert r.returncode and not (cap/'csc.json').exists();rejected.append('memory_cap')
 result=dict(passed=True,java_webgraph_actual_transpose=True,fixture_nodes=n,fixture_source_edges=len(pairs),skew_duplicate_edges=sum(map(len,rows)),rejections=rejected,production_full_graph=False)
 (OUT/'checks/transpose.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
