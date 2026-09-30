import ast,subprocess,tempfile,unittest
from pathlib import Path
from .common import HERE,ROOT,read
class Tests(unittest.TestCase):
    def test_geometry_and_dma(self):
        source=r'''
#include "policy_math.h"
#include "legacy_geometry.h"
#include <cassert>
int main(){
 static_assert(digit_policy::page_bytes==1024,"cache line");
 static_assert(digit_policy::scratch_bytes==16*1024*1024,"preload scratch");
 for(uint64_t r: {0ULL,1ULL,2ULL,3ULL,4ULL,293946283ULL}){
   assert(digit_policy::page_of(r)==r);
   assert(digit_policy::byte_in_page(r)==0);
   auto off=legacy_dma_offset(r,1024,0,4096);
   assert(off==r*1024);assert(off%4096+1024<=4096);
 }
 assert(!digit_policy::row_valid(-1,0,293946284));
 assert(!digit_policy::row_valid(293946284,0,293946284));
 assert(digit_policy::blocks_for(65536,true)*4<=digit_policy::scratch_pages);
 assert(4ULL*1024*1024*1024/1024==4194304);
}
'''
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'test.cc').write_text(source)
            subprocess.run(['g++','-std=c++11','-I'+str(HERE/'native/gids_module/include'),'-I'+str(HERE/'native/bam/include'),str(p/'test.cc'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)
    def test_one_row_geometry(self):
        from .runtime import setup_sampling_imports,loader_kwargs
        setup_sampling_imports()
        from digit.io_geometry import IOGeometry
        from candidates.uks_native_v1.binding import protocol
        g=IOGeometry.create(feature_row_bytes=1024,group_size=1,minimum_transfer_bytes=1024,target_request_bytes=1024)
        p=protocol();pool=dict(passed=True,feature_mode='logical_synthetic',row_bytes=1024,offset=4*2**40,verified_bytes=8192)
        for arm in ('gids','digit'):
            k=loader_kwargs(p,arm,8,pool,g)
            self.assertEqual(k['page_size'],1024);self.assertEqual(k['mixed_io_geometry']['cache_slot_bytes'],1024)
            self.assertEqual(k['mixed_io_geometry']['full_valid_mask'],1)
            self.assertFalse(k['window_buffer']);self.assertFalse(k['accumulator_flag'])
    def test_syntax(self):
        for p in HERE.glob('*.py'):ast.parse(p.read_text())
    def test_schedule(self):
        from .controller import JOBS
        self.assertEqual([m for _,_,m in JOBS[:2]],['smoke','smoke'])
        self.assertEqual(len(JOBS),12)
        self.assertEqual([arm for _,arm,_ in JOBS[2:]],['gids','digit']*5)
    def test_policies_and_addresses(self):
        p=read(HERE/'protocol.json')
        self.assertEqual(p['graph_group_size'],2);self.assertEqual(p['io_rows_per_cache_line'],1)
        self.assertFalse(p['raw_ssd_writes']);self.assertEqual(p['gids_policy'],'legacy');self.assertEqual(p['digit_policy'],'fifo')
if __name__=='__main__':unittest.main()
