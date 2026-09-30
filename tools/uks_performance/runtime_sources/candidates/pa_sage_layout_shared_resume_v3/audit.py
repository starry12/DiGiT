"""Untimed smoke oracle: real logical features or the actual shared physical row."""
import numpy as np

def feature_audit(inp,out,blocks,x,features,bundle,mode):
    import runner as r
    if mode=='real':
        result=r.batch_audit(inp,out,blocks,x,features,bundle)
    else:
        from digit.sampler import DIGIT_STORAGE_ROW
        rows=blocks[0].srcdata[DIGIT_STORAGE_ROW].cpu().numpy()
        expected=np.array(features[rows],copy=True)
        logical=inp.cpu().numpy()
        class Oracle:
            def __getitem__(self,ids):
                assert np.array_equal(ids,logical)
                return expected
        result=r.batch_audit(inp,out,blocks,x,Oracle(),bundle)
    result['feature_oracle']=mode
    return result
