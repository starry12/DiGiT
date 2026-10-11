"""Bind sparse overlays to the existing anonymous loader and memory budget."""
from .prepare import check_prepared
def metadata(dataset):
    r=check_prepared(dataset);v=dict(r['base']['base_metadata'])
    v['units']=r['validation']['output_units'];return v
def binding(dataset):
    r=check_prepared(dataset);v={k:dict(x) for k,x in r['base']['base_specs'].items()}
    for key,name in [('group_ptr','ptr.i64'),('group_ids','units.i32'),('covered','covered.i64')]:
        v[key]={k:r['files'][name][k] for k in ('path','offset','length','identity','sha256')}
    return v
