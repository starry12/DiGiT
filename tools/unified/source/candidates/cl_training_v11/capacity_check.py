"""Exercise the actual compiled production validator without allocations or I/O."""
def run(module,arm):
    maximum=978408104 if arm=='gids' else 1174089720
    if list(module.cl_capacity_abi())!=[1,97840809,maximum,128]:
        raise RuntimeError('Wrong arm capacity ABI')
    validate=module.cl_external_cache_validate
    valid=[(4096,8192,97840809,maximum,maximum*128,False,0,0),
           (4096,8192,1,8,1024,False,0,0),
           (4096,8192,78780147,maximum,maximum*128,False,0,0)]
    for args in valid:validate(*args)
    base=list(valid[0]);invalid=[]
    for index,value in [(0,0),(1,0),(2,0),(2,97840810),(3,0),
                        (4,0),(4,maximum*128+1),(5,True),(6,1),(7,1)]:
        args=base.copy();args[index]=value;invalid.append(args)
    for rows in (maximum+8,maximum-1):
        args=base.copy();args[3]=rows;args[4]=rows*128;invalid.append(args)
    args=base.copy();args[3]=8;args[4]=1024;invalid.append(args) # hot rows exceed extent
    args=base.copy();args[4]=(maximum-8)*128;invalid.append(args)
    args=base.copy();args[4]=2**64-1;invalid.append(args)
    for args in invalid:
        try:validate(*args)
        except ValueError:continue
        raise RuntimeError('Native capacity validator accepted invalid boundary: '+str(args))
    return dict(passed=True,cases=len(valid)+len(invalid),valid_cases=len(valid),
        invalid_cases=len(invalid),full_hot_rows=97840809,storage_max=maximum,
        production_validator=True,large_allocation=False,raw_ssd_access=False)

if __name__=='__main__':
    import argparse,importlib.util,json
    from .build import binary
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('gids','digit'),required=True);a=p.parse_args()
    s=importlib.util.spec_from_file_location('BAM_Feature_Store',str(binary(a.arm)));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    print(json.dumps(run(m,a.arm)))
