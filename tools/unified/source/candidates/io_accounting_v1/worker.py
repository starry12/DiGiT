"""Fresh-process bounded native gate; never writes raw SSD data."""
import argparse,json,sys
from pathlib import Path
from candidates.io_accounting_v1.common import HERE,ROOT,verify_release

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('gids','digit_full'),required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--data',default=str(ROOT/'data/papers_g2_random_v2'))
    args=parser.parse_args();args.seed=0;args.smoke=True;args.repeat=0
    execution=verify_release()
    from ae.pa_sage.common import setup_imports
    setup_imports();sys.path.insert(0,str(HERE/'runtime'))
    import BAM_Feature_Store
    expected=(HERE/'runtime/BAM_Feature_Store/BAM_Feature_Store.so').resolve()
    actual=Path(sys.modules['BAM_Feature_Store.BAM_Feature_Store'].__file__).resolve()
    if actual!=expected:raise RuntimeError('Wrong native feature store imported')
    import runner
    from candidates.io_accounting_v1.accounting import install_runner
    install_runner(runner)
    from candidates.io_accounting_v1 import train
    train.run(args)
    if verify_release()!=execution:raise RuntimeError('Candidate changed while running')
if __name__=='__main__':main()
