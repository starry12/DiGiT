"""UKL protocol/budget preview; full preparation and training remain disabled."""
import argparse,json
from pathlib import Path
from .protocol import plan,budget

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=('plan','budget'),nargs='?',default='plan');a=p.parse_args()
 available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
 print(json.dumps(plan() if a.action=='plan' else budget(available),indent=2))
if __name__=='__main__':main()
