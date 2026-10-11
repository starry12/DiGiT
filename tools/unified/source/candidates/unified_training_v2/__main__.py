"""Read-only plan interface until native dataset adapters are validated."""
import argparse
import json
from .config import plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['PA','IG','UKS','UKL','CL'], required=True)
    parser.add_argument('--mode', choices=['short1','short5','full20'], default='short1')
    args=parser.parse_args()
    print(json.dumps(plan(args.dataset,args.mode),indent=2))


if __name__=='__main__':main()
