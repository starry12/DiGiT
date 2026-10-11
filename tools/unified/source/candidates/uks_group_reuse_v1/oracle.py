"""Independent, small-graph Python reference; never used for UKS production."""
from collections import Counter

def expand(units,n,members):
    answer=[]
    for unit in units:
        if unit<n:answer.append(int(unit))
        else:answer.extend(map(int,members[unit-n]))
    return Counter(answer)

def rewrite(row,n,members):
    # Deliberately scan every group instead of using the C++ inverted index.
    remaining=Counter(int(x) for x in row if x<n)
    existing={int(x-n) for x in row if x>=n};added=[]
    for gid,(a,b) in enumerate(members):
        if gid not in existing and remaining[a]>0 and remaining[b]>0:
            remaining[a]-=1;remaining[b]-=1;added.append(n+gid)
    consumed=Counter(int(x) for x in row if x<n)-remaining
    kept=[]
    for x in row:
        if x<n and consumed[x]:consumed[x]-=1
        else:kept.append(int(x))
    return kept+added
