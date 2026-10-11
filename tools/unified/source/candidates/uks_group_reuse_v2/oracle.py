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
    # Independent dictionary/set reference; production uses dense run indexes.
    eligible=[g for g,(a,b) in enumerate(members)
              if g not in existing and a in remaining and b in remaining]
    chosen={x-n for x in added};checks=0;swaps=0;changed=True
    while changed and checks<50000 and swaps<64:
        changed=False
        for gid in eligible:
            if checks>=50000:break
            if gid not in chosen:continue
            a,b=members[gid]
            left=[g for g in eligible if a in members[g]]
            right=[g for g in eligible if b in members[g]]
            if len(left)<2 or len(right)<2:continue
            remaining[a]+=1;remaining[b]+=1;chosen.remove(gid)
            found=False
            for j in left:
                if checks>=50000:break
                checks+=1;x,y=members[j]
                if j==gid or j in chosen or not remaining[x] or not remaining[y]:continue
                remaining[x]-=1;remaining[y]-=1;chosen.add(j)
                for k in right:
                    if checks>=50000:break
                    checks+=1;u,v=members[k]
                    if k==gid or k in chosen or not remaining[u] or not remaining[v]:continue
                    remaining[u]-=1;remaining[v]-=1;chosen.add(k);found=True;break
                if found:break
                remaining[x]+=1;remaining[y]+=1;chosen.remove(j)
            if found:
                swaps+=1;changed=True;break
            remaining[a]-=1;remaining[b]-=1;chosen.add(gid)
    added=[n+g for g in sorted(chosen)]
    consumed=Counter(int(x) for x in row if x<n)-remaining
    kept=[]
    for x in row:
        if x<n and consumed[x]:consumed[x]-=1
        else:kept.append(int(x))
    return kept+added
