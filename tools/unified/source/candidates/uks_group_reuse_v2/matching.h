// Bounded cardinality improvements over v1. No feature/layout changes.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

struct PairEdge {size_t a,b; int64_t gid;};
struct Selection {
 std::vector<unsigned char> chosen;
 size_t greedy=0,swaps=0,checks=0;
};

inline Selection select_pairs(std::vector<size_t> free,
                              const std::vector<PairEdge>& edges,
                              bool improve=true) {
 Selection s;s.chosen.assign(edges.size(),0);
 auto available=[&](size_t i){return free[edges[i].a] && free[edges[i].b];};
 auto take=[&](size_t i){--free[edges[i].a];--free[edges[i].b];s.chosen[i]=1;};
 auto release=[&](size_t i){++free[edges[i].a];++free[edges[i].b];s.chosen[i]=0;};
 for(size_t i=0;i<edges.size();++i)if(available(i)){take(i);++s.greedy;}
 if(!improve)return s;
 std::vector<std::vector<size_t>> adjacent(free.size());
 for(size_t i=0;i<edges.size();++i){adjacent[edges[i].a].push_back(i);adjacent[edges[i].b].push_back(i);}
 // Bound pathological per-owner work. Exhausting a bound retains the current
 // valid selection, whose cardinality is always >= the v1 greedy selection.
 constexpr size_t max_checks=50000,max_swaps=64;
 bool changed=true;
 while(changed && s.checks<max_checks && s.swaps<max_swaps){
  changed=false;
  for(size_t i=0;i<edges.size() && s.checks<max_checks;++i){
   if(!s.chosen[i])continue;
   const auto e=edges[i];
   if(adjacent[e.a].size()<2 || adjacent[e.b].size()<2)continue;
   release(i);bool found=false;
   for(size_t j:adjacent[e.a]){
    if(s.checks>=max_checks)break;
    ++s.checks;
    if(j==i || s.chosen[j] || !available(j))continue;
    take(j);
    for(size_t k:adjacent[e.b]){
     if(s.checks>=max_checks)break;
     ++s.checks;
     if(k==i || s.chosen[k] || !available(k))continue;
     take(k);found=true;break;
    }
    if(found)break;
    release(j);
   }
   if(found){++s.swaps;changed=true;break;}
   take(i); // Roll back even if the budget expired during this attempt.
  }
 }
 return s;
}
