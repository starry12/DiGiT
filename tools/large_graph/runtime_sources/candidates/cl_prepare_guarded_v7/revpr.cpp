// Reverse PageRank on original incoming CSC: i64 offsets, i32 neighbors, implicit EIDs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
using namespace std;
#include "direct.hpp"
void req(bool b,const string&s){if(!b)throw runtime_error(s);}
void req(bool b,const char*s){if(!b)throw runtime_error(s);}
void progress(string out,string stage,int iteration,uint64_t owners,double seconds){poll(owners,0,iteration);string s="{\"stage\":\""+stage+"\",\"iteration\":"+to_string(iteration)+",\"owners_processed\":"+to_string(owners)+",\"seconds\":"+to_string(seconds)+"}\n";save(out+"/progress.next",s.data(),s.size());req(rename((out+"/progress.next").c_str(),(out+"/progress.json").c_str())==0,"progress rename");}
extern "C" int run_revpr(const int64_t*p,const int32_t*x,uint64_t n,uint64_t e,const char*output,int iterations,Tick cb,uint64_t rate){try{
 tick_fn=cb;write_rate=rate;req(rate>0,"write rate");string out(output);req(n>=10&&n<2147483647&&iterations>0&&iterations<=20,"bounds");
 req(p[0]==0&&p[n]>=0&&uint64_t(p[n])==e,"CSC extent");
 for(uint64_t i=0;i<n;i++){req(p[i]>=0&&p[i]<=p[i+1],"offset monotonicity");if(i%1000000==0)poll(i,n,1);}
 vector<double> rank(n,1./n),next(n);auto begin=chrono::steady_clock::now();auto elapsed=[&](){return chrono::duration<double>(chrono::steady_clock::now()-begin).count();};
 for(int it=0;it<iterations;it++){
  double dangling=0;for(uint64_t d=0;d<n;d++){if(d%1000000==0)poll(d,n,3);if(p[d]==p[d+1])dangling+=rank[d];}fill(next.begin(),next.end(),(.15+.85*dangling)/n);
  for(uint64_t d=0;d<n;d++){
   auto a=p[d],b=p[d+1];if(a!=b){double value=.85*rank[d]/(b-a);for(auto j=a;j<b;j++){if(j>a&&(j-a)%1048576==0)poll(d,n,2);int32_t v=x[j];req(v>=0&&uint64_t(v)<n,"neighbor outside node range");next[v]+=value;}}
   if((d+1)%1000000==0)progress(out,"rank",it+1,d+1,elapsed());
  }
  rank.swap(next);double mass=0;for(double v:rank){req(isfinite(v)&&v>=0,"nonfinite/negative rank");mass+=v;}req(abs(mass-1)<1e-5,"rank mass drift");
  progress(out,"rank",it+1,n,elapsed());cout<<"iteration "<<it+1<<"/"<<iterations<<" mass "<<mass<<" elapsed "<<elapsed()<<endl;
 }
 save(out+"/revpr.f64",rank.data(),8*n);vector<double>().swap(next);progress(out,"top_k",iterations,n,elapsed());
 vector<uint32_t> order(n);iota(order.begin(),order.end(),0);uint64_t compared=0;auto cmp=[&](uint32_t a,uint32_t b){if((++compared%1048576)==0)poll(compared,0,4);return rank[a]!=rank[b]?rank[a]>rank[b]:a<b;};uint64_t k=n/10;
 // Same selected hot set as full sort; avoid sorting the other 90% of nodes.
 nth_element(order.begin(),order.begin()+k,order.end(),cmp);vector<int64_t> hot(k);for(uint64_t i=0;i<k;i++)hot[i]=order[i];sort(hot.begin(),hot.end());save(out+"/hot_nodes.i64",hot.data(),8*k);
 string s="{\"complete\":true,\"nodes\":"+to_string(n)+",\"edges\":"+to_string(e)+",\"iterations\":"+to_string(iterations)+",\"hot_nodes\":"+to_string(k)+",\"damping\":0.85,\"rank_dtype\":\"float64\",\"neighbor_dtype\":\"int32\",\"direction\":\"PageRank of reverse graph over incoming CSC\",\"tie_break\":\"node_id_ascending\"}\n";save(out+"/rank.json",s.data(),s.size());progress(out,"complete",iterations,n,elapsed());return 0;
 }catch(const exception&e){cerr<<"ERROR: "<<e.what()<<endl;return 1;}}
