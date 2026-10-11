// PA bidirectional multigraph is symmetric: reverse PageRank equals PageRank.
#define main reuse_cli_main
#include "reuse.cpp"
#undef main
#include <cmath>
#include <omp.h>
int main(int argc,char**argv){try{
 need(argc==5,"rank source.spec output.f64 iterations threads");Input x(argv[1]);
 int iters=std::stoi(argv[3]),threads=std::stoi(argv[4]);need(iters>0&&threads>0&&threads<=8,"limits");omp_set_num_threads(threads);
 const auto&p=x.at("ptr");const auto&ix=x.at("idx");I n=x.n;
 std::vector<double> rank(n,1./n),next(n),weighted(n);const double damping=.85;
 for(int it=0;it<iters;++it){
  double dangling=0.;
  #pragma omp parallel for reduction(+:dangling) schedule(static)
  for(I v=0;v<n;++v){I degree=p[v+1]-p[v];weighted[v]=degree?rank[v]/degree:0.;if(!degree)dangling+=rank[v];}
  double base=(1.-damping+damping*dangling)/n;
  #pragma omp parallel for schedule(dynamic,4096)
  for(I v=0;v<n;++v){double sum=0.;for(I e=p[v];e<p[v+1];++e)sum+=weighted[ix[e]];next[v]=base+damping*sum;}
  double mass=0.;for(double v:next){need(std::isfinite(v)&&v>=0.,"nonfinite rank");mass+=v;}need(std::abs(mass-1.)<1e-7,"rank mass");
  rank.swap(next);std::cerr<<"revpr iteration "<<it+1<<"/"<<iters<<" mass "<<mass<<std::endl;
 }
 int fd=open(argv[2],O_CREAT|O_EXCL|O_WRONLY,0644);need(fd>=0,"preserve rank output");
 size_t bytes=n*sizeof(double),done=0;while(done<bytes){ssize_t w=write(fd,(char*)rank.data()+done,std::min<size_t>(8<<20,bytes-done));need(w>0,"rank write");done+=w;}need(!fsync(fd),"rank sync");close(fd);
 std::cout<<"{\"passed\":true,\"iterations\":"<<iters<<",\"nodes\":"<<n<<",\"damping\":0.85,\"symmetric_graph_required\":true}"<<std::endl;
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
