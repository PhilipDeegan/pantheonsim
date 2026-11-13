#define _GNU_SOURCE
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#include <stdint.h>
#define MAXS 4000000
static uintptr_t *buf; static volatile long n;
static void h(int s, siginfo_t *i, void *u){ ucontext_t *c=u; if(n<MAXS) buf[n++]=(uintptr_t)c->uc_mcontext.gregs[REG_RIP]; }
static void dump(void){
  struct itimerval z={{0,0},{0,0}}; setitimer(ITIMER_PROF,&z,0);
  const char *o=getenv("PROF_OUT"); if(!o) return;
  FILE*f=fopen(o,"w"); FILE*m=fopen("/proc/self/maps","r"); char l[512];
  while(fgets(l,sizeof l,m)) if(strstr(l,"r-xp")) fprintf(f,"M %s",l);
  for(long k=0;k<n;k++) fprintf(f,"S %lx\n",(unsigned long)buf[k]); fclose(f);
}
__attribute__((constructor)) static void init(void){
  buf=malloc(MAXS*sizeof *buf); struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_sigaction=h; sa.sa_flags=SA_SIGINFO|SA_RESTART; sigaction(SIGPROF,&sa,0);
  struct itimerval t={{0,1000},{0,1000}}; setitimer(ITIMER_PROF,&t,0); atexit(dump);
}
