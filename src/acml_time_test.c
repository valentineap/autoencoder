#include <acml.h>
#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<pthreads.h>
int lstate=633;
int state[633];

#define MAX_ISIZE 15
#define N_ITER 100

double *make_matrix(int n,int m,double stdev){
  double *p=malloc(n*m*sizeof(double));
  int i,j;
  int info;
  drandgaussian(n*m,0.,stdev,state,p,&info);
  return p;
}

int main(int argc,char **argv){
  int lseed=1;
  int seed=(unsigned)time(NULL);
  int info;
  drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
  
  int n,m,i,j,isize,iter;


  double *m1,*m2;
  time_t t1,t2;
  
  for(isize=0;isize<MAX_ISIZE;isize++){
    n=pow(2,isize);
    m=pow(2,isize+1);
    m1=make_matrix(n,m,1.0);
    m2=make_matrix(n,1,1.0);
    time(&t1); 
    for(iter=0;iter<N_ITER;iter++){
      for(i=0;i<n;i++){
	for(j=0;j<m;j++){
	  *(m1+(i*m)+j)+=*(m2+i);
	}
      }
    }
  
    time(&t2);
    printf("%ix%i (%i) - %d\n",n,m,n*m,((int) (t2-t1)));
    free(m1);
    free(m2);
    m1=make_matrix(n,m,1.0);
    m2=make_matrix(n,1,1.0);
    time(&t1); 
#pragma omp parallel
    {
#pragma omp for schedule(static)     
      for(iter=0;iter<N_ITER;iter++){
   	for(i=0;i<n;i++){
	  for(j=0;j<m;j++){
	    *(m1+(i*m)+j)+=*(m2+i);
	  }
	}
      }
    }
  
    time(&t2);
    printf("%ix%i (%i) - %d\n",n,m,n*m,((int) (t2-t1)));
    free(m1);
    free(m2);

  }

	  
    


}
