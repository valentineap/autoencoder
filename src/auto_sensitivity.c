#include<stdio.h>
#include<stdlib.h>
#include "autoencoder.h"
#ifndef ACML
#include<cblas.h>
#endif
#include<math.h>
#include<string.h>
int main(int argc, char * argv[]){
  if(argc!=5){printf("auto_sensitivity DATFILE NETFILE IREC OUTFILE\n");exit(-1);}
  dataset_t data=load_dataset(argv[1]);
  autoenc_t *a=load_autoencoder(argv[2]);
  int irec;
  sscanf(argv[3],"%i",&irec);
  char *outfile=argv[4];
  printf("%i\n",irec);
  int i,j,k;
  node_val_t nodes[a->nlayers+1];
  make_nodevals(a,nodes,1);

  for(i=0;i<a->layers[0].nin;i++){
    nodes[0].values[i]=*(data.data+(i*data.nrecs)+irec);
  }
  


  printf("A\n");
  
  for(i=1;i<a->nlayers+1;i++){
    for(j=0;j<nodes[i].n;j++){
	nodes[i].values[j]=*((a->layers[i-1].b)+j);
    }
#ifdef ACML
    dgemm('N','N',1,nodes[i].n,nodes[i-1].n,1.0,nodes[i-1].values,1,a->layers[i-1].w,nodes[i-1].n,1.0,nodes[i].values,1);
#else
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nodes[i].n,1,nodes[i-1].n,1.0,a->layers[i-1].w,nodes[i-1].n,nodes[i-1].values,1,1.0,nodes[i].values,1);
#endif

    for(j=0;j<nodes[i].n;j++){
      nodes[i].values[j]=a->layers[i-1].loglo+(a->layers[i-1].logup-a->layers[i-1].loglo)*(LOGISTIC(nodes[i].values[j]* *(a->layers[i-1].a+j)));
      nodes[i].derivs[j]=(nodes[i].values[j]-a->layers[i-1].loglo)*(a->layers[i-1].logup-nodes[i].values[j])/(a->layers[i-1].logup-a->layers[i-1].loglo);
    }
  }
  //  FILE * tmp1=fopen("data.tmp","w");
  //for(i=0;i<a->layers[0].nin;i++){
  //  fprintf(tmp1,"%i %f\n",i,nodes[a->nlayers].values[i]);
  //}
  //fclose(tmp1);

  printf("B\n");
  double * d1,*d2,*d3;
  d1=malloc(nodes[0].n*nodes[1].n*sizeof(double));
  memcpy(d1,a->layers[0].w,nodes[0].n*nodes[1].n*sizeof(double));
  for(j=0;j<nodes[1].n;j++){
    for(i=0;i<nodes[0].n;i++){
      *(d1+(nodes[0].n*j)+i) *=( *(a->layers[0].a+j)*nodes[1].derivs[j]);
    }
  }   
  printf("C\n");
  for(i=2;i<a->nlayers+1;i++){
    d2=malloc(nodes[i-1].n*nodes[i].n*sizeof(double));
    d3=malloc(nodes[0].n*nodes[i].n*sizeof(double));
    printf("%i %i\n",nodes[0].n,nodes[i].n);
    memcpy(d2,a->layers[i-1].w,nodes[i-1].n*nodes[i].n*sizeof(double));
    printf("D\n");
    for(j=0;j<nodes[i].n;j++){
      for(k=0;k<nodes[i-1].n;k++){
	*(d2+(nodes[i-1].n*j)+k) *=( *(a->layers[i-1].a+j)*nodes[i].derivs[j]);
      }
    }
    printf("E\n");
#ifdef ACML
    dgemm('N','N',nodes[0].n,nodes[i].n,nodes[i-1].n,d1,nodes[0].n,d2,nodes[i-1].n,0.0,d3, nodes[0].n);
#else
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nodes[i].n,nodes[0].n,nodes[i-1].n,1.0,d2,nodes[i-1].n,d1,nodes[0].n,0.,d3,nodes[0].n);
#endif
    free(d1);
    free(d2);
    d1=d3;
  }
  
  FILE * fp=fopen(outfile,"w");
  for(i=0;i<nodes[0].n;i++){
    for(j=0;j<nodes[0].n;j++){
      fprintf(fp,"%i %i %e\n",i,j,*(d3+(i*nodes[0].n)+j));
    }
  }
  fclose(fp);

  

}
      
