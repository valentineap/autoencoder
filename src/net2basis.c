#include<stdio.h>
#include<stdlib.h>
#include<cblas.h>
#include "autoencoder.h"

int main(int argc, char *argv[]){
  if(argc!=3){
    printf("Usage: net2basis NETFILE DATFILE\n");
    exit(-1);
  }
  char *netfile=argv[1];
  char *datfile=argv[2];
 
  printf("Loading autoencoder...\n");
  autoenc_t *a=load_autoencoder(netfile);
  printf("Autoencoder from file: %s\n",netfile);
  int i,j;
  for(i=0;i<a->nlayers;i++){
    printf("%i-",a->layers[i].nin);
  }
  printf("%i\n",a->layers[a->nlayers-1].nout);

  printf("...done!\n");
  dataset_t ones,basis;
  dataset_alloc(&ones,a->layers[a->nlayers/2].nin,a->layers[a->nlayers/2].nin);
  for(i=0;i<a->layers[a->nlayers/2].nin;i++){
    for(j=0;j<a->layers[a->nlayers/2].nin;j++){
      if(i==j){
	*(ones.data+(i*a->layers[a->nlayers/2].nin)+j)=1.0;
      } else {
	*(ones.data+(i*a->layers[a->nlayers/2].nin)+j)=0.0;
      }
    }
  }
  dataset_alloc(&basis,a->layers[a->nlayers-1].nout,a->layers[a->nlayers/2].nin);
  autoencoder_decode(a,&ones,&basis);
  
  double *correls=malloc(a->layers[a->nlayers/2].nin*a->layers[a->nlayers/2].nin*sizeof(double));
  cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,a->layers[a->nlayers/2].nin,a->layers[a->nlayers/2].nin,a->layers[a->nlayers-1].nout,1.0,basis.data,a->layers[a->nlayers/2].nin,basis.data,a->layers[a->nlayers/2].nin,0.0,correls,a->layers[a->nlayers/2].nin);
  FILE * fp=fopen(datfile,"w");
  fprintf(fp,"# %i %s\n",a->layers[a->nlayers/2].nin,netfile);
  for(i=0;i<a->layers[a->nlayers/2].nin;i++){
    for(j=0;j<a->layers[a->nlayers/2].nin;j++){
      fprintf(fp,"%i %i %f\n",i+1,j+1,*(correls+(i*a->layers[a->nlayers/2].nin)+j)/ *(correls+(i*a->layers[a->nlayers/2].nin)+i));
    }
  }
  
  fclose(fp);
}
