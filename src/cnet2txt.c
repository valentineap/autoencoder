#include<stdio.h>
#include<stdlib.h>
#include "autoencoder.h"

int main(int argc, char *argv[]){
  autoenc_t *a;
  printf("%s\n",argv[1]);
  a=load_autoencoder(argv[1]);
  int i,j,k,n;
  FILE * fp;
  char outfile[20];
  for(i=0;i<a->nlayers;i++){
    sprintf((char*)&outfile,"layer_%02i_wt.dat",i);
    printf("%s\n",outfile);
    fp=fopen(outfile,"w");
    n=a->layers[i].nin*a->layers[i].nout;
    for(j=0;j<n;j++){
      fprintf(fp,"%.5f\n",*(a->layers[i].w+j));
    }
    fclose(fp);
    sprintf((char*)&outfile,"layer_%02i_bi.dat",i);
    printf("%s\n",outfile);
    fp=fopen(outfile,"w");
    n=a->layers[i].nout;
    for(j=0;j<n;j++){
      fprintf(fp,"%.5f\n",*(a->layers[i].b+j));
    }
    fclose(fp);
    sprintf((char*)&outfile,"layer_%02i_se.dat",i);
    printf("%s\n",outfile);
    fp=fopen(outfile,"w");
    n=a->layers[i].nout;
    for(j=0;j<n;j++){
      fprintf(fp,"%.5f\n",*(a->layers[i].a+j));
    }
    fclose(fp);

  }
}
