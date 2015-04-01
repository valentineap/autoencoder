#include<stdio.h>
#include<stdlib.h>
#include "autoencoder.h"

int main(int argc, char *argv[]){
  set_binary_mode(0);
  autoenc_t *a;
  printf("Loading autoencoder from: %s\n",argv[1]);
  a=load_autoencoder(argv[1]);
  printf("Writing dataset to: %s\n",argv[2]);
  dataset_t d;
  dataset_alloc(&d,a->layers[0].nin,a->layers[0].nout);
  int irec,ipt;
  double minval=0.,maxval=0.;
  for(irec=0;irec<a->layers[0].nout;irec++){
    for(ipt=0;ipt<a->layers[0].nin;ipt++){
      if(*(a->layers[0].w+(irec*a->layers[0].nin)+ipt)<minval){minval=*(a->layers[0].w+(irec*a->layers[0].nin)+ipt);}
      if(*(a->layers[0].w+(irec*a->layers[0].nin)+ipt)>maxval){maxval=*(a->layers[0].w+(irec*a->layers[0].nin)+ipt);}
      *(d.data+(ipt*a->layers[0].nout)+irec)=*(a->layers[0].w+(irec*a->layers[0].nin)+ipt);
    }
    *(d.scale+irec)=1.;
    *(d.weight+irec)=1.;
  }
  printf("min: %f max %f\n",minval,maxval);
  writedata(argv[2],&d);
}
    
      
