//////////////////////////////////////////////////////////////
// auto_encdec.c - Encode & decode dataset with autoencoder //
//////////////////////////////////////////////////////////////
// Andrew Valentine                                         //
// Universiteit Utrecht                                     //
// 2011-2012                                                //
//////////////////////////////////////////////////////////////
// $Id: auto_encode.c,v 1.2 2012/03/31 15:21:00 andrew Exp $


#include<stdlib.h>
#include<stdio.h>
#include "autoencoder.h"

int main(int argc, char *argv[]){
  if(argc!=4){
    printf("Error: incorrect arguments supplied.\n");
    printf("Usage: auto_encdec INPUTFILE AUTOENCODER OUTPUTFILE\n");
    exit(-1);
  }
  char *inputfile=argv[1];
  char *netfile=argv[2];
  char *outputfile=argv[3];
  dataset_t inputdata=load_dataset(inputfile);
  printf("Encoding dataset from file:%s\n",inputfile);
  printf("File contains %i records of length %i\n",inputdata.nrecs,inputdata.npoints);
  dataset_t encdata;
  autoenc_t *a=load_autoencoder(netfile);
  printf("Autoencoder from file: %s\n",netfile);
  int i;
  for(i=0;i<a->nlayers;i++){
    printf("%i-",a->layers[i].nin);
  }
  printf("%i\n",a->layers[a->nlayers-1].nout);
  dataset_alloc(&encdata,a->layers[a->nlayers/2].nin,inputdata.nrecs);
  autoencoder_encode(a,&inputdata,&encdata);
  autoencoder_decode(a,&encdata,&inputdata);
  writedata(outputfile,&inputdata);
  printf("%i records of length %i written to: %s\n",inputdata.nrecs,inputdata.npoints,outputfile);
}
