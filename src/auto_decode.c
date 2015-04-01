/////////////////////////////////////////////////////
// auto_decode.c - Decode dataset with autoencoder //
/////////////////////////////////////////////////////
// Andrew Valentine                                //
// Universiteit Utrecht                            //
// 2011-2012                                       //
/////////////////////////////////////////////////////
// $Id: auto_decode.c,v 1.2 2012/03/31 15:21:07 andrew Exp $

#include<stdlib.h>
#include<stdio.h>
#include "autoencoder.h"

int main(int argc, char *argv[]){
  if(argc!=4){
    printf("Error: incorrect arguments supplied.\n");
    printf("Usage: auto_decode INPUTFILE AUTOENCODER OUTPUTFILE\n");
    exit(-1);
  }
  char *inputfile=argv[1];
  char *netfile=argv[2];
  char *outputfile=argv[3];
  dataset_t inputdata=load_dataset(inputfile);
  printf("Decoding dataset from file:%s\n",inputfile);
  printf("File contains %i records of length %i\n",inputdata.nrecs,inputdata.npoints);
  dataset_t outputdata;
  autoenc_t *a=load_autoencoder(netfile);
  printf("Autoencoder from file: %s\n",netfile);
  int i;
  for(i=0;i<a->nlayers;i++){
    printf("%i-",a->layers[i].nin);
  }
  printf("%i\n",a->layers[a->nlayers-1].nout);
  dataset_alloc(&outputdata,a->layers[a->nlayers-1].nout,inputdata.nrecs);
  autoencoder_decode(a,&inputdata,&outputdata);
  writedata(outputfile,&outputdata);
  printf("%i records of length %i written to: %s\n",outputdata.nrecs,outputdata.npoints,outputfile);
}
