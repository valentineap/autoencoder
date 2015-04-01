#include <stdio.h>
#include <stdlib.h>
#include "autoencoder.h"

int main(int argc, char *argv[]){
  if(argc<2){
    printf("Usage: convert_dataset_format FILE1 FILE2\n");
    abort();
  }
  dataset_t data=load_dataset(argv[1]);
  int bin_mode=get_binary_mode();
  printf("Read dataset in ");
  if(bin_mode==1){
    printf("binary");
  } else {
    printf("ascii");
  }
  printf(" mode (nrecs=%i, npoints=%i).\nConverting to ",data.nrecs,data.npoints);
  if(bin_mode==1){
    printf("ascii");
  } else {
    printf("binary");
  }
  printf(" mode...\n");
  //Flip AUTO_BINARY_MODE
  set_binary_mode(1-bin_mode);
  writedata(argv[2],&data);
}
