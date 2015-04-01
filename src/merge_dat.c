#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include "autoencoder.h"
int main(int argc,const char **argv){
  if(argc<4){
    printf("Usage: merge_dat file1 file2 [...] out");
  }
  int ninputs=argc-2;  
  char * output_file=argv[ninputs+1];
  FILE *fp;
  //Check file does not exist
  fp=fopen(output_file,"r");
  if(fp!=NULL){
    printf("*** Output file already exists. Refusing to overwrite: delete and re-run ***");
    abort();
  }  
  dataset_t * input_files=malloc(ninputs*sizeof(dataset_t));
  int i;
  for(i=0;i<ninputs;i++){
    *(input_files+i)=load_dataset((char *)*(argv+1+i));
  }
  int npts=input_files->npoints;
  int nrecs=input_files->nrecs;

  for(i=1;i<ninputs;i++){
    if((input_files+i)->nrecs!=nrecs){
      printf("Difference in number of records between:\n%s\n%s\n",*(argv+1),*(argv+1+i));
      abort();
    }
    npts+=(input_files+i)->npoints;
  }
  dataset_t * output_data=malloc(sizeof(dataset_t));
  dataset_alloc(output_data,npts,nrecs);
  int next=0,nthis;
  for(i=0;i<ninputs;i++){
    nthis=(input_files+i)->nrecs*(input_files+i)->npoints;
    memcpy(output_data->data+next,(input_files+i)->data,nthis*sizeof(double));
    dataset_free(input_files+i);
    next+=nthis;
  }
  for(i=0;i<nrecs;i++){
    *(output_data->weight+i)=1;
    *(output_data->scale+i)=1;
  }
  writedata(output_file,output_data);
  dataset_free(output_data);
}
  
