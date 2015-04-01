#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include<popt.h>
#include "autoencoder.h"
int main(int argc, const char **argv){
  if(argc==1){
    printf("No input files...\n");
    abort();
  }
  char * output_file=NULL;
  struct poptOption optionsTable[]={
    {"output-file",'o',POPT_ARG_STRING,&output_file,0,"Filename for output","FILE"},
    POPT_AUTOHELP
    POPT_TABLEEND};
  poptContext poptc=poptGetContext(NULL,argc,argv,optionsTable,0);
  int pp;
  if((pp=poptGetNextOpt(poptc))<-1){
    printf("%s: %s\n",poptBadOption(poptc,POPT_BADOPTION_NOALIAS),poptStrerror(pp));
    exit(-1);
  }
  const char **leftovers=poptGetArgs(poptc);
  int nfiles=0;
  while(*(leftovers+nfiles)!=NULL){nfiles++;}
  if(nfiles==0){
    printf("No input files...\n");
    abort();
  }
  if(output_file==NULL){
    printf("No output file specified...\n");
    abort();
  }
  printf("%i files to be merged...\n",nfiles);
  dataset_t *infiles=malloc(nfiles*sizeof(dataset_t));
  int ifile;
  int gnrecs=0,gnpts;
  for(ifile=0;ifile<nfiles;ifile++){
    *(infiles+ifile)=load_dataset((char *)*(leftovers+ifile));
    if((infiles+ifile)->npoints==0){
      printf("File %i (%s) does not contain any records!\n",ifile+1,*(leftovers+ifile));
    }
    if(ifile==0){
      gnpts=infiles->npoints;
    } else {
      if((infiles+ifile)->npoints!=gnpts){
	printf("File %i (%s) does not share npoints value with predecessors.\nGot %i; expected %i...\n",ifile+1,*(leftovers+ifile),(infiles+ifile)->npoints,gnpts);
	abort();
      }
    }
    gnrecs+=(infiles+ifile)->nrecs;
  }
  printf("All files read in. Combined size: %i records of %i points each.\n",gnrecs,gnpts);
  dataset_t dout;
  dataset_alloc(&dout,gnpts,gnrecs);
  int ioffset=0;
  int ipt;
  for(ipt=0;ipt<gnpts;ipt++){	
	  for(ifile=0;ifile<nfiles;ifile++){
		if((infiles+ifile)->nrecs>0){
		  memcpy(dout.data+ioffset,(infiles+ifile)->data+(ipt*(infiles+ifile)->nrecs),(infiles+ifile)->nrecs*sizeof(double));
		  ioffset+=(infiles+ifile)->nrecs;
	  	}	
	  }
  }


ioffset=0;
  for(ifile=0;ifile<nfiles;ifile++){
    if((infiles+ifile)->nrecs>0){
      
//      memcpy(dout.data+(ioffset*gnpts),(infiles+ifile)->data,gnpts*(infiles+ifile)->nrecs*sizeof(double));
      memcpy(dout.scale+ioffset,(infiles+ifile)->scale,(infiles+ifile)->nrecs*sizeof(double));
      memcpy(dout.weight+ioffset,(infiles+ifile)->weight,(infiles+ifile)->nrecs*sizeof(double));
      ioffset+=(infiles+ifile)->nrecs;
    }
    dataset_free(infiles+ifile);
  }
  writedata(output_file,&dout);
      



  free(infiles);

}
