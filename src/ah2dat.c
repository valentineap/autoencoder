#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ahdefs.h"
#include "autoencoder.h"


int main(int argc, char *argv[]){
  if(argc!=4){printf("Incorrect arguments.\nUsage: ah2dat AHFILE DATFILE NPOINTS\n");exit(-1);}
  char *ahfile=argv[1];
  char *datfile=argv[2];
  int npoints;
  sscanf(argv[3],"%i",&npoints);

  int unit=1;
  int status=-1;
  int ifail,nrecs;
  c_openfile_(&unit,ahfile,&status,&ifail);
  if(ifail!=0){abort();}
  c_countrecs_(&unit,&nrecs,&ifail);
  if(ifail!=0){abort();}
  printf("Nrecs: %i\n",nrecs);
  int startbyte=1;
  ah_header_t header;
  FORTRAN_REAL *data=malloc(npoints*sizeof(FORTRAN_REAL));
  dataset_t dataset;
  dataset_alloc(&dataset,npoints,nrecs);
  int iirec=0,irec,ipt;
  for(irec=0;irec<nrecs;irec++){
    c_readheader_(&unit,(void *)&header,&startbyte,&ifail);
    startbyte+=1080;
    if(ifail!=0){printf("ifail: %i\n",ifail);abort();}
    if(header.ndt>=npoints){
      c_readdata_(&unit,(void *)data,&startbyte,&npoints,&ifail);
      if(ifail!=0){printf("ifail: %i\n",ifail);abort();}
      *(dataset.scale+iirec)=0.;
      for(ipt=0;ipt<npoints;ipt++){
	if(fabs((double) *(data+ipt))>*(dataset.scale+iirec)){
	  *(dataset.scale+iirec)=fabs((double) *(data+ipt));
	}
      } 
      for(ipt=0;ipt<npoints;ipt++){*(dataset.data+(ipt*nrecs)+iirec)=(double) *(data+ipt)/ *(dataset.scale+iirec);}
      *(dataset.weight+iirec)=*(dataset.scale+iirec);
      iirec++;
    }
    startbyte+=(4*header.ndt);
  }
  if(iirec!=nrecs){printf("Resizing...: %i\n",iirec);dataset_resize(&dataset,iirec);}
  writedata(datfile,&dataset);

}
