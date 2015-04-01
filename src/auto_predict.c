#include<stdlib.h>
#include<string.h>
#include<math.h>
#include "autoencoder.h"

#define NRAND 10000
#define NPDF 51

#define BESTFIT 0
#define PDF 1
int main(int argc,char *argv[]){
  int output=PDF;
  dataset_t data_in;
  dataset_t data_in_red;
  dataset_t enc_in;
  dataset_t enc_out;
  dataset_t data_out;
  dataset_t data_out_2;
  autoenc_t *a_enc;
  autoenc_t *a_dec;
  
  data_in=load_dataset(argv[1]);
  a_enc=load_autoencoder(argv[2]);
  a_dec=load_autoencoder(argv[3]);

  int n_enc_in=a_enc->layers[a_enc->nlayers/2 -1].nout;
  int n_dec_in=a_enc->layers[0].nin;
  int n_enc_out=a_dec->layers[a_dec->nlayers/2 -1].nout;
  int n_dec_out=a_dec->layers[0].nin;

  
  dataset_alloc(&enc_in,n_enc_in,data_in.nrecs);
  dataset_alloc(&enc_out,n_enc_out,NRAND);
  dataset_alloc(&data_out,n_dec_out,NRAND);
  if(output==BESTFIT){
    dataset_alloc(&data_out_2,n_dec_out,data_in.nrecs);
  } else if(output==PDF){
    dataset_alloc(&data_out_2,n_dec_out*NPDF,data_in.nrecs);
  }
  dataset_alloc(&data_in_red,n_dec_in,data_in.nrecs);
 

  //Copy initial chunk of data to new dataset
  memcpy(data_in_red.data,data_in.data,n_dec_in*data_in.nrecs*sizeof(double));
  memcpy(data_in_red.scale,data_in.scale,data_in.nrecs*sizeof(double));
  memcpy(data_in_red.weight,data_in.weight,data_in.nrecs*sizeof(double));
  
  //Now we can encode...
  autoencoder_encode(a_enc,&data_in_red,&enc_in);
  
  //Copy encodings into expanded encoding; add random numbers
  int irec,ipt,ir;
  for(irec=0;irec<data_in.nrecs;irec++){
    printf("Record %i of %i\n",irec,data_in.nrecs);
    //#pragma omp parallel for if(n_enc_in*NRAND>=MIN_ARRSIZE_PARALLEL) schedule(static) private(ipt,ir) shared(enc_out,enc_in,data_in)
    for(ipt=0;ipt<n_enc_in;ipt++){
      for(ir=0;ir<NRAND;ir++){
	*(enc_out.data+(ipt*NRAND)+ir)=*(enc_in.data+(ipt*data_in.nrecs)+irec)+(0.05*random_normal());
	if(ipt==0){
	  *(enc_out.weight+ir)=*(enc_in.weight+irec);
	  *(enc_out.scale+ir)=*(enc_in.scale+irec);
	}
      }
      
    }
    //    random_uniform_mem(enc_out.data+(n_enc_in*NRAND),(n_enc_out-n_enc_in)*NRAND,-1.1,1.1);
    random_normal_mem(enc_out.data+(n_enc_in*NRAND),(n_enc_out-n_enc_in)*NRAND,0.1);
    autoencoder_decode(a_dec,&enc_out,&data_out);

    //We now have NRAND possible seismograms.
    

    if(output==BESTFIT){
      double dist,best_dist;
      int ibest;
      dist=0.;
      ibest=0;
      for(ipt=0;ipt<n_dec_out;ipt++){
	dist+=pow(*(data_out.data+(ipt*NRAND))-*(data_in.data+(ipt*data_in.nrecs)+irec),2.);
      }
      best_dist=dist;
      for(ir=1;ir<NRAND;ir++){
	dist=0.;
	for(ipt=0;ipt<n_dec_out;ipt++){
	  dist+=pow(*(data_out.data+(ipt*NRAND)+ir)-*(data_in.data+(ipt*data_in.nrecs)+irec),2.);
	}
	if(dist<best_dist){
	  best_dist=dist;
	ibest=ir;
	}
      }
      for(ipt=0;ipt<data_out.npoints;ipt++){
	*(data_out_2.data+(ipt*data_in.nrecs)+irec)=*(data_out.data+(ipt*NRAND)+ibest);
      }
      *(data_out_2.scale+irec)=*(data_out.scale+irec);
      *(data_out_2.weight+irec)=*(data_out_2.weight+irec);
      printf("For record %i, preferred %i (%f)\n",irec,ibest,0.5*best_dist/(double) n_dec_in);
    } else if (output==PDF) {
      //First, zero array...
      int ivert;
      for(ipt=0;ipt<n_dec_out;ipt++){
	for(ivert=0;ivert<NPDF;ivert++){
	  *(data_out_2.data+(((ipt*NPDF)+ivert)*data_in.nrecs)+irec)=0.;
	}
      }
      for(ir=0;ir<NRAND;ir++){
	for(ipt=0;ipt<n_dec_out;ipt++){
	  ivert=(int) ((1.1+*(data_out.data+(ipt*NRAND)+ir))*(double) NPDF/(2.2));
	  *(data_out_2.data+(((ipt*NPDF)+ivert)*data_in.nrecs)+irec)+=1.;
	}
      }
      ////Normalise
      for(ipt=0;ipt<n_dec_out;ipt++){
	for(ivert=0;ivert<NPDF;ivert++){
	  *(data_out_2.data+(((ipt*NPDF)+ivert)*data_in.nrecs)+irec)/=(double) NRAND;
	}
      }
      *(data_out_2.scale+irec)=*(data_out.scale+irec);
      *(data_out_2.weight+irec)=*(data_out_2.weight+irec);
    }
  }

  printf("Write...\n");
  writedata(argv[4],&data_out_2);
  dataset_free(&data_in);
  dataset_free(&data_in_red);
  dataset_free(&data_out);
  dataset_free(&enc_in);
  dataset_free(&enc_out);
  autoencoder_free(a_enc);
  autoencoder_free(a_dec);
}
