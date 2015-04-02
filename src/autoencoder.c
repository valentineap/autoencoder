////////////////////////////////////////////////////////////
// autoencoder.c - Main library for autoencoder framework //
////////////////////////////////////////////////////////////
// Andrew Valentine                                       //
// Universiteit Utrecht                                   //
// 2011-2012                                              //
//                                                        //
// With contributions from:                               //
// - Paul Kaeufl                                          //
//                                                        //
////////////////////////////////////////////////////////////
// $Id: autoencoder.c,v 1.3 2012/03/31 15:14:51 andrew Exp andrew $
//
// Compile with -DACML if using ACML library; otherwise will require cblas library.
// If not using ACML, requires libtwist (Mersenne twister code) or compile with -DNATIVE_RAND to use the C standard 'rand' function
// Turn on openmp directives in compiler to enable full parallelism (shared-memory only)

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef ACML
#include <acml.h>
#else
#include <cblas.h>
#ifndef NATIVE_RAND //Use Mersenne Twister; only if not ACML
#include "mt19937ar.h"
#endif
#endif
#include <assert.h>
#include <signal.h>
#include "autoencoder.h"

#ifdef TIMING
#include <time.h>
#endif

volatile sig_atomic_t cease_training=0; //Flag to handle Ctrl-C
volatile int AUTO_BINARY_MODE=1;

///////////////////////
// Dataset functions //
///////////////////////
void dataset_alloc(dataset_t *data,int npts,int nrecs){
  //Allocate memory for dataset storage: nrecs records, each containing npts points
  data->npoints=npts;
  data->nrecs=nrecs;
  data->data=malloc(npts*nrecs*sizeof(double));
  data->scale=malloc(nrecs*sizeof(double));
  data->weight=malloc(nrecs*sizeof(double));
}
void dataset_free(dataset_t *data){
  //Free previously allocated dataset
  data->npoints=-1;
  data->nrecs=-1;
  free(data->data);
  free(data->scale);
  free(data->weight);
}

void dataset_copy(dataset_t *dest,dataset_t *src){
  //Make a copy of existing data structure in memory
  dataset_alloc(dest,src->npoints,src->nrecs);
  memcpy(dest->data,src->data,src->npoints*src->nrecs*sizeof(double));
  memcpy(dest->weight,src->weight,src->nrecs*sizeof(double));
  memcpy(dest->scale,src->scale,src->nrecs*sizeof(double));
}

void dataset_resize(dataset_t *data,int newnrecs){
  //Resize dataset (change nrecs only). realloc *should* preserve existing memory contents.
  data->data=realloc(data->data,newnrecs*data->npoints*sizeof(double));
  data->weight=realloc(data->weight,newnrecs*sizeof(double));
  data->scale=realloc(data->scale,newnrecs*sizeof(double));
  int i,j;
  if(newnrecs>data->nrecs){
    for(j=data->nrecs;j<newnrecs;j++){
      for(i=0;i<data->npoints;i++){
	*(data->data+(i*newnrecs)+j)=0.;
      }
      *(data->weight+j)=0.;
      *(data->scale+j)=0.;
    }
  }
  data->nrecs=newnrecs;
}

dataset_t load_dataset(char *filename){
  //Open file and read in dataset
  dataset_t data;
  int npts,nrecs,check;
  FILE *fp;
  if((fp=fopen(filename,"rb"))==NULL){
    printf("Error opening file: %s\n",filename);
    abort();
  }
  fread(&check,sizeof(int),1,fp);
  if(check==0){ 
    AUTO_BINARY_MODE=1;
    if(fread(&npts,sizeof(int),1,fp)!=1){printf("Error reading from file: %s\n",filename);printf("Failed to read npts\n");abort();}
    if(fread(&nrecs,sizeof(int),1,fp)!=1){printf("Error reading from file: %s\n",filename);printf("Failed to read nrecs\n");abort();}
    dataset_alloc(&data,npts,nrecs);
    if(fread(data.data,sizeof(double),npts*nrecs,fp)!=npts*nrecs){printf("Error reading from file: %s\n",filename);printf("Failed to read data\n");abort();}
    if(fread(data.scale,sizeof(double),nrecs,fp)!=nrecs){printf("Error reading from file: %s\n",filename);printf("Failed to read scale\n");abort();}
    if(fread(data.weight,sizeof(double),nrecs,fp)!=nrecs){printf("Error reading from file: %s\n",filename);printf("Failed to read weights\n");abort();}
    fclose(fp);
  } else {
    AUTO_BINARY_MODE=0;
    fclose(fp);
    if((fp=fopen(filename,"r"))==NULL){
      printf("Error opening file: %s\n",filename);
      abort();
    }
    if(fscanf(fp,"%d%d",&npts,&nrecs)!=2){printf("Error reading from file: %s\n",filename);printf("Have npts=%i & nrecs=%i\n",npts,nrecs);abort();}
    dataset_alloc(&data,npts,nrecs);
    int i,j;
    for (j=0;j<nrecs;j++){
      for(i=0;i<npts;i++){
	//      printf("%i %i\n",i,j);
	if(fscanf(fp,"%lf",data.data+(i*(nrecs)+j))!=1){printf("Error reading data from file: %s\n",filename);printf("Failed to read data[%i*(nrecs)+%i]\n",i,j);abort();}
      }
      if(fscanf(fp,"%le",data.scale+j)!=1){printf("Error reading scale from file: %s\n",filename);printf("Failed to read scale[%i]\n",j);abort();}
      if(fscanf(fp,"%le",data.weight+j)!=1){printf("Error reading weight from file: %s\n",filename);printf("Failed to read weight[%i]\n",j);abort();}
    }
    fclose(fp);
  }
  double t;
  int ii,jj;
  printf("npts=%i,nrecs=%i\n",npts,nrecs);
  t=0.;
  for (jj=0;jj<nrecs;jj++){
    for(ii=0;ii<npts;ii++){
      t+=*(data.data+(ii*nrecs)+jj);
    }
  }
  printf("tda=%f\n",t);
  t=0.;
  for(jj=0;jj<nrecs;jj++){
    t+=*(data.scale+jj);
  }
  printf("tsc=%f\n",t);
  t=0.;
  for(jj=0;jj<nrecs;jj++){
    t+=*(data.weight+jj);
  }
  printf("twt=%f\n\n");


  return data;
}

void writedata(char *filename,dataset_t *data){
  //Write dataset to a file
  if(AUTO_BINARY_MODE==1){
    FILE *fp;
    int check=0;
    if((fp=fopen(filename,"wb"))==NULL){printf("Error opening file: %s\n",filename);abort();}
    fwrite(&check,sizeof(int),1,fp); //To indicate that this is a binary file...
    fwrite(&(data->npoints),sizeof(int),1,fp);
    fwrite(&(data->nrecs),sizeof(int),1,fp);
    fwrite(data->data,sizeof(double),data->nrecs*data->npoints,fp);
    fwrite(data->scale,sizeof(double),data->nrecs,fp);
    fwrite(data->weight,sizeof(double),data->nrecs,fp);
    fclose(fp);
  } else {
    FILE *fp;
    if((fp=fopen(filename,"w"))==NULL){printf("Error opening file: %s\n",filename);abort();}
    int i,j;
    fprintf(fp,"%06i %04i\n",data->npoints,data->nrecs);
    for(j=0;j<data->nrecs;j++){
      for(i=0;i<data->npoints;i++){
	fprintf(fp,"%8.5f ",*(data->data+(i*data->nrecs)+j));
      }
      fprintf(fp,"%11.5e %11.5e\n",*(data->scale+j),*(data->weight+j));
    }
    fclose(fp);
  }
}

int get_binary_mode(){
  return AUTO_BINARY_MODE;
}

void set_binary_mode(int m){
  AUTO_BINARY_MODE=m;
}

/////////////////////////////
// Random number functions //
/////////////////////////////
volatile int seeded_random=0;//Has random seed been initialised?
#ifdef ACML
int lstate=633;
int state[633];
void random_uniform_mem(double *start,int n, double lo,double hi){
  //Fill memory block with uniformly-distributed random numbers
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  dranduniform(n,lo,hi,state,start,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
}

void random_normal_mem(double *start, int n, double stdev){
  //Fill memory block with gaussian-distributed random numbers
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  drandgaussian(n,0.0,stdev,state,start,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
}

inline double random_uniform(double lo,double hi){
  //Return a random number from a uniform distribution in [lo,hi]
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  double r;
  dranduniform(1,lo,hi,state,&r,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
  return r;
}

inline double random_normal(){
  //Return a random number from a Gaussian with unit stdev
  int info=0;
  if(seeded_random==0){
    int lseed=1;
    int seed=(unsigned)time(NULL)+getpid();
    drandinitialize(3,0,&seed,&lseed,state,&lstate,&info);
    if(info!=0) {
      printf("Error initializing random number generator; aborting...\n");
      exit(-1);
    }
    seeded_random=1;
  }
  double r;
  drandgaussian(1,0.,1.0,state,&r,&info);
  if(info!=0){
    printf("Error in random number generation; aborting...\n");
    exit(-1);
  }
  return r;
}

#else
int have_stored_rand=0;
double stored_rand;

inline double random_normal(){
  //Return a random number drawn from a Gaussian distribution with standard deviation 1.0
  //Marsaglia's Polar Method
  double u,v,x1,x2,w;
#ifdef NATIVE_RAND
  int seedval;
  if(seeded_random==0){
    seedval=(unsigned)time(NULL)+getpid();
    srand(seedval);
    seeded_random=1;
  }
#else //Mersenne Twister
  unsigned long seedval;
  if(seeded_random==0){
    seedval=(unsigned long)time(NULL)+getpid();
    init_genrand(seedval);
    seeded_random=1;
  }
#endif
  if(have_stored_rand){
    have_stored_rand=0;
    return stored_rand;
  } else {
    do{
#ifdef NATIVE_RAND 
      u=(double)rand()/(double)RAND_MAX;
      v=(double)rand()/(double)RAND_MAX;
#else
      u=genrand_real1();
      v=genrand_real2();
#endif
      x1=2.0*u-1.0;
      x2=2.0*v-1.0;
      w=x1*x1+x2*x2;
    } while (w>=1.0);
    stored_rand=x2*sqrtl((-2.0*logl(w))/w);
    have_stored_rand=1;
    return x1*sqrtl((-2.0*logl(w))/w);
  }
}
#endif

////////////////////
// CRBM functions //
////////////////////
void crbm_init(crbm_t *c,int nlv, int nlh, double stdev, double loglo, double logup, FILE *logfp){
  //Set up CRBM. Allocate memory, initialise weights.
  if(logfp!=NULL){fprintf(logfp,"# crbm_init  >> nlv=%i nlh=%i init-stdev=%f f0=%f f1=%f\n",nlv,nlh,stdev,loglo,logup);}
  c->nlv=nlv;
  c->nlh=nlh;
  if((c->w=calloc(c->nlv * c->nlh,sizeof(double)))==NULL){printf("Error allocating w\n");abort();};
  if((c->b_v2h=calloc(c->nlh,sizeof(double)))==NULL){printf("Error allocating b_v2h\n");abort();};
  if((c->a_v2h=calloc(c->nlh,sizeof(double)))==NULL){printf("Error allocating a_v2h\n");abort();};
  if((c->b_h2v=calloc(c->nlv,sizeof(double)))==NULL){printf("Error allocating b_h2v\n");abort();};
  if((c->a_h2v=calloc(c->nlv,sizeof(double)))==NULL){printf("Error allocating a_h2v\n");abort();};
  int i;
  if(stdev>=0.){for(i=0;i<c->nlv*c->nlh;i++){*(c->w+i)+=stdev*random_normal();}}
  for(i=0;i<c->nlv;i++){*(c->a_h2v+i)=1.0;}
  for(i=0;i<c->nlh;i++){*(c->a_v2h+i)=1.0;}
  c->loglo=loglo;
  c->logup=logup;
}

void crbm_free(crbm_t *c){
  //Free memory used by CRBM
  c->nlv=0;
  c->nlh=0;
  free(c->w);
  free(c->b_v2h);
  free(c->b_h2v);
  free(c->a_v2h);
  free(c->a_h2v);
}


void crbm_encode(crbm_t *c,dataset_t *data_in, dataset_t *data_out,double stdev){
  //Encode data_in using CRBM c; result in data_out. stdev specifies width of Gaussian random number distribution to use (-1 for no randomness)
  int i,j;
  assert(data_in->nrecs==data_out->nrecs);
  assert(data_in->npoints==c->nlv);
  assert(data_out->npoints==c->nlh);
#pragma omp parallel for if((c->nlh*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(c,data_in,data_out)
  for(i=0;i<c->nlh;i++){
    for(j=0;j<data_in->nrecs;j++){
      *(data_out->data+(i*data_in->nrecs)+j)=*(c->b_v2h+i);
    }
  }
#ifdef ACML
  dgemm('N','T',data_in->nrecs,c->nlh,c->nlv,1.0,data_in->data,data_in->nrecs,c->w,c->nlh,1.0,data_out->data,data_in->nrecs);
#else
  cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,c->nlh,data_in->nrecs,c->nlv,1.0,c->w,c->nlh,data_in->data,data_in->nrecs,1.0,data_out->data,data_in->nrecs);
#endif
  memcpy(data_out->weight,data_in->weight,data_in->nrecs*sizeof(double)); //Propagate weights and scales through successive CRBMs
  memcpy(data_out->scale,data_in->scale,data_in->nrecs*sizeof(double));
  if(stdev<0.){
#pragma omp parallel for if((c->nlh*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(data_in,data_out,c)
    for(i=0;i<c->nlh;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*ACTIVATION(*(c->a_v2h+i)*(*(data_out->data+(i*data_in->nrecs)+j)));
      }
    }
  } else {
#pragma omp parallel for if((c->nlh*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(data_in,data_out,c)
    for(i=0;i<c->nlh;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*ACTIVATION(*(c->a_v2h+i)*(*(data_out->data+(i*data_in->nrecs)+j)+(stdev*random_normal())));
      }
    }
  }
}


void crbm_decode(crbm_t *c,dataset_t *data_in, dataset_t *data_out,double stdev){
  //Decode data_in using c; result in data_out
  int i,j;
  assert(data_in->nrecs==data_out->nrecs);
  assert(data_in->npoints==c->nlh);
  assert(data_out->npoints==c->nlv);
#pragma omp parallel for if((c->nlv*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(c,data_in,data_out)
  for(i=0;i<c->nlv;i++){
    for(j=0;j<data_in->nrecs;j++){
      *(data_out->data+(i*data_in->nrecs)+j)=*(c->b_h2v+i);
    }
  }
#ifdef ACML
  dgemm('N','N',data_in->nrecs,c->nlv,c->nlh,1.0,data_in->data,data_in->nrecs,c->w,c->nlh,1.0,data_out->data,data_in->nrecs);
#else
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,c->nlv,data_in->nrecs,c->nlh,1.0,c->w,c->nlh,data_in->data,data_in->nrecs,1.0,data_out->data,data_in->nrecs);
#endif
  memcpy(data_out->weight,data_in->weight,data_in->nrecs*sizeof(double)); //Propagate weights and scales through successive CRBMs
  memcpy(data_out->scale,data_in->scale,data_in->nrecs*sizeof(double));

  if(stdev<0.){
#pragma omp parallel for if((c->nlv*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(data_in,data_out,c)
    for(i=0;i<c->nlv;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*ACTIVATION(*(c->a_h2v+i)*(*(data_out->data+(i*data_in->nrecs)+j)));
      }
    }
  } else {
#pragma omp parallel for if((c->nlv*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(data_in,data_out,c)
    for(i=0;i<c->nlv;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*ACTIVATION(*(c->a_h2v+i)*(*(data_out->data+(i*data_in->nrecs)+j)+(stdev*random_normal())));
      }
    }
  }
}

void _tmp_dbg(crbm_t *c){
  int ii;
  for (ii=0;ii<4;ii++) {
    printf("a_h2v[%i]: %f\t", ii,c->a_h2v[ii]);
    printf("a_v2h[%i]: %f\t", ii,c->a_v2h[ii]);
    printf("b_h2v[%i]: %f\t", ii,c->b_h2v[ii]);
    printf("b_v2h[%i]: %f\t\n", ii,c->b_v2h[ii]);
  }
  for (ii=0;ii<=10;ii++) {
    printf("w[%i]: %f\n", ii,c->w[ii]);
  }

}


void crbm_train(crbm_t *c,dataset_t *data,state_settings_t * settings,FILE *logfp,dataset_t *enc_force){
  //Train CRBM c using dataset data. Write log data to logfp. If enc_force!=NULL, start of encodings must match those in dataset enc_force.
  dataset_t enc,dec,encdec;
  double *cdata,*cdec;
  if(enc_force!=NULL){
    assert(enc_force->nrecs==data->nrecs);
    assert(enc_force->npoints<=c->nlh);
  }
  dataset_alloc(&enc,c->nlh,data->nrecs);
  dataset_alloc(&dec,c->nlv,data->nrecs);
  dataset_alloc(&encdec,c->nlh,data->nrecs);
  cdata=malloc(c->nlv*c->nlh*sizeof(double));
  cdec=malloc(c->nlv*c->nlh*sizeof(double));
  if(logfp!=NULL){fprintf(logfp,"# crbm_train >> nrecs=%i niter=%i stdev=%f lrate=%f\n",data->nrecs,settings->niter_crbm,settings->noise_crbm,settings->lrate_crbm);}
  int i,j;
  double scale,denom;
  double wtnorm=0.;
  for(i=0;i<data->nrecs;i++){wtnorm+=*(data->weight+i);}
  while(settings->iter<settings->niter_crbm){
    crbm_encode(c,data,&enc,settings->noise_crbm);
    if(enc_force!=NULL){
      memcpy(enc.data,enc_force->data,enc_force->npoints*enc_force->nrecs*sizeof(double));
    }
    crbm_decode(c,&enc,&dec,settings->noise_crbm);
    crbm_encode(c,&dec,&encdec,settings->noise_crbm);
    if(enc_force!=NULL){
      memcpy(encdec.data,enc_force->data,enc_force->npoints*enc_force->nrecs*sizeof(double));
    }
    if(settings->crbm_verbose_flag==1){
        printf("CRBM (%i->%i) training iteration %3i :: E=%f\n",c->nlv,c->nlh,settings->iter+1,error_dataset(data,&dec));
    }
    if(logfp!=NULL){fprintf(logfp,"%i %f\n",settings->iter+1,error_dataset(data,&dec));}
    //for (i=0;i<10;i++) {printf("Data[%i]: %f\n", i, *(data->data+i));}
#pragma omp parallel for if((c->nlh*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i,j) shared(c,data,enc,wtnorm,encdec)
    for(i=0;i<c->nlh;i++){
      for(j=0;j<data->nrecs;j++){ //Scale encodings by weights so that when we sum over all records (in dgemm) weighting is taken into account
	*(enc.data+(i*data->nrecs)+j)*=*(data->weight+j)/wtnorm; 
	*(encdec.data+(i*data->nrecs)+j)*=*(data->weight+j)/wtnorm;
      }
    }
#ifdef ACML
    dgemm('T','N',c->nlh,c->nlv,data->nrecs,1.0,enc.data,data->nrecs,data->data,data->nrecs,0.0,cdata,c->nlh);
    dgemm('T','N',c->nlh,c->nlv,data->nrecs,1.0,encdec.data,data->nrecs,dec.data,data->nrecs,0.0,cdec,c->nlh);
#else
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,c->nlv,c->nlh,data->nrecs,1.0,data->data,data->nrecs,enc.data,data->nrecs,0.0,cdata,c->nlh);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,c->nlv,c->nlh,data->nrecs,1.0,dec.data,data->nrecs,encdec.data,data->nrecs,0.0,cdec,c->nlh);
#endif
    scale=settings->lrate_crbm; //wtnorm implies division by number of traces (or equivalent)
#pragma omp parallel if(((c->nlv*c->nlh)>=MIN_ARRSIZE_PARALLEL)||((c->nlv*data->nrecs)>=MIN_ARRSIZE_PARALLEL)||((c->nlh*data->nrecs)>=MIN_ARRSIZE_PARALLEL)) private(i,j,denom) shared(c,scale,cdata,cdec,enc,encdec,dec,wtnorm)
    {
#pragma omp for schedule(static)
      for(i=0;i<c->nlv;i++){
	for(j=0;j<c->nlh;j++){
	  *(c->w+(i*c->nlh)+j)+=scale*(*(cdata+(i*c->nlh)+j)-*(cdec+(i*c->nlh)+j));//Weighting already handled
	}
      }
#pragma omp for schedule(static)
      for(i=0;i<c->nlh;i++){ 
	for(j=0;j<data->nrecs;j++){
	  *(c->b_v2h+i)+=scale*(*(enc.data+(i*data->nrecs)+j)-*(encdec.data+(i*data->nrecs)+j));//Weighting already handled
	}
      }
#pragma omp for schedule(static)
      for(i=0;i<c->nlv;i++){
	for(j=0;j<data->nrecs;j++){
	  *(c->b_h2v+i)+=scale*(*(data->data+(i*data->nrecs)+j)-*(dec.data+(i*data->nrecs)+j))* *(data->weight+j)/wtnorm;
	}	
      }
#pragma omp for schedule(static)
      for(i=0;i<c->nlh;i++){
	denom=pow(*(c->a_v2h+i),2.);
	for(j=0;j<data->nrecs;j++){
	  *(c->a_v2h+i)+=(scale/denom)*(pow(*(enc.data+(i*data->nrecs)+j),2.)-pow(*(encdec.data+(i*data->nrecs)+j),2.))*wtnorm/ *(data->weight+j);//Avoid double-weighting the squared terms
	}
      }
#pragma omp for schedule(static)
      for(i=0;i<c->nlv;i++){
	denom=pow(*(c->a_h2v+i),2.);
	for(j=0;j<data->nrecs;j++){
	  *(c->a_h2v+i)+=(scale/denom)*(pow(*(data->data+(i*data->nrecs)+j),2.)-pow(*(dec.data+(i*data->nrecs)+j),2.))* *(data->weight+j)/wtnorm;
	}
      }
    }
    
    settings->iter++;
    if(cease_training){
      FILE * fp=fopen("crbm.dump","w");
      //Need to dump all crbms up to here...
      int i;
      for(i=0;i<settings->stage+1;i++){
	write_crbm(fp,c-(settings->stage)+i);
      }
      fclose(fp);
      write_state("crbm.dump.state",*settings,"crbm.dump");
      exit(0);
    }
  }
  
  dataset_free(&enc);
  dataset_free(&dec);
  dataset_free(&encdec);
  free(cdata);
  free(cdec);
}

void write_crbm(FILE *fp,crbm_t *c){
  //Write crbm structure to file
  fwrite(&(c->nlv),sizeof(int),1,fp);
  fwrite(&(c->nlh),sizeof(int),1,fp);
  fwrite(c->w,sizeof(double),c->nlv*c->nlh,fp);
  fwrite(c->b_v2h,sizeof(double),c->nlh,fp);
  fwrite(c->b_h2v,sizeof(double),c->nlv,fp);
  fwrite(c->a_v2h,sizeof(double),c->nlh,fp);
  fwrite(c->a_h2v,sizeof(double),c->nlv,fp);
  fwrite(&(c->loglo),sizeof(double),1,fp);
  fwrite(&(c->logup),sizeof(double),1,fp);
}

void save_crbm(char *filename, crbm_t *c){
  //Write crbm structure to named file
  FILE *fp=fopen(filename,"w");
  fseek(fp,0,SEEK_SET);
  write_crbm(fp,c);
  fclose(fp);
} 


crbm_t read_crbm(FILE *fp){
  //Read crbm structure from file
  crbm_t c;
  int nlv,nlh;
  fread(&nlv,sizeof(int),1,fp);
  fread(&nlh,sizeof(int),1,fp);
  c.nlv=nlv;
  c.nlh=nlh;
  c.w=malloc(nlv*nlh*sizeof(double));
  c.b_v2h=malloc(nlh*sizeof(double));
  c.b_h2v=malloc(nlv*sizeof(double));
  c.a_v2h=malloc(nlh*sizeof(double));
  c.a_h2v=malloc(nlv*sizeof(double));
  fread(c.w,sizeof(double),nlv*nlh,fp);
  fread(c.b_v2h,sizeof(double),nlh,fp);
  fread(c.b_h2v,sizeof(double),nlv,fp);
  fread(c.a_v2h,sizeof(double),nlh,fp);
  fread(c.a_h2v,sizeof(double),nlv,fp);
  fread(&(c.loglo),sizeof(double),1,fp);
  fread(&(c.logup),sizeof(double),1,fp);
  return c;
}

crbm_t load_crbm(char *filename){
  //Read crbm structure from named file
  FILE *fp=fopen(filename,"r");
  fseek(fp,0,SEEK_SET);
  crbm_t c=read_crbm(fp);
  fclose(fp);
  return c;
}


///////////////////////////
// Autoencoder functions //
///////////////////////////
autoenc_t * make_autoencoder(int ncrbms,crbm_t *crbms,FILE *logfp){
  //Assemble autoencoder from collection of crbms
  autoenc_t * myauto=malloc(sizeof(autoenc_t)+2*ncrbms*sizeof(layer_t));
  myauto->nlayers=2*ncrbms;
  int i,j,k;
  if(logfp!=NULL){
    fprintf(logfp, "# nlayers=%i : ",myauto->nlayers);
  }
  for(i=0;i<ncrbms;i++){
    myauto->layers[i].nin=(crbms+i)->nlv;
    if(logfp!=NULL){
      fprintf(logfp,"%i ",myauto->layers[i].nin);
    }
    myauto->layers[i].nout=(crbms+i)->nlh;
    myauto->layers[i].loglo=(crbms+i)->loglo;
    myauto->layers[i].logup=(crbms+i)->logup;
    myauto->layers[i].w=malloc((crbms+i)->nlv * (crbms+i)->nlh*sizeof(double));
#pragma omp parallel for if(((crbms+i)->nlh*(crbms+i)->nlv)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(myauto,crbms,i) 
    for(j=0;j<(crbms+i)->nlh;j++){
      for(k=0;k<(crbms+i)->nlv;k++){
	*(myauto->layers[i].w+(j*(crbms+i)->nlv)+k)=*((crbms+i)->w+(k*(crbms+i)->nlh)+j);
      }
    }
    myauto->layers[i].a=malloc((crbms+i)->nlh*sizeof(double));
    memcpy(myauto->layers[i].a,(crbms+i)->a_v2h,(crbms+i)->nlh*sizeof(double));
    myauto->layers[i].b=malloc((crbms+i)->nlh*sizeof(double));
    memcpy(myauto->layers[i].b,(crbms+i)->b_v2h,(crbms+i)->nlh*sizeof(double));
  }
  int icrbm;
  for(i=ncrbms;i<2*ncrbms;i++){
    icrbm=2*ncrbms-i-1;
    myauto->layers[i].nin=(crbms+icrbm)->nlh;
    if(logfp!=NULL){
      fprintf(logfp, "%i ",myauto->layers[i].nin);
    }
    myauto->layers[i].nout=(crbms+icrbm)->nlv;
    myauto->layers[i].loglo=(crbms+icrbm)->loglo;
    myauto->layers[i].logup=(crbms+icrbm)->logup;
    myauto->layers[i].w=malloc((crbms+icrbm)->nlv*(crbms+icrbm)->nlh*sizeof(double));
    memcpy(myauto->layers[i].w,(crbms+icrbm)->w,(crbms+icrbm)->nlv*(crbms+icrbm)->nlh*sizeof(double));
    myauto->layers[i].a=malloc((crbms+icrbm)->nlv*sizeof(double));
    memcpy(myauto->layers[i].a,(crbms+icrbm)->a_h2v,(crbms+icrbm)->nlv*sizeof(double));
    myauto->layers[i].b=malloc((crbms+icrbm)->nlv*sizeof(double));
    memcpy(myauto->layers[i].b,(crbms+icrbm)->b_h2v,(crbms+icrbm)->nlv*sizeof(double));
  }
  if(logfp!=NULL){
    fprintf(logfp,"%i\n",myauto->layers[2*ncrbms-1].nout);
  }
  return myauto;
}

void write_autoencoder(FILE *fp,autoenc_t *a){
  //Write autoencoder structure to file
  fwrite(&(a->nlayers),sizeof(int),1,fp);
  int i;
  for(i=0;i<a->nlayers;i++){
    fwrite(&(a->layers[i].nin),sizeof(int),1,fp);
    fwrite(&(a->layers[i].nout),sizeof(int),1,fp);
    fwrite(&(a->layers[i].loglo),sizeof(double),1,fp);
    fwrite(&(a->layers[i].logup),sizeof(double),1,fp);
    fwrite(a->layers[i].w,sizeof(double),a->layers[i].nin*a->layers[i].nout,fp);
    fwrite(a->layers[i].a,sizeof(double),a->layers[i].nout,fp);
    fwrite(a->layers[i].b,sizeof(double),a->layers[i].nout,fp);
  }
}

void save_autoencoder(char *filename,autoenc_t *a){
  //Write autoencoder structure to named file
  FILE *fp=fopen(filename,"w");
  fseek(fp,0,SEEK_SET);
  write_autoencoder(fp,a);
  fclose(fp);
}

autoenc_t *read_autoencoder(FILE *fp){
  //Read autoencoder from file
  //Assume that fp is correctly positioned...
  int nlayers;
  autoenc_t *a;
  fread(&nlayers,sizeof(int),1,fp);
  a=malloc(sizeof(autoenc_t)+nlayers*sizeof(layer_t));
  a->nlayers=nlayers;
  int i;
  for(i=0;i<nlayers;i++){
    fread(&(a->layers[i].nin),sizeof(int),1,fp);
    fread(&(a->layers[i].nout),sizeof(int),1,fp);
    fread(&(a->layers[i].loglo),sizeof(double),1,fp);
    fread(&(a->layers[i].logup),sizeof(double),1,fp);
    a->layers[i].w=malloc(a->layers[i].nin*a->layers[i].nout*sizeof(double));
    fread(a->layers[i].w,sizeof(double),a->layers[i].nin*a->layers[i].nout,fp);
    a->layers[i].a=malloc(a->layers[i].nout*sizeof(double));
    fread(a->layers[i].a,sizeof(double),a->layers[i].nout,fp);
    a->layers[i].b=malloc(a->layers[i].nout*sizeof(double));
    fread(a->layers[i].b,sizeof(double),a->layers[i].nout,fp);
  }
  return a;
}

autoenc_t *load_autoencoder(char *filename){
  //Read autoencoder from named file
  FILE *fp=fopen(filename,"r");
  fseek(fp,0,SEEK_SET);
  autoenc_t *a=read_autoencoder(fp);
  fclose(fp);
  return a;
}

void autoencoder_free(autoenc_t *a){
  //free memory used by autoencoder
  int i;
  for(i=0;i<a->nlayers;i++){
    free(a->layers[i].w);
    free(a->layers[i].a);
    free(a->layers[i].b);
  }
}

void autoencoder_encode(autoenc_t *a,dataset_t *data_in,dataset_t *data_out){
  //Encode data_in using autoencoder
  assert(data_in->npoints==a->layers[0].nin);
  assert(data_out->npoints==a->layers[a->nlayers/2 -1].nout);
  assert(data_in->nrecs==data_out->nrecs);
  int i,j,k;
  double *layer_in=malloc(a->layers[0].nin*data_in->nrecs*sizeof(double));
  double *layer_out;
  double *tmp;
  memcpy(layer_in,data_in->data,a->layers[0].nin*data_in->nrecs*sizeof(double));
  for(i=0;i<a->nlayers/2;i++){
    layer_out=malloc(a->layers[i].nout*data_in->nrecs*sizeof(double));
#pragma omp parallel for if((a->layers[i].nout*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(data_in,layer_out,i,a)
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=*(a->layers[i].b+j);
      }
    }
#ifdef ACML
    dgemm('N','N',data_in->nrecs,a->layers[i].nout,a->layers[i].nin,1.0,layer_in,data_in->nrecs,a->layers[i].w,a->layers[i].nin,1.0,layer_out,data_in->nrecs);
#else
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i].nout,data_in->nrecs,a->layers[i].nin,1.0,a->layers[i].w,a->layers[i].nin,layer_in,data_in->nrecs,1.0,layer_out,data_in->nrecs);
#endif
#pragma omp parallel for if((a->layers[i].nout*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(layer_out,data_in,a,i)
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=a->layers[i].loglo+(a->layers[i].logup-a->layers[i].loglo)*ACTIVATION(*(layer_out+(j*data_in->nrecs)+k)* *(a->layers[i].a+j));
      }
    }
    free(layer_in);
    layer_in=layer_out;
    layer_out=NULL;
  }
  memcpy(data_out->data,layer_in,data_in->nrecs*a->layers[a->nlayers/2 -1].nout*sizeof(double)); //Layer_in due to swap earlier
  memcpy(data_out->weight,data_in->weight,data_in->nrecs*sizeof(double));
  memcpy(data_out->scale,data_in->scale,data_in->nrecs*sizeof(double));
  free(layer_in);

} 
  
void autoencoder_decode(autoenc_t *a,dataset_t *data_in, dataset_t *data_out){
  //Decode data_in using autoencoder
  assert(data_out->npoints==a->layers[0].nin);
  assert(data_in->npoints==a->layers[a->nlayers/2 -1].nout);
  assert(data_in->nrecs==data_out->nrecs);

  int i,j,k;
  double *layer_in=malloc(a->layers[a->nlayers/2].nin*data_in->nrecs*sizeof(double));
  double *layer_out;
  double *tmp;
  memcpy(layer_in,data_in->data,a->layers[a->nlayers/2].nin*data_in->nrecs*sizeof(double));
  for(i=a->nlayers/2;i<a->nlayers;i++){
    layer_out=malloc(a->layers[i].nout*data_in->nrecs*sizeof(double));
#pragma omp parallel for if((a->layers[i].nout*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(data_in,layer_out,i,a)
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=*(a->layers[i].b+j);
      }
    }
#ifdef ACML
    dgemm('N','N',data_in->nrecs,a->layers[i].nout,a->layers[i].nin,1.0,layer_in,data_in->nrecs,a->layers[i].w,a->layers[i].nin,1.0,layer_out,data_in->nrecs);
#else
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i].nout,data_in->nrecs,a->layers[i].nin,1.0,a->layers[i].w,a->layers[i].nin,layer_in,data_in->nrecs,1.0,layer_out,data_in->nrecs);
#endif
#pragma omp parallel for if((a->layers[i].nout*data_in->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(layer_out,data_in,a,i)
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=a->layers[i].loglo+(a->layers[i].logup-a->layers[i].loglo)*ACTIVATION(*(layer_out+(j*data_in->nrecs)+k)* *(a->layers[i].a+j));
      }
    }
    free(layer_in);
    layer_in=layer_out;
    layer_out=NULL;
  }
  memcpy(data_out->data,layer_in,data_in->nrecs*a->layers[a->nlayers-1].nout*sizeof(double)); //Layer_in due to swap earlier
  memcpy(data_out->weight,data_in->weight,data_in->nrecs*sizeof(double));
  memcpy(data_out->scale,data_in->scale,data_in->nrecs*sizeof(double));
  free(layer_in);
}

void autoencoder_encdec(autoenc_t *a,node_val_t *nodevals,int nrecs){
  //Propagate information through autoencoder network, storing both node values and their derivatives in structure 'nodevals'
  //Assume that nodevals[0] is already populated...
  int i,j,k;
  for(i=1;i<a->nlayers+1;i++){
#pragma omp parallel for if(((nodevals+i)->n*nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(nodevals,i,a,nrecs)
    for(j=0;j<(nodevals+i)->n;j++){
      for(k=0;k<nrecs;k++){
	*(((nodevals+i)->values)+(j*nrecs)+k)=*((a->layers[i-1].b)+j);
      }
    }
#ifdef ACML
    dgemm('N','N',nrecs,(nodevals+i)->n,(nodevals+i-1)->n,1.0,(nodevals+i-1)->values,nrecs,a->layers[i-1].w,(nodevals+i-1)->n,1.0,(nodevals+i)->values,nrecs);
#else
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,(nodevals+i)->n,nrecs,(nodevals+i-1)->n,1.0,a->layers[i-1].w,(nodevals+i-1)->n,(nodevals+i-1)->values,nrecs,1.0,(nodevals+i)->values,nrecs);
#endif
#pragma omp parallel for if(((nodevals+i)->n*nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(nodevals,nrecs,a,i)
    for(j=0;j<(nodevals+i)->n;j++){
      for(k=0;k<nrecs;k++){
	*((nodevals+i)->values+(j*nrecs)+k)=a->layers[i-1].loglo+(a->layers[i-1].logup-a->layers[i-1].loglo)*(ACTIVATION(*((nodevals+i)->values+(j*nrecs)+k)* *(a->layers[i-1].a+j)));
	*((nodevals+i)->derivs+(j*nrecs)+k)=(*((nodevals+i)->values+(j*nrecs)+k)-a->layers[i-1].loglo)*(a->layers[i-1].logup-*((nodevals+i)->values+(j*nrecs)+k))/(a->layers[i-1].logup-a->layers[i-1].loglo);
      }
    } 
  }
}

void autoencoder_batchtrain(autoenc_t *a,dataset_t *data,state_settings_t *settings,dataset_t *monitor,FILE *logfp,dataset_t *enc_force){
  //Main training routine. Read the paper...
#ifdef TANH_ACTIVATION
  printf("ABORTING: Tanh activation function is not available for backprop training yet.\n");
  exit(-1);
#endif
  //Write log file if required
  if(logfp!=NULL){
    if(monitor==NULL){
      fprintf(logfp,"# nrecs=%i niter=%i stdev=%f lrate=%f nrecs_mon=0\n",data->nrecs,settings->niter_auto,settings->noise_auto,settings->lrate_auto);
    } else {
      fprintf(logfp,"# nrecs=%i niter=%i stdev=%f lrate=%f nrecs_mon=%i\n",data->nrecs,settings->niter_auto,settings->noise_auto,settings->lrate_auto,monitor->nrecs);
    }
  }
  
  //Set up structure to hold values of each neuron within network
  node_val_t nodes[a->nlayers+1];
  make_nodevals(a,nodes,data->nrecs);
  int i,j,k;
  
  //Set up structure for back-propagation of errors
  double *delta[a->nlayers+1];
  for(i=1;i<a->nlayers+1;i++){
    delta[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
  }
  
  //Set up structure for back-propagation of cost term if required
  double *cost[a->nlayers/2+1];
  if(settings->costwt>0.){
    for(i=1;i<a->nlayers/2+1;i++){
      cost[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
    }
  }
  
  //Set up structure for back-propagation of enc-force term if required
  double *gamma[a->nlayers/2+1];
  if(enc_force!=NULL){
    for(i=1;i<a->nlayers/2+1;i++){
      gamma[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double)); //top-level gamma is oversize
    }
  }
  
  //Set up to handle monitoring dataset if required
  dataset_t mon_enc,mon_dec;
  double olderr,err;
  if(monitor!=NULL){
    dataset_alloc(&mon_enc,a->layers[a->nlayers/2].nin,monitor->nrecs);
    dataset_alloc(&mon_dec,a->layers[0].nin,monitor->nrecs);
  }
  double wtnorm=0;
  for(i=0;i<nodes[0].n;i++){wtnorm+=*(data->weight+i);}
  double mon_base;
  double *tmp=calloc(data->nrecs*data->npoints,sizeof(double));
  double err_base=error_array(data->data,tmp,data->npoints,data->nrecs,data->weight);
  free(tmp);
  printf("Base error for training dataset: %f\n",err_base);
  if(monitor!=NULL){
    tmp=calloc(monitor->nrecs*monitor->npoints,sizeof(double));
    mon_base=error_array(monitor->data,tmp,monitor->npoints,monitor->nrecs,monitor->weight);
    free(tmp);
    printf("Base error for monitor dataset: %f\n",mon_base);
  }

  //Initialise adaptive learning rate
  settings->N_reduced=0;
  settings->i_red_sum=0;
  settings->iter_reset=0;
  int n_increasing=0;
  double eta;

#ifdef TIMING
  clock_t tstart = clock(), tdiff;
#endif
  //Main training loop
  while(settings->iter<settings->niter_auto && cease_training==0){
    //Evaluate encoding/reconstruction of dataset for current weights
    memcpy(nodes[0].values,data->data,nodes[0].n*data->nrecs*sizeof(double));
    if(settings->noise_auto>=0.){for(i=0;i<data->nrecs*nodes[0].n;i++){*(nodes[0].values+i)+=settings->noise_auto*random_normal();}}//Add noise if requested
    autoencoder_encdec(a,nodes,data->nrecs);
    //Compute reconstruction error
    if(settings->iter>0){olderr=err;}
    err=error_array(nodes[0].values,nodes[a->nlayers].values,nodes[0].n,data->nrecs,data->weight);
    if(enc_force!=NULL){
      err+=settings->enc_force_weight*error_array(enc_force->data,nodes[a->nlayers/2].values,enc_force->npoints,data->nrecs,data->weight);
    }
    //Compute average encoding length
    double lenc=len_enc_array(nodes[a->nlayers/2].values,nodes[a->nlayers/2].n,data->nrecs,data->weight);

    //Learning rate is adaptive to handle instabilities etc. This is a bit of a mess, probably needs a rewrite/rethink.
    //See paper for description of what's (supposed to be) happening
    if(settings->iter==0){
      settings->i_adapt_rate=0;
    } else {
      if(err<=olderr || (settings->noise_auto>=0. && err<=(olderr+2.*settings->noise_auto)) && n_increasing<5){
        if(err>olderr){
          n_increasing++;
        } else {
          n_increasing=0;
        }
        if(settings->i_adapt_rate<settings->N_adapt_rate){
          settings->i_adapt_rate++;
        }
      } else {
        n_increasing=0;
        settings->N_reduced++;
        settings->i_red_sum+=settings->i_adapt_rate;
        if((settings->N_reduced>(int)(0.015*(settings->iter-settings->iter_reset)))&&((settings->iter-settings->iter_reset)>100)){
          settings->lrate_auto=settings->eta0+0.95*((double)settings->i_red_sum/(double)settings->N_reduced)*(settings->lrate_auto-settings->eta0)/settings->N_adapt_rate;
          printf("*** Permanently reducing learning rate: %f ***\n",settings->lrate_auto);
          settings->iter_reset=settings->iter;
          settings->N_reduced=0;
          settings->i_red_sum=0;
          settings->i_adapt_rate=0;
        } else {
          settings->i_adapt_rate/=2;
        }
      }
    }
    //Final learning rate for this iteration:
    eta = settings->eta0+settings->i_adapt_rate*(settings->lrate_auto-settings->eta0)/settings->N_adapt_rate;

    double mon_err;
    //Compute reconstruction error for monitor dataset if reqd, write progress to file
    if(monitor!=NULL){
      autoencoder_encode(a,monitor,&mon_enc);
      autoencoder_decode(a,&mon_enc,&mon_dec);
      mon_err=error_dataset(monitor,&mon_dec);
      printf("Autoencoder training iteration %4i :: E=%.5f (%.3f%%) Emon=%.5f (%.3f%%) L=%.3f eta=%.6f\n",settings->iter+1,err,err*100./err_base,mon_err,mon_err*100./mon_base,lenc,eta);
      if(logfp!=NULL){fprintf(logfp,"%i %f %f %f\n",settings->iter+1,eta, err,mon_err);}
    } else {
      printf("Autoencoder training iteration %4i :: E=%.5f (%.3f%%) L=%.3f\n",settings->iter+1,err,err*100./err_base,lenc);
      if(logfp!=NULL){fprintf(logfp,"%i %f %f\n",eta,settings->iter+1,err);}
    }
    
    // APV adopted PK's backpropagate_errors() function. Subsequent commented-out code can be deleted after successful test
    backpropagate_errors(a,data,settings,nodes,enc_force,delta,cost,gamma);
      
/*     //Back-propagate errors */
/* #pragma omp parallel for if((data->nrecs*nodes[0].n)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i) shared(delta,nodes,a) */
/*     for(i=0;i<(data->nrecs*nodes[0].n);i++){ */
/*       *(delta[a->nlayers]+i)=*(nodes[a->nlayers].values+i)- *(nodes[0].values+i); */
/*     } */
/*     double *tmp; */
/*     for(i=a->nlayers-1;i>0;i--){ */
/*       tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double)); */
/* #pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(tmp,data,delta,i,nodes) */
/*       for(j=0;j<nodes[i+1].n;j++){ */
/* 	for(k=0;k<data->nrecs;k++){ */
/* 	  *(tmp+(j*data->nrecs)+k)=*(delta[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j); */
/* 	} */
/*       } */
/* #ifdef ACML */
/*       dgemm('N','T',data->nrecs,nodes[i].n,nodes[i+1].n,1.0,tmp,data->nrecs,a->layers[i].w,nodes[i].n,0.0,delta[i],data->nrecs); */
/* #else */
/*       cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,delta[i],data->nrecs); */
/* #endif */
/*       free(tmp); */
/*     } */
/*     if(enc_force!=NULL){ */
/*       //Also back-propagate errors from the encoding stage */
/* #pragma omp parallel for if((data->nrecs*nodes[a->nlayers/2].n)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i) shared(gamma,a,nodes,enc_force) */
/*       for(i=0;i<(data->nrecs*nodes[a->nlayers/2].n);i++){ */
/* 	if(i<(data->nrecs*enc_force->npoints)){ */
/* 	    *(gamma[a->nlayers/2]+i)=*(nodes[a->nlayers/2].values+i)- *(enc_force->data+i); */
/* 	  } else { */
/* 	    *(gamma[a->nlayers/2]+i)=0.; */
/* 	  } */
/*       } */
      
/*       double *tmp; */
/*       for(i=a->nlayers/2-1;i>0;i--){ */
/* 	tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double)); */
/* #pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(tmp,data,gamma,nodes,a) */
/* 	for(j=0;j<nodes[i+1].n;j++){ */
/* 	  for(k=0;k<data->nrecs;k++){ */
/* 	    *(tmp+(j*data->nrecs)+k)=*(gamma[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j); */
/* 	  } */
/* 	} */
/* #ifdef ACML */
/* 	dgemm('N','T',data->nrecs,nodes[i].n,nodes[i+1].n,1.0,tmp,data->nrecs,a->layers[i].w,nodes[i].n,0.0,gamma[i],data->nrecs); */
/* #else */
/* 	cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,gamma[i],data->nrecs); */
/* #endif */
/* 	free(tmp); */
/*       } */
/*     } */
/*     if(settings->costwt>0.){ */
/* 	//Back-propagate cost term */
/* #pragma omp parallel for if((nodes[a->nlayers/2].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i) shared(cost,a,nodes) */
/*       for(i=0;i<(data->nrecs*nodes[a->nlayers/2].n);i++){ */
/* 	if(enc_force!=NULL && i<(data->nrecs*enc_force->npoints)){  */
/* 	  *(cost[a->nlayers/2]+i)=0.; */
/* 	} else { */
/* 	  *(cost[a->nlayers/2]+i)=copysign(1.0,*(nodes[a->nlayers/2].values+i)); */
/* 	} */
/*       } */
/*       for(i=a->nlayers/2-1;i>0;i--){ */
/* 	tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double)); */
/* #pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(nodes,data,tmp,cost,i,a) */
/* 	for(j=0;j<nodes[i+1].n;j++){ */
/* 	  for(k=0;k<data->nrecs;k++){ */
/* 	    *(tmp+(j*data->nrecs)+k)=*(cost[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j); */
/* 	  } */
/* 	} */
/* #ifdef ACML */
/* 	dgemm('N','T',data->nrecs,nodes[i].n,nodes[i+1].n,1.0,tmp,data->nrecs,a->layers[i].w,nodes[i].n,0.0,cost[i],data->nrecs); */
/* #else */
/* 	cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,cost[i],data->nrecs); */
/* #endif */
/* 	free(tmp); */
/*       } */
/*     } */
/*     if(enc_force!=NULL){ */
/*       //delta -> delta + (phi . gamma) */
/*       for(i=1;i<a->nlayers/2;i++){ */
/* #pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(data,delta,settings,gamma,i) */
/* 	for(j=0;j<nodes[i+1].n;j++){ */
/* 	  for(k=0;k<data->nrecs;k++){ */
/* 	    *(delta[i]+(j*data->nrecs)+k)+=settings->enc_force_weight* *(gamma[i]+(j*data->nrecs)+k); */
/* 	  } */
/* 	} */
/*       } */
/*     } */

    //Compute weight updates from backpropagated errors
    double *dw,*da,*db,*work;
    for(i=1;i<a->nlayers+1;i++){
      dw=calloc(a->layers[i-1].nin*a->layers[i-1].nout,sizeof(double));
      da=calloc(a->layers[i-1].nout,sizeof(double));
      db=calloc(a->layers[i-1].nout,sizeof(double));
      work=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
      memcpy(work,delta[i],a->layers[i-1].nout*data->nrecs*sizeof(double));
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>=MIN_ARRSIZE_PARALLEL) private(j,k) shared(a,data,work,i,nodes,wtnorm,db)
      {
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(work+(j*data->nrecs)+k)*=*(nodes[i].derivs+(j*data->nrecs)+k)* *(a->layers[i-1].a+j)* *(data->weight+k)/wtnorm;
	  }
	}
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  *(db+j)=*(work+(j*data->nrecs));
	  for(k=1;k<data->nrecs;k++){
	    *(db+j)+=*(work+(j*data->nrecs)+k);
	  }
	}
      }
#ifdef ACML
      dgemm('T','N',a->layers[i-1].nin,a->layers[i-1].nout,data->nrecs,1.0,nodes[i-1].values,data->nrecs,work,data->nrecs,0.0,dw,a->layers[i-1].nin);
#else
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,a->layers[i-1].nout,a->layers[i-1].nin,data->nrecs,1.0,work,data->nrecs,nodes[i-1].values,data->nrecs,0.0,dw,a->layers[i-1].nin);
#endif 
#pragma omp parallel for if((a->layers[i-1].nout*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(a,data,work,i)
      for(j=0;j<a->layers[i-1].nout;j++){
	for(k=0;k<data->nrecs;k++){
	  *(work+(j*data->nrecs)+k)=*(a->layers[i-1].b+j);
	}
      }
#ifdef ACML
      dgemm('N','N',data->nrecs,a->layers[i-1].nout,a->layers[i-1].nin,1.0,nodes[i-1].values,data->nrecs,a->layers[i-1].w,a->layers[i-1].nin,1.0,work,data->nrecs);
#else
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i-1].nout,data->nrecs,a->layers[i-1].nin,1.0,a->layers[i-1].w,a->layers[i-1].nin,nodes[i-1].values,data->nrecs,1.0,work,data->nrecs);
#endif
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>MIN_ARRSIZE_PARALLEL) private(j,k) shared(work,delta,nodes,a,data,da,wtnorm)
      {
#pragma omp for schedule(static)
	for(j=0;j<(a->layers[i-1].nout*data->nrecs);j++){
	  *(work+j)*=*(delta[i]+j)* *(nodes[i].derivs+j);
	}
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  *(da+j)=*(work+(j*data->nrecs))* *(data->weight)/wtnorm;
	  for(k=1;k<data->nrecs;k++){
	    *(da+j)+=*(work+(j*data->nrecs)+k)* *(data->weight+k)/wtnorm;
	  }
	}
      }
      if(settings->costwt>0. && i<=a->nlayers/2){
	memcpy(work,cost[i],a->layers[i-1].nout*data->nrecs*sizeof(double));
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>MIN_ARRSIZE_PARALLEL) private(j,k) shared(a,data,work,nodes,wtnorm,db,settings)
	{
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(work+(j*data->nrecs)+k)*=*(nodes[i].derivs+(j*data->nrecs)+k)* *(a->layers[i-1].a+j)* *(data->weight+k)/wtnorm;
	    }
	  }
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(db+j)+=*(work+(j*data->nrecs)+k)*settings->costwt;
	    }
	  }
	}
	//Update dw - note funny scaling to accommodate costwt within dgemm call
#ifdef ACML
	dgemm('T','N',a->layers[i-1].nin,a->layers[i-1].nout,data->nrecs,1.0,nodes[i-1].values,data->nrecs,work,data->nrecs,1.0/settings->costwt,dw,a->layers[i-1].nin);
#else
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,a->layers[i-1].nout,a->layers[i-1].nin,data->nrecs,1.0,work,data->nrecs,nodes[i-1].values,data->nrecs,1.0/settings->costwt,dw,a->layers[i-1].nin);
#endif

#pragma omp parallel if ((a->layers[i-1].nin*a->layers[i-1].nout)>=MIN_ARRSIZE_PARALLEL) private(j,k) shared(a,dw,settings,work,data,i)
	{
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
	    *(dw+j)=settings->costwt* *(dw+j);
	  }
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(work+(j*data->nrecs)+k)=*(a->layers[i-1].b+j);
	    }
	  }
	}
#ifdef ACML
	dgemm('N','N',data->nrecs,a->layers[i-1].nout,a->layers[i-1].nin,1.0,nodes[i-1].values,data->nrecs,a->layers[i-1].w,a->layers[i-1].nin,1.0,work,data->nrecs);
#else
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i-1].nout,data->nrecs,a->layers[i-1].nin,1.0,a->layers[i-1].w,a->layers[i-1].nin,nodes[i-1].values,data->nrecs,1.0,work,data->nrecs);
#endif
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>=MIN_ARRSIZE_PARALLEL) private(j,k) shared(i,a,work,cost,nodes,da,data,settings,wtnorm)
	{
#pragma omp for schedule(static)
	  for(j=0;j<(a->layers[i-1].nout*data->nrecs);j++){
	    *(work+j)*=*(cost[i]+j)* *(nodes[i].derivs+j);
	  }
#pragma omp for schedule(static)	 
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(da+j)+=*(work+(j*data->nrecs)+k)*settings->costwt* *(data->weight+k)/wtnorm;
	    }
	  }
	}
      }
      free(work);
#pragma omp parallel if((a->layers[i-1].nin*a->layers[i-1].nout)>=MIN_ARRSIZE_PARALLEL) private(j) shared(a,eta,dw,da,db,i)
      {
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
	  *(a->layers[i-1].w+j)-=eta* *(dw+j);
	}
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  *(a->layers[i-1].a+j)-=eta* *(da+j);
	}
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  *(a->layers[i-1].b+j)-=eta* *(db+j);
	}
      }
      free(dw);
      free(da);
      free(db);
    }
    settings->iter++;
    //And let's go around for another iteration...
  }
#ifdef TIMING
  tdiff = clock() - tstart;
  int num_procs=1;
  if (settings->num_procs>0) {
    num_procs=settings->num_procs;
  }
  int msec = tdiff * 1000 / CLOCKS_PER_SEC / settings->iter / num_procs;
  printf("%d seconds %d milliseconds per iteration\n", msec/1000, msec%1000);
#endif
  //Clean up...
  if(monitor!=NULL){
    dataset_free(&mon_enc);
    dataset_free(&mon_dec);
  }
  for(i=1;i<a->nlayers+1;i++){
    free(delta[i]);
  }
  if(settings->costwt>0.){
    for(i=1;i<a->nlayers/2+1;i++){
      free(cost[i]);
    }
  }
  if(enc_force!=NULL){
    for(i=1;i<a->nlayers/2+1;i++){
      free(gamma[i]);
    }
  }
  nodevals_free(a,nodes);
}

double avg_step_size(double **Dw, double **Da, double **Db, autoenc_t *a)
{
  int i,j;
  double Davg=0.0;
  for(i=1;i<a->nlayers+1;i++){
    int nrecs=a->layers[i-1].nin*a->layers[i-1].nout;
    double tmp=0.0;
    for(j=0;j<nrecs;j++){
      tmp+=*(Dw[i]+j);
    }
    Davg+=tmp/nrecs;
    nrecs=a->layers[i-1].nout;
    tmp=0.0;
    for(j=0;j<nrecs;j++){
      tmp+=*(Da[i]+j);
      tmp+=*(Db[i]+j);
    }
    Davg+=tmp/nrecs;
  }
  return Davg/(3.0*a->nlayers);
}

void backpropagate_errors(autoenc_t *a,dataset_t *data,state_settings_t *settings,node_val_t *nodes,dataset_t *enc_force,double **delta,double **cost, double **gamma){
  int i,j,k;
 //Back-propagate errors
#pragma omp parallel for if((data->nrecs*nodes[0].n)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i) shared(delta,nodes,a)
  for(i=0;i<(data->nrecs*nodes[0].n);i++){
    *(delta[a->nlayers]+i)=*(nodes[a->nlayers].values+i)- *(nodes[0].values+i);
  }
  double *tmp;
  for(i=a->nlayers-1;i>0;i--){
    tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double));
#pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(tmp,data,delta,i,nodes)
    for(j=0;j<nodes[i+1].n;j++){
      for(k=0;k<data->nrecs;k++){
        *(tmp+(j*data->nrecs)+k)=*(delta[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j);
      }
    }
#ifdef ACML
    dgemm('N','T',data->nrecs,nodes[i].n,nodes[i+1].n,1.0,tmp,data->nrecs,a->layers[i].w,nodes[i].n,0.0,delta[i],data->nrecs);
#else
    cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,delta[i],data->nrecs);
#endif
    free(tmp);
  }
  if(enc_force!=NULL){
    //Also back-propagate errors from the encoding stage
#pragma omp parallel for if((data->nrecs*nodes[a->nlayers/2].n)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i) shared(gamma,a,nodes,enc_force)
    for(i=0;i<(data->nrecs*nodes[a->nlayers/2].n);i++){
      if(i<(data->nrecs*enc_force->npoints)){
          *(gamma[a->nlayers/2]+i)=*(nodes[a->nlayers/2].values+i)- *(enc_force->data+i);
        } else {
          *(gamma[a->nlayers/2]+i)=0.;
        }
    }

    double *tmp;
    for(i=a->nlayers/2-1;i>0;i--){
      tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double));
#pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(tmp,data,gamma,nodes,a)
      for(j=0;j<nodes[i+1].n;j++){
        for(k=0;k<data->nrecs;k++){
          *(tmp+(j*data->nrecs)+k)=*(gamma[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j);
        }
      }
#ifdef ACML
      dgemm('N','T',data->nrecs,nodes[i].n,nodes[i+1].n,1.0,tmp,data->nrecs,a->layers[i].w,nodes[i].n,0.0,gamma[i],data->nrecs);
#else
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,gamma[i],data->nrecs);
#endif
      free(tmp);
    }
  }
  if(settings->costwt>0.){
      //Back-propagate cost term
#pragma omp parallel for if((nodes[a->nlayers/2].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(i) shared(cost,a,nodes)
    for(i=0;i<(data->nrecs*nodes[a->nlayers/2].n);i++){
      if(enc_force!=NULL && i<(data->nrecs*enc_force->npoints)){
        *(cost[a->nlayers/2]+i)=0.;
      } else {
        *(cost[a->nlayers/2]+i)=copysign(1.0,*(nodes[a->nlayers/2].values+i));
      }
    }
    for(i=a->nlayers/2-1;i>0;i--){
      tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double));
#pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(nodes,data,tmp,cost,i,a)
      for(j=0;j<nodes[i+1].n;j++){
        for(k=0;k<data->nrecs;k++){
          *(tmp+(j*data->nrecs)+k)=*(cost[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j);
        }
      }
#ifdef ACML
      dgemm('N','T',data->nrecs,nodes[i].n,nodes[i+1].n,1.0,tmp,data->nrecs,a->layers[i].w,nodes[i].n,0.0,cost[i],data->nrecs);
#else
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,cost[i],data->nrecs);
#endif
      free(tmp);
    }
  }
  if(enc_force!=NULL){
    //delta -> delta + (phi . gamma)
    for(i=1;i<a->nlayers/2;i++){
#pragma omp parallel for if((nodes[i+1].n*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(data,delta,settings,gamma,i)
      for(j=0;j<nodes[i+1].n;j++){
        for(k=0;k<data->nrecs;k++){
          *(delta[i]+(j*data->nrecs)+k)+=settings->enc_force_weight* *(gamma[i]+(j*data->nrecs)+k);
        }
      }
    }
  }
}

void autoencoder_batchtrain_rprop(autoenc_t *a,dataset_t *data,state_settings_t *settings,
    dataset_t *monitor,FILE *logfp,dataset_t *enc_force){
  /*
   * Uses the Rprop algorithm (c.f. Igel&Huesken,
   * 2003, Neurocomputing 50) for autoencoder training stages.
   *
   * The algorithm uses a per-weight step-size, which is decreased or increased,
   * depending on the sign of the error function derivative. If the derivative
   * changes sign between two consecutive iterations, the step-size is
   * decreased, otherwise increased.
   *
   * Default behaviour is to use the improved rProp+ algorithm (Igel&Huesken,
   * 2003, Neurocomputing 50). The algorithm takes back weight updates, if they
   * cross a local minimum and lead to an increase in error (weight-backtracking).
   * If --no-wb is set, we use the more simple rProp- algorithm, without weight-
   * backtracking.
   *
   * Note that Igel&Huesken (2003) find that for very small condition numbers
   * (a<=3), that is the error surface locally has a very pronounced valley-like
   * shape, simple rProp- might lead to a better convergence performance than
   * the improved algorithm.
   *
   * Todo: optimization: (1) we only need to know the sign of the derivatives,
   * could we save a few multiplications that do not affect the sign? (2) Do the
   * many if statements in the rProp code section negatively influence performance?
   * Todo: Massive cleanup! There is a lot of duplicate code now which overlaps
   * with autoencoder_batchtrain.
   * Todo: parallel stuff does not work properly (?)
   * */
#ifdef TANH_ACTIVATION
  printf("ABORTING: Tanh activation function is not available for backprop training yet.\n");
  exit(-1);
#endif

  int i,j,k;

  // init rprop specific stuff
  double etap = settings->etap;
  double etam = settings->etam;
  double D0 = settings->delta0;
  double Dmin = settings->delta_min;
  double Dmax = settings->delta_max;

  double *Dw[a->nlayers+1], *Da[a->nlayers+1], *Db[a->nlayers+1];
  double *dwold[a->nlayers+1], *daold[a->nlayers+1], *dbold[a->nlayers+1];

  for(i=1;i<a->nlayers+1;i++){
    Dw[i]=calloc(a->layers[i-1].nin*a->layers[i-1].nout,sizeof(double));
    Da[i]=calloc(a->layers[i-1].nout,sizeof(double));
    Db[i]=calloc(a->layers[i-1].nout,sizeof(double));
    // init step-sizes with D0
    for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
      *(Dw[i]+j)=D0;
    }
    for(j=0;j<a->layers[i-1].nout;j++){
      *(Da[i]+j)=D0;
      *(Db[i]+j)=D0;
    }
    dwold[i]=calloc(a->layers[i-1].nin*a->layers[i-1].nout,sizeof(double));
    daold[i]=calloc(a->layers[i-1].nout,sizeof(double));
    dbold[i]=calloc(a->layers[i-1].nout,sizeof(double));
  }

  //printf("avg step size %f\n", avg_step_size(Dw, Da, Db, a));

  //Main training routine. Read the paper...
  //Write log file if required
  if(logfp!=NULL){
    if(monitor==NULL){
      fprintf(logfp,"# nrecs=%i niter=%i stdev=%f lrate=%f nrecs_mon=0\n",data->nrecs,settings->niter_auto,settings->noise_auto,settings->lrate_auto);
    } else {
      fprintf(logfp,"# nrecs=%i niter=%i stdev=%f lrate=%f nrecs_mon=%i\n",data->nrecs,settings->niter_auto,settings->noise_auto,settings->lrate_auto,monitor->nrecs);
    }
  }

  //Set up structure to hold values of each neuron within network
  node_val_t nodes[a->nlayers+1];
  make_nodevals(a,nodes,data->nrecs);

  //Set up structure for back-propagation of errors
  double *delta[a->nlayers+1];
  for(i=1;i<a->nlayers+1;i++){
    delta[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
  }

  //Set up structure for back-propagation of cost term if required
  double *cost[a->nlayers/2+1];
  if(settings->costwt>0.){
    for(i=1;i<a->nlayers/2+1;i++){
      cost[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
    }
  }

  //Set up structure for back-propagation of enc-force term if required
  double *gamma[a->nlayers/2+1];
  if(enc_force!=NULL){
    for(i=1;i<a->nlayers/2+1;i++){
      gamma[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double)); //top-level gamma is oversize
    }
  }

  //Set up to handle monitoring dataset if required
  dataset_t mon_enc,mon_dec;
  double olderr,err;
  if(monitor!=NULL){
    dataset_alloc(&mon_enc,a->layers[a->nlayers/2].nin,monitor->nrecs);
    dataset_alloc(&mon_dec,a->layers[0].nin,monitor->nrecs);
  }
  double wtnorm=0;
  for(i=0;i<nodes[0].n;i++){wtnorm+=*(data->weight+i);}
  double mon_base;
  double *tmp=calloc(data->nrecs*data->npoints,sizeof(double));
  double err_base=error_array(data->data,tmp,data->npoints,data->nrecs,data->weight);
  free(tmp);
  printf("Base error for training dataset: %f\n",err_base);
  if(monitor!=NULL){
    tmp=calloc(monitor->nrecs*monitor->npoints,sizeof(double));
    mon_base=error_array(monitor->data,tmp,monitor->npoints,monitor->nrecs,monitor->weight);
    free(tmp);
    printf("Base error for monitor dataset: %f\n",mon_base);
  }
#ifdef TIMING
  clock_t tstart = clock(), tdiff;
#endif
  //Main training loop
  while(settings->iter<settings->niter_auto && cease_training==0){
    //Evaluate encoding/reconstruction of dataset for current weights
    memcpy(nodes[0].values,data->data,nodes[0].n*data->nrecs*sizeof(double));
    if(settings->noise_auto>=0.){for(i=0;i<data->nrecs*nodes[0].n;i++){*(nodes[0].values+i)+=settings->noise_auto*random_normal();}}//Add noise if requested
    autoencoder_encdec(a,nodes,data->nrecs);
    //Compute reconstruction error
    if(settings->iter>0){olderr=err;}
    err=error_array(nodes[0].values,nodes[a->nlayers].values,nodes[0].n,data->nrecs,data->weight);
    if(enc_force!=NULL){
      err+=settings->enc_force_weight*error_array(enc_force->data,nodes[a->nlayers/2].values,enc_force->npoints,data->nrecs,data->weight);
    }
    //Compute average encoding length
    double lenc=len_enc_array(nodes[a->nlayers/2].values,nodes[a->nlayers/2].n,data->nrecs,data->weight);

    //Average step size for this iteration:
    double eta = avg_step_size(Dw, Da, Db, a);

    double mon_err;
    //Compute reconstruction error for monitor dataset if reqd, write progress to file
    if(monitor!=NULL){
      autoencoder_encode(a,monitor,&mon_enc);
      autoencoder_decode(a,&mon_enc,&mon_dec);
      mon_err=error_dataset(monitor,&mon_dec);
      printf("Autoencoder training iteration %4i :: E=%.5f (%.3f%%) Emon=%.5f (%.3f%%) L=%.3f avg step size is %.6f\n",settings->iter+1,err,err*100./err_base,mon_err,mon_err*100./mon_base,lenc,eta);
      if(logfp!=NULL){fprintf(logfp,"%i %f %f %f\n",settings->iter+1,eta, err,mon_err);}
    } else {
        printf("Autoencoder training iteration %4i :: E=%.5f (%.3f%%) L=%.3f avg step size is %.6f\n",settings->iter+1,err,err*100./err_base,lenc,eta);
      if(logfp!=NULL){fprintf(logfp,"%i %f %f\n",eta,settings->iter+1,err);}
    }

    backpropagate_errors(a,data,settings,nodes,enc_force,delta,cost,gamma);

    //Compute weight updates from backpropagated errors
    double *dw,*da,*db,*work;
    for(i=1;i<a->nlayers+1;i++){
      dw=calloc(a->layers[i-1].nin*a->layers[i-1].nout,sizeof(double));
      da=calloc(a->layers[i-1].nout,sizeof(double));
      db=calloc(a->layers[i-1].nout,sizeof(double));
      work=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
      memcpy(work,delta[i],a->layers[i-1].nout*data->nrecs*sizeof(double));
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>=MIN_ARRSIZE_PARALLEL) private(j,k) shared(a,data,work,i,nodes,wtnorm,db)
      {
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(work+(j*data->nrecs)+k)*=*(nodes[i].derivs+(j*data->nrecs)+k)* *(a->layers[i-1].a+j)* *(data->weight+k)/wtnorm;
	  }
	}
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  *(db+j)=*(work+(j*data->nrecs));
	  for(k=1;k<data->nrecs;k++){
	    *(db+j)+=*(work+(j*data->nrecs)+k);
	  }
	}
      }
#ifdef ACML
      dgemm('T','N',a->layers[i-1].nin,a->layers[i-1].nout,data->nrecs,1.0,nodes[i-1].values,data->nrecs,work,data->nrecs,0.0,dw,a->layers[i-1].nin);
#else
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,a->layers[i-1].nout,a->layers[i-1].nin,data->nrecs,1.0,work,data->nrecs,nodes[i-1].values,data->nrecs,0.0,dw,a->layers[i-1].nin);
#endif
#pragma omp parallel for if((a->layers[i-1].nout*data->nrecs)>=MIN_ARRSIZE_PARALLEL) schedule(static) private(j,k) shared(a,data,work,i)
      for(j=0;j<a->layers[i-1].nout;j++){
	for(k=0;k<data->nrecs;k++){
	  *(work+(j*data->nrecs)+k)=*(a->layers[i-1].b+j);
	}
      }
#ifdef ACML
      dgemm('N','N',data->nrecs,a->layers[i-1].nout,a->layers[i-1].nin,1.0,nodes[i-1].values,data->nrecs,a->layers[i-1].w,a->layers[i-1].nin,1.0,work,data->nrecs);
#else
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i-1].nout,data->nrecs,a->layers[i-1].nin,1.0,a->layers[i-1].w,a->layers[i-1].nin,nodes[i-1].values,data->nrecs,1.0,work,data->nrecs);
#endif
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>MIN_ARRSIZE_PARALLEL) private(j,k) shared(work,delta,nodes,a,data,da,wtnorm)
      {
#pragma omp for schedule(static)
	for(j=0;j<(a->layers[i-1].nout*data->nrecs);j++){
	  *(work+j)*=*(delta[i]+j)* *(nodes[i].derivs+j);
	}
#pragma omp for schedule(static)
	for(j=0;j<a->layers[i-1].nout;j++){
	  *(da+j)=*(work+(j*data->nrecs))* *(data->weight)/wtnorm;
	  for(k=1;k<data->nrecs;k++){
	    *(da+j)+=*(work+(j*data->nrecs)+k)* *(data->weight+k)/wtnorm;
	  }
	}
      }
      if(settings->costwt>0. && i<=a->nlayers/2){
	memcpy(work,cost[i],a->layers[i-1].nout*data->nrecs*sizeof(double));
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>MIN_ARRSIZE_PARALLEL) private(j,k) shared(a,data,work,nodes,wtnorm,db,settings)
	{
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(work+(j*data->nrecs)+k)*=*(nodes[i].derivs+(j*data->nrecs)+k)* *(a->layers[i-1].a+j)* *(data->weight+k)/wtnorm;
	    }
	  }
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(db+j)+=*(work+(j*data->nrecs)+k)*settings->costwt;
	    }
	  }
	}
	//Update dw - note funny scaling to accommodate costwt within dgemm call
#ifdef ACML
	dgemm('T','N',a->layers[i-1].nin,a->layers[i-1].nout,data->nrecs,1.0,nodes[i-1].values,data->nrecs,work,data->nrecs,1.0/settings->costwt,dw,a->layers[i-1].nin);
#else
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,a->layers[i-1].nout,a->layers[i-1].nin,data->nrecs,1.0,work,data->nrecs,nodes[i-1].values,data->nrecs,1.0/settings->costwt,dw,a->layers[i-1].nin);
#endif

#pragma omp parallel if ((a->layers[i-1].nin*a->layers[i-1].nout)>=MIN_ARRSIZE_PARALLEL) private(j,k) shared(a,dw,settings,work,data,i)
	{
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
	    *(dw+j)=settings->costwt* *(dw+j);
	  }
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(work+(j*data->nrecs)+k)=*(a->layers[i-1].b+j);
	    }
	  }
	}
#ifdef ACML
	dgemm('N','N',data->nrecs,a->layers[i-1].nout,a->layers[i-1].nin,1.0,nodes[i-1].values,data->nrecs,a->layers[i-1].w,a->layers[i-1].nin,1.0,work,data->nrecs);
#else
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i-1].nout,data->nrecs,a->layers[i-1].nin,1.0,a->layers[i-1].w,a->layers[i-1].nin,nodes[i-1].values,data->nrecs,1.0,work,data->nrecs);
#endif
#pragma omp parallel if((a->layers[i-1].nout*data->nrecs)>=MIN_ARRSIZE_PARALLEL) private(j,k) shared(i,a,work,cost,nodes,da,data,settings,wtnorm)
	{
#pragma omp for schedule(static)
	  for(j=0;j<(a->layers[i-1].nout*data->nrecs);j++){
	    *(work+j)*=*(cost[i]+j)* *(nodes[i].derivs+j);
	  }
#pragma omp for schedule(static)
	  for(j=0;j<a->layers[i-1].nout;j++){
	    for(k=0;k<data->nrecs;k++){
	      *(da+j)+=*(work+(j*data->nrecs)+k)*settings->costwt* *(data->weight+k)/wtnorm;
	    }
	  }
	}
      }
      free(work);
      double signcur, prd;
#pragma omp parallel if((a->layers[i-1].nin*a->layers[i-1].nout)>=MIN_ARRSIZE_PARALLEL) private(j,signcur,prd) shared(a,dw,da,db,i,Dw,Da,Db,dwold,daold,dbold,settings)
        {
        if (settings->no_wb==1) {
          ////////////////////////////////////////////////////////
          // use Rprop without weight backtracking (a.k.a. Rprop-)
          ////////////////////////////////////////////////////////
#pragma omp for schedule(static)
          for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
            // update weights
            signcur = copysign(1.0, *(dw+j));
            prd = *(dw+j)* *(dwold[i]+j);

            if (prd > 0) {
              *(Dw[i]+j) = fmin(etap* *(Dw[i]+j), Dmax);
            } else if (prd < 0) {
              *(Dw[i]+j) = fmax(etam* *(Dw[i]+j), Dmin);
            } // do nothing if partial derivative is zero
            *(a->layers[i-1].w+j)-=*(Dw[i]+j)* signcur;
          }

#pragma omp for schedule(static)
          for(j=0;j<a->layers[i-1].nout;j++){
            // update sensitivities
            signcur = copysign(1.0, *(da+j));
            prd = *(da+j)* *(daold[i]+j);
            if (prd > 0) {
              *(Da[i]+j) = fmin(etap* *(Da[i]+j), Dmax);
            } else if (prd < 0) {
              *(Da[i]+j) = fmax(etam* *(Da[i]+j), Dmin);
            }
            *(a->layers[i-1].a+j)-=*(Da[i]+j)* signcur;

            // update biases
            signcur = copysign(1.0, *(db+j));
            prd = *(db+j)* *(dbold[i]+j);
            if (prd > 0) {
              *(Db[i]+j) = fmin(etap* *(Db[i]+j), Dmax);
            } else if (prd < 0) {
              *(Db[i]+j) = fmax(etam* *(Db[i]+j), Dmin);
            } // do nothing if partial derivative is zero
            *(a->layers[i-1].b+j)-=*(Db[i]+j)* signcur;
          }

        } else {
          ///////////////////////////////////////////////////////////////
          // use improved Rprop with weight backtracking (a.k.a. iRprop+)
          ///////////////////////////////////////////////////////////////
#pragma omp for schedule(static)
          for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
            // update weights
            signcur = copysign(1.0, *(dw+j));
            prd = *(dw+j)* *(dwold[i]+j);
            if (prd > 0) {
              *(Dw[i]+j) = fmin(etap* *(Dw[i]+j), Dmax);
            } else if (prd < 0) {
              // take back previous weight update, if it lead to an error increase
              if (err>olderr) {
                *(a->layers[i-1].w+j)+=*(Dw[i]+j);
              }
              // calculate new weight update
              *(Dw[i]+j) = fmax(etam* *(Dw[i]+j), Dmin);
              // set derivative to zero, to avoid step size being overwritten in the next iteration
              *(dw+j)=0.0;
            }
            // apply current weight update
            if (prd >= 0) {
              *(a->layers[i-1].w+j)-=*(Dw[i]+j)* signcur;
            }
          }
#pragma omp for schedule(static)
          for(j=0;j<a->layers[i-1].nout;j++){
            // update sensitivities
            signcur = copysign(1.0, *(da+j));
            prd = *(da+j)* *(daold[i]+j);
            if (prd > 0) {
              *(Da[i]+j) = fmin(etap* *(Da[i]+j), Dmax);
            } else if (prd < 0) {
              *(Da[i]+j) = fmax(etam* *(Da[i]+j), Dmin);
            }
            *(a->layers[i-1].a+j)-=*(Da[i]+j)* signcur;
            if (prd > 0) {
              *(Da[i]+j) = fmin(etap* *(Da[i]+j), Dmax);
            } else if (prd < 0) {
              if (err>olderr) {
                *(a->layers[i-1].a+j)+=*(Da[i]+j);
              }
              *(Da[i]+j) = fmax(etam* *(Da[i]+j), Dmin);
              *(da+j)=0.0;
            }
            if (prd >= 0) {
              *(a->layers[i-1].a+j)-=*(Da[i]+j)* signcur;
            }

            // update biases
            signcur = copysign(1.0, *(db+j));
            prd = *(db+j)* *(dbold[i]+j);
            if (prd > 0) {
              *(Db[i]+j) = fmin(etap* *(Db[i]+j), Dmax);
            } else if (prd < 0) {
              if (err>olderr) {
                *(a->layers[i-1].b+j)+=*(Db[i]+j);
              }
              *(Db[i]+j) = fmax(etam* *(Db[i]+j), Dmin);
              *(db+j)=0.0;
            }
            if (prd >= 0) {
              *(a->layers[i-1].b+j)-=*(Db[i]+j)* signcur;
            }
          }
        }
      }
      memcpy(dwold[i],dw,a->layers[i-1].nin*a->layers[i-1].nout*sizeof(double));
      memcpy(daold[i],da,a->layers[i-1].nout*sizeof(double));
      memcpy(dbold[i],db,a->layers[i-1].nout*sizeof(double));

      free(dw);
      free(da);
      free(db);
    }
    settings->iter++;
    //And let's go around for another iteration...
  }

#ifdef TIMING
  tdiff = clock() - tstart;
  int num_procs=1;
  if (settings->num_procs>0) {
    num_procs=settings->num_procs;
  }
  int msec = tdiff * 1000 / CLOCKS_PER_SEC / settings->iter / num_procs;
  printf("%d seconds %d milliseconds per iteration\n", msec/1000, msec%1000);
#endif
  //Clean up...
  if(monitor!=NULL){
    dataset_free(&mon_enc);
    dataset_free(&mon_dec);
  }
  for(i=1;i<a->nlayers+1;i++){
    free(delta[i]);
  }
  if(settings->costwt>0.){
    for(i=1;i<a->nlayers/2+1;i++){
      free(cost[i]);
    }
  }
  if(enc_force!=NULL){
    for(i=1;i<a->nlayers/2+1;i++){
      free(gamma[i]);
    }
  }
  nodevals_free(a,nodes);
}

/////////////////////////////
// Miscellaneous functions //
/////////////////////////////	
void make_nodevals(autoenc_t *a,node_val_t *nodevals,int nrecs){
  //Allocate memory for nodevals structure
  int i;
  for(i=0;i<a->nlayers;i++){
    (nodevals+i)->n=a->layers[i].nin;
    (nodevals+i)->values=malloc(nrecs*((nodevals+i)->n)*sizeof(double));
    (nodevals+i)->derivs=malloc(nrecs*((nodevals+i)->n)*sizeof(double));
  }
  (nodevals+(a->nlayers))->n=a->layers[a->nlayers-1].nout;
  (nodevals+(a->nlayers))->values=malloc(nrecs*((nodevals+(a->nlayers))->n)*sizeof(double));
  (nodevals+(a->nlayers))->derivs=malloc(nrecs*((nodevals+(a->nlayers))->n)*sizeof(double));
}

void nodevals_free(autoenc_t *a, node_val_t *nodevals){
  int i;
  for(i=0;i<a->nlayers+1;i++){
    free((nodevals+i)->values);
    free((nodevals+i)->derivs);
  }
}

double error_dataset(dataset_t *d1,dataset_t *d2){
  //Compute mean-squared error for entire dataset
  int i,j;
  double e=0.0;
  double wtnorm=0.;

  for(i=0;i<d1->nrecs;i++){wtnorm+=*(d1->weight+i);}
#pragma omp parallel for if((d1->npoints*d1->nrecs)>=MIN_ARRSIZE_PARALLEL) reduction(+:e) private(i,j) shared(d1,d2,wtnorm)
  for(i=0;i<d1->npoints;i++){
    for(j=0;j<d1->nrecs;j++){
      e+=*(d1->weight+j)*pow(*(d1->data+(i*d1->nrecs)+j)-*(d2->data+(i*d1->nrecs)+j),2.)/wtnorm;
    }
  }
  return 0.5*e;
}

double error_array(double *d1,double *d2,int npoints,int nrecs, double *weights){
  int i,j;
  double e=0.0;
  double wtnorm=0.;
  for(i=0;i<nrecs;i++){wtnorm+=*(weights+i);}
#pragma omp parallel for if((npoints*nrecs)>=MIN_ARRSIZE_PARALLEL) reduction(+:e) private(i,j) shared(weights,d1,d2,nrecs,wtnorm)
  for(i=0;i<npoints;i++){
    for(j=0;j<nrecs;j++){
      e+=*(weights+j)*pow(*(d1+(i*nrecs)+j)-*(d2+(i*nrecs)+j),2.)/wtnorm;
    }
  }
  return 0.5*e;
}

double len_enc_array(double *d1,int npoints,int nrecs, double *weights){
  int i,j;
  double l=0.0;
  double wtnorm=0.;
  //printf("%i %i\n",nrecs,npoints);
  for(i=0;i<nrecs;i++){wtnorm+=*(weights+i);}
  //printf("wtnorm:%f\n",wtnorm);
  for(i=0;i<npoints;i++){
    for(j=0;j<nrecs;j++){
      l+=*(weights+j)*pow(*(d1+(i*nrecs)+j),2.)/wtnorm;
      //l+=pow(*(d1+(i*nrecs)+j),2.);
    }
  }
  //printf("%f\n",l);
  return 0.5*l;
}

void abort_handler(int signum){
  //signal(signum,SIG_DFL); //Try to avoid infinite loops...
  printf("Training interupted by SIGINT. Options:\n(r)esume training\n(c)omplete current iteration, save network and exit normally\n(a)bort\n\nPlease press r, c or a...");
  char resp[80]="";
  strncpy(resp," ",80);
  int loop=1;
  while(loop){
    scanf("%s",resp);
    switch(resp[0]){
    case 'a': 
      exit(-1);
      loop=0;
      break;
    case 'r':
      signal(signum,abort_handler);
      loop=0;
      break;
    case 'c':
      cease_training=1;
      loop=0;
      break;
    case '\n':
      break;
    default:
      printf("Unrecognised option. Please press r, c or a...");
      break;
    }
    //loop++;
    //if(loop==15){
    //  printf("Apparently stuck in loop...exiting...\n");
    //  abort();
    //}
  }
}

void write_state(char * filename,state_settings_t settings,char * storefile){
  FILE * fp=fopen(filename,"w");
  setbuf(fp,NULL);
  int sw,lstr;
  //Write out settings
  fwrite(&settings.niter_crbm,sizeof(int),2,fp);
  fwrite(&settings.stdev_init,sizeof(double),7,fp);
  if(settings.datafile==NULL){lstr=0;} else {lstr=strlen(settings.datafile);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr>0){fwrite(settings.datafile,sizeof(char),lstr,fp);}
  if(settings.monitorfile==NULL){lstr=0;} else {lstr=strlen(settings.monitorfile);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr>0){fwrite(settings.monitorfile,sizeof(char),lstr,fp);}
  if(settings.force_file==NULL){lstr=0;} else {lstr=strlen(settings.force_file);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr>0){fwrite(settings.force_file,sizeof(char),lstr,fp);}
  if(settings.outputfile==NULL){lstr=0;} else {lstr=strlen(settings.outputfile);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr>0){fwrite(settings.outputfile,sizeof(char),lstr,fp);}
  if(settings.logbase==NULL){lstr=0;} else {lstr=strlen(settings.logbase);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr>0){fwrite(settings.logbase,sizeof(char),lstr,fp);}
  fwrite(&settings.ncrbms,sizeof(int),1,fp);
  fwrite(settings.sizes,sizeof(int),settings.ncrbms+1,fp);
  fwrite(&settings.stage,sizeof(int),3,fp);
  fwrite(&settings.eta0,sizeof(double),1,fp);
  fwrite(&settings.N_adapt_rate,sizeof(int),5,fp);
  fwrite(&settings.costwt,sizeof(double),1,fp);
  fwrite(&settings.num_procs,sizeof(int),1,fp);
  fwrite(&settings.enc_force_weight,sizeof(double),1,fp);
  //RProp stuff
  fwrite(&settings.etap,sizeof(double),1,fp);
  fwrite(&settings.etam,sizeof(double),1,fp);
  fwrite(&settings.delta0,sizeof(double),1,fp);
  fwrite(&settings.delta_max,sizeof(double),1,fp);
  fwrite(&settings.delta_min,sizeof(double),1,fp);
  //end RProp stuff
  if(storefile==NULL){lstr=0;} else {lstr=strlen(storefile);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr!=0){fwrite(storefile,sizeof(char),lstr,fp);}
  //And deal with rng
#ifdef NATIVE_RAND
  sw=0;
#else
#ifdef ACML
  sw=2;
#else
  sw=1;
#endif
#endif
  fwrite(&sw,sizeof(int),1,fp);
#ifndef NATIVE_RAND
#ifdef ACML
  fwrite(&lstate,sizeof(int),1,fp);
  fwrite(&state,sizeof(int),lstate,fp);
#else
  dump_twister_state(fp);
#endif
#endif
  fclose(fp);
}

state_settings_t load_state(char * filename,char **storefile){
  state_settings_t s;
  FILE * fp=fopen(filename,"r");
  int sw,lstr;
  fread(&s.niter_crbm,sizeof(int),2,fp);
  fread(&s.stdev_init,sizeof(double),7,fp);
  fread(&lstr,sizeof(int),1,fp);
  if(lstr>0){
    s.datafile=malloc(sizeof(char)*lstr);
    fread(s.datafile,sizeof(char),lstr,fp);
  } else {
    s.datafile=NULL;
  }
  fread(&lstr,sizeof(int),1,fp);
  if(lstr>0){
    s.monitorfile=malloc(sizeof(char)*lstr);
    fread(s.monitorfile,sizeof(char),lstr,fp);
  } else {
    s.monitorfile=NULL;
  }
  fread(&lstr,sizeof(int),1,fp);
  if(lstr>0){
    s.force_file=malloc(sizeof(char)*lstr);
    fread(s.force_file,sizeof(char),lstr,fp);
  } else {
    s.force_file=NULL;
  }
  fread(&lstr,sizeof(int),1,fp);
  if(lstr>0){
    s.outputfile=malloc(sizeof(char)*lstr);
    fread(s.outputfile,sizeof(char),lstr,fp);
  } else {
    s.outputfile="out.cnet";
  }
  fread(&lstr,sizeof(int),1,fp);
  if(lstr>0){
    s.logbase=malloc(sizeof(char)*lstr);
    fread(s.logbase,sizeof(char),lstr,fp);
  } else {
    s.logbase=NULL;
  }
  fread(&s.ncrbms,sizeof(int),1,fp);
  s.sizes=malloc(sizeof(int)*(s.ncrbms+1));
  fread(s.sizes,sizeof(int),s.ncrbms+1,fp);
  fread(&s.stage,sizeof(int),3,fp);
  fread(&s.eta0,sizeof(double),1,fp);
  fread(&s.N_adapt_rate,sizeof(int),5,fp);
  fread(&s.costwt,sizeof(double),1,fp);
  fread(&s.num_procs,sizeof(int),1,fp);
  fread(&s.enc_force_weight,sizeof(double),1,fp);

  //RProp stuff
  fread(&s.etap,sizeof(double),1,fp);
  fread(&s.etam,sizeof(double),1,fp);
  fread(&s.delta0,sizeof(double),1,fp);
  fread(&s.delta_max,sizeof(double),1,fp);
  fread(&s.delta_min,sizeof(double),1,fp);
  //end RProp stuff

  fread(&lstr,sizeof(int),1,fp);
  printf("lstr: %i\n",lstr);
  if(lstr>0){
    *storefile=malloc(lstr*sizeof(char));
    fread(*storefile,sizeof(char),lstr,fp);
  }
  fread(&sw,sizeof(int),1,fp);
#ifdef NATIVE_RAND
  if(sw!=0){printf("*** Settings file is for different prng ***\n");}
#else
  if(sw==0){
    printf("*** No random number state provided in settings file ***\n");
  } else if(sw==1){
#ifdef ACML
    printf("*** Settings file is for non-ACML version; ignoring prng information... ***\n");
#else
    load_twister_state(fp);
    seeded_random=1;
#endif
  } else if(sw==2){
#ifdef ACML
    fread(&lstate,sizeof(int),1,fp);
    fread(&state,sizeof(int),lstate,fp);
    seeded_random=1;
#else
    printf("*** Settings file is for ACML version; ignoring prng information... ***\n");
#endif
  } else {
    printf("*** Unrecognised prng code; ignoring... ***\n");
  }
#endif
  fclose(fp);
  return s;
}

void fprint_state(state_settings_t s,FILE * fp){
  if(fp==NULL){fp=stdout;}
  
  fprintf(fp,"niter_crbm:  %4i     niter_auto: %5i\n",s.niter_crbm,s.niter_auto);
  fprintf(fp,"stdev_init:  %.4f\n",s.stdev_init);
  fprintf(fp,"noise_crbm:  %.4f   noise_auto: %.4f\n",s.noise_crbm,s.noise_auto);
  fprintf(fp,"lrate_crbm:  %.4f   lrate_auto: %.4f\n",s.lrate_crbm,s.lrate_auto);
  fprintf(fp,"f0:          %5.3f   f1:         %5.3f\n",s.f0,s.f1);
  
  if(s.datafile==NULL){
    fprintf(fp,"datafile:    --\n");
  } else {
    fprintf(fp,"datafile:    %s\n",s.datafile);
  }
  if(s.monitorfile==NULL){
    fprintf(fp,"monitorfile: --\n");
  } else {
    fprintf(fp,"monitorfile: %s\n",s.monitorfile);
  }
  if(s.outputfile==NULL){
    fprintf(fp,"outputfile:  --\n");
  } else {
    fprintf(fp,"outputfile:  %s\n",s.outputfile);
  }
  if(s.force_file==NULL){
    fprintf(fp,"force-file:  --\n");
  } else {
    fprintf(fp,"force-file:  %s\n",s.force_file);
  }
  if(s.logbase==NULL){
    fprintf(fp,"logbase:     --\n");
  } else {
    fprintf(fp,"logbase:     %s\n",s.logbase);
  }
  fprintf(fp,"ncrbms:      %i\n",s.ncrbms);
  fprintf(fp,"sizes:       ");
  int i;
  for(i=0;i<s.ncrbms+1;i++){
    fprintf(fp,"%i ",*(s.sizes+i));
  }
  fprintf(fp,"\n");
  fprintf(fp,"stage:       %2i       iter:%i\n",s.stage,s.iter);
}
  

autoenc_t * autoencoder_make_and_train(crbm_t *crbms,autoenc_t * a,state_settings_t settings){
  dataset_t data,monitor,encoded1,encoded2,enc_force,*tdata1=&encoded1,*tdata2=&encoded2,*tmp,*p2data=NULL,*p2monitor=NULL,*p2enc_force=NULL;
  char logfile[80];
  FILE *logfp=NULL;
  int i;
  signal(SIGINT,abort_handler); // Handle Ctrl-C
#ifdef _OPENMP
  if(settings.num_procs>0){
    omp_set_num_threads(settings.num_procs);
    printf("Running in parallel on %i processors\n",settings.num_procs);
  } else {
    printf("Running in parallel on %i processors\n",omp_get_num_procs());
  }
#endif
  if(settings.datafile!=NULL){
    data=load_dataset(settings.datafile);
    if(settings.ignore_weights){
      for(i=0;i<data.nrecs;i++){*(data.weight+i)=1.;}
    }
    p2data=&data;
    dataset_copy(&encoded1,&data);
  } else {
    printf("No training dataset provided. Aborting...\n");
    exit(-1);
  }
  if(settings.monitorfile!=NULL){
    monitor=load_dataset(settings.monitorfile);
    if(settings.ignore_weights){
      for(i=0;i<monitor.nrecs;i++){*(monitor.weight+i)=1.;}
    }
    p2monitor=&monitor;
  }
  if(settings.force_file!=NULL){
    enc_force=load_dataset(settings.force_file);
    p2enc_force=&enc_force;
  }
  if(settings.stage<settings.ncrbms){
    //There is some CRBM training to do...
    //Use any CRBMs that are provided to encode dataset
    for(i=0;i<settings.stage;i++){
      dataset_alloc(tdata2,*(settings.sizes+i+1),tdata1->nrecs);
      crbm_encode(crbms+i,tdata1,tdata2,-1);
      dataset_free(tdata1);
      tmp=tdata1;
      tdata1=tdata2;
      tdata2=tmp;
    }
    //So tdata1 now points at dataset to use for training current CRBM...
    for(i=settings.stage;i<settings.ncrbms;i++){
      if(settings.logbase!=NULL){
	if(strlen(settings.logbase)>68){printf("Error: log file variable not large enough\n");exit(-1);}
	sprintf(&logfile[0],"%s.crbm.%02i.log",settings.logbase,i+1);
      }
      if(settings.iter==0){
	//We're beginning this training from scratch
	if(settings.logbase!=NULL){logfp=fopen(&logfile[0],"w");}
	printf("Making level-%i CRBM (%i->%i)...\n",i+1,*(settings.sizes+i),*(settings.sizes+i+1));
	crbm_init(crbms+i,*(settings.sizes+i),*(settings.sizes+i+1),settings.stdev_init,settings.f0,settings.f1,logfp);
      } else {
	if(settings.logbase!=NULL){logfp=fopen(&logfile[0],"w+");}
      }
      if(settings.datafile!=NULL){
	printf("Training for %i iterations...\n",settings.niter_crbm-settings.iter);
	if(i==settings.ncrbms-1){
	  //Only use force-file on bottom encoding layer
	  crbm_train(crbms+i,tdata1,&settings,logfp,p2enc_force);
	} else {
	  crbm_train(crbms+i,tdata1,&settings,logfp,NULL);
	}
	printf("...done!\n");
      }
      if(i<settings.ncrbms-1 && settings.datafile!=NULL){
	dataset_alloc(tdata2,*(settings.sizes+i+1),tdata1->nrecs);
	crbm_encode(crbms+i,tdata1,tdata2,-1);
	dataset_free(tdata1);
	tmp=tdata1;
	tdata1=tdata2;
	tdata2=tmp;
      }
      if(settings.logbase!=NULL){fclose(logfp);}
      settings.iter=0; //Next iteration should begin afresh
      settings.stage++; //And we are moving on to the next level
    }
    if(settings.datafile!=NULL){dataset_free(tdata1);}
  }
  if(settings.stage==settings.ncrbms){
    if(settings.logbase!=NULL){
      if(strlen(settings.logbase)>72){printf("Error: log file variable not large enough\n");exit(-1);}
      sprintf(&logfile[0],"%s.auto.log",settings.logbase,i+1);
    }
    if(settings.iter==0 && crbms!=NULL){
      if(settings.logbase!=NULL){logfp=fopen(&logfile[0],"w");}
      a=make_autoencoder(settings.ncrbms,crbms,logfp);
      for(i=0;i<settings.ncrbms;i++){crbm_free(crbms+i);}
    } else {
      //Hopefully something useful exists in 'a'
      if(settings.logbase!=NULL){logfp=fopen(&logfile[0],"w+");}
    }
    if(settings.datafile!=NULL){
      if(settings.rprop_flag==0) {
        printf("Using gradient descent trainer.\n");
        autoencoder_batchtrain(a,p2data,&settings,p2monitor,logfp,p2enc_force);
      } else {
        printf("Using RProp- trainer.\n");
        autoencoder_batchtrain_rprop(a,p2data,&settings,p2monitor,logfp,p2enc_force);
      }
    }
    save_autoencoder(settings.outputfile,a);
    char statefile[80];
    sprintf(&statefile[0],"%s.state",settings.outputfile);
    write_state(statefile,settings,settings.outputfile);
    if(settings.logbase!=NULL){fclose(logfp);}
  }
  return a;
}
