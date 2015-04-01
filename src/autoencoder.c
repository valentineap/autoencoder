#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef ACML
#include <acml.h>
#else
#include <cblas.h>
#endif
#include <assert.h>
#include <signal.h>
#include "autoencoder.h"
#ifndef NATIVE_RAND //Use Mersenne Twister
#include "mt19937ar.h"
#endif

volatile cease_training=0;


///////////////////////
// Dataset functions //
///////////////////////
void dataset_alloc(dataset_t *data,int npts,int nrecs){
  data->npoints=npts;
  data->nrecs=nrecs;
  data->data=malloc(npts*nrecs*sizeof(double));
  data->scale=malloc(nrecs*sizeof(double));
  data->weight=malloc(nrecs*sizeof(double));
}
void dataset_free(dataset_t *data){
  data->npoints=-1;
  data->nrecs=-1;
  free(data->data);
  free(data->scale);
  free(data->weight);
}

void dataset_copy(dataset_t *dest,dataset_t *src){
  dataset_alloc(dest,src->npoints,src->nrecs);
  memcpy(dest->data,src->data,src->npoints*src->nrecs*sizeof(double));
  memcpy(dest->weight,src->weight,src->nrecs*sizeof(double));
  memcpy(dest->scale,src->scale,src->nrecs*sizeof(double));
}

void dataset_resize(dataset_t *data,int newnrecs){
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
  FILE *fp;
  if((fp=fopen(filename,"r"))==NULL){
    printf("Error opening file: %s\n",*filename);
    abort();
  }
  int npts,nrecs;
  if(fscanf(fp,"%d%d",&npts,&nrecs)!=2){printf("Error reading from file: %s\n",*filename);printf("Have npts=%i & nrecs=%i\n",npts,nrecs);abort();}
  dataset_t data;
  dataset_alloc(&data,npts,nrecs);
  int i,j;
  for (j=0;j<nrecs;j++){
    for(i=0;i<npts;i++){
      //      printf("%i %i\n",i,j);
      if(fscanf(fp,"%lf",data.data+(i*(nrecs)+j))!=1){printf("Error reading data from file: %s\n",*filename);printf("Failed to read data[%i*(nrecs)+%i]\n",i,j);abort();}
    }
    if(fscanf(fp,"%le",data.scale+j)!=1){printf("Error reading scale from file: %s\n",*filename);abort();}
    if(fscanf(fp,"%le",data.weight+j)!=1){printf("Error reading weight from file: %s\n",*filename);abort();}
  }
  fclose(fp);
  return data;
}

void writedata(char *filename,dataset_t *data){
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



/////////////////////////////
// Random number functions //
/////////////////////////////
int have_stored_rand=0;
double stored_rand;
int seeded_random=0;//Has random seed been initialised?

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
  int i,j;
  assert(data_in->nrecs==data_out->nrecs);
  assert(data_in->npoints==c->nlv);
  assert(data_out->npoints==c->nlh);
  for(i=0;i<c->nlh;i++){
    for(j=0;j<data_in->nrecs;j++){
      *(data_out->data+(i*data_in->nrecs)+j)=*(c->b_v2h+i);
    }
  }
  cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,c->nlh,data_in->nrecs,c->nlv,1.0,c->w,c->nlh,data_in->data,data_in->nrecs,1.0,data_out->data,data_in->nrecs);
  memcpy(data_out->weight,data_in->weight,data_in->nrecs*sizeof(double)); //Propagate weights and scales through successive CRBMs
  memcpy(data_out->scale,data_in->scale,data_in->nrecs*sizeof(double));
  if(stdev<0.){
    for(i=0;i<c->nlh;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*LOGISTIC(*(c->a_v2h+i)*(*(data_out->data+(i*data_in->nrecs)+j)));
      }
    }
  } else {
    for(i=0;i<c->nlh;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*LOGISTIC(*(c->a_v2h+i)*(*(data_out->data+(i*data_in->nrecs)+j)+(stdev*random_normal())));
      }
    }
  }
}


void crbm_decode(crbm_t *c,dataset_t *data_in, dataset_t *data_out,double stdev){
  int i,j;
  assert(data_in->nrecs==data_out->nrecs);
  assert(data_in->npoints==c->nlh);
  assert(data_out->npoints==c->nlv);

  for(i=0;i<c->nlv;i++){
    for(j=0;j<data_in->nrecs;j++){
      *(data_out->data+(i*data_in->nrecs)+j)=*(c->b_h2v+i);
    }
  }
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,c->nlv,data_in->nrecs,c->nlh,1.0,c->w,c->nlh,data_in->data,data_in->nrecs,1.0,data_out->data,data_in->nrecs);
  memcpy(data_out->weight,data_in->weight,data_in->nrecs*sizeof(double)); //Propagate weights and scales through successive CRBMs
  memcpy(data_out->scale,data_in->scale,data_in->nrecs*sizeof(double));
  if(stdev<0.){
    for(i=0;i<c->nlv;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*LOGISTIC(*(c->a_h2v+i)*(*(data_out->data+(i*data_in->nrecs)+j)));
      }
    }
  } else {
    for(i=0;i<c->nlv;i++){
      for(j=0;j<data_in->nrecs;j++){
	*(data_out->data+(i*data_in->nrecs)+j)=c->loglo+(c->logup-c->loglo)*LOGISTIC(*(c->a_h2v+i)*(*(data_out->data+(i*data_in->nrecs)+j)+(stdev*random_normal())));
      }
    }
  }
}

void crbm_train(crbm_t *c,dataset_t *data,state_settings_t * settings,FILE *logfp){
  dataset_t enc,dec,encdec;
  double *cdata,*cdec;
  dataset_alloc(&enc,c->nlh,data->nrecs);
  dataset_alloc(&dec,c->nlv,data->nrecs);
  dataset_alloc(&encdec,c->nlh,data->nrecs);
  cdata=malloc(c->nlv*c->nlh*sizeof(double));
  cdec=malloc(c->nlv*c->nlh*sizeof(double));
  if(logfp!=NULL){fprintf(logfp,"# crbm_train >> nrecs=%i niter=%i stdev=%f lrate=%f\n",data->nrecs,settings->niter_crbm,settings->noise_crbm,settings->lrate_crbm);}
  int iter,i,j,k;
  double wtnorm=0.;
  for(i=0;i<data->nrecs;i++){wtnorm+=*(data->weight+i);}
  while(settings->iter<settings->niter_crbm){
    crbm_encode(c,data,&enc,settings->noise_crbm);
    crbm_decode(c,&enc,&dec,settings->noise_crbm);
    crbm_encode(c,&dec,&encdec,settings->noise_crbm);
    printf("CRBM (%i->%i) training iteration %3i :: E=%f\n",c->nlv,c->nlh,settings->iter+1,error_dataset(data,&dec));
    if(logfp!=NULL){fprintf(logfp,"%i %f\n",settings->iter+1,error_dataset(data,&dec));}
    for(i=0;i<c->nlh;i++){
      for(j=0;j<data->nrecs;j++){ //Scale encodings by weights so that when we sum over all records (in dgemm) weighting is taken into account
	*(enc.data+(i*data->nrecs)+j)*=*(data->weight+j)/wtnorm; 
	*(encdec.data+(i*data->nrecs)+j)*=*(data->weight+j)/wtnorm;
      }
    }
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,c->nlv,c->nlh,data->nrecs,1.0,data->data,data->nrecs,enc.data,data->nrecs,0.0,cdata,c->nlh);
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,c->nlv,c->nlh,data->nrecs,1.0,dec.data,data->nrecs,encdec.data,data->nrecs,0.0,cdec,c->nlh);
    double scale=settings->lrate_crbm; //wtnorm implies division by number of traces (or equivalent)
    for(i=0;i<c->nlv;i++){
      for(j=0;j<c->nlh;j++){
	*(c->w+(i*c->nlh)+j)+=scale*(*(cdata+(i*c->nlh)+j)-*(cdec+(i*c->nlh)+j));//Weighting already handled
      }
    }
    for(i=0;i<c->nlh;i++){ 
      for(j=0;j<data->nrecs;j++){
	*(c->b_v2h+i)+=scale*(*(enc.data+(i*data->nrecs)+j)-*(encdec.data+(i*data->nrecs)+j));//Weighting already handled
      }
    }
    for(i=0;i<c->nlv;i++){
      for(j=0;j<data->nrecs;j++){
	*(c->b_h2v+i)+=scale*(*(data->data+(i*data->nrecs)+j)-*(dec.data+(i*data->nrecs)+j))* *(data->weight+j)/wtnorm;
      }	
    }

    double denom;
    for(i=0;i<c->nlh;i++){
      denom=pow(*(c->a_v2h+i),2.);
      for(j=0;j<data->nrecs;j++){
	*(c->a_v2h+i)+=(scale/denom)*(pow(*(enc.data+(i*data->nrecs)+j),2.)-pow(*(encdec.data+(i*data->nrecs)+j),2.))*wtnorm/ *(data->weight+j);//Avoid double-weighting the squared terms
      }
    }
    for(i=0;i<c->nlv;i++){
      denom=pow(*(c->a_h2v+i),2.);
       for(j=0;j<data->nrecs;j++){
	 *(c->a_h2v+i)+=(scale/denom)*(pow(*(data->data+(i*data->nrecs)+j),2.)-pow(*(dec.data+(i*data->nrecs)+j),2.))* *(data->weight+j)/wtnorm;
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
  FILE *fp=fopen(filename,"w");
  fseek(fp,0,SEEK_SET);
  write_crbm(fp,c);
  fclose(fp);
} 


crbm_t read_crbm(FILE *fp){
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
  FILE *fp=fopen(filename,"w");
  fseek(fp,0,SEEK_SET);
  write_autoencoder(fp,a);
  fclose(fp);
}

autoenc_t *read_autoencoder(FILE *fp){
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
  FILE *fp=fopen(filename,"r");
  fseek(fp,0,SEEK_SET);
  autoenc_t *a=read_autoencoder(fp);
  fclose(fp);
  return a;
}

void autoencoder_free(autoenc_t *a){
  int i;
  for(i=0;i<a->nlayers;i++){
    free(a->layers[i].w);
    free(a->layers[i].a);
    free(a->layers[i].b);
  }
}

void autoencoder_encode(autoenc_t *a,dataset_t *data_in,dataset_t *data_out){
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
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=*(a->layers[i].b+j);
      }
    }
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i].nout,data_in->nrecs,a->layers[i].nin,1.0,a->layers[i].w,a->layers[i].nin,layer_in,data_in->nrecs,1.0,layer_out,data_in->nrecs);
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=a->layers[i].loglo+(a->layers[i].logup-a->layers[i].loglo)*LOGISTIC(*(layer_out+(j*data_in->nrecs)+k)* *(a->layers[i].a+j));
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
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=*(a->layers[i].b+j);
      }
    }
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i].nout,data_in->nrecs,a->layers[i].nin,1.0,a->layers[i].w,a->layers[i].nin,layer_in,data_in->nrecs,1.0,layer_out,data_in->nrecs);
    for(j=0;j<a->layers[i].nout;j++){
      for(k=0;k<data_in->nrecs;k++){
	*(layer_out+(j*data_in->nrecs)+k)=a->layers[i].loglo+(a->layers[i].logup-a->layers[i].loglo)*LOGISTIC(*(layer_out+(j*data_in->nrecs)+k)* *(a->layers[i].a+j));
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
  //Assume that nodevals[0] is already populated...
  int i,j,k;
  for(i=1;i<a->nlayers+1;i++){
    for(j=0;j<(nodevals+i)->n;j++){
      for(k=0;k<nrecs;k++){
	*(((nodevals+i)->values)+(j*nrecs)+k)=*((a->layers[i-1].b)+j);
      }
    }
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,(nodevals+i)->n,nrecs,(nodevals+i-1)->n,1.0,a->layers[i-1].w,(nodevals+i-1)->n,(nodevals+i-1)->values,nrecs,1.0,(nodevals+i)->values,nrecs);
    for(j=0;j<(nodevals+i)->n;j++){
      for(k=0;k<nrecs;k++){
	*((nodevals+i)->values+(j*nrecs)+k)=a->layers[i-1].loglo+(a->layers[i-1].logup-a->layers[i-1].loglo)*(LOGISTIC(*((nodevals+i)->values+(j*nrecs)+k)* *(a->layers[i-1].a+j)));
	*((nodevals+i)->derivs+(j*nrecs)+k)=(*((nodevals+i)->values+(j*nrecs)+k)-a->layers[i-1].loglo)*(a->layers[i-1].logup-*((nodevals+i)->values+(j*nrecs)+k))/(a->layers[i-1].logup-a->layers[i-1].loglo);
      }
    }
  }
}

void autoencoder_batchtrain(autoenc_t *a,dataset_t *data,state_settings_t *settings,dataset_t *monitor,FILE *logfp){
  //Main training routine

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

  //Set up structure for back-propagation of cost term
  double *cost[a->nlayers/2+1];
  if(settings->costwt>0.){
    for(i=1;i<a->nlayers/2+1;i++){
      cost[i]=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
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

  settings->N_reduced=0;
  settings->i_red_sum=0;
  settings->iter_reset=0;
  int n_increasing=0;
  double eta;

  //Main training loop
  while(settings->iter<settings->niter_auto && cease_training==0){
    //Evaluate encoding/reconstruction of dataset for current weights
    memcpy(nodes[0].values,data->data,nodes[0].n*data->nrecs*sizeof(double));
    if(settings->noise_auto>=0.){for(i=0;i<data->nrecs*nodes[0].n;i++){*(nodes[0].values+i)+=settings->noise_auto*random_normal();}}//Add noise if requested
    autoencoder_encdec(a,nodes,data->nrecs);
    //Compute reconstruction error
    if(settings->iter>0){olderr=err;}
    err=error_array(nodes[0].values,nodes[a->nlayers].values,nodes[0].n,data->nrecs,data->weight);
    //Compute average encoding length
    double lenc=len_enc_array(nodes[a->nlayers/2].values,nodes[a->nlayers/2].n,data->nrecs,data->weight);
    //Learning rate is adaptive to handle instabilities etc.
    if(settings->iter==0){
      settings->i_adapt_rate=0;
    } else {
      if(err<=olderr || (settings->noise_auto>=0. && err<=(olderr+2.*settings->noise_auto)) && n_increasing<5){
	if(err>olderr){
	  n_increasing++;
	} else {
	  n_increasing=0;
	}
	if(settings->i_adapt_rate<settings->N_adapt_rate){settings->i_adapt_rate++;}
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
    eta=settings->eta0+settings->i_adapt_rate*(settings->lrate_auto-settings->eta0)/settings->N_adapt_rate;

    double mon_err;
    //Compute reconstruction error for monitor dataset if reqd, write progress to file
    if(monitor!=NULL){
      autoencoder_encode(a,monitor,&mon_enc);
      autoencoder_decode(a,&mon_enc,&mon_dec);
      mon_err=error_dataset(monitor,&mon_dec);
      printf("Autoencoder training iteration %4i :: E=%.5f (%.3f%%) Emon=%.5f (%.3f%%) L=%.3f\n",settings->iter+1,err,err*100./err_base,mon_err,mon_err*100./mon_base,lenc);
      if(logfp!=NULL){fprintf(logfp,"%i %f %f %f\n",settings->iter+1,eta, err,mon_err);}
    } else {
      printf("Autoencoder training iteration %4i :: E=%.5f (%.3f%%) L=%.3f\n",settings->iter+1,err,err*100./err_base,lenc);
      if(logfp!=NULL){fprintf(logfp,"%i %f %f\n",eta,settings->iter+1,err);}
    }

    //Back-propagate errors
    for(i=0;i<(data->nrecs*nodes[0].n);i++){*(delta[a->nlayers]+i)=*(nodes[a->nlayers].values+i)- *(nodes[0].values+i);}
    double *tmp;
    for(i=a->nlayers-1;i>0;i--){
      tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double));
      for(j=0;j<nodes[i+1].n;j++){
	for(k=0;k<data->nrecs;k++){
	  *(tmp+(j*data->nrecs)+k)=*(delta[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j);
	}
      }
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,delta[i],data->nrecs);
      free(tmp);
    }
    if(settings->costwt>0.){
      //Back-propagate cost term
      //printf("%i\n",nodes[a->nlayers/2].n);
      //for(i=0;i<(data->nrecs*nodes[a->nlayers/2].n);i++){*(cost[a->nlayers/2]+i)=*(nodes[a->nlayers/2].values+i);}
      for(i=0;i<(data->nrecs*nodes[a->nlayers/2].n);i++){*(cost[a->nlayers/2]+i)=copysign(1.0,*(nodes[a->nlayers/2].values+i));}
      //printf("A\n");
      for(i=a->nlayers/2-1;i>0;i--){
	tmp=malloc(nodes[i+1].n*data->nrecs*sizeof(double));
	for(j=0;j<nodes[i+1].n;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(tmp+(j*data->nrecs)+k)=*(cost[i+1]+(j*data->nrecs)+k)* *(nodes[i+1].derivs+(j*data->nrecs)+k) * *(a->layers[i].a+j);
	  }
	}
	cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nodes[i].n,data->nrecs,nodes[i+1].n,1.0,a->layers[i].w,nodes[i].n,tmp,data->nrecs,0.0,cost[i],data->nrecs);
	free(tmp);
      }
      //printf("B\n");
    }

    //Compute weight updates from backpropagated errors
    double *dw,*da,*db,*work;
    for(i=1;i<a->nlayers+1;i++){
      dw=calloc(a->layers[i-1].nin*a->layers[i-1].nout,sizeof(double));
      da=calloc(a->layers[i-1].nout,sizeof(double));
      db=calloc(a->layers[i-1].nout,sizeof(double));
      work=malloc(a->layers[i-1].nout*data->nrecs*sizeof(double));
      memcpy(work,delta[i],a->layers[i-1].nout*data->nrecs*sizeof(double));
      for(j=0;j<a->layers[i-1].nout;j++){
	for(k=0;k<data->nrecs;k++){
	  *(work+(j*data->nrecs)+k)*=*(nodes[i].derivs+(j*data->nrecs)+k)* *(a->layers[i-1].a+j)* *(data->weight+k)/wtnorm;
	}
      }
      for(j=0;j<a->layers[i-1].nout;j++){
	*(db+j)=*(work+(j*data->nrecs));
	for(k=1;k<data->nrecs;k++){
	  *(db+j)+=*(work+(j*data->nrecs)+k);
	}
      }
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,a->layers[i-1].nout,a->layers[i-1].nin,data->nrecs,1.0,work,data->nrecs,nodes[i-1].values,data->nrecs,0.0,dw,a->layers[i-1].nin);
      
      for(j=0;j<a->layers[i-1].nout;j++){
	for(k=0;k<data->nrecs;k++){
	  *(work+(j*data->nrecs)+k)=*(a->layers[i-1].b+j);
	}
      }
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i-1].nout,data->nrecs,a->layers[i-1].nin,1.0,a->layers[i-1].w,a->layers[i-1].nin,nodes[i-1].values,data->nrecs,1.0,work,data->nrecs);
      for(j=0;j<(a->layers[i-1].nout*data->nrecs);j++){
	*(work+j)*=*(delta[i]+j)* *(nodes[i].derivs+j);
      }
      for(j=0;j<a->layers[i-1].nout;j++){
	*(da+j)=*(work+(j*data->nrecs))* *(data->weight)/wtnorm;
	for(k=1;k<data->nrecs;k++){
	  *(da+j)+=*(work+(j*data->nrecs)+k)* *(data->weight+k)/wtnorm;
	}
      }

      if(settings->costwt>0. && i<=a->nlayers/2){
	//printf("%i\n",a->layers[i-1].nout);
	//printf("C\n");
	memcpy(work,cost[i],a->layers[i-1].nout*data->nrecs*sizeof(double));
	//printf("D\n");
	for(j=0;j<a->layers[i-1].nout;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(work+(j*data->nrecs)+k)*=*(nodes[i].derivs+(j*data->nrecs)+k)* *(a->layers[i-1].a+j)* *(data->weight+k)/wtnorm;
	  }
	}
	//printf("E\n");
	for(j=0;j<a->layers[i-1].nout;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(db+j)+=*(work+(j*data->nrecs)+k)*settings->costwt;
	  }
	}
	//printf("F\n");
	//Update dw - note funny scaling to accommodate costwt within dgemm call
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,a->layers[i-1].nout,a->layers[i-1].nin,data->nrecs,1.0,work,data->nrecs,nodes[i-1].values,data->nrecs,1.0/settings->costwt,dw,a->layers[i-1].nin);
	for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){*(dw+j)=settings->costwt* *(dw+j);}
	
	for(j=0;j<a->layers[i-1].nout;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(work+(j*data->nrecs)+k)=*(a->layers[i-1].b+j);
	  }
	}
	cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,a->layers[i-1].nout,data->nrecs,a->layers[i-1].nin,1.0,a->layers[i-1].w,a->layers[i-1].nin,nodes[i-1].values,data->nrecs,1.0,work,data->nrecs);
	for(j=0;j<(a->layers[i-1].nout*data->nrecs);j++){
	  *(work+j)*=*(cost[i]+j)* *(nodes[i].derivs+j);
	}
	for(j=0;j<a->layers[i-1].nout;j++){
	  for(k=0;k<data->nrecs;k++){
	    *(da+j)+=*(work+(j*data->nrecs)+k)*settings->costwt* *(data->weight+k)/wtnorm;
	  }
	}
      }
      free(work);
      for(j=0;j<a->layers[i-1].nin*a->layers[i-1].nout;j++){
	*(a->layers[i-1].w+j)-=eta* *(dw+j);
      }
      for(j=0;j<a->layers[i-1].nout;j++){
	*(a->layers[i-1].a+j)-=eta* *(da+j);
      }
      for(j=0;j<a->layers[i-1].nout;j++){
	*(a->layers[i-1].b+j)-=eta* *(db+j);
      }
      free(dw);
      free(da);
      free(db);
    }
    settings->iter++;
    //And let's go around for another iteration...
  }
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
  nodevals_free(a,nodes);
}

/////////////////////////////
// Miscellaneous functions //
/////////////////////////////	
void make_nodevals(autoenc_t *a,node_val_t *nodevals,int nrecs){
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
  int i,j;
  double e=0.0;
  double wtnorm=0.;
  for(i=0;i<d1->nrecs;i++){wtnorm+=*(d1->weight+i);}
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
  printf("Training interupted by SIGINT. Options:\n(r)esume training\n(c)omplete current iteration, save network and exit normally\n(a)bort\n\nPlease press r, c or a...");
  char resp='\0';
  int loop=1;
  while(loop){
    scanf("%c",&resp);
    switch(resp){
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
  fwrite(&settings.N_adapt_rate,sizeof(int),4,fp);
  if(storefile==NULL){lstr=0;} else {lstr=strlen(storefile);}
  fwrite(&lstr,sizeof(int),1,fp);
  if(lstr!=0){fwrite(storefile,sizeof(char),lstr,fp);}
  //And deal with rng
#ifdef NATIVE_RAND
  sw=0;
#else
  sw=1;
#endif
  fwrite(&sw,sizeof(int),1,fp);
#ifndef NATIVE_RAND
  dump_twister_state(fp);
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
  fread(&s.N_adapt_rate,sizeof(int),4,fp);
  fread(&lstr,sizeof(int),1,fp);
  printf("lstr: %i\n",lstr);
  if(lstr>0){
    *storefile=malloc(lstr*sizeof(char));
    fread(*storefile,sizeof(char),lstr,fp);
  }
  fread(&sw,sizeof(int),1,fp);
#ifdef NATIVE_RAND
  if(sw==1){printf("*** Settings file is for Mersenne Twister prng ***\n");}
#else
  if(sw==0){
    printf("*** No random number state provided in settings file ***\n");
  } else {
    load_twister_state(fp);
    seeded_random=1;
  }
#endif
  fclose(fp);
  return s;
}

void print_state(state_settings_t s){
  printf("niter_crbm:  %4i     niter_auto: %5i\n",s.niter_crbm,s.niter_auto);
  printf("stdev_init:  %.4f\n",s.stdev_init);
  printf("noise_crbm:  %.4f   noise_auto: %.4f\n",s.noise_crbm,s.noise_auto);
  printf("lrate_crbm:  %.4f   lrate_auto: %.4f\n",s.lrate_crbm,s.lrate_auto);
  printf("f0:          %5.3f   f1:         %5.3f\n",s.f0,s.f1);
  if(s.datafile==NULL){
    printf("datafile:    --\n");
  } else {
    printf("datafile:    %s\n",s.datafile);
  }
  if(s.monitorfile==NULL){
    printf("monitorfile: --\n");
  } else {
    printf("monitorfile: %s\n",s.monitorfile);
  }
  if(s.outputfile==NULL){
    printf("outputfile:  --\n");
  } else {
    printf("outputfile:  %s\n",s.outputfile);
  }
  if(s.logbase==NULL){
    printf("logbase:     --\n");
  } else {
    printf("logbase:     %s\n",s.logbase);
  }
  printf("ncrbms:      %i\n",s.ncrbms);
  printf("sizes:       ");
  int i;
  for(i=0;i<s.ncrbms+1;i++){
    printf("%i ",*(s.sizes+i));
  }
  printf("\n");
  printf("stage:       %2i       iter:%i\n",s.stage,s.iter);
}
  

autoenc_t * autoencoder_make_and_train(crbm_t *crbms,autoenc_t * a,state_settings_t settings){
  dataset_t data,monitor,encoded1,encoded2,*tdata1=&encoded1,*tdata2=&encoded2,*tmp,*p2data=NULL,*p2monitor=NULL;
  char logfile[80];
  FILE *logfp=NULL;
  int i;
  signal(SIGINT,abort_handler);
  if(settings.datafile!=NULL){
    data=load_dataset(settings.datafile);
    if(settings.ignore_weights){
      for(i=0;i<data.nrecs;i++){*(data.weight+i)=1.;}
    }
    p2data=&data;
    dataset_copy(&encoded1,&data);
  }
  if(settings.monitorfile!=NULL){
    monitor=load_dataset(settings.monitorfile);
    if(settings.ignore_weights){
      for(i=0;i<monitor.nrecs;i++){*(monitor.weight+i)=1.;}
    }
    p2monitor=&monitor;
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
	crbm_train(crbms+i,tdata1,&settings,logfp);
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
      logfp=fopen(&logfile[0],"w");
      a=make_autoencoder(settings.ncrbms,crbms,logfp);
      for(i=0;i<settings.ncrbms;i++){crbm_free(crbms+i);}
    } else {
      //Hopefully something useful exists in 'a'
      logfp=fopen(&logfile[0],"w+");
    }
    if(settings.datafile!=NULL){
      autoencoder_batchtrain(a,p2data,&settings,p2monitor,logfp);
    }
    save_autoencoder(settings.outputfile,a);
    char statefile[80];
    sprintf(&statefile[0],"%s.state",settings.outputfile);
    write_state(statefile,settings,settings.outputfile);
    if(settings.logbase!=NULL){fclose(logfp);}
  }
  return a;
}
  
    
  
  

		

   
