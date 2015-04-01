#include <stdlib.h>
#include <string.h>
#include <popt.h>
#include "autoencoder.h"

/* void auto_create(dataset_t *data,dataset_t *monitor,FILE *outfp,state_settings_t settings){ */
/*   crbm_t crbms[settings.ncrbms]; */
/*   crbm_t *crbms_p[settings.ncrbms]; */

/*   dataset_t encoded1,encoded2; */
/*   dataset_t *tdata1=&encoded1,*tdata2=&encoded2,*tmp; */
/*   if(data!=NULL){dataset_copy(&encoded1,data);} */
/*   int i; */
/*   char logfile[80]; */
/*   FILE * logfp=NULL; */
/*   for(i=0;i<settings.ncrbms;i++){ */
/*     if(settings.logbase!=NULL){ */
/*       if(strlen(settings.logbase)>68){printf("Error: log file variable not large enough\n");exit(-1);} */
/*       sprintf(&logfile[0],"%s.crbm.%02i.log",settings.logbase,i+1); */
/*       logfp=fopen(&logfile[0],"w"); */
/*     } */
/*     printf("Making level-%i CRBM (%i->%i)...\n",i+1,*(settings.sizes+i),*(settings.sizes+i+1)); */
/*     crbms_p[i]=&crbms[i]; */
/*     crbm_init(&crbms[i],*(settings.sizes+i),*(settings.sizes+i+1),settings.stdev_init,settings.f0,settings.f1,logfp); */
/*     if(data!=NULL){ */
/*       printf("Training for %i iterations...\n",settings.niter_crbm); */
/*       crbm_train(&crbms[i],tdata1,settings.niter_crbm,settings.noise_crbm,settings.lrate_crbm,logfp); */
/*       printf("...done!\n"); */
/*     } */
/*     if(i<settings.ncrbms-1 && data!=NULL){ */
/*       dataset_alloc(tdata2,*(settings.sizes+i+1),tdata1->nrecs); */
/*       crbm_encode(&crbms[i],tdata1,tdata2,-1); */
/*       dataset_free(tdata1); */
/*       tmp=tdata1; */
/*       tdata1=tdata2; */
/*       tdata2=tmp; */
/*     } */
/*     if(settings.logbase!=NULL){fclose(logfp);} */
/*   } */
/*   if(data!=NULL){dataset_free(tdata1);} */
/*   if(settings.logbase!=NULL){ */
/*     sprintf(&logfile[0],"%s.auto.log",settings.logbase); */
/*     logfp=fopen(&logfile[0],"w"); */
/*   } */
    
/*   autoenc_t *a=make_autoencoder(settings.ncrbms,crbms_p,NULL); */
/*   for(i=0;i<settings.ncrbms;i++){crbm_free(&crbms[i]);} */
/*   if(data!=NULL){ */
/*     printf("Beginning autoencoder training...\n"); */
/*     autoencoder_batchtrain(a,data,settings.niter_auto,settings.noise_auto,settings.lrate_auto,monitor,logfp); */
/*   } else { */
/*     printf("Storing untrained autoencoder\n"); */
/*   } */
/*   write_autoencoder(outfp,a); */
/*   if(settings.logbase!=NULL){ */
/*     fclose(logfp); */
/*   } */
/* } */
		   
      
      
      
int main(int argc,const char **argv){

  state_settings_t settings={500,3000,0.01,0.1,0.1,0.3,0.1,-1.1,1.1,NULL,NULL,"out.cnet",NULL,0,NULL,0,0,0,0.01,100,0,0,0,0,0.0};
  char *statefile=NULL;
  struct poptOption optionsTable[] = {
    {"training-data",'t',POPT_ARG_STRING,&settings.datafile,0,"File containing training data","FILE"},
    {"monitor-data",'m',POPT_ARG_STRING,&settings.monitorfile,0,"File containing validation dataset","FILE"},
    {"niter-crbm",'c',POPT_ARG_INT|POPT_ARGFLAG_SHOW_DEFAULT,&settings.niter_crbm,0,"Number of iterations of CRBM training to perform","NITER"},
    {"niter-auto",'a',POPT_ARG_INT|POPT_ARGFLAG_SHOW_DEFAULT,&settings.niter_auto,0,"(Maximum) number of iterations of autoencoder training to perform","NITER"},
    {"output-file",'o',POPT_ARG_STRING,&settings.outputfile,0,"Filename for storing autoencoder","FILE"},
    {"stdev-init",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.stdev_init,0,"Standard deviation of Gaussian distribution to use when initialising CRBM weights","STDEV"},
    {"noise-crbm",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.noise_crbm,0,"Standard deviation of Gaussian noise to use during CRBM training step","STDEV"},
    {"noise-auto",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.noise_auto,0,"Standard deviation of Gaussian noise to use during autoencoder training step","STDEV"},
    {"lrate-crbm",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.lrate_crbm,0,"Learning rate parameter for CRBM training","RATE"},
    {"lrate-auto",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.lrate_auto,0,"Learning rate parameter for autoencoder training","RATE"},
    {"f0",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.f0,0,"Lower bound of logistic function","F0"},
    {"f1",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.f1,0,"Upper bound of logistic function","F1"},
    {"ignore-weights",'\0',POPT_ARG_NONE,&settings.ignore_weights,0,"Give all training data unit weight, over-riding any weight information in data file",NULL},
    {"logfile-base",'l',POPT_ARG_STRING,&settings.logbase,0,"Base filename for logfiles","FILE"},
    {"resume",'\0',POPT_ARG_STRING,&statefile,0,"Resume previous training","STATE-FILE"},
    {"cost-weight",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.costwt,0,"Weight given to cost term during training","WEIGHT"},
    POPT_AUTOHELP
    POPT_TABLEEND};



  poptContext poptc=poptGetContext(NULL,argc,argv,optionsTable,0);
  int pp,i;
  if((pp=poptGetNextOpt(poptc))<-1){printf("%s: %s\n",poptBadOption(poptc,POPT_BADOPTION_NOALIAS),poptStrerror(pp));exit(-1);}
  crbm_t *crbms=NULL;
  autoenc_t *a=NULL;
  if(statefile!=NULL){
    //We have been asked to resume
    char *storefile;
    printf("Previous state loaded from %s\n",statefile);
    settings=load_state(statefile,&storefile);
    print_state(settings);
    printf("This will be modified to reflect anything specified on the command line.\n");
    if(storefile==NULL){printf("Error: unable to find stored network\n");exit(-1);}
    //Load storefile data
    if(settings.stage<settings.ncrbms){ 
      //We are trying to load crbms
      printf("Loading CRBMs from: %s\n",storefile);
      FILE * fp=fopen(storefile,"r");
      crbms=malloc(settings.ncrbms*sizeof(crbm_t));
      int i;
      for(i=0;i<settings.stage+1;i++){
	*(crbms+i)=read_crbm(fp);
      }
    } else {
      //We are loading an autoencoder
      printf("Loading autoencoder from: %s\n",storefile);
      a=load_autoencoder(storefile);
    }
    poptResetContext(poptc);
    if((pp=poptGetNextOpt(poptc))<-1){printf("%s: %s\n",poptBadOption(poptc,POPT_BADOPTION_NOALIAS),poptStrerror(pp));exit(-1);}
    //Don't reprocess statefile variable...
    //Special handling of layer numbers...
    //Load crbm/autoencoder
  } else{
  const char **leftovers=poptGetArgs(poptc);
    while(*(leftovers+settings.ncrbms)!=NULL){settings.ncrbms++;}
    if(settings.ncrbms>1){
      settings.sizes=malloc(settings.ncrbms*sizeof(int));
      for(i=0;i<settings.ncrbms;i++){sscanf(*(leftovers+i),"%i",settings.sizes+i);}
      settings.ncrbms--; //Actually, one less crbm than sizes
      crbms=malloc(settings.ncrbms*sizeof(crbm_t));
      a=NULL;
    } else {
      printf("No layer sizes provided\n");
      exit(-1);
    }
  }  
  a=autoencoder_make_and_train(crbms,a,settings);

}
