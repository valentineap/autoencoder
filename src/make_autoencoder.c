////////////////////////////////////////////////////////
// make_autoencoder.c - Generate/train an autoencoder //
////////////////////////////////////////////////////////
// Andrew Valentine                                   //
// Universiteit Utrecht                               //
// 2011-2013                                          //
//                                                    //
// With contributions from:                           //
// - Paul Kaeufl                                      //
//                                                    //
////////////////////////////////////////////////////////
// $Id: make_autoencoder.c,v 1.3 2012/03/31 15:23:22 andrew Exp andrew $
#include <stdlib.h>
#include <string.h>
#include <popt.h>
#include "autoencoder.h"

          
int main(int argc,const char **argv){
  // 13/04/1012 PK changed N_adapt_rate from 100 to 1000
  state_settings_t settings={500,3000,0.01,0.1,0.1,0.3,0.1,-1.1,1.1,NULL,NULL,NULL,"out.cnet",NULL,0,NULL,0,0,0,0.01,100,0,0,0,0,0.0,0,1.0,0,1.2,0.5,0.5,5.0,1.0e-6,0,1};
  char *statefile=NULL;
  

  // Ultimately we may end up with a lot of different training algorithms etc; therefore keep options tables nicely structured...

  struct poptOption crbmOptionsTable[] = { //Options governing CRBM stage
    {"niter-crbm",'c',POPT_ARG_INT|POPT_ARGFLAG_SHOW_DEFAULT,&settings.niter_crbm,0,"Number of iterations of CRBM training to perform","NITER"},
    {"noise-crbm",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.noise_crbm,0,"Standard deviation of Gaussian noise to use during CRBM training step","STDEV"},
    {"lrate-crbm",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.lrate_crbm,0,"Learning rate parameter for CRBM training","RATE"},
    POPT_TABLEEND};

  struct poptOption fileOptionsTable[] = { //Anything dealing with files and their contents
    {"training-data",'t',POPT_ARG_STRING,&settings.datafile,0,"File containing training data","FILE"},
    {"ignore-weights",'\0',POPT_ARG_NONE,&settings.ignore_weights,0,"Give all training data unit weight, over-riding any weight information in data file",NULL},
    {"monitor-data",'m',POPT_ARG_STRING,&settings.monitorfile,0,"File containing validation dataset","FILE"},
    {"logfile-base",'l',POPT_ARG_STRING,&settings.logbase,0,"Base filename for logfiles","FILE"},
    {"output-file",'o',POPT_ARG_STRING,&settings.outputfile,0,"Filename for storing autoencoder","FILE"},
    POPT_TABLEEND};

  struct poptOption networkOptionsTable[] = { //Anything dealing with overall network architecture
    {"stdev-init",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.stdev_init,0,"Standard deviation of Gaussian distribution to use when initialising network weights","STDEV"},
    {"f0",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.f0,0,"Lower bound of logistic function","F0"},
    {"f1",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.f1,0,"Upper bound of logistic function","F1"},
    POPT_TABLEEND};

  struct poptOption experimentalOptionsTable[] = { //A home for options under developement. Presumably these will eventually move elsewhere, or be removed
    {"force-encodings",'\0',POPT_ARG_STRING,&settings.force_file,0,"File containing start of all encodings","FILE"},
    {"force-enc-weight",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.enc_force_weight,0,"Weight to attach to force-encodings term in error function"},
    {"cost-weight",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.costwt,0,"Weight given to cost term during training","WEIGHT"},
    POPT_TABLEEND};

  struct poptOption rpropOptionsTable[] = { //RProp family options (PK)
    {"rprop",'\0',POPT_ARG_NONE,&settings.rprop_flag,0,"Use iRprop+ trainer. If set, eta0, Imax and lrate-auto are ignored",NULL},
    {"etap",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.etap,0,"RProp: Amount by which step-size is increased when following gradient","ETA"},
    {"etam",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.etam,0,"RProp:Amount by which step-size is decreased when overstepping","ETA"},
    {"delta0",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.delta0,0,"Rprop: Initial step-size","DELTA"},
    {"delta-max",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.delta_max,0,"Rprop: Maximum step-size","DELTA"},
    {"delta-min",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.delta_min,0,"Rprop: Minimum step-size","DELTA"},
    {"no-wb",'\0',POPT_ARG_NONE,&settings.no_wb,0,"Don't use weight-backtracking (Rprop-).",NULL},    
    POPT_TABLEEND};

  // Now construct main program options table
  struct poptOption optionsTable[] = {
    {"niter-auto",'a',POPT_ARG_INT|POPT_ARGFLAG_SHOW_DEFAULT,&settings.niter_auto,0,"(Maximum) number of iterations of autoencoder training to perform","NITER"},
    {"noise-auto",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.noise_auto,0,"Standard deviation of Gaussian noise to use during autoencoder training step","STDEV"},
    {"lrate-auto",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.lrate_auto,0,"Learning rate parameter for autoencoder training","RATE"},
    {"eta0",'\0',POPT_ARG_DOUBLE|POPT_ARGFLAG_SHOW_DEFAULT,&settings.eta0,0,"Base for adaptive learning rate","ETA"},
    {"Imax",'\0',POPT_ARG_INT|POPT_ARGFLAG_SHOW_DEFAULT,&settings.N_adapt_rate,0,"maximum number of learning rate increments","I"},
    {"resume",'\0',POPT_ARG_STRING,&statefile,0,"Resume previous training","STATE-FILE"},
    {"num-procs",'\0',POPT_ARG_INT|POPT_ARGFLAG_SHOW_DEFAULT,&settings.num_procs,0,"Number of processors to use (if 0, read value from OMP_NUM_THREADS environment variable)"},
    {NULL,'\0',POPT_ARG_INCLUDE_TABLE,&fileOptionsTable,0,"Options controlling input and output files",NULL},
    {NULL,'\0',POPT_ARG_INCLUDE_TABLE,&networkOptionsTable,0,"Options governing network architecture",NULL},
    {NULL,'\0',POPT_ARG_INCLUDE_TABLE,&crbmOptionsTable,0,"Options controlling CRBM pretraining",NULL},
    {NULL,'\0',POPT_ARG_INCLUDE_TABLE,&rpropOptionsTable,0,"RProp trainer options",NULL},
    {NULL,'\0',POPT_ARG_INCLUDE_TABLE,&experimentalOptionsTable,0,"Experimental options; performance not guaranteed! Use at your own risk...",NULL},
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
    fprint_state(settings,NULL);
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
    // Read network dimensions from command line and allocate data structures
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
      FILE * fpsettings=fopen("make_autoencoder.settings","w");
      fprint_state(settings,fpsettings);
      fclose(fpsettings);
    }
    
  }  

  // And call the main worker routine
  a=autoencoder_make_and_train(crbms,a,settings);

}
