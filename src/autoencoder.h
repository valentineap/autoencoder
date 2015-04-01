#ifndef _autoencoder_h_
#define _autoencoder_h_
#include <stdio.h>

// There is relatively little point in saving system state during CRBM training - each layer of CRBMs depends entirely on the
// result of training the previous layer, so it is not possible to 'extend' the training.
// Only possibility is to dump system state at end of first CRBM layer, so that this can be taken further?
typedef struct state_settings {
  int niter_crbm;
  int niter_auto;
  double stdev_init;
  double noise_crbm;
  double noise_auto;
  double lrate_crbm;
  double lrate_auto;
  double f0;
  double f1;
  char * datafile;
  char * monitorfile;
  char * outputfile;
  char * logbase;
  int ncrbms;
  int *sizes;
  int stage;
  int iter;
  int i_adapt_rate;
  double eta0;
  int N_adapt_rate;
  int N_reduced;
  int i_red_sum;
  int iter_reset;
  int ignore_weights;
  double costwt;
} state_settings_t;


typedef struct dataset {
  int npoints;
  int nrecs;
  double *data;
  double *scale;
  double *weight;
} dataset_t;

typedef struct crbm {
  int nlv;
  int nlh;
  double * w;
  double * b_v2h;
  double * b_h2v;
  double * a_v2h;
  double * a_h2v;
  double loglo;
  double logup;
} crbm_t;

typedef struct layer {
  int nin;
  int nout;
  double loglo;
  double logup;
  double *w;
  double *a;
  double *b;
} layer_t;

typedef struct autoenc {
  int nlayers;
  layer_t layers[];
} autoenc_t;

typedef struct node_val {
  int n;
  double * values;
  double * derivs;
} node_val_t;



///////////////////////
// Logistic function //
///////////////////////
#define LOGISTIC(x) ((double)(1.L/(1.L+expl(-(long double)(x)))))


/////////////////////////
// Function prototypes //
/////////////////////////
// Dataset functions
void dataset_alloc(dataset_t *,int,int);
void dataset_free(dataset_t *);
void dataset_copy(dataset_t *,dataset_t *);
void dataset_resize(dataset_t *,int);
dataset_t load_dataset(char *);
void writedata(char *,dataset_t *);
// Random number functions
inline double random_normal();
//CRBM functions
void crbm_init(crbm_t *,int,int,double,double,double,FILE *);
void crbm_free(crbm_t *);
void crbm_encode(crbm_t *,dataset_t *,dataset_t *,double);
void crbm_decode(crbm_t *,dataset_t *,dataset_t *,double);
void crbm_train(crbm_t *,dataset_t *,state_settings_t *,FILE *);
void write_crbm(FILE *,crbm_t *);
void save_crbm(char *,crbm_t *);
crbm_t read_crbm(FILE *);
crbm_t load_crbm(char *);
// Autoencoder functions
autoenc_t * make_autoencoder(int,crbm_t *,FILE *);
void write_autoencoder(FILE *,autoenc_t *);
void save_autoencoder(char *,autoenc_t *);
autoenc_t *read_autoencoder(FILE *);
autoenc_t *load_autoencoder(char *);
void autoencoder_free(autoenc_t *);
void autoencoder_encode(autoenc_t *,dataset_t *,dataset_t *);
void autoencoder_decode(autoenc_t *,dataset_t *,dataset_t *);
void autoencoder_encdec(autoenc_t *,node_val_t *,int);
void  autoencoder_batchtrain(autoenc_t *,dataset_t*,state_settings_t *,dataset_t *,FILE *);
// Miscellaneous functions
void make_nodevals(autoenc_t *,node_val_t *,int);
void nodevals_free(autoenc_t *,node_val_t *);
double error_dataset(dataset_t *,dataset_t *);
double error_array(double *,double *,int,int,double *);
double len_enc_array(double *,int,int,double *);
void abort_handler(int);
void write_state(char *,state_settings_t,char *);
state_settings_t load_state(char *,char **);
autoenc_t * autoencoder_make_and_train(crbm_t *,autoenc_t *,state_settings_t);
#endif
