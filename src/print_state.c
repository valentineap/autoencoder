#include "autoencoder.h"
int main(int argc,char *argv[]){
  char * statefile=argv[1];
  char * storefile;
  state_settings_t s=load_state(statefile, &storefile);
  fprint_state(s,NULL);
}
