#include "mt19937ar.h"
#include<stdio.h>

void dump_twister_state(FILE *fp){
  fwrite(&mt[0],sizeof(unsigned long),N,fp);
  fwrite(&mti,sizeof(int),1,fp);
}

void load_twister_state(FILE *fp){
  fread(&mt[0],sizeof(unsigned long),N,fp);
  fread(&mti,sizeof(int),1,fp);
}
