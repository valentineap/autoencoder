#include "stdio.h"
#include "string.h"
int main(int argc, char *argv[]){
  union{
    int i;
    char x[4];
  } u;
  printf("%i\n",sizeof(int));
  memcpy(&u.x[0],argv[1],4);
  printf("%i\n",u.i);
  u.i=0;
  printf("|%s|\n",u.x);
  
}
