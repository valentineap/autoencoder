
#define AH_STA_CODE 0
#define AH_STA_NAME 4
#define AH_CHA_CODE 12
#define AH_CHA_NAME 16
#define AH_TYP_CODE 24
#define AH_TYP_NAME 28
#define AH_STA_LOC  36
#define AH_INST_RES 48
#define AH_EV_LOC   536
#define AH_EV_DATE  548
#define AH_L_EV_COM 572
#define AH_EV_COM   576
#define AH_INST_TYP 656
#define AH_NPOINTS  660
#define AH_SMPINT   664
#define AH_MAX_AMP  668
#define AH_STA_DATE 672
#define AH_L_DATCOM 700
#define AH_DATCOM   704
#define AH_L_AHLOG  784
#define AH_AHLOG    788
#define AH_L_EXTRAS 992
#define AH_EXTRAS   996
#define AH_L_HEADER 1080 

#define AH_N_PAZ 29
#define AH_N_AHLOG 51
#define AH_N_AHCOM 20
#define AH_N_EXTRAS 21


#define FORTRAN_REAL float
#define FORTRAN_INTEGER int
#define FORTRAN_DOUBLE double

#define FP_TYPE_4B float
#define IN_TYPE_4B int

typedef struct ah_location{
  FP_TYPE_4B lat;
  FP_TYPE_4B lon;
  FP_TYPE_4B z;
} ah_location_t;

typedef struct ah_time{
  IN_TYPE_4B day;
  IN_TYPE_4B month;
  IN_TYPE_4B year;
  IN_TYPE_4B hours;
  IN_TYPE_4B minutes;
  FP_TYPE_4B seconds;
} ah_time_t;

typedef struct ah_response{
  FP_TYPE_4B digsen;
  FP_TYPE_4B a0;
  FP_TYPE_4B npoles;
  FP_TYPE_4B dum1;
  FP_TYPE_4B nzeros;
  FP_TYPE_4B dum2;
  FP_TYPE_4B paz[4*AH_N_PAZ];
}ah_response_t;

typedef struct ah_header{
  IN_TYPE_4B stacode;
  char staname[8];
  IN_TYPE_4B chacode;
  char chaname[8];
  IN_TYPE_4B typcode;
  char typname[8];
  ah_location_t staloc;
  ah_response_t response;
  ah_location_t evloc;
  ah_time_t evdate;
  IN_TYPE_4B levcom;
  char evcom[4*AH_N_AHCOM];
  IN_TYPE_4B ityp;
  IN_TYPE_4B ndt;
  FP_TYPE_4B del;
  FP_TYPE_4B amax;
  ah_time_t stadate;
  IN_TYPE_4B nl3;
  IN_TYPE_4B ldcom;
  char datacom[4*AH_N_AHCOM];
  IN_TYPE_4B lahlog;
  char ahlog[4*AH_N_AHLOG];
  IN_TYPE_4B nxtras;
  IN_TYPE_4B xtras[AH_N_EXTRAS];
} ah_header_t;

typedef struct ah_fortran_location{
  FORTRAN_REAL lat;
  FORTRAN_REAL lon;
  FORTRAN_REAL z;
} ah_fortran_location_t;

typedef struct ah_fortran_time{
  FORTRAN_INTEGER day;
  FORTRAN_INTEGER month;
  FORTRAN_INTEGER year;
  FORTRAN_INTEGER hours;
  FORTRAN_INTEGER minutes;
  FORTRAN_REAL seconds;
} ah_fortran_time_t;

typedef struct ah_fortran_response{
  FORTRAN_REAL digsen;
  FORTRAN_REAL a0;
  FORTRAN_REAL npoles;
  FORTRAN_REAL dum1;
  FORTRAN_REAL nzeros;
  FORTRAN_REAL dum2;
  FORTRAN_REAL paz[4*AH_N_PAZ];
}ah_fortran_response_t;

typedef struct ah_fortran_header{
  FORTRAN_INTEGER stacode;
  char staname[8];
  FORTRAN_INTEGER chacode;
  char chaname[8];
  FORTRAN_INTEGER typcode;
  char typname[8];
  ah_fortran_location_t staloc;
  ah_fortran_response_t response;
  ah_fortran_location_t evloc;
  ah_fortran_time_t evdate;
  FORTRAN_INTEGER levcom;
  char evcom[4*AH_N_AHCOM];
  FORTRAN_INTEGER ityp;
  FORTRAN_INTEGER ndt;
  FORTRAN_REAL del;
  FORTRAN_REAL amax;
  ah_fortran_time_t stadate;
  FORTRAN_INTEGER nl3;
  FORTRAN_INTEGER ldcom;
  char datacom[4*AH_N_AHCOM];
  FORTRAN_INTEGER lahlog;
  char ahlog[4*AH_N_AHLOG];
  FORTRAN_INTEGER nxtras;
  FORTRAN_INTEGER xtras[AH_N_EXTRAS];
} ah_fortran_header_t;

inline void byswap4(char *,int);
void c_check_type_size(FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void buffswap(char *);
void c_openfile_(FORTRAN_INTEGER *,char *,FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_closefile_(FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_readheader_(FORTRAN_INTEGER *,void *,FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_readdata_(FORTRAN_INTEGER *,void *,FORTRAN_INTEGER *,FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_writeheader_(FORTRAN_INTEGER *,void *,FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_writedata_(FORTRAN_INTEGER *,void *,FORTRAN_INTEGER *,FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_countrecs_(FORTRAN_INTEGER *,FORTRAN_INTEGER *,FORTRAN_INTEGER *);
void c_byteswap_(FORTRAN_INTEGER *,FORTRAN_INTEGER *);

