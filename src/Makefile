BINDIR=../bin
OBJDIR=../obj
LIBDIR=../lib
MODDIR=../mod

CCOMPILE=gcc -c -O3 -DNATIVE_RAND
CLINK=gcc -O3 -L../lib -L/opt/local/lib

LAPACK_COMPILE=-I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers/
LAPACK_LINK=-L/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/ -lblas

.PHONY: all clean ah2dat autoencoder auto_decode auto_encode auto_encdec auto_sensitivity auto_train convert_dataset_format cnet2dat cnet2txt make_autoencoder net2basis auto_predict merge_datasets print_state merge_dat



#Aliases
all:  auto_decode auto_encode auto_encdec auto_sensitivity convert_dataset_format cnet2dat cnet2txt make_autoencoder net2basis merge_datasets print_state merge_dat
clean:
	rm $(BINDIR)/*
	rm $(OBJDIR)/*
	rm $(MODDIR)/*



#ah2dat: $(BINDIR)/ah2dat
autoencoder: $(OBJDIR)/autoencoder.o
auto_decode: $(BINDIR)/auto_decode
auto_encode: $(BINDIR)/auto_encode
auto_encdec: $(BINDIR)/auto_encdec
auto_predict: $(BINDIR)/auto_predict
auto_sensitivity: $(BINDIR)/auto_sensitivity
auto_train: $(BINDIR)/auto_train
convert_dataset_format: $(BINDIR)/convert_dataset_format
cnet2dat: $(BINDIR)/cnet2dat
cnet2txt: $(BINDIR)/cnet2txt
make_autoencoder: $(BINDIR)/make_autoencoder
merge_datasets: $(BINDIR)/merge_datasets
merge_dat: $(BINDIR)/merge_dat
net2basis: $(BINDIR)/net2basis
print_state: $(BINDIR)/print_state

#Binaries
#$(BINDIR)/ah2dat: $(OBJDIR)/autoencoder.o $(OBJDIR)/ah2dat.o
#	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -L$(LIBDIR) -lah64 -ltwist
$(BINDIR)/auto_decode: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_decode.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/auto_encode: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_encode.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/auto_encdec: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_encdec.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/auto_predict: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_predict.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK)
$(BINDIR)/auto_sensitivity: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_sensitivity.o
	$(CLINK) $+ -o $@  $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/auto_train: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_train.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -lpopt -ltwist
$(BINDIR)/convert_dataset_format: $(OBJDIR)/autoencoder.o $(OBJDIR)/convert_dataset_format.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/cnet2dat: $(OBJDIR)/autoencoder.o $(OBJDIR)/cnet2dat.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/cnet2txt: $(OBJDIR)/autoencoder.o $(OBJDIR)/cnet2txt.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/make_autoencoder: $(OBJDIR)/autoencoder.o $(OBJDIR)/make_autoencoder.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -lpopt -ltwist
$(BINDIR)/merge_datasets: $(OBJDIR)/autoencoder.o $(OBJDIR)/merge_datasets.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -lpopt -ltwist
$(BINDIR)/merge_dat: $(OBJDIR)/autoencoder.o $(OBJDIR)/merge_dat.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/net2basis: $(OBJDIR)/autoencoder.o $(OBJDIR)/net2basis.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist
$(BINDIR)/print_state: $(OBJDIR)/autoencoder.o $(OBJDIR)/print_state.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(LAPACK_LINK) -ltwist

#Objects
$(OBJDIR)/ah2dat.o: ah2dat.c autoencoder.h
	$(CCOMPILE) $(LAPACK_COMPILE) ah2dat.c -o $@
$(OBJDIR)/autoencoder.o: autoencoder.c autoencoder.h
	$(CCOMPILE) $(LAPACK_COMPILE) autoencoder.c -o $@
$(OBJDIR)/auto_decode.o: auto_decode.c autoencoder.h
	$(CCOMPILE) auto_decode.c -o $@
$(OBJDIR)/auto_encode.o: auto_encode.c autoencoder.h
	$(CCOMPILE) auto_encode.c -o $@
$(OBJDIR)/auto_encdec.o: auto_encdec.c autoencoder.h
	$(CCOMPILE) auto_encdec.c -o $@
$(OBDJIR)/auto_predict.o: auto_predict.c autoencoder.h
	$(CCOMPILE) auto_predict.c -o $@
$(OBJDIR)/auto_sensitivity.o: auto_sensitivity.c autoencoder.h
	$(CCOMPILE) $(LAPACK_COMPILE) auto_sensitivity.c -o $@
$(OBJDIR)/auto_train.o: auto_train.c autoencoder.h
	$(CCOMPILE) auto_train.c -o $@
$(OBJDIR)/convert_dataset_format.o: convert_dataset_format.c autoencoder.h
	$(CCOMPILE) convert_dataset_format.c -o $@
$(OBJDIR)/cnet2dat.o: cnet2dat.c autoencoder.h
	$(CCOMPILE) cnet2dat.c -o $@
$(OBJDIR)/cnet2txt.o: cnet2txt.c autoencoder.h
	$(CCOMPILE) cnet2txt.c -o $@
$(OBJDIR)/make_autoencoder.o: make_autoencoder.c autoencoder.h
	$(CCOMPILE) make_autoencoder.c -o $@
$(OBJDIR)/merge_datasets.o: merge_datasets.c autoencoder.h
	$(CCOMPILE) merge_datasets.c -o $@
$(OBJDIR)/merge_dat.o: merge_dat.c autoencoder.h
	$(CCOMPILE) merge_dat.c -o $@
$(OBJDIR)/net2basis.o: net2basis.c
	$(CCOMPILE) $(LAPACK_COMPILE) $+ -o $@
$(OBJDIR)/print_state.o: print_state.c
	$(CCOMPILE) $(LAPACK_COMPILE) $+ -o $@
