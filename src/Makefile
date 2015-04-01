include $(MAKEDEFS)

.PHONY: all ah2dat autoencoder auto_decode auto_encode auto_sensitivity auto_train cnet2txt make_autoencoder net2basis

#Aliases
all: ah2dat auto_decode auto_encode auto_sensitivity cnet2txt make_autoencoder net2basis

ah2dat: $(BINDIR)/ah2dat
autoencoder: $(OBJDIR)/autoencoder.o
auto_decode: $(BINDIR)/auto_decode
auto_encode: $(BINDIR)/auto_encode
auto_sensitivity: $(BINDIR)/auto_sensitivity
auto_train: $(BINDIR)/auto_train
cnet2txt: $(BINDIR)/cnet2txt
make_autoencoder: $(BINDIR)/make_autoencoder
net2basis: $(BINDIR)/net2basis

#Binaries
$(BINDIR)/ah2dat: $(OBJDIR)/autoencoder.o $(OBJDIR)/ah2dat.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -lah64 -ltwist
$(BINDIR)/auto_decode: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_decode.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -ltwist
$(BINDIR)/auto_encode: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_encode.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -ltwist
$(BINDIR)/auto_sensitivity: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_sensitivity.o
	$(CLINK) $+ -o $@  $(CMATHLIBS) $(CATLAS) -ltwist
$(BINDIR)/auto_train: $(OBJDIR)/autoencoder.o $(OBJDIR)/auto_train.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -lpopt -ltwist
$(BINDIR)/cnet2txt: $(OBJDIR)/autoencoder.o $(OBJDIR)/cnet2txt.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -ltwist
$(BINDIR)/make_autoencoder: $(OBJDIR)/autoencoder.o $(OBJDIR)/make_autoencoder.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -lpopt -ltwist
$(BINDIR)/net2basis: $(OBJDIR)/autoencoder.o $(OBJDIR)/net2basis.o
	$(CLINK) $+ -o $@ $(CMATHLIBS) $(CATLAS) -ltwist

#Objects
$(OBJDIR)/ah2dat.o: ah2dat.c autoencoder.h
	$(CCOMPILE) $(ATLAS_INC) ah2dat.c -o $@
$(OBJDIR)/autoencoder.o: autoencoder.c autoencoder.h
	$(CCOMPILE) $(ATLAS_INC) autoencoder.c -o $@
$(OBJDIR)/auto_decode.o: auto_decode.c autoencoder.h
	$(CCOMPILE) auto_decode.c -o $@
$(OBJDIR)/auto_encode.o: auto_encode.c autoencoder.h
	$(CCOMPILE) auto_encode.c -o $@
$(OBJDIR)/auto_sensitivity.o: auto_sensitivity.c autoencoder.h
	$(CCOMPILE) $(ATLAS_INC) auto_sensitivity.c -o $@
$(OBJDIR)/auto_train.o: auto_train.c autoencoder.h
	$(CCOMPILE) auto_train.c -o $@
$(OBJDIR)/cnet2txt.o: cnet2txt.c autoencoder.h
	$(CCOMPILE) cnet2txt.c -o $@
$(OBJDIR)/make_autoencoder.o: make_autoencoder.c autoencoder.h
	$(CCOMPILE) make_autoencoder.c -o $@
$(OBJDIR)/net2basis.o: net2basis.c
	$(CCOMPILE) $(ATLAS_INC) $+ -o $@




