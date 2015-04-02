include $(MAKEDEFS)

.PHONY: all libtwist mt19937ar mt19937ar_test twisttools
all: libtwist mt19937ar mt19937ar_test twisttools

mt19937ar_test: $(BINDIR)/mt19937ar_test

$(BINDIR)/mt19937ar_test: $(OBJDIR)/mt19937ar.o $(OBJDIR)/mt19937ar_test.o
	$(CLINK) -o $@ $+

mt19937ar: $(OBJDIR)/mt19937ar.o

libtwist: $(LIBDIR)/libtwist.a

$(LIBDIR)/libtwist.a: $(OBJDIR)/mt19937ar.o $(OBJDIR)/twisttools.o
	ar rcs $@ $+

$(OBJDIR)/mt19937ar.o: mt19937ar.c mt19937ar.h
	$(CCOMPILE) mt19937ar.c -o $@
$(OBJDIR)/mt19937ar_test.o: mt19937ar_test.c mt19937ar.h
	$(CCOMPILE) mt19937ar_test.c -o $@
$(OBJDIR)/twisttools.o: twisttools.c mt19937ar.h
	$(CCOMPILE) twisttools.c -o $@

