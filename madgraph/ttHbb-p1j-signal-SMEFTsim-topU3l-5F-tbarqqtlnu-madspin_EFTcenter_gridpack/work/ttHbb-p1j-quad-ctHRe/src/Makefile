LIBDIR=../lib
CXXFLAGS= -Ofast -I.

target=$(LIBDIR)/libmodel_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.a

all: $(target)

objects=HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.o Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.o rambo.o read_slha.o

$(target): $(objects)
	$(AR) cru $@ $(objects)
	ranlib $(target)

.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(objects)

