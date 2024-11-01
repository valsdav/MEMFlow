LIBDIR=../lib
CXXFLAGS= -std=c++17 -Ofast -I. -fPIC

target=$(LIBDIR)/libmodel_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.so

all: $(target)

objects=HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.o Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.o rambo.o read_slha.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(target): $(objects)
	$(CXX) -shared -o $@ $(objects) 

.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(objects)
