LIBDIR=lib
INCDIR=src
MODELLIB=model_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless
CXXFLAGS= -std=c++17 -Ofast -I$(INCDIR) -I. -fPIC
LIBFLAGS= -L$(LIBDIR) -l$(MODELLIB)

SRCDIR=SubProcesses

# Automatically find all .cc files in the directory
sources=$(wildcard $(SRCDIR)/*.cc)

# Generate object file names by replacing .cc with .o
objects=$(sources:.cc=.o)

# Rule to compile .cc to .o
%.o: %.cc  $(LIBDIR)/lib$(MODELLIB).so
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBFLAGS) 

# Target to build the shared library
sharedlib=$(LIBDIR)/libSMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.so

# Compile the main target with shared library
$(main): $(objects) $(sharedlib) $(LIBDIR)/lib$(MODELLIB).a
	$(CXX) -o $@ $(objects) $(LIBFLAGS) -Wl,-rpath,$(LIBDIR)

# Build the shared library
$(sharedlib): $(objects) $(LIBDIR)/lib$(MODELLIB).so
	$(CXX) -shared -o $@ $(objects) $(LIBFLAGS) -Wl,-rpath,$(LIBDIR)

# If the static library is still needed
$(LIBDIR)/lib$(MODELLIB).so:
	cd src && make

.PHONY: clean

python: $(sharedlib)
	cd python && make

clean:
	rm -f $(main)
	rm -f $(objects)
	rm -f $(LIBDIR)/*
	rm -f src/*.o

