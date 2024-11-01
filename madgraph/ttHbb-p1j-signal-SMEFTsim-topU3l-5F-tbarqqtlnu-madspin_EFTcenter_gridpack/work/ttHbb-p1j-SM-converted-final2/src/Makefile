LIBDIR=../lib
CXXFLAGS= -std=c++17 -Ofast -I. -fPIC

target=$(LIBDIR)/libmodel_sm.so

all: $(target)

objects=HelAmps_sm.o Parameters_sm.o rambo.o read_slha.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(target): $(objects)
	$(CXX) -shared -o $@ $(objects) 

.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(objects)
