PYTHON_VERSION := $(shell python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYBIND11_INCLUDE := $(shell python -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE := $(shell python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
NUMPY_INCLUDE := $(shell python -c "import numpy; print(numpy.get_include())")
CUSTOMLIB_INCLUDE = -I../SubProcesses -I../src

CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC  -I$(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) $(CUSTOMLIB_INCLUDE)

PROCESSLIB = /afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow/madgraph/ttHbb-p1j-signal-SMEFTsim-topU3l-5F-tbarqqtlnu-madspin_EFTcenter_gridpack/work/ttHbb-p1j-quad-ctHre-converted-final2/lib

LDFLAGS = -lSMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless -lmodel_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless -L$(PROCESSLIB)  -Wl,-rpath,$(shell python -c "import site; print(site.getsitepackages()[0])") -Wl,-rpath,$(PROCESSLIB)

TARGET = mem_eval.so 

all: $(TARGET)

$(TARGET): mem_eval.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf *.so
