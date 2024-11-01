import os
from os import path
import argparse
# Read all the subprocesses from the folders and create a new folder with all the files renamed with the process of the file.
# Prepare a new makefile

parser = argparse.ArgumentParser(description='Convert folder')
parser.add_argument('-i','--input', type=str, help='Folder to convert')
parser.add_argument('-o','--output', type=str, help='Folder to save the converted files')
parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of the output folder')
parser.add_argument('-m', '--model_name', type=str, help='Model name')
parser.add_argument("-ln", '--lib-name',type=str, help='Library name')
args = parser.parse_args()

input_folder = args.input
output_folder = args.output
# current directory
basedir = os.path.dirname(os.path.abspath(__file__))

if not path.exists(output_folder):
    os.makedirs(output_folder)
    os.makedirs(output_folder+"/SubProcesses")
else:
    if args.force:
        print("Folder already exists. Overwriting")
    else:
        print("Folder already exists")
        exit(1)

process_names = []
  
for root, dirs, files in os.walk(input_folder+"/SubProcesses"):
    for dirname in dirs:
        splits = dirname.split("_")
        process_name = splits[-2]+"_"+splits[-1]
        basename = "_".join(splits[:-2])
        print(f"Process name: {process_name} Basename: {basename}")
        process_names.append(process_name)

        with open(input_folder+"/SubProcesses/"+dirname+"/CPPProcess.h", "r") as f:
            lines = f.read()
            lines = lines.replace("CPPProcess", process_name)
            # Setup the base class
            lines = lines.replace(f"class {process_name}", f'#include "baseclass.h"\n\nclass {process_name} : public ProcessClass')
            # Replace the constructor
            lines = lines.replace(f"{process_name}() {{}}", f'{process_name}(): ProcessClass("{process_name}") {{}};')
            
            with open(output_folder+"/SubProcesses/"+process_name +".h", "w") as f:
                f.write(lines)
        with open(input_folder+"/SubProcesses/"+dirname+"/CPPProcess.cc", "r") as f:
            lines = f.read()
            lines = lines.replace("CPPProcess", process_name)
            with open(output_folder+"/SubProcesses/"+process_name +".cc", "w") as f:
                f.write(lines)

# Create the base class
baseclass = """#ifndef __baseclass_h
#define __baseclass_h

#include <string>
#include <vector>

class ProcessClass {
public:
  ProcessClass(std::string process_name);
  ~ProcessClass();
  
  virtual void initProc(std::string param_card_name) = 0;
  virtual void sigmaKin() = 0;
  virtual double sigmaHat() = 0;
  virtual std::string name() const = 0;
  virtual int code() const = 0;
  virtual const std::vector<double> & getMasses() const = 0;
  virtual std::vector < double * > getMomenta() = 0;
  virtual void setMomenta(std::vector < double * > & momenta) = 0;
  virtual void setInitial(int inid1, int inid2) = 0;
  virtual const double * getMatrixElements() const = 0;
  std::string getProcessName() const;
private:
  std::string _process_name;
};

#endif // __baseclass_h
"""
with open(output_folder+"/SubProcesses/baseclass.h", "w") as f:
    f.write(baseclass)

baseclass_cc = """#include "baseclass.h"

ProcessClass::ProcessClass(std::string process_name) : _process_name(process_name){}

ProcessClass::~ProcessClass(){}

std::string ProcessClass::getProcessName() const{
  return _process_name;
}
"""
with open(output_folder+"/SubProcesses/baseclass.cc", "w") as f:
    f.write(baseclass_cc)



# Create the processes.h
processes_h = """#ifndef __processes_h
#define __processes_h"""
for process in process_names:
    processes_h += f'\n#include "{process}.h"'

# We add a function to get a vector of initialized processes
processes_h += "\n#include <vector>\n"
processes_h += """std::vector<ProcessClass*> getProcesses(std::string param_card){
        std::vector<ProcessClass*> processes;\n"""
for i, process in enumerate(process_names):
    processes_h += f"""\n{process} * process_{i} = new {process}();
process_{i}->initProc(param_card);
processes.push_back(process_{i});"""

processes_h += "\nreturn processes;\n}"
processes_h += "\n#endif // __processes_h"
    
with open(output_folder+"/SubProcesses/processes.h", "w") as f:
    f.write(processes_h)
    

makefile = f"""LIBDIR=lib
INCDIR=src
MODELLIB=model_{args.model_name}
CXXFLAGS= -std=c++17 -Ofast -I$(INCDIR) -I. -fPIC
LIBFLAGS= -L$(LIBDIR) -l$(MODELLIB)

SRCDIR=SubProcesses

# Automatically find all .cc files in the directory
sources=$(wildcard $(SRCDIR)/*.cc)

# Generate object file names by replacing .cc with .o
objects=$(sources:.cc=.o)

# Rule to compile .cc to .o
%.o: %.cc  $(LIBDIR)/lib$(MODELLIB).so
\t$(CXX) $(CXXFLAGS) -c $< -o $@ $(LIBFLAGS) 

# Target to build the shared library
sharedlib=$(LIBDIR)/lib{args.lib_name}.so

# Compile the main target with shared library
$(main): $(objects) $(sharedlib) $(LIBDIR)/lib$(MODELLIB).a
\t$(CXX) -o $@ $(objects) $(LIBFLAGS) -Wl,-rpath,$(LIBDIR)

# Build the shared library
$(sharedlib): $(objects) $(LIBDIR)/lib$(MODELLIB).so
\t$(CXX) -shared -o $@ $(objects) $(LIBFLAGS) -Wl,-rpath,$(LIBDIR)

# If the static library is still needed
$(LIBDIR)/lib$(MODELLIB).so:
\tcd src && make

.PHONY: clean

python: $(sharedlib)
\tcd python && make

clean:
\trm -f $(main)
\trm -f $(objects)
\trm -f $(LIBDIR)/*
\trm -f src/*.o

"""
with open(output_folder+"/Makefile", "w") as f:
    f.write(makefile)

# Make file for src folder

#Copy the full src folder
os.system(f"cp -r {input_folder}/src {output_folder}/src")
src_makefile = f"""LIBDIR=../lib
CXXFLAGS= -std=c++17 -Ofast -I. -fPIC

target=$(LIBDIR)/libmodel_{args.model_name}.so

all: $(target)

objects=HelAmps_{args.model_name}.o Parameters_{args.model_name}.o rambo.o read_slha.o

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(target): $(objects)
	$(CXX) -shared -o $@ $(objects) 

.PHONY: clean

clean:
	rm -f $(target)
	rm -f $(objects)
"""
with open(output_folder+"/src/Makefile", "w") as f:
    f.write(src_makefile)

               
os.system(f"cp -r {input_folder}/Cards {output_folder}/Cards")
os.makedirs(output_folder+"/lib", exist_ok=True)
os.makedirs(output_folder+"/python", exist_ok=True)
# mem_eval copy from template
os.system(f"cp mem_eval_template.cc {output_folder}/python/mem_eval.cc")

# Makefile for mem_eval
mem_eval_makefile = f"""PYTHON_VERSION := $(shell python -c "import sys; print(f'{{sys.version_info.major}}.{{sys.version_info.minor}}')")
PYBIND11_INCLUDE := $(shell python -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE := $(shell python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
NUMPY_INCLUDE := $(shell python -c "import numpy; print(numpy.get_include())")
CUSTOMLIB_INCLUDE = -I../SubProcesses -I../src

CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC  -I$(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) $(CUSTOMLIB_INCLUDE)

PROCESSLIB = {basedir}/{output_folder}/lib

LDFLAGS = -l{args.lib_name} -lmodel_{args.model_name} -L$(PROCESSLIB)  -Wl,-rpath,$(shell python -c "import site; print(site.getsitepackages()[0])") -Wl,-rpath,$(PROCESSLIB)

TARGET = mem_eval.so 

all: $(TARGET)

$(TARGET): mem_eval.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -rf *.so
"""
with open(output_folder+"/python/Makefile", "w") as f:
    f.write(mem_eval_makefile)
