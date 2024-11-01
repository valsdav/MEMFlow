// Load all the subprocesses
#include "processes.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <chrono>

namespace py = pybind11;

using input_array_t = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr std::array<std::pair<int,int>, 31> ids = {
        std::make_pair(21, 21),

	std::make_pair(1, -1),
	std::make_pair(2, -2),
	std::make_pair(3, -3),
	std::make_pair(4, -4),
	std::make_pair(5, -5),

	std::make_pair(-1, 1),
	std::make_pair(-2, 2),
	std::make_pair(-3, 3),
	std::make_pair(-4, 4),
	std::make_pair(-5, 5),
	
	std::make_pair(1, 21),
	std::make_pair(2, 21),
	std::make_pair(3, 21),
	std::make_pair(4, 21),
	std::make_pair(5, 21),

	std::make_pair(21, 1),
	std::make_pair(21, 2),
	std::make_pair(21, 3),
	std::make_pair(21, 4),
	std::make_pair(21, 5),

	std::make_pair(-1, 21),
	std::make_pair(-2, 21),
	std::make_pair(-3, 21),
	std::make_pair(-4, 21),
	std::make_pair(-5, 21),

	std::make_pair(21, -1),
	std::make_pair(21, -2),
	std::make_pair(21, -3),
	std::make_pair(21, -4),
	std::make_pair(21, -5),
};
constexpr int nprocesses = ids.size();

std::array<double, nprocesses> call_madgraph(std::vector<double*> & momenta,
		   const std::vector<ProcessClass*> & processes){

  // Loop over moments
  for (auto & process : processes) {
    // Call the matrix element
    process->setMomenta(momenta);
    // Compute the matrix element
    process->sigmaKin();
  }
  // Now storing the result
  
  std::array<double, nprocesses> results;

  int z = 0;
  for (auto & [i, j] : ids) {
    for (auto & process : processes) {
	process->setInitial(i, j);
	//std::cout << "Initial particles: " << i << " " << j << " : " << process->sigmaHat() << std::endl;
	double M = process->sigmaHat();
	if (M != 0) {
	  results[z++] = M;
	  break;
	}
    }
  }
  return results;
  
}

// Dummy computation function that processes a 2D slice
py::array_t<double> compute_mem(input_array_t array) {
        auto buffer = array.request();
        double* ptr = static_cast<double*>(buffer.ptr);

        long int shape[3] = {buffer.shape[0], buffer.shape[1], buffer.shape[2]};
	long unsigned int strides[3] = {
	  buffer.strides[0]/sizeof(double),
	  buffer.strides[1]/sizeof(double),
	  buffer.strides[2]/sizeof(double)};

	std::cout << "Shape: " << shape[0] << " " << shape[1] << " " << shape[2] << std::endl;
	std::cout << "Strides: " << strides[0] << " " << strides[1] << " " << strides[2] << std::endl;

	// Initialize the processes
	auto processes = getProcesses("../Cards/param_card.dat");

	auto start_time = std::chrono::high_resolution_clock::now();
	// Collect the results
	double* results = new double[shape[0] * nprocesses];
	
	// Process each slice along the inner dimension
	for (auto i = 0; i < shape[0]; ++i) {
	  //std::cout << "============\nEvent: " << i << std::endl;
	  std::vector<double*> momenta;

	  double* event_ptr = ptr + i * strides[0];
	  for (auto j = 0; j < shape[1]; ++j) {
	    double* particle_ptr = event_ptr + j * strides[1];
	    momenta.push_back(particle_ptr);
	  }

	  // Call MadGraph
	  auto out = call_madgraph(momenta, processes);
	  for (auto j = 0; j < nprocesses; ++j) {
	    results[i * nprocesses + j] = out[j];
	  }
	}
	auto end_time = std::chrono::high_resolution_clock::now();
	// Compute the elapsed time
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "Elapsed time for " << shape[0] << " events =" << duration.count() << " us" << std::endl;
	// Time by event
	std::cout << "Time by event: " << duration.count() / shape[0] << " us" << std::endl;

	py::ssize_t out_shape[] = {shape[0], nprocesses};
	py::ssize_t out_strides[] = {nprocesses * sizeof(double), sizeof(double)};
	py::capsule free_when_done(results, [](void* p) { delete[] reinterpret_cast<double*>(p); });
	py::array_t<double> result(out_shape, out_strides, results, free_when_done);
    
    return result;
	
    }


// Pybind11 module definition
PYBIND11_MODULE(mem_eval, m) {
    m.def("compute_mem", &compute_mem, "Compute mem on input");
}
