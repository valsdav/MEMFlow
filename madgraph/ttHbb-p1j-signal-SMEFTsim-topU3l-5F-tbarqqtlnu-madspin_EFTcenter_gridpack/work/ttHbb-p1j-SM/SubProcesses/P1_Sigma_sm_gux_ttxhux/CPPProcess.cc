//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "CPPProcess.h"
#include "HelAmps_sm.h"

using namespace MG5_sm; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g u~ > t t~ h u~ WEIGHTED<=5 @1
// Process: g c~ > t t~ h c~ WEIGHTED<=5 @1
// Process: g d~ > t t~ h d~ WEIGHTED<=5 @1
// Process: g s~ > t t~ h s~ WEIGHTED<=5 @1

//--------------------------------------------------------------------------
// Initialize process.

void CPPProcess::initProc(string param_card_name) 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_sm::getInstance(); 
  SLHAReader slha(param_card_name); 
  pars->setIndependentParameters(slha); 
  pars->setIndependentCouplings(); 
  pars->printIndependentParameters(); 
  pars->printIndependentCouplings(); 
  // Set external particle masses for this matrix element
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->ZERO); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MH); 
  mME.push_back(pars->ZERO); 
  jamp2[0] = new double[4]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void CPPProcess::sigmaKin() 
{
  // Set the parameters which change event by event
  pars->setDependentParameters(); 
  pars->setDependentCouplings(); 
  static bool firsttime = true; 
  if (firsttime)
  {
    pars->printDependentParameters(); 
    pars->printDependentCouplings(); 
    firsttime = false; 
  }

  // Reset color flows
  for(int i = 0; i < 4; i++ )
    jamp2[0][i] = 0.; 

  // Local variables and constants
  const int ncomb = 32; 
  static bool goodhel[ncomb] = {ncomb * false}; 
  static int ntry = 0, sum_hel = 0, ngood = 0; 
  static int igood[ncomb]; 
  static int jhel; 
  std::complex<double> * * wfs; 
  double t[nprocesses]; 
  // Helicities for the process
  static const int helicities[ncomb][nexternal] = {{-1, -1, -1, -1, 0, -1},
      {-1, -1, -1, -1, 0, 1}, {-1, -1, -1, 1, 0, -1}, {-1, -1, -1, 1, 0, 1},
      {-1, -1, 1, -1, 0, -1}, {-1, -1, 1, -1, 0, 1}, {-1, -1, 1, 1, 0, -1},
      {-1, -1, 1, 1, 0, 1}, {-1, 1, -1, -1, 0, -1}, {-1, 1, -1, -1, 0, 1}, {-1,
      1, -1, 1, 0, -1}, {-1, 1, -1, 1, 0, 1}, {-1, 1, 1, -1, 0, -1}, {-1, 1, 1,
      -1, 0, 1}, {-1, 1, 1, 1, 0, -1}, {-1, 1, 1, 1, 0, 1}, {1, -1, -1, -1, 0,
      -1}, {1, -1, -1, -1, 0, 1}, {1, -1, -1, 1, 0, -1}, {1, -1, -1, 1, 0, 1},
      {1, -1, 1, -1, 0, -1}, {1, -1, 1, -1, 0, 1}, {1, -1, 1, 1, 0, -1}, {1,
      -1, 1, 1, 0, 1}, {1, 1, -1, -1, 0, -1}, {1, 1, -1, -1, 0, 1}, {1, 1, -1,
      1, 0, -1}, {1, 1, -1, 1, 0, 1}, {1, 1, 1, -1, 0, -1}, {1, 1, 1, -1, 0,
      1}, {1, 1, 1, 1, 0, -1}, {1, 1, 1, 1, 0, 1}};
  // Denominators: spins, colors and identical particles
  const int denominators[nprocesses] = {96, 96}; 

  ntry = ntry + 1; 

  // Reset the matrix elements
  for(int i = 0; i < nprocesses; i++ )
  {
    matrix_element[i] = 0.; 
  }
  // Define permutation
  int perm[nexternal]; 
  for(int i = 0; i < nexternal; i++ )
  {
    perm[i] = i; 
  }

  if (sum_hel == 0 || ntry < 10)
  {
    // Calculate the matrix element for all helicities
    for(int ihel = 0; ihel < ncomb; ihel++ )
    {
      if (goodhel[ihel] || ntry < 2)
      {
        calculate_wavefunctions(perm, helicities[ihel]); 
        t[0] = matrix_1_gux_ttxhux(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel]); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        // Calculate matrix elements
        t[1] = matrix_1_gux_ttxhux(); 
        double tsum = 0; 
        for(int iproc = 0; iproc < nprocesses; iproc++ )
        {
          matrix_element[iproc] += t[iproc]; 
          tsum += t[iproc]; 
        }
        // Store which helicities give non-zero result
        if (tsum != 0. && !goodhel[ihel])
        {
          goodhel[ihel] = true; 
          ngood++; 
          igood[ngood] = ihel; 
        }
      }
    }
    jhel = 0; 
    sum_hel = min(sum_hel, ngood); 
  }
  else
  {
    // Only use the "good" helicities
    for(int j = 0; j < sum_hel; j++ )
    {
      jhel++; 
      if (jhel >= ngood)
        jhel = 0; 
      double hwgt = double(ngood)/double(sum_hel); 
      int ihel = igood[jhel]; 
      calculate_wavefunctions(perm, helicities[ihel]); 
      t[0] = matrix_1_gux_ttxhux(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel]); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      // Calculate matrix elements
      t[1] = matrix_1_gux_ttxhux(); 
      for(int iproc = 0; iproc < nprocesses; iproc++ )
      {
        matrix_element[iproc] += t[iproc] * hwgt; 
      }
    }
  }

  for (int i = 0; i < nprocesses; i++ )
    matrix_element[i] /= denominators[i]; 



}

//--------------------------------------------------------------------------
// Evaluate |M|^2, including incoming flavour dependence.

double CPPProcess::sigmaHat() 
{
  // Select between the different processes
  if(id1 == -4 && id2 == 21)
  {
    // Add matrix elements for processes with beams (-4, 21)
    return matrix_element[1]; 
  }
  else if(id1 == -3 && id2 == 21)
  {
    // Add matrix elements for processes with beams (-3, 21)
    return matrix_element[1]; 
  }
  else if(id1 == -2 && id2 == 21)
  {
    // Add matrix elements for processes with beams (-2, 21)
    return matrix_element[1]; 
  }
  else if(id1 == -1 && id2 == 21)
  {
    // Add matrix elements for processes with beams (-1, 21)
    return matrix_element[1]; 
  }
  else if(id1 == 21 && id2 == -4)
  {
    // Add matrix elements for processes with beams (21, -4)
    return matrix_element[0]; 
  }
  else if(id1 == 21 && id2 == -3)
  {
    // Add matrix elements for processes with beams (21, -3)
    return matrix_element[0]; 
  }
  else if(id1 == 21 && id2 == -2)
  {
    // Add matrix elements for processes with beams (21, -2)
    return matrix_element[0]; 
  }
  else if(id1 == 21 && id2 == -1)
  {
    // Add matrix elements for processes with beams (21, -1)
    return matrix_element[0]; 
  }
  else
  {
    // Return 0 if not correct initial state assignment
    return 0.; 
  }
}

//==========================================================================
// Private class member functions

//--------------------------------------------------------------------------
// Evaluate |M|^2 for each subprocess

void CPPProcess::calculate_wavefunctions(const int perm[], const int hel[])
{
  // Calculate wavefunctions for all processes
  int i, j; 

  // Calculate all wavefunctions
  vxxxxx(p[perm[0]], mME[0], hel[0], -1, w[0]); 
  oxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[3]); 
  sxxxxx(p[perm[4]], +1, w[4]); 
  ixxxxx(p[perm[5]], mME[5], hel[5], -1, w[5]); 
  FFV1_2(w[5], w[0], pars->GC_11, pars->ZERO, pars->ZERO, w[6]); 
  FFS4_1(w[2], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[7]); 
  FFV1P0_3(w[6], w[1], pars->GC_11, pars->ZERO, pars->ZERO, w[8]); 
  FFS4_2(w[3], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[9]); 
  FFV1_1(w[2], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[10]); 
  FFV1P0_3(w[5], w[1], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFS4_1(w[10], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[12]); 
  FFV1_2(w[3], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[13]); 
  FFS4_2(w[13], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[14]); 
  FFV1_1(w[1], w[0], pars->GC_11, pars->ZERO, pars->ZERO, w[15]); 
  FFV1P0_3(w[5], w[15], pars->GC_11, pars->ZERO, pars->ZERO, w[16]); 
  VVV1P0_1(w[0], w[11], pars->GC_10, pars->ZERO, pars->ZERO, w[17]); 
  FFV1_1(w[7], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[18]); 
  FFV1_2(w[9], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[19]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[3], w[7], w[8], pars->GC_11, amp[0]); 
  FFV1_0(w[9], w[2], w[8], pars->GC_11, amp[1]); 
  FFV1_0(w[3], w[12], w[11], pars->GC_11, amp[2]); 
  FFV1_0(w[9], w[10], w[11], pars->GC_11, amp[3]); 
  FFV1_0(w[14], w[2], w[11], pars->GC_11, amp[4]); 
  FFV1_0(w[13], w[7], w[11], pars->GC_11, amp[5]); 
  FFV1_0(w[3], w[7], w[16], pars->GC_11, amp[6]); 
  FFV1_0(w[9], w[2], w[16], pars->GC_11, amp[7]); 
  FFV1_0(w[3], w[7], w[17], pars->GC_11, amp[8]); 
  FFV1_0(w[3], w[18], w[11], pars->GC_11, amp[9]); 
  FFV1_0(w[9], w[2], w[17], pars->GC_11, amp[10]); 
  FFV1_0(w[19], w[2], w[11], pars->GC_11, amp[11]); 

}
double CPPProcess::matrix_1_gux_ttxhux() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 12; 
  const int ncolor = 4; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (+amp[4] + amp[5] + amp[6] + amp[7] + std::complex<double>
      (0, 1) * amp[8] + std::complex<double> (0, 1) * amp[10] + amp[11]);
  jamp[1] = +1./2. * (-1./3. * amp[0] - 1./3. * amp[1] - 1./3. * amp[6] - 1./3.
      * amp[7]);
  jamp[2] = +1./2. * (-1./3. * amp[2] - 1./3. * amp[3] - 1./3. * amp[4] - 1./3.
      * amp[5] - 1./3. * amp[9] - 1./3. * amp[11]);
  jamp[3] = +1./2. * (+amp[0] + amp[1] + amp[2] + amp[3] - std::complex<double>
      (0, 1) * amp[8] + amp[9] - std::complex<double> (0, 1) * amp[10]);

  // Sum and square the color flows to get the matrix element
  double matrix = 0; 
  for(i = 0; i < ncolor; i++ )
  {
    ztemp = 0.; 
    for(j = 0; j < ncolor; j++ )
      ztemp = ztemp + cf[i][j] * jamp[j]; 
    matrix = matrix + real(ztemp * conj(jamp[i]))/denom[i]; 
  }

  // Store the leading color flows for choice of color
  for(i = 0; i < ncolor; i++ )
    jamp2[0][i] += real(jamp[i] * conj(jamp[i])); 

  return matrix; 
}



