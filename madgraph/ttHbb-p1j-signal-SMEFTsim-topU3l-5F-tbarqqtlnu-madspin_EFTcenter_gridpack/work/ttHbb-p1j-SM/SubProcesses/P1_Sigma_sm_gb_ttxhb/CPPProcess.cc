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
// Process: g b > t t~ h b WEIGHTED<=5 @1

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
  mME.push_back(pars->mdl_MB); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MH); 
  mME.push_back(pars->mdl_MB); 
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
        t[0] = matrix_1_gb_ttxhb(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel]); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        // Calculate matrix elements
        t[1] = matrix_1_gb_ttxhb(); 
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
      t[0] = matrix_1_gb_ttxhb(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel]); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      // Calculate matrix elements
      t[1] = matrix_1_gb_ttxhb(); 
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
  if(id1 == 5 && id2 == 21)
  {
    // Add matrix elements for processes with beams (5, 21)
    return matrix_element[1]; 
  }
  else if(id1 == 21 && id2 == 5)
  {
    // Add matrix elements for processes with beams (21, 5)
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
  ixxxxx(p[perm[1]], mME[1], hel[1], +1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[3]); 
  sxxxxx(p[perm[4]], +1, w[4]); 
  oxxxxx(p[perm[5]], mME[5], hel[5], +1, w[5]); 
  FFV1_2(w[1], w[0], pars->GC_11, pars->mdl_MB, pars->ZERO, w[6]); 
  FFV1P0_3(w[3], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[7]); 
  FFS4_2(w[6], w[4], pars->GC_83, pars->mdl_MB, pars->ZERO, w[8]); 
  FFS4_1(w[5], w[4], pars->GC_83, pars->mdl_MB, pars->ZERO, w[9]); 
  FFS4_1(w[2], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[10]); 
  FFV1P0_3(w[6], w[5], pars->GC_11, pars->ZERO, pars->ZERO, w[11]); 
  FFS4_2(w[3], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[12]); 
  FFV1_1(w[2], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[13]); 
  FFS4_2(w[1], w[4], pars->GC_83, pars->mdl_MB, pars->ZERO, w[14]); 
  FFV1P0_3(w[3], w[13], pars->GC_11, pars->ZERO, pars->ZERO, w[15]); 
  FFV1P0_3(w[1], w[5], pars->GC_11, pars->ZERO, pars->ZERO, w[16]); 
  FFS4_1(w[13], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[17]); 
  FFV1_2(w[3], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[18]); 
  FFV1P0_3(w[18], w[2], pars->GC_11, pars->ZERO, pars->ZERO, w[19]); 
  FFS4_2(w[18], w[4], pars->GC_94, pars->mdl_MT, pars->mdl_WT, w[20]); 
  FFV1_1(w[5], w[0], pars->GC_11, pars->mdl_MB, pars->ZERO, w[21]); 
  FFS4_1(w[21], w[4], pars->GC_83, pars->mdl_MB, pars->ZERO, w[22]); 
  FFV1P0_3(w[1], w[21], pars->GC_11, pars->ZERO, pars->ZERO, w[23]); 
  FFV1_2(w[14], w[0], pars->GC_11, pars->mdl_MB, pars->ZERO, w[24]); 
  VVV1P0_1(w[0], w[7], pars->GC_10, pars->ZERO, pars->ZERO, w[25]); 
  VVV1P0_1(w[0], w[16], pars->GC_10, pars->ZERO, pars->ZERO, w[26]); 
  FFV1_1(w[10], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[27]); 
  FFV1_2(w[12], w[0], pars->GC_11, pars->mdl_MT, pars->mdl_WT, w[28]); 
  FFV1_1(w[9], w[0], pars->GC_11, pars->mdl_MB, pars->ZERO, w[29]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[8], w[5], w[7], pars->GC_11, amp[0]); 
  FFV1_0(w[6], w[9], w[7], pars->GC_11, amp[1]); 
  FFV1_0(w[3], w[10], w[11], pars->GC_11, amp[2]); 
  FFV1_0(w[12], w[2], w[11], pars->GC_11, amp[3]); 
  FFV1_0(w[14], w[5], w[15], pars->GC_11, amp[4]); 
  FFV1_0(w[3], w[17], w[16], pars->GC_11, amp[5]); 
  FFV1_0(w[12], w[13], w[16], pars->GC_11, amp[6]); 
  FFV1_0(w[1], w[9], w[15], pars->GC_11, amp[7]); 
  FFV1_0(w[14], w[5], w[19], pars->GC_11, amp[8]); 
  FFV1_0(w[20], w[2], w[16], pars->GC_11, amp[9]); 
  FFV1_0(w[18], w[10], w[16], pars->GC_11, amp[10]); 
  FFV1_0(w[1], w[9], w[19], pars->GC_11, amp[11]); 
  FFV1_0(w[14], w[21], w[7], pars->GC_11, amp[12]); 
  FFV1_0(w[1], w[22], w[7], pars->GC_11, amp[13]); 
  FFV1_0(w[3], w[10], w[23], pars->GC_11, amp[14]); 
  FFV1_0(w[12], w[2], w[23], pars->GC_11, amp[15]); 
  FFV1_0(w[24], w[5], w[7], pars->GC_11, amp[16]); 
  FFV1_0(w[14], w[5], w[25], pars->GC_11, amp[17]); 
  FFV1_0(w[3], w[10], w[26], pars->GC_11, amp[18]); 
  FFV1_0(w[3], w[27], w[16], pars->GC_11, amp[19]); 
  FFV1_0(w[12], w[2], w[26], pars->GC_11, amp[20]); 
  FFV1_0(w[28], w[2], w[16], pars->GC_11, amp[21]); 
  FFV1_0(w[1], w[9], w[25], pars->GC_11, amp[22]); 
  FFV1_0(w[1], w[29], w[7], pars->GC_11, amp[23]); 

}
double CPPProcess::matrix_1_gb_ttxhb() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 24; 
  const int ncolor = 4; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (-amp[0] - amp[1] - amp[2] - amp[3] - amp[4] - amp[5] -
      amp[6] - amp[7] - amp[16] - std::complex<double> (0, 1) * amp[17] +
      std::complex<double> (0, 1) * amp[18] - amp[19] + std::complex<double>
      (0, 1) * amp[20] - std::complex<double> (0, 1) * amp[22]);
  jamp[1] = +1./2. * (+1./3. * amp[4] + 1./3. * amp[5] + 1./3. * amp[6] + 1./3.
      * amp[7] + 1./3. * amp[8] + 1./3. * amp[9] + 1./3. * amp[10] + 1./3. *
      amp[11] + 1./3. * amp[19] + 1./3. * amp[21]);
  jamp[2] = +1./2. * (+1./3. * amp[0] + 1./3. * amp[1] + 1./3. * amp[2] + 1./3.
      * amp[3] + 1./3. * amp[12] + 1./3. * amp[13] + 1./3. * amp[14] + 1./3. *
      amp[15] + 1./3. * amp[16] + 1./3. * amp[23]);
  jamp[3] = +1./2. * (-amp[8] - amp[9] - amp[10] - amp[11] - amp[12] - amp[13]
      - amp[14] - amp[15] + std::complex<double> (0, 1) * amp[17] -
      std::complex<double> (0, 1) * amp[18] - std::complex<double> (0, 1) *
      amp[20] - amp[21] + std::complex<double> (0, 1) * amp[22] - amp[23]);

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



