//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "gg_ttxhg.h"
#include "HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.h"

using namespace MG5_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g g > t t~ h g NProp=0 SMHLOOP=0 NP==1

//--------------------------------------------------------------------------
// Initialize process.

void gg_ttxhg::initProc(string param_card_name) 
{
  // Instantiate the model class and set parameters that stay fixed during run
  pars = Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::getInstance(); 
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
  jamp2[0] = new double[6]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void gg_ttxhg::sigmaKin() 
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
  for(int i = 0; i < 6; i++ )
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
  const int denominators[nprocesses] = {256}; 

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
        t[0] = matrix_gg_ttxhg(); 

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
      t[0] = matrix_gg_ttxhg(); 

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

double gg_ttxhg::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 21 && id2 == 21)
  {
    // Add matrix elements for processes with beams (21, 21)
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

void gg_ttxhg::calculate_wavefunctions(const int perm[], const int hel[])
{
  // Calculate wavefunctions for all processes
  int i, j; 

  // Calculate all wavefunctions
  vxxxxx(p[perm[0]], mME[0], hel[0], -1, w[0]); 
  vxxxxx(p[perm[1]], mME[1], hel[1], -1, w[1]); 
  oxxxxx(p[perm[2]], mME[2], hel[2], +1, w[2]); 
  ixxxxx(p[perm[3]], mME[3], hel[3], -1, w[3]); 
  sxxxxx(p[perm[4]], +1, w[4]); 
  vxxxxx(p[perm[5]], mME[5], hel[5], +1, w[5]); 
  VVV5P0_1(w[0], w[1], pars->GC_7, pars->ZERO, pars->ZERO, w[6]); 
  FFS2_1(w[2], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[7]); 
  FFV1_2(w[3], w[6], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[8]); 
  VVV5P0_1(w[6], w[5], pars->GC_7, pars->ZERO, pars->ZERO, w[9]); 
  FFV1_2(w[3], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[10]); 
  FFV1_1(w[2], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[11]); 
  FFS2_2(w[3], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[12]); 
  FFV1_1(w[2], w[6], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[13]); 
  FFV1_1(w[2], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[14]); 
  FFV1_2(w[3], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[15]); 
  FFS2_1(w[14], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[16]); 
  FFV1_1(w[14], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[17]); 
  VVV5P0_1(w[1], w[5], pars->GC_7, pars->ZERO, pars->ZERO, w[18]); 
  FFV1_1(w[14], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[19]); 
  FFV1_2(w[3], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[20]); 
  FFV1_1(w[2], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[21]); 
  FFS2_2(w[20], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[22]); 
  FFV1_2(w[20], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[23]); 
  FFV1_2(w[20], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[24]); 
  VVV5P0_1(w[0], w[5], pars->GC_7, pars->ZERO, pars->ZERO, w[25]); 
  FFV1_2(w[3], w[25], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[26]); 
  FFV1_1(w[2], w[25], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[27]); 
  VVV5P0_1(w[25], w[1], pars->GC_7, pars->ZERO, pars->ZERO, w[28]); 
  FFV1_1(w[21], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[29]); 
  FFV1_2(w[12], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[30]); 
  FFV1_2(w[10], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[31]); 
  FFV1_2(w[15], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[32]); 
  FFV1_1(w[7], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[33]); 
  FFV1_1(w[11], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[34]); 
  VVV5P0_1(w[0], w[18], pars->GC_7, pars->ZERO, pars->ZERO, w[35]); 
  VVVV1P0_1(w[0], w[1], w[5], pars->GC_8, pars->ZERO, pars->ZERO, w[36]); 
  VVVV9P0_1(w[0], w[1], w[5], pars->GC_8, pars->ZERO, pars->ZERO, w[37]); 
  VVVV10P0_1(w[0], w[1], w[5], pars->GC_8, pars->ZERO, pars->ZERO, w[38]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[8], w[7], w[5], pars->GC_6, amp[0]); 
  FFV1_0(w[3], w[7], w[9], pars->GC_6, amp[1]); 
  FFV1_0(w[10], w[7], w[6], pars->GC_6, amp[2]); 
  FFS2_0(w[8], w[11], w[4], pars->GC_461, amp[3]); 
  FFV1_0(w[12], w[11], w[6], pars->GC_6, amp[4]); 
  FFV1_0(w[12], w[13], w[5], pars->GC_6, amp[5]); 
  FFV1_0(w[12], w[2], w[9], pars->GC_6, amp[6]); 
  FFS2_0(w[10], w[13], w[4], pars->GC_461, amp[7]); 
  FFV1_0(w[15], w[16], w[5], pars->GC_6, amp[8]); 
  FFS2_0(w[15], w[17], w[4], pars->GC_461, amp[9]); 
  FFV1_0(w[3], w[16], w[18], pars->GC_6, amp[10]); 
  FFV1_0(w[12], w[14], w[18], pars->GC_6, amp[11]); 
  FFV1_0(w[12], w[19], w[5], pars->GC_6, amp[12]); 
  FFV1_0(w[12], w[17], w[1], pars->GC_6, amp[13]); 
  FFS2_0(w[10], w[19], w[4], pars->GC_461, amp[14]); 
  FFV1_0(w[10], w[16], w[1], pars->GC_6, amp[15]); 
  FFV1_0(w[22], w[21], w[5], pars->GC_6, amp[16]); 
  FFS2_0(w[23], w[21], w[4], pars->GC_461, amp[17]); 
  FFV1_0(w[22], w[2], w[18], pars->GC_6, amp[18]); 
  FFV1_0(w[20], w[7], w[18], pars->GC_6, amp[19]); 
  FFV1_0(w[24], w[7], w[5], pars->GC_6, amp[20]); 
  FFV1_0(w[23], w[7], w[1], pars->GC_6, amp[21]); 
  FFS2_0(w[24], w[11], w[4], pars->GC_461, amp[22]); 
  FFV1_0(w[22], w[11], w[1], pars->GC_6, amp[23]); 
  FFS2_0(w[26], w[21], w[4], pars->GC_461, amp[24]); 
  FFV1_0(w[12], w[21], w[25], pars->GC_6, amp[25]); 
  FFS2_0(w[15], w[27], w[4], pars->GC_461, amp[26]); 
  FFV1_0(w[15], w[7], w[25], pars->GC_6, amp[27]); 
  FFV1_0(w[3], w[7], w[28], pars->GC_6, amp[28]); 
  FFV1_0(w[26], w[7], w[1], pars->GC_6, amp[29]); 
  FFV1_0(w[12], w[2], w[28], pars->GC_6, amp[30]); 
  FFV1_0(w[12], w[27], w[1], pars->GC_6, amp[31]); 
  FFV1_0(w[12], w[29], w[5], pars->GC_6, amp[32]); 
  FFV1_0(w[30], w[21], w[5], pars->GC_6, amp[33]); 
  FFS2_0(w[10], w[29], w[4], pars->GC_461, amp[34]); 
  FFS2_0(w[31], w[21], w[4], pars->GC_461, amp[35]); 
  FFV1_0(w[32], w[7], w[5], pars->GC_6, amp[36]); 
  FFV1_0(w[15], w[33], w[5], pars->GC_6, amp[37]); 
  FFS2_0(w[32], w[11], w[4], pars->GC_461, amp[38]); 
  FFS2_0(w[15], w[34], w[4], pars->GC_461, amp[39]); 
  FFV1_0(w[3], w[7], w[35], pars->GC_6, amp[40]); 
  FFV1_0(w[3], w[33], w[18], pars->GC_6, amp[41]); 
  FFV1_0(w[12], w[2], w[35], pars->GC_6, amp[42]); 
  FFV1_0(w[30], w[2], w[18], pars->GC_6, amp[43]); 
  FFV1_0(w[10], w[33], w[1], pars->GC_6, amp[44]); 
  FFV1_0(w[31], w[7], w[1], pars->GC_6, amp[45]); 
  FFV1_0(w[12], w[34], w[1], pars->GC_6, amp[46]); 
  FFV1_0(w[30], w[11], w[1], pars->GC_6, amp[47]); 
  FFV1_0(w[3], w[7], w[36], pars->GC_6, amp[48]); 
  FFV1_0(w[3], w[7], w[37], pars->GC_6, amp[49]); 
  FFV1_0(w[3], w[7], w[38], pars->GC_6, amp[50]); 
  FFV1_0(w[12], w[2], w[36], pars->GC_6, amp[51]); 
  FFV1_0(w[12], w[2], w[37], pars->GC_6, amp[52]); 
  FFV1_0(w[12], w[2], w[38], pars->GC_6, amp[53]); 

}
double gg_ttxhg::matrix_gg_ttxhg() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 54; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {9, 9, 9, 9, 9, 9}; 
  static const double cf[ncolor][ncolor] = {{64, -8, -8, 1, 1, 10}, {-8, 64, 1,
      10, -8, 1}, {-8, 1, 64, -8, 10, 1}, {1, 10, -8, 64, 1, -8}, {1, -8, 10,
      1, 64, -8}, {10, 1, 1, -8, -8, 64}};

  // Calculate color flows
  jamp[0] = +amp[1] + std::complex<double> (0, 1) * amp[2] +
      std::complex<double> (0, 1) * amp[5] + amp[6] + std::complex<double> (0,
      1) * amp[7] + std::complex<double> (0, 1) * amp[10] +
      std::complex<double> (0, 1) * amp[11] - amp[12] - amp[14] - amp[15] +
      amp[40] + std::complex<double> (0, 1) * amp[41] + amp[42] - amp[44] +
      amp[48] - amp[50] + amp[51] - amp[53];
  jamp[1] = -amp[8] - amp[9] - std::complex<double> (0, 1) * amp[10] -
      std::complex<double> (0, 1) * amp[11] - amp[13] + std::complex<double>
      (0, 1) * amp[26] + std::complex<double> (0, 1) * amp[27] + amp[28] +
      amp[30] + std::complex<double> (0, 1) * amp[31] - amp[37] - amp[40] -
      std::complex<double> (0, 1) * amp[41] - amp[42] - amp[48] - amp[49] -
      amp[51] - amp[52];
  jamp[2] = -amp[1] - std::complex<double> (0, 1) * amp[2] -
      std::complex<double> (0, 1) * amp[5] - amp[6] - std::complex<double> (0,
      1) * amp[7] + std::complex<double> (0, 1) * amp[24] +
      std::complex<double> (0, 1) * amp[25] - amp[28] + std::complex<double>
      (0, 1) * amp[29] - amp[30] - amp[32] - amp[34] - amp[35] - amp[45] +
      amp[49] + amp[50] + amp[52] + amp[53];
  jamp[3] = -amp[16] - amp[17] + std::complex<double> (0, 1) * amp[18] +
      std::complex<double> (0, 1) * amp[19] - amp[21] - std::complex<double>
      (0, 1) * amp[24] - std::complex<double> (0, 1) * amp[25] + amp[28] -
      std::complex<double> (0, 1) * amp[29] + amp[30] - amp[33] - amp[40] -
      amp[42] + std::complex<double> (0, 1) * amp[43] - amp[48] - amp[49] -
      amp[51] - amp[52];
  jamp[4] = +std::complex<double> (0, 1) * amp[0] - amp[1] +
      std::complex<double> (0, 1) * amp[3] + std::complex<double> (0, 1) *
      amp[4] - amp[6] - std::complex<double> (0, 1) * amp[26] -
      std::complex<double> (0, 1) * amp[27] - amp[28] - amp[30] -
      std::complex<double> (0, 1) * amp[31] - amp[36] - amp[38] - amp[39] -
      amp[46] + amp[49] + amp[50] + amp[52] + amp[53];
  jamp[5] = -std::complex<double> (0, 1) * amp[0] + amp[1] -
      std::complex<double> (0, 1) * amp[3] - std::complex<double> (0, 1) *
      amp[4] + amp[6] - std::complex<double> (0, 1) * amp[18] -
      std::complex<double> (0, 1) * amp[19] - amp[20] - amp[22] - amp[23] +
      amp[40] + amp[42] - std::complex<double> (0, 1) * amp[43] - amp[47] +
      amp[48] - amp[50] + amp[51] - amp[53];

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



