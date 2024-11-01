//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "gb_ttxhb.h"
#include "HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.h"

using namespace MG5_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g b > t t~ h b NProp=0 SMHLOOP=0 NP^2==1

//--------------------------------------------------------------------------
// Initialize process.

void gb_ttxhb::initProc(string param_card_name) 
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
  mME.push_back(pars->mdl_MB); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MT); 
  mME.push_back(pars->mdl_MH); 
  mME.push_back(pars->mdl_MB); 
  jamp2[0] = new double[4]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void gb_ttxhb::sigmaKin() 
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
        t[0] = matrix_gb_ttxhb(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel]); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        // Calculate matrix elements
        t[1] = matrix_gb_ttxhb(); 
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
      t[0] = matrix_gb_ttxhb(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel]); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      // Calculate matrix elements
      t[1] = matrix_gb_ttxhb(); 
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

double gb_ttxhb::sigmaHat() 
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

void gb_ttxhb::calculate_wavefunctions(const int perm[], const int hel[])
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
  FFV1_2(w[1], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[6]); 
  FFV1P0_3(w[3], w[2], pars->GC_2, pars->ZERO, pars->ZERO, w[7]); 
  FFS2_2(w[6], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[8]); 
  FFV1_3_3(w[3], w[2], pars->GC_264, pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[9]);
  FFV1_3_3(w[6], w[5], pars->GC_263, -pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[10]);
  FFV1P0_3(w[3], w[2], pars->GC_6, pars->ZERO, pars->ZERO, w[11]); 
  FFS2_3(w[3], w[2], pars->GC_461, pars->mdl_MH, pars->mdl_WH, w[12]); 
  FFS2_3(w[6], w[5], pars->GC_566, pars->mdl_MH, pars->mdl_WH, w[13]); 
  FFS2_3(w[3], w[2], pars->GC_1025, pars->mdl_MH, pars->mdl_WH, w[14]); 
  FFS2_1(w[5], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[15]); 
  FFS2_1(w[2], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[16]); 
  FFV1P0_3(w[6], w[5], pars->GC_1, pars->ZERO, pars->ZERO, w[17]); 
  FFV1P0_3(w[6], w[5], pars->GC_6, pars->ZERO, pars->ZERO, w[18]); 
  FFS2_1(w[2], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[19]); 
  FFV3_3(w[3], w[5], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[20]); 
  FFS2_2(w[3], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[21]); 
  FFV3_3(w[6], w[2], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[22]); 
  FFS2_2(w[3], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[23]); 
  FFSS2_3(w[3], w[2], w[4], pars->GC_364, pars->mdl_MH, pars->mdl_WH, w[24]); 
  FFV1_1(w[2], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[25]); 
  FFS2_2(w[1], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[26]); 
  FFV1P0_3(w[3], w[25], pars->GC_2, pars->ZERO, pars->ZERO, w[27]); 
  FFV1_3_3(w[3], w[25], pars->GC_264, pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[28]);
  FFV1P0_3(w[3], w[25], pars->GC_6, pars->ZERO, pars->ZERO, w[29]); 
  FFS2_3(w[3], w[25], pars->GC_461, pars->mdl_MH, pars->mdl_WH, w[30]); 
  FFS2_3(w[3], w[25], pars->GC_1025, pars->mdl_MH, pars->mdl_WH, w[31]); 
  FFV1P0_3(w[1], w[5], pars->GC_1, pars->ZERO, pars->ZERO, w[32]); 
  FFS2_1(w[25], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[33]); 
  FFS2_1(w[25], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[34]); 
  FFV1_3_3(w[1], w[5], pars->GC_263, -pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[35]);
  FFV1P0_3(w[1], w[5], pars->GC_6, pars->ZERO, pars->ZERO, w[36]); 
  FFS2_3(w[1], w[5], pars->GC_566, pars->mdl_MH, pars->mdl_WH, w[37]); 
  FFV3_3(w[1], w[25], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[38]); 
  FFV1_2(w[3], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[39]); 
  FFV3_3(w[1], w[2], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[40]); 
  FFS2_2(w[39], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[41]); 
  FFS2_2(w[39], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[42]); 
  FFV3_3(w[39], w[5], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[43]); 
  FFV1P0_3(w[39], w[2], pars->GC_2, pars->ZERO, pars->ZERO, w[44]); 
  FFV1_3_3(w[39], w[2], pars->GC_264, pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[45]);
  FFV1P0_3(w[39], w[2], pars->GC_6, pars->ZERO, pars->ZERO, w[46]); 
  FFS2_3(w[39], w[2], pars->GC_461, pars->mdl_MH, pars->mdl_WH, w[47]); 
  FFS2_3(w[39], w[2], pars->GC_1025, pars->mdl_MH, pars->mdl_WH, w[48]); 
  FFV1_1(w[5], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[49]); 
  FFV3_3(w[3], w[49], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[50]); 
  FFS2_1(w[49], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[51]); 
  FFV1_3_3(w[1], w[49], pars->GC_263, -pars->GC_203, pars->mdl_MZ,
      pars->mdl_WZ, w[52]);
  FFS2_3(w[1], w[49], pars->GC_566, pars->mdl_MH, pars->mdl_WH, w[53]); 
  FFV1P0_3(w[1], w[49], pars->GC_1, pars->ZERO, pars->ZERO, w[54]); 
  FFV1P0_3(w[1], w[49], pars->GC_6, pars->ZERO, pars->ZERO, w[55]); 
  FFV1_2(w[21], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[56]); 
  FFV1_2(w[23], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[57]); 
  FFV1_1(w[15], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[58]); 
  FFV1_2(w[26], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[59]); 
  VVV5P0_1(w[0], w[11], pars->GC_7, pars->ZERO, pars->ZERO, w[60]); 
  FFV1_1(w[16], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[61]); 
  FFV1_1(w[19], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[62]); 
  VVV5P0_1(w[0], w[36], pars->GC_7, pars->ZERO, pars->ZERO, w[63]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[8], w[5], w[7], pars->GC_1, amp[0]); 
  FFV1_3_0(w[8], w[5], w[9], pars->GC_263, -pars->GC_203, amp[1]); 
  VVS3_0(w[9], w[10], w[4], pars->GC_392, amp[2]); 
  FFV1_0(w[8], w[5], w[11], pars->GC_6, amp[3]); 
  FFS2_0(w[8], w[5], w[12], pars->GC_566, amp[4]); 
  SSS1_0(w[12], w[4], w[13], pars->GC_336, amp[5]); 
  FFS2_0(w[8], w[5], w[14], pars->GC_566, amp[6]); 
  SSS1_0(w[14], w[4], w[13], pars->GC_336, amp[7]); 
  FFV1_0(w[6], w[15], w[7], pars->GC_1, amp[8]); 
  FFV1_3_0(w[6], w[15], w[9], pars->GC_263, -pars->GC_203, amp[9]); 
  FFV1_0(w[6], w[15], w[11], pars->GC_6, amp[10]); 
  FFS2_0(w[6], w[15], w[12], pars->GC_566, amp[11]); 
  FFS2_0(w[6], w[15], w[14], pars->GC_566, amp[12]); 
  FFV1_0(w[3], w[16], w[17], pars->GC_2, amp[13]); 
  FFV1_3_0(w[3], w[16], w[10], pars->GC_264, pars->GC_203, amp[14]); 
  FFV1_0(w[3], w[16], w[18], pars->GC_6, amp[15]); 
  FFS2_0(w[3], w[16], w[13], pars->GC_1025, amp[16]); 
  FFV1_0(w[3], w[19], w[17], pars->GC_2, amp[17]); 
  FFV1_3_0(w[3], w[19], w[10], pars->GC_264, pars->GC_203, amp[18]); 
  FFV1_0(w[3], w[19], w[18], pars->GC_6, amp[19]); 
  FFS2_0(w[3], w[19], w[13], pars->GC_461, amp[20]); 
  FFS2_0(w[3], w[19], w[13], pars->GC_1025, amp[21]); 
  FFV3_0(w[6], w[16], w[20], pars->GC_202, amp[22]); 
  FFV3_0(w[6], w[19], w[20], pars->GC_202, amp[23]); 
  FFV3_0(w[21], w[5], w[22], pars->GC_202, amp[24]); 
  FFV1_0(w[21], w[2], w[17], pars->GC_2, amp[25]); 
  FFV1_3_0(w[21], w[2], w[10], pars->GC_264, pars->GC_203, amp[26]); 
  FFV1_0(w[21], w[2], w[18], pars->GC_6, amp[27]); 
  FFS2_0(w[21], w[2], w[13], pars->GC_1025, amp[28]); 
  FFV3_0(w[23], w[5], w[22], pars->GC_202, amp[29]); 
  FFV1_0(w[23], w[2], w[17], pars->GC_2, amp[30]); 
  FFV1_3_0(w[23], w[2], w[10], pars->GC_264, pars->GC_203, amp[31]); 
  FFV1_0(w[23], w[2], w[18], pars->GC_6, amp[32]); 
  FFS2_0(w[23], w[2], w[13], pars->GC_461, amp[33]); 
  FFS2_0(w[23], w[2], w[13], pars->GC_1025, amp[34]); 
  VVS3_0(w[20], w[22], w[4], pars->GC_391, amp[35]); 
  FFV3_0(w[8], w[2], w[20], pars->GC_202, amp[36]); 
  FFV3_0(w[3], w[15], w[22], pars->GC_202, amp[37]); 
  FFS2_0(w[6], w[5], w[24], pars->GC_566, amp[38]); 
  FFV1_0(w[26], w[5], w[27], pars->GC_1, amp[39]); 
  FFV1_3_0(w[26], w[5], w[28], pars->GC_263, -pars->GC_203, amp[40]); 
  FFV1_0(w[26], w[5], w[29], pars->GC_6, amp[41]); 
  FFS2_0(w[26], w[5], w[30], pars->GC_566, amp[42]); 
  FFS2_0(w[26], w[5], w[31], pars->GC_566, amp[43]); 
  FFV3_0(w[26], w[25], w[20], pars->GC_202, amp[44]); 
  FFV1_0(w[3], w[33], w[32], pars->GC_2, amp[45]); 
  FFV1_0(w[3], w[34], w[32], pars->GC_2, amp[46]); 
  VVS3_0(w[35], w[28], w[4], pars->GC_392, amp[47]); 
  FFV1_3_0(w[3], w[33], w[35], pars->GC_264, pars->GC_203, amp[48]); 
  FFV1_3_0(w[3], w[34], w[35], pars->GC_264, pars->GC_203, amp[49]); 
  FFV1_0(w[3], w[33], w[36], pars->GC_6, amp[50]); 
  FFV1_0(w[3], w[34], w[36], pars->GC_6, amp[51]); 
  FFSS2_0(w[3], w[25], w[37], w[4], pars->GC_364, amp[52]); 
  SSS1_0(w[37], w[4], w[30], pars->GC_336, amp[53]); 
  SSS1_0(w[37], w[4], w[31], pars->GC_336, amp[54]); 
  FFS2_0(w[3], w[33], w[37], pars->GC_1025, amp[55]); 
  FFS2_0(w[3], w[34], w[37], pars->GC_461, amp[56]); 
  FFS2_0(w[3], w[34], w[37], pars->GC_1025, amp[57]); 
  FFV1_0(w[21], w[25], w[32], pars->GC_2, amp[58]); 
  FFV1_0(w[23], w[25], w[32], pars->GC_2, amp[59]); 
  FFV1_3_0(w[21], w[25], w[35], pars->GC_264, pars->GC_203, amp[60]); 
  FFV1_3_0(w[23], w[25], w[35], pars->GC_264, pars->GC_203, amp[61]); 
  FFV1_0(w[21], w[25], w[36], pars->GC_6, amp[62]); 
  FFV1_0(w[23], w[25], w[36], pars->GC_6, amp[63]); 
  FFS2_0(w[21], w[25], w[37], pars->GC_1025, amp[64]); 
  FFS2_0(w[23], w[25], w[37], pars->GC_461, amp[65]); 
  FFS2_0(w[23], w[25], w[37], pars->GC_1025, amp[66]); 
  FFV3_0(w[21], w[5], w[38], pars->GC_202, amp[67]); 
  FFV3_0(w[23], w[5], w[38], pars->GC_202, amp[68]); 
  VVS3_0(w[20], w[38], w[4], pars->GC_391, amp[69]); 
  FFV3_0(w[1], w[33], w[20], pars->GC_202, amp[70]); 
  FFV3_0(w[1], w[34], w[20], pars->GC_202, amp[71]); 
  FFV3_0(w[3], w[15], w[38], pars->GC_202, amp[72]); 
  FFV1_0(w[1], w[15], w[27], pars->GC_1, amp[73]); 
  FFV1_3_0(w[1], w[15], w[28], pars->GC_263, -pars->GC_203, amp[74]); 
  FFV1_0(w[1], w[15], w[29], pars->GC_6, amp[75]); 
  FFS2_0(w[1], w[15], w[30], pars->GC_566, amp[76]); 
  FFS2_0(w[1], w[15], w[31], pars->GC_566, amp[77]); 
  FFV3_0(w[41], w[5], w[40], pars->GC_202, amp[78]); 
  FFV3_0(w[42], w[5], w[40], pars->GC_202, amp[79]); 
  VVS3_0(w[43], w[40], w[4], pars->GC_391, amp[80]); 
  FFV3_0(w[39], w[15], w[40], pars->GC_202, amp[81]); 
  FFV1_0(w[26], w[5], w[44], pars->GC_1, amp[82]); 
  FFV1_3_0(w[26], w[5], w[45], pars->GC_263, -pars->GC_203, amp[83]); 
  FFV1_0(w[26], w[5], w[46], pars->GC_6, amp[84]); 
  FFS2_0(w[26], w[5], w[47], pars->GC_566, amp[85]); 
  FFS2_0(w[26], w[5], w[48], pars->GC_566, amp[86]); 
  FFV3_0(w[26], w[2], w[43], pars->GC_202, amp[87]); 
  FFV1_0(w[41], w[2], w[32], pars->GC_2, amp[88]); 
  FFV1_0(w[42], w[2], w[32], pars->GC_2, amp[89]); 
  VVS3_0(w[35], w[45], w[4], pars->GC_392, amp[90]); 
  FFV1_3_0(w[41], w[2], w[35], pars->GC_264, pars->GC_203, amp[91]); 
  FFV1_3_0(w[42], w[2], w[35], pars->GC_264, pars->GC_203, amp[92]); 
  FFV1_0(w[41], w[2], w[36], pars->GC_6, amp[93]); 
  FFV1_0(w[42], w[2], w[36], pars->GC_6, amp[94]); 
  FFSS2_0(w[39], w[2], w[37], w[4], pars->GC_364, amp[95]); 
  SSS1_0(w[37], w[4], w[47], pars->GC_336, amp[96]); 
  SSS1_0(w[37], w[4], w[48], pars->GC_336, amp[97]); 
  FFS2_0(w[41], w[2], w[37], pars->GC_1025, amp[98]); 
  FFS2_0(w[42], w[2], w[37], pars->GC_461, amp[99]); 
  FFS2_0(w[42], w[2], w[37], pars->GC_1025, amp[100]); 
  FFV1_0(w[39], w[16], w[32], pars->GC_2, amp[101]); 
  FFV1_0(w[39], w[19], w[32], pars->GC_2, amp[102]); 
  FFV1_3_0(w[39], w[16], w[35], pars->GC_264, pars->GC_203, amp[103]); 
  FFV1_3_0(w[39], w[19], w[35], pars->GC_264, pars->GC_203, amp[104]); 
  FFV1_0(w[39], w[16], w[36], pars->GC_6, amp[105]); 
  FFV1_0(w[39], w[19], w[36], pars->GC_6, amp[106]); 
  FFS2_0(w[39], w[16], w[37], pars->GC_1025, amp[107]); 
  FFS2_0(w[39], w[19], w[37], pars->GC_461, amp[108]); 
  FFS2_0(w[39], w[19], w[37], pars->GC_1025, amp[109]); 
  FFV3_0(w[1], w[16], w[43], pars->GC_202, amp[110]); 
  FFV3_0(w[1], w[19], w[43], pars->GC_202, amp[111]); 
  FFV1_0(w[1], w[15], w[44], pars->GC_1, amp[112]); 
  FFV1_3_0(w[1], w[15], w[45], pars->GC_263, -pars->GC_203, amp[113]); 
  FFV1_0(w[1], w[15], w[46], pars->GC_6, amp[114]); 
  FFS2_0(w[1], w[15], w[47], pars->GC_566, amp[115]); 
  FFS2_0(w[1], w[15], w[48], pars->GC_566, amp[116]); 
  VVS3_0(w[50], w[40], w[4], pars->GC_391, amp[117]); 
  FFV3_0(w[3], w[51], w[40], pars->GC_202, amp[118]); 
  FFV3_0(w[21], w[49], w[40], pars->GC_202, amp[119]); 
  FFV3_0(w[23], w[49], w[40], pars->GC_202, amp[120]); 
  FFV3_0(w[26], w[2], w[50], pars->GC_202, amp[121]); 
  FFV1_0(w[26], w[49], w[7], pars->GC_1, amp[122]); 
  FFV1_3_0(w[26], w[49], w[9], pars->GC_263, -pars->GC_203, amp[123]); 
  FFV1_0(w[26], w[49], w[11], pars->GC_6, amp[124]); 
  FFS2_0(w[26], w[49], w[12], pars->GC_566, amp[125]); 
  FFS2_0(w[26], w[49], w[14], pars->GC_566, amp[126]); 
  FFV1_0(w[1], w[51], w[7], pars->GC_1, amp[127]); 
  VVS3_0(w[9], w[52], w[4], pars->GC_392, amp[128]); 
  FFV1_3_0(w[1], w[51], w[9], pars->GC_263, -pars->GC_203, amp[129]); 
  FFV1_0(w[1], w[51], w[11], pars->GC_6, amp[130]); 
  SSS1_0(w[12], w[4], w[53], pars->GC_336, amp[131]); 
  FFS2_0(w[1], w[51], w[12], pars->GC_566, amp[132]); 
  SSS1_0(w[14], w[4], w[53], pars->GC_336, amp[133]); 
  FFS2_0(w[1], w[51], w[14], pars->GC_566, amp[134]); 
  FFV1_0(w[3], w[16], w[54], pars->GC_2, amp[135]); 
  FFV1_3_0(w[3], w[16], w[52], pars->GC_264, pars->GC_203, amp[136]); 
  FFV1_0(w[3], w[16], w[55], pars->GC_6, amp[137]); 
  FFS2_0(w[3], w[16], w[53], pars->GC_1025, amp[138]); 
  FFV3_0(w[1], w[16], w[50], pars->GC_202, amp[139]); 
  FFV1_0(w[3], w[19], w[54], pars->GC_2, amp[140]); 
  FFV1_3_0(w[3], w[19], w[52], pars->GC_264, pars->GC_203, amp[141]); 
  FFV1_0(w[3], w[19], w[55], pars->GC_6, amp[142]); 
  FFS2_0(w[3], w[19], w[53], pars->GC_461, amp[143]); 
  FFS2_0(w[3], w[19], w[53], pars->GC_1025, amp[144]); 
  FFV3_0(w[1], w[19], w[50], pars->GC_202, amp[145]); 
  FFV1_0(w[21], w[2], w[54], pars->GC_2, amp[146]); 
  FFV1_3_0(w[21], w[2], w[52], pars->GC_264, pars->GC_203, amp[147]); 
  FFV1_0(w[21], w[2], w[55], pars->GC_6, amp[148]); 
  FFS2_0(w[21], w[2], w[53], pars->GC_1025, amp[149]); 
  FFV1_0(w[23], w[2], w[54], pars->GC_2, amp[150]); 
  FFV1_3_0(w[23], w[2], w[52], pars->GC_264, pars->GC_203, amp[151]); 
  FFV1_0(w[23], w[2], w[55], pars->GC_6, amp[152]); 
  FFS2_0(w[23], w[2], w[53], pars->GC_461, amp[153]); 
  FFS2_0(w[23], w[2], w[53], pars->GC_1025, amp[154]); 
  FFS2_0(w[1], w[49], w[24], pars->GC_566, amp[155]); 
  FFV3_0(w[56], w[5], w[40], pars->GC_202, amp[156]); 
  FFV3_0(w[57], w[5], w[40], pars->GC_202, amp[157]); 
  FFV3_0(w[3], w[58], w[40], pars->GC_202, amp[158]); 
  FFV1_0(w[59], w[5], w[7], pars->GC_1, amp[159]); 
  FFV1_3_0(w[59], w[5], w[9], pars->GC_263, -pars->GC_203, amp[160]); 
  FFV1_0(w[59], w[5], w[11], pars->GC_6, amp[161]); 
  FFV1_0(w[26], w[5], w[60], pars->GC_6, amp[162]); 
  FFS2_0(w[59], w[5], w[12], pars->GC_566, amp[163]); 
  FFS2_0(w[59], w[5], w[14], pars->GC_566, amp[164]); 
  FFV3_0(w[59], w[2], w[20], pars->GC_202, amp[165]); 
  FFV1_0(w[3], w[61], w[32], pars->GC_2, amp[166]); 
  FFV1_0(w[3], w[62], w[32], pars->GC_2, amp[167]); 
  FFV1_3_0(w[3], w[61], w[35], pars->GC_264, pars->GC_203, amp[168]); 
  FFV1_3_0(w[3], w[62], w[35], pars->GC_264, pars->GC_203, amp[169]); 
  FFV1_0(w[3], w[16], w[63], pars->GC_6, amp[170]); 
  FFV1_0(w[3], w[61], w[36], pars->GC_6, amp[171]); 
  FFV1_0(w[3], w[19], w[63], pars->GC_6, amp[172]); 
  FFV1_0(w[3], w[62], w[36], pars->GC_6, amp[173]); 
  FFS2_0(w[3], w[61], w[37], pars->GC_1025, amp[174]); 
  FFS2_0(w[3], w[62], w[37], pars->GC_461, amp[175]); 
  FFS2_0(w[3], w[62], w[37], pars->GC_1025, amp[176]); 
  FFV1_0(w[56], w[2], w[32], pars->GC_2, amp[177]); 
  FFV1_0(w[57], w[2], w[32], pars->GC_2, amp[178]); 
  FFV1_3_0(w[56], w[2], w[35], pars->GC_264, pars->GC_203, amp[179]); 
  FFV1_3_0(w[57], w[2], w[35], pars->GC_264, pars->GC_203, amp[180]); 
  FFV1_0(w[21], w[2], w[63], pars->GC_6, amp[181]); 
  FFV1_0(w[56], w[2], w[36], pars->GC_6, amp[182]); 
  FFV1_0(w[23], w[2], w[63], pars->GC_6, amp[183]); 
  FFV1_0(w[57], w[2], w[36], pars->GC_6, amp[184]); 
  FFS2_0(w[56], w[2], w[37], pars->GC_1025, amp[185]); 
  FFS2_0(w[57], w[2], w[37], pars->GC_461, amp[186]); 
  FFS2_0(w[57], w[2], w[37], pars->GC_1025, amp[187]); 
  FFV1_0(w[1], w[58], w[7], pars->GC_1, amp[188]); 
  FFV1_3_0(w[1], w[58], w[9], pars->GC_263, -pars->GC_203, amp[189]); 
  FFV1_0(w[1], w[15], w[60], pars->GC_6, amp[190]); 
  FFV1_0(w[1], w[58], w[11], pars->GC_6, amp[191]); 
  FFS2_0(w[1], w[58], w[12], pars->GC_566, amp[192]); 
  FFS2_0(w[1], w[58], w[14], pars->GC_566, amp[193]); 
  FFV3_0(w[1], w[61], w[20], pars->GC_202, amp[194]); 
  FFV3_0(w[1], w[62], w[20], pars->GC_202, amp[195]); 

}
double gb_ttxhb::matrix_gb_ttxhb() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 196; 
  const int ncolor = 4; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = -1./2. * amp[3] - 1./2. * amp[10] - 1./2. * amp[15] - 1./2. *
      amp[19] + amp[22] + amp[23] + amp[24] - 1./2. * amp[27] + amp[29] - 1./2.
      * amp[32] + amp[35] + amp[36] + amp[37] - 1./2. * amp[41] + amp[44] -
      1./2. * amp[50] - 1./2. * amp[51] - 1./2. * amp[62] - 1./2. * amp[63] +
      amp[67] + amp[68] + amp[69] + amp[70] + amp[71] + amp[72] - 1./2. *
      amp[75] - 1./2. * amp[161] - 1./2. * std::complex<double> (0, 1) *
      amp[162] + amp[165] + 1./2. * std::complex<double> (0, 1) * amp[170] -
      1./2. * amp[171] + 1./2. * std::complex<double> (0, 1) * amp[172] - 1./2.
      * amp[173] + 1./2. * std::complex<double> (0, 1) * amp[181] + 1./2. *
      std::complex<double> (0, 1) * amp[183] - 1./2. * std::complex<double> (0,
      1) * amp[190] + amp[194] + amp[195];
  jamp[1] = -amp[39] - amp[40] + 1./6. * amp[41] - amp[42] - amp[43] - amp[45]
      - amp[46] - amp[47] - amp[48] - amp[49] + 1./6. * amp[50] + 1./6. *
      amp[51] - amp[52] - amp[53] - amp[54] - amp[55] - amp[56] - amp[57] -
      amp[58] - amp[59] - amp[60] - amp[61] + 1./6. * amp[62] + 1./6. * amp[63]
      - amp[64] - amp[65] - amp[66] - amp[73] - amp[74] + 1./6. * amp[75] -
      amp[76] - amp[77] - amp[82] - amp[83] + 1./6. * amp[84] - amp[85] -
      amp[86] - amp[88] - amp[89] - amp[90] - amp[91] - amp[92] + 1./6. *
      amp[93] + 1./6. * amp[94] - amp[95] - amp[96] - amp[97] - amp[98] -
      amp[99] - amp[100] - amp[101] - amp[102] - amp[103] - amp[104] + 1./6. *
      amp[105] + 1./6. * amp[106] - amp[107] - amp[108] - amp[109] - amp[112] -
      amp[113] + 1./6. * amp[114] - amp[115] - amp[116] - amp[166] - amp[167] -
      amp[168] - amp[169] + 1./6. * amp[171] + 1./6. * amp[173] - amp[174] -
      amp[175] - amp[176] - amp[177] - amp[178] - amp[179] - amp[180] + 1./6. *
      amp[182] + 1./6. * amp[184] - amp[185] - amp[186] - amp[187];
  jamp[2] = -amp[0] - amp[1] - amp[2] + 1./6. * amp[3] - amp[4] - amp[5] -
      amp[6] - amp[7] - amp[8] - amp[9] + 1./6. * amp[10] - amp[11] - amp[12] -
      amp[13] - amp[14] + 1./6. * amp[15] - amp[16] - amp[17] - amp[18] + 1./6.
      * amp[19] - amp[20] - amp[21] - amp[25] - amp[26] + 1./6. * amp[27] -
      amp[28] - amp[30] - amp[31] + 1./6. * amp[32] - amp[33] - amp[34] -
      amp[38] - amp[122] - amp[123] + 1./6. * amp[124] - amp[125] - amp[126] -
      amp[127] - amp[128] - amp[129] + 1./6. * amp[130] - amp[131] - amp[132] -
      amp[133] - amp[134] - amp[135] - amp[136] + 1./6. * amp[137] - amp[138] -
      amp[140] - amp[141] + 1./6. * amp[142] - amp[143] - amp[144] - amp[146] -
      amp[147] + 1./6. * amp[148] - amp[149] - amp[150] - amp[151] + 1./6. *
      amp[152] - amp[153] - amp[154] - amp[155] - amp[159] - amp[160] + 1./6. *
      amp[161] - amp[163] - amp[164] - amp[188] - amp[189] + 1./6. * amp[191] -
      amp[192] - amp[193];
  jamp[3] = +amp[78] + amp[79] + amp[80] + amp[81] - 1./2. * amp[84] + amp[87]
      - 1./2. * amp[93] - 1./2. * amp[94] - 1./2. * amp[105] - 1./2. * amp[106]
      + amp[110] + amp[111] - 1./2. * amp[114] + amp[117] + amp[118] + amp[119]
      + amp[120] + amp[121] - 1./2. * amp[124] - 1./2. * amp[130] - 1./2. *
      amp[137] + amp[139] - 1./2. * amp[142] + amp[145] - 1./2. * amp[148] -
      1./2. * amp[152] + amp[156] + amp[157] + amp[158] + 1./2. *
      std::complex<double> (0, 1) * amp[162] - 1./2. * std::complex<double> (0,
      1) * amp[170] - 1./2. * std::complex<double> (0, 1) * amp[172] - 1./2. *
      std::complex<double> (0, 1) * amp[181] - 1./2. * amp[182] - 1./2. *
      std::complex<double> (0, 1) * amp[183] - 1./2. * amp[184] + 1./2. *
      std::complex<double> (0, 1) * amp[190] - 1./2. * amp[191];

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



