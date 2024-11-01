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
// Process: g b > t t~ h b NProp=0 SMHLOOP=0 NP==1

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
  FFS2_3(w[3], w[2], pars->GC_461, pars->mdl_MH, pars->mdl_WH, w[7]); 
  FFS2_2(w[6], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[8]); 
  FFS2_3(w[6], w[5], pars->GC_566, pars->mdl_MH, pars->mdl_WH, w[9]); 
  FFS2_1(w[5], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[10]); 
  FFS2_1(w[2], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[11]); 
  FFV1P0_3(w[6], w[5], pars->GC_1, pars->ZERO, pars->ZERO, w[12]); 
  FFV1_3_3(w[6], w[5], pars->GC_263, -pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[13]);
  FFV1P0_3(w[6], w[5], pars->GC_6, pars->ZERO, pars->ZERO, w[14]); 
  FFS2_1(w[2], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[15]); 
  FFV3_3(w[3], w[5], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[16]); 
  FFS2_2(w[3], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[17]); 
  FFV3_3(w[6], w[2], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[18]); 
  FFS2_2(w[3], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[19]); 
  FFSS2_3(w[3], w[2], w[4], pars->GC_364, pars->mdl_MH, pars->mdl_WH, w[20]); 
  FFV1_1(w[2], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[21]); 
  FFS2_2(w[1], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[22]); 
  FFS2_3(w[3], w[21], pars->GC_461, pars->mdl_MH, pars->mdl_WH, w[23]); 
  FFV1P0_3(w[1], w[5], pars->GC_1, pars->ZERO, pars->ZERO, w[24]); 
  FFS2_1(w[21], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[25]); 
  FFV1_3_3(w[1], w[5], pars->GC_263, -pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[26]);
  FFV1P0_3(w[1], w[5], pars->GC_6, pars->ZERO, pars->ZERO, w[27]); 
  FFS2_3(w[1], w[5], pars->GC_566, pars->mdl_MH, pars->mdl_WH, w[28]); 
  FFS2_1(w[21], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[29]); 
  FFV3_3(w[1], w[21], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[30]); 
  FFV1_2(w[3], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[31]); 
  FFV3_3(w[1], w[2], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[32]); 
  FFS2_2(w[31], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[33]); 
  FFS2_3(w[31], w[2], pars->GC_461, pars->mdl_MH, pars->mdl_WH, w[34]); 
  FFS2_2(w[31], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[35]); 
  FFV3_3(w[31], w[5], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[36]); 
  FFV1_1(w[5], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[37]); 
  FFS2_3(w[1], w[37], pars->GC_566, pars->mdl_MH, pars->mdl_WH, w[38]); 
  FFS2_1(w[37], w[4], pars->GC_566, pars->mdl_MB, pars->ZERO, w[39]); 
  FFV1P0_3(w[1], w[37], pars->GC_1, pars->ZERO, pars->ZERO, w[40]); 
  FFV1_3_3(w[1], w[37], pars->GC_263, -pars->GC_203, pars->mdl_MZ,
      pars->mdl_WZ, w[41]);
  FFV1P0_3(w[1], w[37], pars->GC_6, pars->ZERO, pars->ZERO, w[42]); 
  FFV3_3(w[3], w[37], pars->GC_202, pars->mdl_MW, pars->mdl_WW, w[43]); 
  FFV1_2(w[17], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[44]); 
  FFV1_2(w[22], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[45]); 
  FFV1_1(w[11], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[46]); 
  VVV5P0_1(w[0], w[27], pars->GC_7, pars->ZERO, pars->ZERO, w[47]); 
  FFV1_1(w[15], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[48]); 
  FFV1_2(w[19], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[49]); 
  FFV1_1(w[10], w[0], pars->GC_6, pars->mdl_MB, pars->ZERO, w[50]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFS2_0(w[8], w[5], w[7], pars->GC_566, amp[0]); 
  SSS1_0(w[7], w[4], w[9], pars->GC_336, amp[1]); 
  FFS2_0(w[6], w[10], w[7], pars->GC_566, amp[2]); 
  FFV1_0(w[3], w[11], w[12], pars->GC_2, amp[3]); 
  FFV1_3_0(w[3], w[11], w[13], pars->GC_264, pars->GC_203, amp[4]); 
  FFV1_0(w[3], w[11], w[14], pars->GC_6, amp[5]); 
  FFS2_0(w[3], w[11], w[9], pars->GC_1025, amp[6]); 
  FFS2_0(w[3], w[15], w[9], pars->GC_461, amp[7]); 
  FFV3_0(w[6], w[11], w[16], pars->GC_202, amp[8]); 
  FFV3_0(w[17], w[5], w[18], pars->GC_202, amp[9]); 
  FFV1_0(w[17], w[2], w[12], pars->GC_2, amp[10]); 
  FFV1_3_0(w[17], w[2], w[13], pars->GC_264, pars->GC_203, amp[11]); 
  FFV1_0(w[17], w[2], w[14], pars->GC_6, amp[12]); 
  FFS2_0(w[17], w[2], w[9], pars->GC_1025, amp[13]); 
  FFS2_0(w[19], w[2], w[9], pars->GC_461, amp[14]); 
  FFS2_0(w[6], w[5], w[20], pars->GC_566, amp[15]); 
  FFS2_0(w[22], w[5], w[23], pars->GC_566, amp[16]); 
  FFV1_0(w[3], w[25], w[24], pars->GC_2, amp[17]); 
  FFV1_3_0(w[3], w[25], w[26], pars->GC_264, pars->GC_203, amp[18]); 
  FFV1_0(w[3], w[25], w[27], pars->GC_6, amp[19]); 
  FFSS2_0(w[3], w[21], w[28], w[4], pars->GC_364, amp[20]); 
  SSS1_0(w[28], w[4], w[23], pars->GC_336, amp[21]); 
  FFS2_0(w[3], w[25], w[28], pars->GC_1025, amp[22]); 
  FFS2_0(w[3], w[29], w[28], pars->GC_461, amp[23]); 
  FFV1_0(w[17], w[21], w[24], pars->GC_2, amp[24]); 
  FFV1_3_0(w[17], w[21], w[26], pars->GC_264, pars->GC_203, amp[25]); 
  FFV1_0(w[17], w[21], w[27], pars->GC_6, amp[26]); 
  FFS2_0(w[17], w[21], w[28], pars->GC_1025, amp[27]); 
  FFS2_0(w[19], w[21], w[28], pars->GC_461, amp[28]); 
  FFV3_0(w[17], w[5], w[30], pars->GC_202, amp[29]); 
  FFV3_0(w[1], w[25], w[16], pars->GC_202, amp[30]); 
  FFS2_0(w[1], w[10], w[23], pars->GC_566, amp[31]); 
  FFV3_0(w[33], w[5], w[32], pars->GC_202, amp[32]); 
  FFS2_0(w[22], w[5], w[34], pars->GC_566, amp[33]); 
  FFV1_0(w[33], w[2], w[24], pars->GC_2, amp[34]); 
  FFV1_3_0(w[33], w[2], w[26], pars->GC_264, pars->GC_203, amp[35]); 
  FFV1_0(w[33], w[2], w[27], pars->GC_6, amp[36]); 
  FFSS2_0(w[31], w[2], w[28], w[4], pars->GC_364, amp[37]); 
  SSS1_0(w[28], w[4], w[34], pars->GC_336, amp[38]); 
  FFS2_0(w[33], w[2], w[28], pars->GC_1025, amp[39]); 
  FFS2_0(w[35], w[2], w[28], pars->GC_461, amp[40]); 
  FFV1_0(w[31], w[11], w[24], pars->GC_2, amp[41]); 
  FFV1_3_0(w[31], w[11], w[26], pars->GC_264, pars->GC_203, amp[42]); 
  FFV1_0(w[31], w[11], w[27], pars->GC_6, amp[43]); 
  FFS2_0(w[31], w[11], w[28], pars->GC_1025, amp[44]); 
  FFS2_0(w[31], w[15], w[28], pars->GC_461, amp[45]); 
  FFV3_0(w[1], w[11], w[36], pars->GC_202, amp[46]); 
  FFS2_0(w[1], w[10], w[34], pars->GC_566, amp[47]); 
  FFV3_0(w[17], w[37], w[32], pars->GC_202, amp[48]); 
  FFS2_0(w[22], w[37], w[7], pars->GC_566, amp[49]); 
  SSS1_0(w[7], w[4], w[38], pars->GC_336, amp[50]); 
  FFS2_0(w[1], w[39], w[7], pars->GC_566, amp[51]); 
  FFV1_0(w[3], w[11], w[40], pars->GC_2, amp[52]); 
  FFV1_3_0(w[3], w[11], w[41], pars->GC_264, pars->GC_203, amp[53]); 
  FFV1_0(w[3], w[11], w[42], pars->GC_6, amp[54]); 
  FFS2_0(w[3], w[11], w[38], pars->GC_1025, amp[55]); 
  FFV3_0(w[1], w[11], w[43], pars->GC_202, amp[56]); 
  FFS2_0(w[3], w[15], w[38], pars->GC_461, amp[57]); 
  FFV1_0(w[17], w[2], w[40], pars->GC_2, amp[58]); 
  FFV1_3_0(w[17], w[2], w[41], pars->GC_264, pars->GC_203, amp[59]); 
  FFV1_0(w[17], w[2], w[42], pars->GC_6, amp[60]); 
  FFS2_0(w[17], w[2], w[38], pars->GC_1025, amp[61]); 
  FFS2_0(w[19], w[2], w[38], pars->GC_461, amp[62]); 
  FFS2_0(w[1], w[37], w[20], pars->GC_566, amp[63]); 
  FFV3_0(w[44], w[5], w[32], pars->GC_202, amp[64]); 
  FFS2_0(w[45], w[5], w[7], pars->GC_566, amp[65]); 
  FFV1_0(w[3], w[46], w[24], pars->GC_2, amp[66]); 
  FFV1_3_0(w[3], w[46], w[26], pars->GC_264, pars->GC_203, amp[67]); 
  FFV1_0(w[3], w[11], w[47], pars->GC_6, amp[68]); 
  FFV1_0(w[3], w[46], w[27], pars->GC_6, amp[69]); 
  FFS2_0(w[3], w[46], w[28], pars->GC_1025, amp[70]); 
  FFS2_0(w[3], w[48], w[28], pars->GC_461, amp[71]); 
  FFV1_0(w[44], w[2], w[24], pars->GC_2, amp[72]); 
  FFV1_3_0(w[44], w[2], w[26], pars->GC_264, pars->GC_203, amp[73]); 
  FFV1_0(w[17], w[2], w[47], pars->GC_6, amp[74]); 
  FFV1_0(w[44], w[2], w[27], pars->GC_6, amp[75]); 
  FFS2_0(w[44], w[2], w[28], pars->GC_1025, amp[76]); 
  FFS2_0(w[49], w[2], w[28], pars->GC_461, amp[77]); 
  FFS2_0(w[1], w[50], w[7], pars->GC_566, amp[78]); 
  FFV3_0(w[1], w[46], w[16], pars->GC_202, amp[79]); 

}
double gb_ttxhb::matrix_gb_ttxhb() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 80; 
  const int ncolor = 4; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = -1./2. * amp[5] + amp[8] + amp[9] - 1./2. * amp[12] - 1./2. *
      amp[19] - 1./2. * amp[26] + amp[29] + amp[30] + 1./2. *
      std::complex<double> (0, 1) * amp[68] - 1./2. * amp[69] + 1./2. *
      std::complex<double> (0, 1) * amp[74] + amp[79];
  jamp[1] = -amp[16] - amp[17] - amp[18] + 1./6. * amp[19] - amp[20] - amp[21]
      - amp[22] - amp[23] - amp[24] - amp[25] + 1./6. * amp[26] - amp[27] -
      amp[28] - amp[31] - amp[33] - amp[34] - amp[35] + 1./6. * amp[36] -
      amp[37] - amp[38] - amp[39] - amp[40] - amp[41] - amp[42] + 1./6. *
      amp[43] - amp[44] - amp[45] - amp[47] - amp[66] - amp[67] + 1./6. *
      amp[69] - amp[70] - amp[71] - amp[72] - amp[73] + 1./6. * amp[75] -
      amp[76] - amp[77];
  jamp[2] = -amp[0] - amp[1] - amp[2] - amp[3] - amp[4] + 1./6. * amp[5] -
      amp[6] - amp[7] - amp[10] - amp[11] + 1./6. * amp[12] - amp[13] - amp[14]
      - amp[15] - amp[49] - amp[50] - amp[51] - amp[52] - amp[53] + 1./6. *
      amp[54] - amp[55] - amp[57] - amp[58] - amp[59] + 1./6. * amp[60] -
      amp[61] - amp[62] - amp[63] - amp[65] - amp[78];
  jamp[3] = +amp[32] - 1./2. * amp[36] - 1./2. * amp[43] + amp[46] + amp[48] -
      1./2. * amp[54] + amp[56] - 1./2. * amp[60] + amp[64] - 1./2. *
      std::complex<double> (0, 1) * amp[68] - 1./2. * std::complex<double> (0,
      1) * amp[74] - 1./2. * amp[75];

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



