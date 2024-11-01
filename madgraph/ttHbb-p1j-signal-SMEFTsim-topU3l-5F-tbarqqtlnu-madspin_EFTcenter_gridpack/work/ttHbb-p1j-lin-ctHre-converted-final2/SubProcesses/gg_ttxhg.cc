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
// Process: g g > t t~ h g NProp=0 SMHLOOP=0 NP^2==1

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
  FFS2_1(w[2], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[10]); 
  FFV1_2(w[3], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[11]); 
  FFV1_1(w[2], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[12]); 
  FFS2_2(w[3], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[13]); 
  FFS2_2(w[3], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[14]); 
  FFV1_1(w[2], w[6], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[15]); 
  FFV1_1(w[2], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[16]); 
  FFV1_2(w[3], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[17]); 
  FFS2_1(w[16], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[18]); 
  FFS2_1(w[16], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[19]); 
  FFV1_1(w[16], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[20]); 
  VVV5P0_1(w[1], w[5], pars->GC_7, pars->ZERO, pars->ZERO, w[21]); 
  FFV1_1(w[16], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[22]); 
  FFV1_2(w[3], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[23]); 
  FFV1_1(w[2], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[24]); 
  FFS2_2(w[23], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[25]); 
  FFS2_2(w[23], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[26]); 
  FFV1_2(w[23], w[5], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[27]); 
  FFV1_2(w[23], w[1], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[28]); 
  VVV5P0_1(w[0], w[5], pars->GC_7, pars->ZERO, pars->ZERO, w[29]); 
  FFV1_2(w[3], w[29], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[30]); 
  FFV1_1(w[2], w[29], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[31]); 
  VVV5P0_1(w[29], w[1], pars->GC_7, pars->ZERO, pars->ZERO, w[32]); 
  FFV1_1(w[24], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[33]); 
  FFV1_2(w[13], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[34]); 
  FFV1_2(w[14], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[35]); 
  FFV1_2(w[11], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[36]); 
  FFV1_2(w[17], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[37]); 
  FFV1_1(w[7], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[38]); 
  FFV1_1(w[10], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[39]); 
  FFV1_1(w[12], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[40]); 
  VVV5P0_1(w[0], w[21], pars->GC_7, pars->ZERO, pars->ZERO, w[41]); 
  VVVV1P0_1(w[0], w[1], w[5], pars->GC_8, pars->ZERO, pars->ZERO, w[42]); 
  VVVV9P0_1(w[0], w[1], w[5], pars->GC_8, pars->ZERO, pars->ZERO, w[43]); 
  VVVV10P0_1(w[0], w[1], w[5], pars->GC_8, pars->ZERO, pars->ZERO, w[44]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  FFV1_0(w[8], w[7], w[5], pars->GC_6, amp[0]); 
  FFV1_0(w[3], w[7], w[9], pars->GC_6, amp[1]); 
  FFV1_0(w[8], w[10], w[5], pars->GC_6, amp[2]); 
  FFV1_0(w[3], w[10], w[9], pars->GC_6, amp[3]); 
  FFV1_0(w[11], w[7], w[6], pars->GC_6, amp[4]); 
  FFV1_0(w[11], w[10], w[6], pars->GC_6, amp[5]); 
  FFS2_0(w[8], w[12], w[4], pars->GC_461, amp[6]); 
  FFS2_0(w[8], w[12], w[4], pars->GC_1025, amp[7]); 
  FFV1_0(w[13], w[12], w[6], pars->GC_6, amp[8]); 
  FFV1_0(w[14], w[12], w[6], pars->GC_6, amp[9]); 
  FFV1_0(w[13], w[15], w[5], pars->GC_6, amp[10]); 
  FFV1_0(w[13], w[2], w[9], pars->GC_6, amp[11]); 
  FFV1_0(w[14], w[15], w[5], pars->GC_6, amp[12]); 
  FFV1_0(w[14], w[2], w[9], pars->GC_6, amp[13]); 
  FFS2_0(w[11], w[15], w[4], pars->GC_461, amp[14]); 
  FFS2_0(w[11], w[15], w[4], pars->GC_1025, amp[15]); 
  FFV1_0(w[17], w[18], w[5], pars->GC_6, amp[16]); 
  FFV1_0(w[17], w[19], w[5], pars->GC_6, amp[17]); 
  FFS2_0(w[17], w[20], w[4], pars->GC_461, amp[18]); 
  FFS2_0(w[17], w[20], w[4], pars->GC_1025, amp[19]); 
  FFV1_0(w[3], w[18], w[21], pars->GC_6, amp[20]); 
  FFV1_0(w[3], w[19], w[21], pars->GC_6, amp[21]); 
  FFV1_0(w[13], w[16], w[21], pars->GC_6, amp[22]); 
  FFV1_0(w[14], w[16], w[21], pars->GC_6, amp[23]); 
  FFV1_0(w[13], w[22], w[5], pars->GC_6, amp[24]); 
  FFV1_0(w[13], w[20], w[1], pars->GC_6, amp[25]); 
  FFV1_0(w[14], w[22], w[5], pars->GC_6, amp[26]); 
  FFV1_0(w[14], w[20], w[1], pars->GC_6, amp[27]); 
  FFS2_0(w[11], w[22], w[4], pars->GC_461, amp[28]); 
  FFS2_0(w[11], w[22], w[4], pars->GC_1025, amp[29]); 
  FFV1_0(w[11], w[18], w[1], pars->GC_6, amp[30]); 
  FFV1_0(w[11], w[19], w[1], pars->GC_6, amp[31]); 
  FFV1_0(w[25], w[24], w[5], pars->GC_6, amp[32]); 
  FFV1_0(w[26], w[24], w[5], pars->GC_6, amp[33]); 
  FFS2_0(w[27], w[24], w[4], pars->GC_461, amp[34]); 
  FFS2_0(w[27], w[24], w[4], pars->GC_1025, amp[35]); 
  FFV1_0(w[25], w[2], w[21], pars->GC_6, amp[36]); 
  FFV1_0(w[26], w[2], w[21], pars->GC_6, amp[37]); 
  FFV1_0(w[23], w[7], w[21], pars->GC_6, amp[38]); 
  FFV1_0(w[23], w[10], w[21], pars->GC_6, amp[39]); 
  FFV1_0(w[28], w[7], w[5], pars->GC_6, amp[40]); 
  FFV1_0(w[27], w[7], w[1], pars->GC_6, amp[41]); 
  FFV1_0(w[28], w[10], w[5], pars->GC_6, amp[42]); 
  FFV1_0(w[27], w[10], w[1], pars->GC_6, amp[43]); 
  FFS2_0(w[28], w[12], w[4], pars->GC_461, amp[44]); 
  FFS2_0(w[28], w[12], w[4], pars->GC_1025, amp[45]); 
  FFV1_0(w[25], w[12], w[1], pars->GC_6, amp[46]); 
  FFV1_0(w[26], w[12], w[1], pars->GC_6, amp[47]); 
  FFS2_0(w[30], w[24], w[4], pars->GC_461, amp[48]); 
  FFS2_0(w[30], w[24], w[4], pars->GC_1025, amp[49]); 
  FFV1_0(w[13], w[24], w[29], pars->GC_6, amp[50]); 
  FFV1_0(w[14], w[24], w[29], pars->GC_6, amp[51]); 
  FFS2_0(w[17], w[31], w[4], pars->GC_461, amp[52]); 
  FFS2_0(w[17], w[31], w[4], pars->GC_1025, amp[53]); 
  FFV1_0(w[17], w[7], w[29], pars->GC_6, amp[54]); 
  FFV1_0(w[17], w[10], w[29], pars->GC_6, amp[55]); 
  FFV1_0(w[3], w[7], w[32], pars->GC_6, amp[56]); 
  FFV1_0(w[30], w[7], w[1], pars->GC_6, amp[57]); 
  FFV1_0(w[3], w[10], w[32], pars->GC_6, amp[58]); 
  FFV1_0(w[30], w[10], w[1], pars->GC_6, amp[59]); 
  FFV1_0(w[13], w[2], w[32], pars->GC_6, amp[60]); 
  FFV1_0(w[13], w[31], w[1], pars->GC_6, amp[61]); 
  FFV1_0(w[14], w[2], w[32], pars->GC_6, amp[62]); 
  FFV1_0(w[14], w[31], w[1], pars->GC_6, amp[63]); 
  FFV1_0(w[13], w[33], w[5], pars->GC_6, amp[64]); 
  FFV1_0(w[34], w[24], w[5], pars->GC_6, amp[65]); 
  FFV1_0(w[14], w[33], w[5], pars->GC_6, amp[66]); 
  FFV1_0(w[35], w[24], w[5], pars->GC_6, amp[67]); 
  FFS2_0(w[11], w[33], w[4], pars->GC_461, amp[68]); 
  FFS2_0(w[11], w[33], w[4], pars->GC_1025, amp[69]); 
  FFS2_0(w[36], w[24], w[4], pars->GC_461, amp[70]); 
  FFS2_0(w[36], w[24], w[4], pars->GC_1025, amp[71]); 
  FFV1_0(w[37], w[7], w[5], pars->GC_6, amp[72]); 
  FFV1_0(w[17], w[38], w[5], pars->GC_6, amp[73]); 
  FFV1_0(w[37], w[10], w[5], pars->GC_6, amp[74]); 
  FFV1_0(w[17], w[39], w[5], pars->GC_6, amp[75]); 
  FFS2_0(w[37], w[12], w[4], pars->GC_461, amp[76]); 
  FFS2_0(w[37], w[12], w[4], pars->GC_1025, amp[77]); 
  FFS2_0(w[17], w[40], w[4], pars->GC_461, amp[78]); 
  FFS2_0(w[17], w[40], w[4], pars->GC_1025, amp[79]); 
  FFV1_0(w[3], w[7], w[41], pars->GC_6, amp[80]); 
  FFV1_0(w[3], w[38], w[21], pars->GC_6, amp[81]); 
  FFV1_0(w[3], w[10], w[41], pars->GC_6, amp[82]); 
  FFV1_0(w[3], w[39], w[21], pars->GC_6, amp[83]); 
  FFV1_0(w[13], w[2], w[41], pars->GC_6, amp[84]); 
  FFV1_0(w[34], w[2], w[21], pars->GC_6, amp[85]); 
  FFV1_0(w[14], w[2], w[41], pars->GC_6, amp[86]); 
  FFV1_0(w[35], w[2], w[21], pars->GC_6, amp[87]); 
  FFV1_0(w[11], w[38], w[1], pars->GC_6, amp[88]); 
  FFV1_0(w[36], w[7], w[1], pars->GC_6, amp[89]); 
  FFV1_0(w[11], w[39], w[1], pars->GC_6, amp[90]); 
  FFV1_0(w[36], w[10], w[1], pars->GC_6, amp[91]); 
  FFV1_0(w[13], w[40], w[1], pars->GC_6, amp[92]); 
  FFV1_0(w[34], w[12], w[1], pars->GC_6, amp[93]); 
  FFV1_0(w[14], w[40], w[1], pars->GC_6, amp[94]); 
  FFV1_0(w[35], w[12], w[1], pars->GC_6, amp[95]); 
  FFV1_0(w[3], w[7], w[42], pars->GC_6, amp[96]); 
  FFV1_0(w[3], w[7], w[43], pars->GC_6, amp[97]); 
  FFV1_0(w[3], w[7], w[44], pars->GC_6, amp[98]); 
  FFV1_0(w[3], w[10], w[42], pars->GC_6, amp[99]); 
  FFV1_0(w[3], w[10], w[43], pars->GC_6, amp[100]); 
  FFV1_0(w[3], w[10], w[44], pars->GC_6, amp[101]); 
  FFV1_0(w[13], w[2], w[42], pars->GC_6, amp[102]); 
  FFV1_0(w[13], w[2], w[43], pars->GC_6, amp[103]); 
  FFV1_0(w[13], w[2], w[44], pars->GC_6, amp[104]); 
  FFV1_0(w[14], w[2], w[42], pars->GC_6, amp[105]); 
  FFV1_0(w[14], w[2], w[43], pars->GC_6, amp[106]); 
  FFV1_0(w[14], w[2], w[44], pars->GC_6, amp[107]); 

}
double gg_ttxhg::matrix_gg_ttxhg() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 108; 
  const int ncolor = 6; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {9, 9, 9, 9, 9, 9}; 
  static const double cf[ncolor][ncolor] = {{64, -8, -8, 1, 1, 10}, {-8, 64, 1,
      10, -8, 1}, {-8, 1, 64, -8, 10, 1}, {1, 10, -8, 64, 1, -8}, {1, -8, 10,
      1, 64, -8}, {10, 1, 1, -8, -8, 64}};

  // Calculate color flows
  jamp[0] = +amp[1] + amp[3] + std::complex<double> (0, 1) * amp[4] +
      std::complex<double> (0, 1) * amp[5] + std::complex<double> (0, 1) *
      amp[10] + amp[11] + std::complex<double> (0, 1) * amp[12] + amp[13] +
      std::complex<double> (0, 1) * amp[14] + std::complex<double> (0, 1) *
      amp[15] + std::complex<double> (0, 1) * amp[20] + std::complex<double>
      (0, 1) * amp[21] + std::complex<double> (0, 1) * amp[22] +
      std::complex<double> (0, 1) * amp[23] - amp[24] - amp[26] - amp[28] -
      amp[29] - amp[30] - amp[31] + amp[80] + std::complex<double> (0, 1) *
      amp[81] + amp[82] + std::complex<double> (0, 1) * amp[83] + amp[84] +
      amp[86] - amp[88] - amp[90] + amp[96] - amp[98] + amp[99] - amp[101] +
      amp[102] - amp[104] + amp[105] - amp[107];
  jamp[1] = -amp[16] - amp[17] - amp[18] - amp[19] - std::complex<double> (0,
      1) * amp[20] - std::complex<double> (0, 1) * amp[21] -
      std::complex<double> (0, 1) * amp[22] - std::complex<double> (0, 1) *
      amp[23] - amp[25] - amp[27] + std::complex<double> (0, 1) * amp[52] +
      std::complex<double> (0, 1) * amp[53] + std::complex<double> (0, 1) *
      amp[54] + std::complex<double> (0, 1) * amp[55] + amp[56] + amp[58] +
      amp[60] + std::complex<double> (0, 1) * amp[61] + amp[62] +
      std::complex<double> (0, 1) * amp[63] - amp[73] - amp[75] - amp[80] -
      std::complex<double> (0, 1) * amp[81] - amp[82] - std::complex<double>
      (0, 1) * amp[83] - amp[84] - amp[86] - amp[96] - amp[97] - amp[99] -
      amp[100] - amp[102] - amp[103] - amp[105] - amp[106];
  jamp[2] = -amp[1] - amp[3] - std::complex<double> (0, 1) * amp[4] -
      std::complex<double> (0, 1) * amp[5] - std::complex<double> (0, 1) *
      amp[10] - amp[11] - std::complex<double> (0, 1) * amp[12] - amp[13] -
      std::complex<double> (0, 1) * amp[14] - std::complex<double> (0, 1) *
      amp[15] + std::complex<double> (0, 1) * amp[48] + std::complex<double>
      (0, 1) * amp[49] + std::complex<double> (0, 1) * amp[50] +
      std::complex<double> (0, 1) * amp[51] - amp[56] + std::complex<double>
      (0, 1) * amp[57] - amp[58] + std::complex<double> (0, 1) * amp[59] -
      amp[60] - amp[62] - amp[64] - amp[66] - amp[68] - amp[69] - amp[70] -
      amp[71] - amp[89] - amp[91] + amp[97] + amp[98] + amp[100] + amp[101] +
      amp[103] + amp[104] + amp[106] + amp[107];
  jamp[3] = -amp[32] - amp[33] - amp[34] - amp[35] + std::complex<double> (0,
      1) * amp[36] + std::complex<double> (0, 1) * amp[37] +
      std::complex<double> (0, 1) * amp[38] + std::complex<double> (0, 1) *
      amp[39] - amp[41] - amp[43] - std::complex<double> (0, 1) * amp[48] -
      std::complex<double> (0, 1) * amp[49] - std::complex<double> (0, 1) *
      amp[50] - std::complex<double> (0, 1) * amp[51] + amp[56] -
      std::complex<double> (0, 1) * amp[57] + amp[58] - std::complex<double>
      (0, 1) * amp[59] + amp[60] + amp[62] - amp[65] - amp[67] - amp[80] -
      amp[82] - amp[84] + std::complex<double> (0, 1) * amp[85] - amp[86] +
      std::complex<double> (0, 1) * amp[87] - amp[96] - amp[97] - amp[99] -
      amp[100] - amp[102] - amp[103] - amp[105] - amp[106];
  jamp[4] = +std::complex<double> (0, 1) * amp[0] - amp[1] +
      std::complex<double> (0, 1) * amp[2] - amp[3] + std::complex<double> (0,
      1) * amp[6] + std::complex<double> (0, 1) * amp[7] + std::complex<double>
      (0, 1) * amp[8] + std::complex<double> (0, 1) * amp[9] - amp[11] -
      amp[13] - std::complex<double> (0, 1) * amp[52] - std::complex<double>
      (0, 1) * amp[53] - std::complex<double> (0, 1) * amp[54] -
      std::complex<double> (0, 1) * amp[55] - amp[56] - amp[58] - amp[60] -
      std::complex<double> (0, 1) * amp[61] - amp[62] - std::complex<double>
      (0, 1) * amp[63] - amp[72] - amp[74] - amp[76] - amp[77] - amp[78] -
      amp[79] - amp[92] - amp[94] + amp[97] + amp[98] + amp[100] + amp[101] +
      amp[103] + amp[104] + amp[106] + amp[107];
  jamp[5] = -std::complex<double> (0, 1) * amp[0] + amp[1] -
      std::complex<double> (0, 1) * amp[2] + amp[3] - std::complex<double> (0,
      1) * amp[6] - std::complex<double> (0, 1) * amp[7] - std::complex<double>
      (0, 1) * amp[8] - std::complex<double> (0, 1) * amp[9] + amp[11] +
      amp[13] - std::complex<double> (0, 1) * amp[36] - std::complex<double>
      (0, 1) * amp[37] - std::complex<double> (0, 1) * amp[38] -
      std::complex<double> (0, 1) * amp[39] - amp[40] - amp[42] - amp[44] -
      amp[45] - amp[46] - amp[47] + amp[80] + amp[82] + amp[84] -
      std::complex<double> (0, 1) * amp[85] + amp[86] - std::complex<double>
      (0, 1) * amp[87] - amp[93] - amp[95] + amp[96] - amp[98] + amp[99] -
      amp[101] + amp[102] - amp[104] + amp[105] - amp[107];

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



