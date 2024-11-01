//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include "gd_ttxhd.h"
#include "HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.h"

using namespace MG5_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless; 

//==========================================================================
// Class member functions for calculating the matrix elements for
// Process: g d > t t~ h d NProp=0 SMHLOOP=0 NP^2==1
// Process: g s > t t~ h s NProp=0 SMHLOOP=0 NP^2==1

//--------------------------------------------------------------------------
// Initialize process.

void gd_ttxhd::initProc(string param_card_name) 
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
  jamp2[0] = new double[4]; 
}

//--------------------------------------------------------------------------
// Evaluate |M|^2, part independent of incoming flavour.

void gd_ttxhd::sigmaKin() 
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
        t[0] = matrix_gd_ttxhd(); 
        // Mirror initial state momenta for mirror process
        perm[0] = 1; 
        perm[1] = 0; 
        // Calculate wavefunctions
        calculate_wavefunctions(perm, helicities[ihel]); 
        // Mirror back
        perm[0] = 0; 
        perm[1] = 1; 
        // Calculate matrix elements
        t[1] = matrix_gd_ttxhd(); 
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
      t[0] = matrix_gd_ttxhd(); 
      // Mirror initial state momenta for mirror process
      perm[0] = 1; 
      perm[1] = 0; 
      // Calculate wavefunctions
      calculate_wavefunctions(perm, helicities[ihel]); 
      // Mirror back
      perm[0] = 0; 
      perm[1] = 1; 
      // Calculate matrix elements
      t[1] = matrix_gd_ttxhd(); 
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

double gd_ttxhd::sigmaHat() 
{
  // Select between the different processes
  if(id1 == 1 && id2 == 21)
  {
    // Add matrix elements for processes with beams (1, 21)
    return matrix_element[1]; 
  }
  else if(id1 == 3 && id2 == 21)
  {
    // Add matrix elements for processes with beams (3, 21)
    return matrix_element[1]; 
  }
  else if(id1 == 21 && id2 == 1)
  {
    // Add matrix elements for processes with beams (21, 1)
    return matrix_element[0]; 
  }
  else if(id1 == 21 && id2 == 3)
  {
    // Add matrix elements for processes with beams (21, 3)
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

void gd_ttxhd::calculate_wavefunctions(const int perm[], const int hel[])
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
  FFV1_2(w[1], w[0], pars->GC_6, pars->ZERO, pars->ZERO, w[6]); 
  FFV1_3_3(w[3], w[2], pars->GC_264, pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[7]);
  FFV1_3_3(w[6], w[5], pars->GC_263, -pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[8]);
  FFS2_1(w[2], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[9]); 
  FFV1P0_3(w[6], w[5], pars->GC_1, pars->ZERO, pars->ZERO, w[10]); 
  FFV1P0_3(w[6], w[5], pars->GC_6, pars->ZERO, pars->ZERO, w[11]); 
  FFS2_1(w[2], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[12]); 
  FFS2_2(w[3], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[13]); 
  FFS2_2(w[3], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[14]); 
  FFV1_1(w[2], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[15]); 
  FFV1P0_3(w[1], w[5], pars->GC_1, pars->ZERO, pars->ZERO, w[16]); 
  FFS2_1(w[15], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[17]); 
  FFS2_1(w[15], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[18]); 
  FFV1_3_3(w[1], w[5], pars->GC_263, -pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[19]);
  FFV1_3_3(w[3], w[15], pars->GC_264, pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[20]);
  FFV1P0_3(w[1], w[5], pars->GC_6, pars->ZERO, pars->ZERO, w[21]); 
  FFV1_2(w[3], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[22]); 
  FFS2_2(w[22], w[4], pars->GC_461, pars->mdl_MT, pars->mdl_WT, w[23]); 
  FFS2_2(w[22], w[4], pars->GC_1025, pars->mdl_MT, pars->mdl_WT, w[24]); 
  FFV1_3_3(w[22], w[2], pars->GC_264, pars->GC_203, pars->mdl_MZ, pars->mdl_WZ,
      w[25]);
  FFV1_1(w[5], w[0], pars->GC_6, pars->ZERO, pars->ZERO, w[26]); 
  FFV1_3_3(w[1], w[26], pars->GC_263, -pars->GC_203, pars->mdl_MZ,
      pars->mdl_WZ, w[27]);
  FFV1P0_3(w[1], w[26], pars->GC_1, pars->ZERO, pars->ZERO, w[28]); 
  FFV1P0_3(w[1], w[26], pars->GC_6, pars->ZERO, pars->ZERO, w[29]); 
  FFV1_1(w[9], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[30]); 
  FFV1_1(w[12], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[31]); 
  VVV5P0_1(w[0], w[21], pars->GC_7, pars->ZERO, pars->ZERO, w[32]); 
  FFV1_2(w[13], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[33]); 
  FFV1_2(w[14], w[0], pars->GC_6, pars->mdl_MT, pars->mdl_WT, w[34]); 

  // Calculate all amplitudes
  // Amplitude(s) for diagram number 0
  VVS3_0(w[7], w[8], w[4], pars->GC_392, amp[0]); 
  FFV1_0(w[3], w[9], w[10], pars->GC_2, amp[1]); 
  FFV1_3_0(w[3], w[9], w[8], pars->GC_264, pars->GC_203, amp[2]); 
  FFV1_0(w[3], w[9], w[11], pars->GC_6, amp[3]); 
  FFV1_0(w[3], w[12], w[10], pars->GC_2, amp[4]); 
  FFV1_3_0(w[3], w[12], w[8], pars->GC_264, pars->GC_203, amp[5]); 
  FFV1_0(w[3], w[12], w[11], pars->GC_6, amp[6]); 
  FFV1_0(w[13], w[2], w[10], pars->GC_2, amp[7]); 
  FFV1_3_0(w[13], w[2], w[8], pars->GC_264, pars->GC_203, amp[8]); 
  FFV1_0(w[13], w[2], w[11], pars->GC_6, amp[9]); 
  FFV1_0(w[14], w[2], w[10], pars->GC_2, amp[10]); 
  FFV1_3_0(w[14], w[2], w[8], pars->GC_264, pars->GC_203, amp[11]); 
  FFV1_0(w[14], w[2], w[11], pars->GC_6, amp[12]); 
  FFV1_0(w[3], w[17], w[16], pars->GC_2, amp[13]); 
  FFV1_0(w[3], w[18], w[16], pars->GC_2, amp[14]); 
  VVS3_0(w[19], w[20], w[4], pars->GC_392, amp[15]); 
  FFV1_3_0(w[3], w[17], w[19], pars->GC_264, pars->GC_203, amp[16]); 
  FFV1_3_0(w[3], w[18], w[19], pars->GC_264, pars->GC_203, amp[17]); 
  FFV1_0(w[3], w[17], w[21], pars->GC_6, amp[18]); 
  FFV1_0(w[3], w[18], w[21], pars->GC_6, amp[19]); 
  FFV1_0(w[13], w[15], w[16], pars->GC_2, amp[20]); 
  FFV1_0(w[14], w[15], w[16], pars->GC_2, amp[21]); 
  FFV1_3_0(w[13], w[15], w[19], pars->GC_264, pars->GC_203, amp[22]); 
  FFV1_3_0(w[14], w[15], w[19], pars->GC_264, pars->GC_203, amp[23]); 
  FFV1_0(w[13], w[15], w[21], pars->GC_6, amp[24]); 
  FFV1_0(w[14], w[15], w[21], pars->GC_6, amp[25]); 
  FFV1_0(w[23], w[2], w[16], pars->GC_2, amp[26]); 
  FFV1_0(w[24], w[2], w[16], pars->GC_2, amp[27]); 
  VVS3_0(w[19], w[25], w[4], pars->GC_392, amp[28]); 
  FFV1_3_0(w[23], w[2], w[19], pars->GC_264, pars->GC_203, amp[29]); 
  FFV1_3_0(w[24], w[2], w[19], pars->GC_264, pars->GC_203, amp[30]); 
  FFV1_0(w[23], w[2], w[21], pars->GC_6, amp[31]); 
  FFV1_0(w[24], w[2], w[21], pars->GC_6, amp[32]); 
  FFV1_0(w[22], w[9], w[16], pars->GC_2, amp[33]); 
  FFV1_0(w[22], w[12], w[16], pars->GC_2, amp[34]); 
  FFV1_3_0(w[22], w[9], w[19], pars->GC_264, pars->GC_203, amp[35]); 
  FFV1_3_0(w[22], w[12], w[19], pars->GC_264, pars->GC_203, amp[36]); 
  FFV1_0(w[22], w[9], w[21], pars->GC_6, amp[37]); 
  FFV1_0(w[22], w[12], w[21], pars->GC_6, amp[38]); 
  VVS3_0(w[7], w[27], w[4], pars->GC_392, amp[39]); 
  FFV1_0(w[3], w[9], w[28], pars->GC_2, amp[40]); 
  FFV1_3_0(w[3], w[9], w[27], pars->GC_264, pars->GC_203, amp[41]); 
  FFV1_0(w[3], w[9], w[29], pars->GC_6, amp[42]); 
  FFV1_0(w[3], w[12], w[28], pars->GC_2, amp[43]); 
  FFV1_3_0(w[3], w[12], w[27], pars->GC_264, pars->GC_203, amp[44]); 
  FFV1_0(w[3], w[12], w[29], pars->GC_6, amp[45]); 
  FFV1_0(w[13], w[2], w[28], pars->GC_2, amp[46]); 
  FFV1_3_0(w[13], w[2], w[27], pars->GC_264, pars->GC_203, amp[47]); 
  FFV1_0(w[13], w[2], w[29], pars->GC_6, amp[48]); 
  FFV1_0(w[14], w[2], w[28], pars->GC_2, amp[49]); 
  FFV1_3_0(w[14], w[2], w[27], pars->GC_264, pars->GC_203, amp[50]); 
  FFV1_0(w[14], w[2], w[29], pars->GC_6, amp[51]); 
  FFV1_0(w[3], w[30], w[16], pars->GC_2, amp[52]); 
  FFV1_0(w[3], w[31], w[16], pars->GC_2, amp[53]); 
  FFV1_3_0(w[3], w[30], w[19], pars->GC_264, pars->GC_203, amp[54]); 
  FFV1_3_0(w[3], w[31], w[19], pars->GC_264, pars->GC_203, amp[55]); 
  FFV1_0(w[3], w[9], w[32], pars->GC_6, amp[56]); 
  FFV1_0(w[3], w[30], w[21], pars->GC_6, amp[57]); 
  FFV1_0(w[3], w[12], w[32], pars->GC_6, amp[58]); 
  FFV1_0(w[3], w[31], w[21], pars->GC_6, amp[59]); 
  FFV1_0(w[33], w[2], w[16], pars->GC_2, amp[60]); 
  FFV1_0(w[34], w[2], w[16], pars->GC_2, amp[61]); 
  FFV1_3_0(w[33], w[2], w[19], pars->GC_264, pars->GC_203, amp[62]); 
  FFV1_3_0(w[34], w[2], w[19], pars->GC_264, pars->GC_203, amp[63]); 
  FFV1_0(w[13], w[2], w[32], pars->GC_6, amp[64]); 
  FFV1_0(w[33], w[2], w[21], pars->GC_6, amp[65]); 
  FFV1_0(w[14], w[2], w[32], pars->GC_6, amp[66]); 
  FFV1_0(w[34], w[2], w[21], pars->GC_6, amp[67]); 

}
double gd_ttxhd::matrix_gd_ttxhd() 
{
  int i, j; 
  // Local variables
  const int ngraphs = 68; 
  const int ncolor = 4; 
  std::complex<double> ztemp; 
  std::complex<double> jamp[ncolor]; 
  // The color matrix;
  static const double denom[ncolor] = {1, 1, 1, 1}; 
  static const double cf[ncolor][ncolor] = {{12, 4, 4, 0}, {4, 12, 0, 4}, {4,
      0, 12, 4}, {0, 4, 4, 12}};

  // Calculate color flows
  jamp[0] = +1./2. * (-amp[3] - amp[6] - amp[9] - amp[12] - amp[18] - amp[19] -
      amp[24] - amp[25] + std::complex<double> (0, 1) * amp[56] - amp[57] +
      std::complex<double> (0, 1) * amp[58] - amp[59] + std::complex<double>
      (0, 1) * amp[64] + std::complex<double> (0, 1) * amp[66]);
  jamp[1] = -amp[13] - amp[14] - amp[15] - amp[16] - amp[17] + 1./6. * amp[18]
      + 1./6. * amp[19] - amp[20] - amp[21] - amp[22] - amp[23] + 1./6. *
      amp[24] + 1./6. * amp[25] - amp[26] - amp[27] - amp[28] - amp[29] -
      amp[30] + 1./6. * amp[31] + 1./6. * amp[32] - amp[33] - amp[34] - amp[35]
      - amp[36] + 1./6. * amp[37] + 1./6. * amp[38] - amp[52] - amp[53] -
      amp[54] - amp[55] + 1./6. * amp[57] + 1./6. * amp[59] - amp[60] - amp[61]
      - amp[62] - amp[63] + 1./6. * amp[65] + 1./6. * amp[67];
  jamp[2] = -amp[0] - amp[1] - amp[2] + 1./6. * amp[3] - amp[4] - amp[5] +
      1./6. * amp[6] - amp[7] - amp[8] + 1./6. * amp[9] - amp[10] - amp[11] +
      1./6. * amp[12] - amp[39] - amp[40] - amp[41] + 1./6. * amp[42] - amp[43]
      - amp[44] + 1./6. * amp[45] - amp[46] - amp[47] + 1./6. * amp[48] -
      amp[49] - amp[50] + 1./6. * amp[51];
  jamp[3] = +1./2. * (-amp[31] - amp[32] - amp[37] - amp[38] - amp[42] -
      amp[45] - amp[48] - amp[51] - std::complex<double> (0, 1) * amp[56] -
      std::complex<double> (0, 1) * amp[58] - std::complex<double> (0, 1) *
      amp[64] - amp[65] - std::complex<double> (0, 1) * amp[66] - amp[67]);

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



