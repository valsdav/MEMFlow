//==========================================================================
// This file has been automatically generated for C++ by
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#include <iostream> 
#include <iomanip> 
#include "Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.h"

// Initialize static instance
Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless *
    Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::instance = 0;

// Function to get static instance - only one instance per program
Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless *
    Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::getInstance()
{
  if (instance == 0)
    instance = new Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless(); 

  return instance; 
}

void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::setIndependentParameters(SLHAReader& slha)
{
  // Define "zero"
  zero = 0; 
  ZERO = 0; 
  // Prepare a vector for indices
  vector<int> indices(2, 0); 
  mdl_WH = slha.get_block_entry("decay", 25, 4.070000e-03); 
  mdl_WW = slha.get_block_entry("decay", 24, 2.085000e+00); 
  mdl_WZ = slha.get_block_entry("decay", 23, 2.495200e+00); 
  mdl_WT = slha.get_block_entry("decay", 6, 1.330000e+00); 
  mdl_ymt = slha.get_block_entry("yukawa", 6, 1.727600e+02); 
  mdl_ymb = slha.get_block_entry("yukawa", 5, 4.180000e+00); 
  aS = slha.get_block_entry("sminputs", 3, 1.179000e-01); 
  mdl_Gf = slha.get_block_entry("sminputs", 2, 1.166379e-05); 
  mdl_MW = slha.get_block_entry("sminputs", 1, 8.038700e+01); 
  mdl_LambdaSMEFT = slha.get_block_entry("smeftcutoff", 1, 1.000000e+03); 
  mdl_ctHRe = slha.get_block_entry("smeft", 11, 1.000000e-01); 
  mdl_MH = slha.get_block_entry("mass", 25, 1.250900e+02); 
  mdl_MZ = slha.get_block_entry("mass", 23, 9.118760e+01); 
  mdl_MT = slha.get_block_entry("mass", 6, 1.727600e+02); 
  mdl_MB = slha.get_block_entry("mass", 5, 4.180000e+00); 
  mdl_dMH2 = 0.; 
  mdl_cHB = 0.; 
  mdl_dGf = 0.; 
  mdl_conjg__cbH = 0.; 
  mdl_dgw = 0.; 
  mdl_cH = 0.; 
  mdl_ctHIm = 0.; 
  mdl_cHW = 0.; 
  mdl_dg1 = 0.; 
  mdl_complexi = std::complex<double> (0., 1.); 
  mdl_ctHH = mdl_ctHRe + mdl_ctHIm * mdl_complexi; 
  mdl_MWsm = mdl_MW; 
  mdl_MW__exp__2 = ((mdl_MW) * (mdl_MW)); 
  mdl_MZ__exp__2 = ((mdl_MZ) * (mdl_MZ)); 
  mdl_sqrt__2 = sqrt(2.); 
  mdl_nb__2__exp__0_25 = pow(2., 0.25); 
  mdl_MH__exp__2 = ((mdl_MH) * (mdl_MH)); 
  mdl_sth2 = 1. - mdl_MW__exp__2/mdl_MZ__exp__2; 
  mdl_nb__10__exp___m_40 = pow(10., -40.); 
  mdl_MZ1 = mdl_MZ; 
  mdl_MH1 = mdl_MH; 
  mdl_MT1 = mdl_MT; 
  mdl_WZ1 = mdl_WZ; 
  mdl_WW1 = mdl_WW; 
  mdl_WH1 = mdl_WH; 
  mdl_WT1 = mdl_WT; 
  mdl_cth = sqrt(1. - mdl_sth2); 
  mdl_MW1 = mdl_MWsm; 
  mdl_sqrt__sth2 = sqrt(mdl_sth2); 
  mdl_sth = mdl_sqrt__sth2; 
  mdl_LambdaSMEFT__exp__2 = ((mdl_LambdaSMEFT) * (mdl_LambdaSMEFT)); 
  mdl_conjg__ctHH = conj(mdl_ctHH); 
  mdl_MT__exp__2 = ((mdl_MT) * (mdl_MT)); 
  mdl_MH__exp__6 = pow(mdl_MH, 6.); 
  mdl_MWsm__exp__6 = pow(mdl_MWsm, 6.); 
  mdl_MH__exp__4 = ((mdl_MH) * (mdl_MH) * (mdl_MH) * (mdl_MH)); 
  mdl_MWsm__exp__4 = ((mdl_MWsm) * (mdl_MWsm) * (mdl_MWsm) * (mdl_MWsm)); 
  mdl_MWsm__exp__2 = ((mdl_MWsm) * (mdl_MWsm)); 
  mdl_MZ__exp__4 = ((mdl_MZ) * (mdl_MZ) * (mdl_MZ) * (mdl_MZ)); 
  mdl_MZ__exp__6 = pow(mdl_MZ, 6.); 
  mdl_cth__exp__2 = ((mdl_cth) * (mdl_cth)); 
  mdl_sth__exp__2 = ((mdl_sth) * (mdl_sth)); 
  mdl_MB__exp__2 = ((mdl_MB) * (mdl_MB)); 
  mdl_MZ__exp__3 = ((mdl_MZ) * (mdl_MZ) * (mdl_MZ)); 
  mdl_sth__exp__4 = ((mdl_sth) * (mdl_sth) * (mdl_sth) * (mdl_sth)); 
  mdl_sth__exp__6 = pow(mdl_sth, 6.); 
  mdl_sth__exp__3 = ((mdl_sth) * (mdl_sth) * (mdl_sth)); 
  mdl_sth__exp__5 = pow(mdl_sth, 5.); 
  mdl_cth__exp__3 = ((mdl_cth) * (mdl_cth) * (mdl_cth)); 
  mdl_aEW = (mdl_Gf * mdl_MW__exp__2 * (1. - mdl_MW__exp__2/mdl_MZ__exp__2) *
      mdl_sqrt__2)/M_PI;
  mdl_sqrt__Gf = sqrt(mdl_Gf); 
  mdl_vevhat = 1./(mdl_nb__2__exp__0_25 * mdl_sqrt__Gf); 
  mdl_lam = (mdl_Gf * mdl_MH__exp__2)/mdl_sqrt__2; 
  mdl_sqrt__aEW = sqrt(mdl_aEW); 
  mdl_ee = 2. * mdl_sqrt__aEW * sqrt(M_PI); 
  mdl_yb = (mdl_ymb * mdl_sqrt__2)/mdl_vevhat; 
  mdl_yt = (mdl_ymt * mdl_sqrt__2)/mdl_vevhat; 
  mdl_vevhat__exp__2 = ((mdl_vevhat) * (mdl_vevhat)); 
  mdl_vevT = (1. + mdl_dGf/2.) * mdl_vevhat; 
  mdl_g1 = mdl_ee/mdl_cth; 
  mdl_gw = mdl_ee/mdl_sth; 
  mdl_yb0 = (1. - mdl_dGf/2.) * mdl_yb + (mdl_vevhat__exp__2 *
      mdl_conjg__cbH)/(2. * mdl_LambdaSMEFT__exp__2);
  mdl_yt0 = (1. - mdl_dGf/2.) * mdl_yt + (mdl_vevhat__exp__2 *
      mdl_conjg__ctHH)/(2. * mdl_LambdaSMEFT__exp__2);
  mdl_ee__exp__2 = ((mdl_ee) * (mdl_ee)); 
  mdl_gHaa = (mdl_ee__exp__2 * (-1.75 + (4. * (0.3333333333333333 + (7. *
      mdl_MH__exp__2)/(360. * mdl_MT__exp__2)))/3. - (29. *
      mdl_MH__exp__6)/(16800. * mdl_MWsm__exp__6) - (19. *
      mdl_MH__exp__4)/(1680. * mdl_MWsm__exp__4) - (11. * mdl_MH__exp__2)/(120.
      * mdl_MWsm__exp__2)))/(8. * ((M_PI) * (M_PI)));
  mdl_gHza = (mdl_ee__exp__2 * (((0.4583333333333333 + (29. *
      mdl_MH__exp__6)/(100800. * mdl_MWsm__exp__6) + (19. *
      mdl_MH__exp__4)/(10080. * mdl_MWsm__exp__4) + (11. *
      mdl_MH__exp__2)/(720. * mdl_MWsm__exp__2) + (mdl_MH__exp__4 *
      mdl_MZ__exp__2)/(2100. * mdl_MWsm__exp__6) + (mdl_MH__exp__2 *
      mdl_MZ__exp__2)/(280. * mdl_MWsm__exp__4) + (7. * mdl_MZ__exp__2)/(180. *
      mdl_MWsm__exp__2) + (67. * mdl_MH__exp__2 * mdl_MZ__exp__4)/(100800. *
      mdl_MWsm__exp__6) + (53. * mdl_MZ__exp__4)/(10080. * mdl_MWsm__exp__4) +
      (43. * mdl_MZ__exp__6)/(50400. * mdl_MWsm__exp__6) - (31. *
      mdl_cth__exp__2)/(24. * mdl_sth__exp__2) - (29. * mdl_cth__exp__2 *
      mdl_MH__exp__6)/(20160. * mdl_MWsm__exp__6 * mdl_sth__exp__2) - (19. *
      mdl_cth__exp__2 * mdl_MH__exp__4)/(2016. * mdl_MWsm__exp__4 *
      mdl_sth__exp__2) - (11. * mdl_cth__exp__2 * mdl_MH__exp__2)/(144. *
      mdl_MWsm__exp__2 * mdl_sth__exp__2) - (mdl_cth__exp__2 * mdl_MH__exp__4 *
      mdl_MZ__exp__2)/(560. * mdl_MWsm__exp__6 * mdl_sth__exp__2) - (31. *
      mdl_cth__exp__2 * mdl_MH__exp__2 * mdl_MZ__exp__2)/(2520. *
      mdl_MWsm__exp__4 * mdl_sth__exp__2) - (mdl_cth__exp__2 *
      mdl_MZ__exp__2)/(9. * mdl_MWsm__exp__2 * mdl_sth__exp__2) - (43. *
      mdl_cth__exp__2 * mdl_MH__exp__2 * mdl_MZ__exp__4)/(20160. *
      mdl_MWsm__exp__6 * mdl_sth__exp__2) - (17. * mdl_cth__exp__2 *
      mdl_MZ__exp__4)/(1120. * mdl_MWsm__exp__4 * mdl_sth__exp__2) - (5. *
      mdl_cth__exp__2 * mdl_MZ__exp__6)/(2016. * mdl_MWsm__exp__6 *
      mdl_sth__exp__2)) * mdl_sth)/mdl_cth + ((0.3333333333333333 + (7. *
      mdl_MH__exp__2)/(360. * mdl_MT__exp__2) + (11. * mdl_MZ__exp__2)/(360. *
      mdl_MT__exp__2)) * (0.5 - (4. * mdl_sth__exp__2)/3.))/(mdl_cth *
      mdl_sth)))/(4. * ((M_PI) * (M_PI)));
  mdl_barlam = (1. - mdl_dGf - mdl_dMH2) * mdl_lam; 
  mdl_gwsh = (mdl_ee * (1. + mdl_dgw - (mdl_cHW *
      mdl_vevhat__exp__2)/mdl_LambdaSMEFT__exp__2))/mdl_sth;
  mdl_vev = (1. - (3. * mdl_cH * mdl_vevhat__exp__2)/(8. * mdl_lam *
      mdl_LambdaSMEFT__exp__2)) * mdl_vevT;
  mdl_g1sh = (mdl_ee * (1. + mdl_dg1 - (mdl_cHB *
      mdl_vevhat__exp__2)/mdl_LambdaSMEFT__exp__2))/mdl_cth;
  mdl_ee__exp__3 = ((mdl_ee) * (mdl_ee) * (mdl_ee)); 
  mdl_vevhat__exp__3 = ((mdl_vevhat) * (mdl_vevhat) * (mdl_vevhat)); 
}
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::setIndependentCouplings()
{
  GC_1 = (mdl_ee * mdl_complexi)/3.; 
  GC_2 = (-2. * mdl_ee * mdl_complexi)/3.; 
  GC_202 = -((mdl_ee * mdl_complexi)/(mdl_sth * mdl_sqrt__2)); 
  GC_203 = -(mdl_ee * mdl_complexi)/(2. * mdl_cth * mdl_sth); 
  GC_263 = -(mdl_ee * mdl_complexi * mdl_sth)/(3. * mdl_cth); 
  GC_264 = (2. * mdl_ee * mdl_complexi * mdl_sth)/(3. * mdl_cth); 
  GC_336 = -6. * mdl_complexi * mdl_lam * mdl_vevhat; 
  GC_364 = (3. * mdl_ctHRe * mdl_complexi *
      mdl_vevhat)/(mdl_LambdaSMEFT__exp__2 * mdl_sqrt__2);
  GC_391 = (mdl_ee__exp__2 * mdl_complexi * mdl_vevhat)/(2. * mdl_sth__exp__2); 
  GC_392 = (mdl_ee__exp__2 * mdl_complexi * mdl_vevhat)/(2. * mdl_cth__exp__2 *
      mdl_sth__exp__2);
  GC_461 = (mdl_ctHRe * mdl_complexi * mdl_vevhat__exp__2)/(mdl_LambdaSMEFT__exp__2 * mdl_sqrt__2); 
  GC_566 = -((mdl_complexi * mdl_yb)/mdl_sqrt__2); 
  GC_1025 = -((mdl_complexi * mdl_yt)/mdl_sqrt__2); 
}
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::setDependentParameters()
{
  mdl_sqrt__aS = sqrt(aS); 
  G = 2. * mdl_sqrt__aS * sqrt(M_PI); 
  mdl_gHgg2 = (-7. * aS)/(720. * M_PI); 
  mdl_gHgg4 = aS/(360. * M_PI); 
  mdl_gHgg5 = aS/(20. * M_PI); 
  mdl_G__exp__2 = ((G) * (G)); 
  mdl_gHgg1 = mdl_G__exp__2/(48. * ((M_PI) * (M_PI))); 
  mdl_gHgg3 = (aS * G)/(60. * M_PI); 
  mdl_G__exp__3 = ((G) * (G) * (G)); 
}
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::setDependentCouplings()
{
  GC_6 = -(mdl_complexi * G); 
  GC_7 = G; 
  GC_8 = mdl_complexi * mdl_G__exp__2; 
}

// Routines for printing out parameters
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::printIndependentParameters()
{
  cout <<  "SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless model parameters independent of event kinematics:" <<
      endl;
  cout << setw(20) <<  "mdl_WH " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WH << endl;
  cout << setw(20) <<  "mdl_WW " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WW << endl;
  cout << setw(20) <<  "mdl_WZ " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WZ << endl;
  cout << setw(20) <<  "mdl_WT " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WT << endl;
  cout << setw(20) <<  "mdl_ymt " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_ymt << endl;
  cout << setw(20) <<  "mdl_ymb " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_ymb << endl;
  cout << setw(20) <<  "aS " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << aS << endl;
  cout << setw(20) <<  "mdl_Gf " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_Gf << endl;
  cout << setw(20) <<  "mdl_MW " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MW << endl;
  cout << setw(20) <<  "mdl_LambdaSMEFT " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_LambdaSMEFT << endl;
  cout << setw(20) <<  "mdl_ctHRe " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_ctHRe << endl;
  cout << setw(20) <<  "mdl_MH " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MH << endl;
  cout << setw(20) <<  "mdl_MZ " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MZ << endl;
  cout << setw(20) <<  "mdl_MT " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MT << endl;
  cout << setw(20) <<  "mdl_MB " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MB << endl;
  cout << setw(20) <<  "mdl_dMH2 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_dMH2 << endl;
  cout << setw(20) <<  "mdl_cHB " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_cHB << endl;
  cout << setw(20) <<  "mdl_dGf " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_dGf << endl;
  cout << setw(20) <<  "mdl_conjg__cbH " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_conjg__cbH << endl;
  cout << setw(20) <<  "mdl_dgw " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_dgw << endl;
  cout << setw(20) <<  "mdl_cH " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_cH << endl;
  cout << setw(20) <<  "mdl_ctHIm " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_ctHIm << endl;
  cout << setw(20) <<  "mdl_cHW " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_cHW << endl;
  cout << setw(20) <<  "mdl_dg1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_dg1 << endl;
  cout << setw(20) <<  "mdl_complexi " <<  "= " << setiosflags(ios::scientific)
      << setw(10) << mdl_complexi << endl;
  cout << setw(20) <<  "mdl_ctHH " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_ctHH << endl;
  cout << setw(20) <<  "mdl_MWsm " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MWsm << endl;
  cout << setw(20) <<  "mdl_MW__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MW__exp__2 << endl;
  cout << setw(20) <<  "mdl_MZ__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MZ__exp__2 << endl;
  cout << setw(20) <<  "mdl_sqrt__2 " <<  "= " << setiosflags(ios::scientific)
      << setw(10) << mdl_sqrt__2 << endl;
  cout << setw(20) <<  "mdl_nb__2__exp__0_25 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_nb__2__exp__0_25 << endl;
  cout << setw(20) <<  "mdl_MH__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MH__exp__2 << endl;
  cout << setw(20) <<  "mdl_sth2 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_sth2 << endl;
  cout << setw(20) <<  "mdl_nb__10__exp___m_40 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_nb__10__exp___m_40 <<
      endl;
  cout << setw(20) <<  "mdl_MZ1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MZ1 << endl;
  cout << setw(20) <<  "mdl_MH1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MH1 << endl;
  cout << setw(20) <<  "mdl_MT1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MT1 << endl;
  cout << setw(20) <<  "mdl_WZ1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WZ1 << endl;
  cout << setw(20) <<  "mdl_WW1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WW1 << endl;
  cout << setw(20) <<  "mdl_WH1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WH1 << endl;
  cout << setw(20) <<  "mdl_WT1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_WT1 << endl;
  cout << setw(20) <<  "mdl_cth " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_cth << endl;
  cout << setw(20) <<  "mdl_MW1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_MW1 << endl;
  cout << setw(20) <<  "mdl_sqrt__sth2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sqrt__sth2 << endl;
  cout << setw(20) <<  "mdl_sth " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_sth << endl;
  cout << setw(20) <<  "mdl_LambdaSMEFT__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_LambdaSMEFT__exp__2 <<
      endl;
  cout << setw(20) <<  "mdl_conjg__ctHH " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_conjg__ctHH << endl;
  cout << setw(20) <<  "mdl_MT__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MT__exp__2 << endl;
  cout << setw(20) <<  "mdl_MH__exp__6 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MH__exp__6 << endl;
  cout << setw(20) <<  "mdl_MWsm__exp__6 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MWsm__exp__6 << endl;
  cout << setw(20) <<  "mdl_MH__exp__4 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MH__exp__4 << endl;
  cout << setw(20) <<  "mdl_MWsm__exp__4 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MWsm__exp__4 << endl;
  cout << setw(20) <<  "mdl_MWsm__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MWsm__exp__2 << endl;
  cout << setw(20) <<  "mdl_MZ__exp__4 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MZ__exp__4 << endl;
  cout << setw(20) <<  "mdl_MZ__exp__6 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MZ__exp__6 << endl;
  cout << setw(20) <<  "mdl_cth__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_cth__exp__2 << endl;
  cout << setw(20) <<  "mdl_sth__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sth__exp__2 << endl;
  cout << setw(20) <<  "mdl_MB__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MB__exp__2 << endl;
  cout << setw(20) <<  "mdl_MZ__exp__3 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_MZ__exp__3 << endl;
  cout << setw(20) <<  "mdl_sth__exp__4 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sth__exp__4 << endl;
  cout << setw(20) <<  "mdl_sth__exp__6 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sth__exp__6 << endl;
  cout << setw(20) <<  "mdl_sth__exp__3 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sth__exp__3 << endl;
  cout << setw(20) <<  "mdl_sth__exp__5 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sth__exp__5 << endl;
  cout << setw(20) <<  "mdl_cth__exp__3 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_cth__exp__3 << endl;
  cout << setw(20) <<  "mdl_aEW " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_aEW << endl;
  cout << setw(20) <<  "mdl_sqrt__Gf " <<  "= " << setiosflags(ios::scientific)
      << setw(10) << mdl_sqrt__Gf << endl;
  cout << setw(20) <<  "mdl_vevhat " <<  "= " << setiosflags(ios::scientific)
      << setw(10) << mdl_vevhat << endl;
  cout << setw(20) <<  "mdl_lam " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_lam << endl;
  cout << setw(20) <<  "mdl_sqrt__aEW " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_sqrt__aEW << endl;
  cout << setw(20) <<  "mdl_ee " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_ee << endl;
  cout << setw(20) <<  "mdl_yb " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_yb << endl;
  cout << setw(20) <<  "mdl_yt " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_yt << endl;
  cout << setw(20) <<  "mdl_vevhat__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_vevhat__exp__2 << endl;
  cout << setw(20) <<  "mdl_vevT " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_vevT << endl;
  cout << setw(20) <<  "mdl_g1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_g1 << endl;
  cout << setw(20) <<  "mdl_gw " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gw << endl;
  cout << setw(20) <<  "mdl_yb0 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_yb0 << endl;
  cout << setw(20) <<  "mdl_yt0 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_yt0 << endl;
  cout << setw(20) <<  "mdl_ee__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_ee__exp__2 << endl;
  cout << setw(20) <<  "mdl_gHaa " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHaa << endl;
  cout << setw(20) <<  "mdl_gHza " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHza << endl;
  cout << setw(20) <<  "mdl_barlam " <<  "= " << setiosflags(ios::scientific)
      << setw(10) << mdl_barlam << endl;
  cout << setw(20) <<  "mdl_gwsh " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gwsh << endl;
  cout << setw(20) <<  "mdl_vev " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_vev << endl;
  cout << setw(20) <<  "mdl_g1sh " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_g1sh << endl;
  cout << setw(20) <<  "mdl_ee__exp__3 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_ee__exp__3 << endl;
  cout << setw(20) <<  "mdl_vevhat__exp__3 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_vevhat__exp__3 << endl;
}
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::printIndependentCouplings()
{
  cout <<  "SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless model couplings independent of event kinematics:" <<
      endl;
  cout << setw(20) <<  "GC_1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_1 << endl;
  cout << setw(20) <<  "GC_2 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_2 << endl;
  cout << setw(20) <<  "GC_202 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_202 << endl;
  cout << setw(20) <<  "GC_203 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_203 << endl;
  cout << setw(20) <<  "GC_263 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_263 << endl;
  cout << setw(20) <<  "GC_264 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_264 << endl;
  cout << setw(20) <<  "GC_336 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_336 << endl;
  cout << setw(20) <<  "GC_364 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_364 << endl;
  cout << setw(20) <<  "GC_391 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_391 << endl;
  cout << setw(20) <<  "GC_392 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_392 << endl;
  cout << setw(20) <<  "GC_461 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_461 << endl;
  cout << setw(20) <<  "GC_566 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_566 << endl;
  cout << setw(20) <<  "GC_1025 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_1025 << endl;
}
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::printDependentParameters()
{
  cout <<  "SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless model parameters dependent on event kinematics:" <<
      endl;
  cout << setw(20) <<  "mdl_sqrt__aS " <<  "= " << setiosflags(ios::scientific)
      << setw(10) << mdl_sqrt__aS << endl;
  cout << setw(20) <<  "G " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << G << endl;
  cout << setw(20) <<  "mdl_gHgg2 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHgg2 << endl;
  cout << setw(20) <<  "mdl_gHgg4 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHgg4 << endl;
  cout << setw(20) <<  "mdl_gHgg5 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHgg5 << endl;
  cout << setw(20) <<  "mdl_G__exp__2 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_G__exp__2 << endl;
  cout << setw(20) <<  "mdl_gHgg1 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHgg1 << endl;
  cout << setw(20) <<  "mdl_gHgg3 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << mdl_gHgg3 << endl;
  cout << setw(20) <<  "mdl_G__exp__3 " <<  "= " <<
      setiosflags(ios::scientific) << setw(10) << mdl_G__exp__3 << endl;
}
void Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless::printDependentCouplings()
{
  cout <<  "SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless model couplings dependent on event kinematics:" <<
      endl;
  cout << setw(20) <<  "GC_6 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_6 << endl;
  cout << setw(20) <<  "GC_7 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_7 << endl;
  cout << setw(20) <<  "GC_8 " <<  "= " << setiosflags(ios::scientific) <<
      setw(10) << GC_8 << endl;
}


