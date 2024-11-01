//==========================================================================
// This file has been automatically generated for C++
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_H
#define Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_H

#include <complex> 

#include "read_slha.h"
using namespace std; 

class Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless
{
  public:

    static Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless *
        getInstance();

    // Define "zero"
    double zero, ZERO; 
    // Model parameters independent of aS
    double mdl_WH, mdl_WW, mdl_WZ, mdl_WT, mdl_ymt, mdl_ymb, aS, mdl_Gf,
        mdl_MW, mdl_LambdaSMEFT, mdl_ctHRe, mdl_MH, mdl_MZ, mdl_MT, mdl_MB,
        mdl_dMH2, mdl_cHW, mdl_ctHIm, mdl_conjg__cbH, mdl_dGf, mdl_cH, mdl_dgw,
        mdl_dg1, mdl_cHB, mdl_MWsm, mdl_MW__exp__2, mdl_MZ__exp__2,
        mdl_sqrt__2, mdl_nb__2__exp__0_25, mdl_MH__exp__2, mdl_sth2,
        mdl_nb__10__exp___m_40, mdl_MZ1, mdl_MH1, mdl_MT1, mdl_WZ1, mdl_WW1,
        mdl_WH1, mdl_WT1, mdl_cth, mdl_MW1, mdl_sqrt__sth2, mdl_sth,
        mdl_LambdaSMEFT__exp__2, mdl_MT__exp__2, mdl_MH__exp__6,
        mdl_MWsm__exp__6, mdl_MH__exp__4, mdl_MWsm__exp__4, mdl_MWsm__exp__2,
        mdl_MZ__exp__4, mdl_MZ__exp__6, mdl_cth__exp__2, mdl_sth__exp__2,
        mdl_MB__exp__2, mdl_MZ__exp__3, mdl_sth__exp__4, mdl_sth__exp__6,
        mdl_sth__exp__3, mdl_sth__exp__5, mdl_cth__exp__3, mdl_aEW,
        mdl_sqrt__Gf, mdl_vevhat, mdl_lam, mdl_sqrt__aEW, mdl_ee, mdl_yb,
        mdl_yt, mdl_vevhat__exp__2, mdl_vevT, mdl_g1, mdl_gw, mdl_ee__exp__2,
        mdl_gHaa, mdl_gHza, mdl_barlam, mdl_gwsh, mdl_vev, mdl_g1sh,
        mdl_ee__exp__3, mdl_vevhat__exp__3;
    std::complex<double> mdl_complexi, mdl_ctHH, mdl_conjg__ctHH, mdl_yb0,
        mdl_yt0;
    // Model parameters dependent on aS
    double mdl_sqrt__aS, G, mdl_gHgg2, mdl_gHgg4, mdl_gHgg5, mdl_G__exp__2,
        mdl_gHgg1, mdl_gHgg3;
    std::complex<double> mdl_G__exp__3; 
    // Model couplings independent of aS
    std::complex<double> GC_1, GC_2, GC_202, GC_203, GC_263, GC_264, GC_336,
        GC_364, GC_461, GC_566, GC_1025;
    // Model couplings dependent on aS
    std::complex<double> GC_6, GC_7, GC_8; 

    // Set parameters that are unchanged during the run
    void setIndependentParameters(SLHAReader& slha); 
    // Set couplings that are unchanged during the run
    void setIndependentCouplings(); 
    // Set parameters that are changed event by event
    void setDependentParameters(); 
    // Set couplings that are changed event by event
    void setDependentCouplings(); 

    // Print parameters that are unchanged during the run
    void printIndependentParameters(); 
    // Print couplings that are unchanged during the run
    void printIndependentCouplings(); 
    // Print parameters that are changed event by event
    void printDependentParameters(); 
    // Print couplings that are changed event by event
    void printDependentCouplings(); 


  private:
    static Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless *
        instance;
}; 

#endif  // Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_H

