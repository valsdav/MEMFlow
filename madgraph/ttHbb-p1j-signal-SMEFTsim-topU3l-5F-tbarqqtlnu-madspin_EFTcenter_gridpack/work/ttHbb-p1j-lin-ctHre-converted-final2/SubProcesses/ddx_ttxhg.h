//==========================================================================
// This file has been automatically generated for C++ Standalone by
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef MG5_Sigma_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_ddx_ttxhg_H
#define MG5_Sigma_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_ddx_ttxhg_H

#include <complex> 
#include <vector> 

#include "Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless.h"

using namespace std; 

//==========================================================================
// A class for calculating the matrix elements for
// Process: d d~ > t t~ h g NProp=0 SMHLOOP=0 NP^2==1
// Process: s s~ > t t~ h g NProp=0 SMHLOOP=0 NP^2==1
//--------------------------------------------------------------------------

#include "baseclass.h"

class ddx_ttxhg : public ProcessClass
{
  public:

    // Constructor.
    ddx_ttxhg(): ProcessClass("ddx_ttxhg") {};

    // Initialize process.
    virtual void initProc(string param_card_name); 

    // Calculate flavour-independent parts of cross section.
    virtual void sigmaKin(); 

    // Evaluate sigmaHat(sHat).
    virtual double sigmaHat(); 

    // Info on the subprocess.
    virtual string name() const {return "d d~ > t t~ h g (SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless)";}

    virtual int code() const {return 0;}

    const vector<double> & getMasses() const {return mME;}

    // Get and set momenta for matrix element evaluation
    vector < double * > getMomenta(){return p;}
    void setMomenta(vector < double * > & momenta){p = momenta;}
    void setInitial(int inid1, int inid2){id1 = inid1; id2 = inid2;}

    // Get matrix element vector
    const double * getMatrixElements() const {return matrix_element;}

    // Constants for array limits
    static const int ninitial = 2; 
    static const int nexternal = 6; 
    static const int nprocesses = 2; 

  private:

    // Private functions to calculate the matrix element for all subprocesses
    // Calculate wavefunctions
    void calculate_wavefunctions(const int perm[], const int hel[]); 
    static const int nwavefuncs = 35; 
    std::complex<double> w[nwavefuncs][18]; 
    static const int namplitudes = 68; 
    std::complex<double> amp[namplitudes]; 
    double matrix_ddx_ttxhg(); 

    // Store the matrix element value from sigmaKin
    double matrix_element[nprocesses]; 

    // Color flows, used when selecting color
    double * jamp2[nprocesses]; 

    // Pointer to the model parameters
    Parameters_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless * pars; 

    // vector with external particle masses
    vector<double> mME; 

    // vector with momenta (to be changed each event)
    vector < double * > p; 
    // Initial particle ids
    int id1, id2; 

}; 


#endif  // MG5_Sigma_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_ddx_ttxhg_H
