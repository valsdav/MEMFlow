//==========================================================================
// This file has been automatically generated for C++ Standalone
// MadGraph5_aMC@NLO v. 2.9.18, 2023-12-08
// By the MadGraph5_aMC@NLO Development Team
// Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch
//==========================================================================

#ifndef HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_H
#define HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_H

#include <cmath> 
#include <complex> 

using namespace std; 

namespace MG5_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless 
{
void oxxxxx(double p[4], double fmass, int nhel, int nsf, std::complex<double>
    fo[6]);

double Sgn(double e, double f); 

void sxxxxx(double p[4], int nss, std::complex<double> sc[3]); 

void ixxxxx(double p[4], double fmass, int nhel, int nsf, std::complex<double>
    fi[6]);

void txxxxx(double p[4], double tmass, int nhel, int nst, std::complex<double>
    fi[18]);

void vxxxxx(double p[4], double vmass, int nhel, int nsv, std::complex<double>
    v[6]);

void VVVV9P0_1(std::complex<double> V2[], std::complex<double> V3[],
    std::complex<double> V4[], std::complex<double> COUP, double M1, double W1,
    std::complex<double> V1[]);

void FFV1_0(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> V3[], std::complex<double> COUP, std::complex<double>
    & vertex);
void FFV1_3_0(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> V3[], std::complex<double> COUP1, std::complex<double>
    COUP2, std::complex<double> & vertex);

void FFV1_1(std::complex<double> F2[], std::complex<double> V3[],
    std::complex<double> COUP, double M1, double W1, std::complex<double> F1[]);

void FFV1_2(std::complex<double> F1[], std::complex<double> V3[],
    std::complex<double> COUP, double M2, double W2, std::complex<double> F2[]);

void FFV1P0_3(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> COUP, double M3, double W3, std::complex<double> V3[]);

void FFV1_3(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> COUP, double M3, double W3, std::complex<double> V3[]);
void FFV1_3_3(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> COUP1, std::complex<double> COUP2, double M3, double
    W3, std::complex<double> V3[]);

void FFV3_0(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> V3[], std::complex<double> COUP, std::complex<double>
    & vertex);

void FFV3_3(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> COUP, double M3, double W3, std::complex<double> V3[]);

void VVVV10P0_1(std::complex<double> V2[], std::complex<double> V3[],
    std::complex<double> V4[], std::complex<double> COUP, double M1, double W1,
    std::complex<double> V1[]);

void FFS2_0(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> S3[], std::complex<double> COUP, std::complex<double>
    & vertex);

void FFS2_1(std::complex<double> F2[], std::complex<double> S3[],
    std::complex<double> COUP, double M1, double W1, std::complex<double> F1[]);

void FFS2_2(std::complex<double> F1[], std::complex<double> S3[],
    std::complex<double> COUP, double M2, double W2, std::complex<double> F2[]);

void FFS2_3(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> COUP, double M3, double W3, std::complex<double> S3[]);

void FFSS2_0(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> S3[], std::complex<double> S4[], std::complex<double>
    COUP, std::complex<double> & vertex);

void FFSS2_3(std::complex<double> F1[], std::complex<double> F2[],
    std::complex<double> S4[], std::complex<double> COUP, double M3, double W3,
    std::complex<double> S3[]);

void VVV5P0_1(std::complex<double> V2[], std::complex<double> V3[],
    std::complex<double> COUP, double M1, double W1, std::complex<double> V1[]);

void VVVV1P0_1(std::complex<double> V2[], std::complex<double> V3[],
    std::complex<double> V4[], std::complex<double> COUP, double M1, double W1,
    std::complex<double> V1[]);

void VVS3_0(std::complex<double> V1[], std::complex<double> V2[],
    std::complex<double> S3[], std::complex<double> COUP, std::complex<double>
    & vertex);

void SSS1_0(std::complex<double> S1[], std::complex<double> S2[],
    std::complex<double> S3[], std::complex<double> COUP, std::complex<double>
    & vertex);

}  // end namespace MG5_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless

#endif  // HelAmps_SMEFTsim_topU3l_MwScheme_UFO_ttHbb_ttHbb_ctHRe_massless_H
