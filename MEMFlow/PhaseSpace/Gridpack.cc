#include "Gridpack.h"
#include "TMathBase.h"
#include "Math/GenVector/LorentzVector.h"
#include <vector>
#include <array>

using namespace std;
using namespace TMath;
using namespace ROOT::Math;

const float TOP_MASS = 172.69;
const float HIGGS_MASS = 125.25;
const float Pmax = 500;
const float pp_En = 6.5e3;


std::array<double, 19> transfromPointsFromCube(std::array<double, 19> input){
  std::array<double, 19> output;
  // Pt --> linear scaling
  for (uint i=0; i<3; i++){
    output[i] =  -Pmax + (2*Pmax) * input[i];
  }
  
  // From theta [0, 2pi] to eta
  for (uint i=3; i< 6;i++){
    output[i] = - Log(Tan(input[i]/2));
  }
  
  // Phi is already between [0, 2pi]
  for (uint i=6; i< 9;i++){
     output[i] = input[i];
  }
  
  // Gluon Pz
  output[9] = -Pmax + (2*Pmax)*input[9];
  
  return output;
}

std::array<PtEtaPhiMVector,12> getVectorsFromPoints(std::array<double, 19> points){
  /*
    The point are ordered by
    - pt (top_had, top_lep, H)  [0,1,2]
    - eta (top_had, top_lep, H) [3,4,5]
    -phi (top_had, top_lep, H) [6,7,8]
    - gluon Z  [9]
   
   */
  auto inputs = transfromPointsFromCube(points)
  
  PtEtaPhiMVector top_had {inputs[0], inputs[3], inputs[6], TOP_MASS};
  PtEtaPhiMVector top_lep {inputs[1], inputs[4], inputs[7], TOP_MASS};
  PtEtaPhiMVector H {inputs[2], inputs[5], inputs[8], HIGGS_MASS};

  auto tot = top_had + top_lep + H;
  auto gluon = -tot;
  gluon.SetPz = inputs[9];

  // Getting the initial gluons fractions
  auto tot_pz = tot.Pz() + gluon.Pz();

  PxPyPzMVector initial_gluon1 {0, 0, +pp_En, 0};
  auto initial_glion2 = -initial_gluon1;
  initial_gluon1 += tot_pz;
  initial_gluon2 -= tot_pz;

  return {
    initial_gluon1,
    initial_gluon2,
    top_had,
    top_lep,
    H,
    gluon
  }
  
}

// std::array<double,19> getPointsFromVector(std::array<PtEtaPhiMVector, 12> vectors){
//   return 
// }
