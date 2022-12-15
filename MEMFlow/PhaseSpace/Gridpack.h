#include "Math/GenVector/LorentzVector.h"
#include <vector>
#include <array>

using namespace ROOT::Math;

// std::array<double, 19> transformPointsToCube(std::array<double, 19> points);
std::array<double, 19> transfromPointsFromCube(std::array<double, 19> points);

std::array<PtEtaPhiMVector,14> getVectorsFromPoints(std::array<double, 19> points);

// std::array<double,19> getPointsFromVector(std::array<PtEtaPhiMVector, 14> vectors);


