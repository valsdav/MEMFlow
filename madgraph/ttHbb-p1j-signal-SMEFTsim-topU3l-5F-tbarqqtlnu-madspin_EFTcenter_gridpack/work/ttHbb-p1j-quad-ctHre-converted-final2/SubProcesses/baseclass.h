#ifndef __baseclass_h
#define __baseclass_h

#include <string>
#include <vector>

class ProcessClass {
public:
  ProcessClass(std::string process_name);
  ~ProcessClass();
  
  virtual void initProc(std::string param_card_name) = 0;
  virtual void sigmaKin() = 0;
  virtual double sigmaHat() = 0;
  virtual std::string name() const = 0;
  virtual int code() const = 0;
  virtual const std::vector<double> & getMasses() const = 0;
  virtual std::vector < double * > getMomenta() = 0;
  virtual void setMomenta(std::vector < double * > & momenta) = 0;
  virtual void setInitial(int inid1, int inid2) = 0;
  virtual const double * getMatrixElements() const = 0;
  std::string getProcessName() const;
private:
  std::string _process_name;
};

#endif // __baseclass_h
