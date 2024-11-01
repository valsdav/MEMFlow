#include "baseclass.h"

ProcessClass::ProcessClass(std::string process_name) : _process_name(process_name){}

ProcessClass::~ProcessClass(){}

std::string ProcessClass::getProcessName() const{
  return _process_name;
}
