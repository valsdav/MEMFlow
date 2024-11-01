#ifndef __processes_h
#define __processes_h
#include "gg_ttxhg.h"
#include "gb_ttxhb.h"
#include "gbx_ttxhbx.h"
#include "bbx_ttxhg.h"
#include "gu_ttxhu.h"
#include "gux_ttxhux.h"
#include "uux_ttxhg.h"
#include <vector>
std::vector<ProcessClass*> getProcesses(std::string param_card){
        std::vector<ProcessClass*> processes;

gg_ttxhg * process_0 = new gg_ttxhg();
process_0->initProc(param_card);
processes.push_back(process_0);
gb_ttxhb * process_1 = new gb_ttxhb();
process_1->initProc(param_card);
processes.push_back(process_1);
gbx_ttxhbx * process_2 = new gbx_ttxhbx();
process_2->initProc(param_card);
processes.push_back(process_2);
bbx_ttxhg * process_3 = new bbx_ttxhg();
process_3->initProc(param_card);
processes.push_back(process_3);
gu_ttxhu * process_4 = new gu_ttxhu();
process_4->initProc(param_card);
processes.push_back(process_4);
gux_ttxhux * process_5 = new gux_ttxhux();
process_5->initProc(param_card);
processes.push_back(process_5);
uux_ttxhg * process_6 = new uux_ttxhg();
process_6->initProc(param_card);
processes.push_back(process_6);
return processes;
}
#endif // __processes_h