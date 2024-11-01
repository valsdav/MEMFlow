#ifndef __processes_h
#define __processes_h
#include "gb_ttxhb.h"
#include "gbx_ttxhbx.h"
#include "bbx_ttxhg.h"
#include "gg_ttxhg.h"
#include "gu_ttxhu.h"
#include "gd_ttxhd.h"
#include "gux_ttxhux.h"
#include "gdx_ttxhdx.h"
#include "uux_ttxhg.h"
#include "ddx_ttxhg.h"
#include <vector>
std::vector<ProcessClass*> getProcesses(std::string param_card){
        std::vector<ProcessClass*> processes;

gb_ttxhb * process_0 = new gb_ttxhb();
process_0->initProc(param_card);
processes.push_back(process_0);
gbx_ttxhbx * process_1 = new gbx_ttxhbx();
process_1->initProc(param_card);
processes.push_back(process_1);
bbx_ttxhg * process_2 = new bbx_ttxhg();
process_2->initProc(param_card);
processes.push_back(process_2);
gg_ttxhg * process_3 = new gg_ttxhg();
process_3->initProc(param_card);
processes.push_back(process_3);
gu_ttxhu * process_4 = new gu_ttxhu();
process_4->initProc(param_card);
processes.push_back(process_4);
gd_ttxhd * process_5 = new gd_ttxhd();
process_5->initProc(param_card);
processes.push_back(process_5);
gux_ttxhux * process_6 = new gux_ttxhux();
process_6->initProc(param_card);
processes.push_back(process_6);
gdx_ttxhdx * process_7 = new gdx_ttxhdx();
process_7->initProc(param_card);
processes.push_back(process_7);
uux_ttxhg * process_8 = new uux_ttxhg();
process_8->initProc(param_card);
processes.push_back(process_8);
ddx_ttxhg * process_9 = new ddx_ttxhg();
process_9->initProc(param_card);
processes.push_back(process_9);
return processes;
}
#endif // __processes_h