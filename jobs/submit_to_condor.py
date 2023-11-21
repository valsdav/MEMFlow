import htcondor
import sys
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--version', type=str, required=True)
parser.add_argument('--dry', action="store_true")
parser.add_argument('--antonio', action="store_true")
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument("--good-gpus", action="store_true")
parser.add_argument("--args", nargs="+", type=str, help="additional args")
args = parser.parse_args()

model = args.model
version = args.version
dry = args.dry
antonio = args.antonio

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

if antonio:
    basedir = "/afs/cern.ch/user/a/adpetre/public/memflow/MEMFlow"
else:
    basedir = "/afs/cern.ch/work/d/dvalsecc/private/MEM/MEMFlow"

sub = htcondor.Submit()

if model == "huber_mmd":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"pretraining_huber_mmd/pretraining_huber_mmd_v{version}.yaml flow_pretraining_huber_mmd"

    
elif model == "mmd_huber":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_mmd_huber.sh"
    sub['Error'] = f"{basedir}/jobs/error/mmd-huber-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/mmd-huber-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/mmd-huber-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"pretraining_mmd_huber/pretraining_mmd_huber_v{version}.yaml flow_pretraining_mmd_huber"  

elif model == "huber_mmd_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['arguments'] = f"pretraining_huber_mmd_labframe/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe"

elif model == "huber_mmd_labframe_gluon":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_huber_mmd_labframe_gluon.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-gluon-{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-gluon-{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-gluon-{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['arguments'] = f"pretraining_huber_mmd_labframe_gluon/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe_gluon"

elif model == "huber_mmd_labframe_gluon_distributed":
    sub['Executable'] = f"{basedir}/jobs/script_condor_pretraining_distributed_labframe_gluon.sh"
    sub['Error'] = f"{basedir}/jobs/error/huber-mmd-labframe-gluon-distributed-{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/huber-mmd-labframe-gluon-distributed-{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/huber-mmd-labframe-gluon-distributed-{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"testmatch"'
    sub['arguments'] = f"pretraining_huber_mmd_labframe_gluon/pretraining_huber_mmd_labframe_v{version}.yaml flow_pretraining_huber_mmd_labframe_gluon/distrubuted_lxplus"


elif model == "flow_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-pretrain-labframe-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-pretrain-labframe-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-pretrain-labframe-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_pretrain_labframe_logit/flow_pretrain_labframe_logit_v{version}.yaml flow_pretrain_labframe_logit"


elif model == "flow_labframe_psloss":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_labframe_psloss.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-pretrain-labframe-psloss-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-pretrain-labframe-psloss-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-pretrain-labframe-psloss-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_pretrain_labframe_psloss_logit/flow_labframe_psloss_v{version}.yaml flow_labframe_psloss_logit"


elif model == "flow_labframe_sampling":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_labframe_sampling.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-pretrain-labframe-sampling-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-pretrain-labframe-sampling-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-pretrain-labframe-sampling-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_pretrain_labframe_sampling/flow_labframe_sampling_v{version}.yaml flow_labframe_sampling"


elif model == "flow_nopretrain":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_nopretrain.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-nopretrain-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-nopretrain-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-nopretrain-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"flow_nopretrain_spanet_logit/flow_nopretrain_v{version}.yaml flow_nopretrain_spanet_logit"


elif model == "flow_evaluation_labframe":
    sub['Executable'] = f"{basedir}/jobs/script_condor_flow_evaluation_labframe.sh"
    sub['Error'] = f"{basedir}/jobs/error/flow-evaluation-labframe-v{version}-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/flow-evaluation-labframe-v{version}-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/flow-evaluation-labframe-v{version}-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"workday"'
    sub['arguments'] = " ".join(args.args)

elif model == "transfer_flow_firstVersion":
    sub['Executable'] = f"{basedir}/jobs/script_condor_transfer_flow.sh"
    sub['Error'] = f"{basedir}/jobs/error/transfer_flow-$(ClusterId).$(ProcId).err"
    sub['Output'] = f"{basedir}/jobs/output/transfer_flow-$(ClusterId).$(Proc1Id).out"
    sub['Log'] = f"{basedir}/jobs/log/transfer_flow-$(ClusterId).log"
    sub['MY.SendCredential'] = True
    sub['MY.SingularityImage'] = '"/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/dvalsecc/memflow:latest"'
    sub['+JobFlavour'] = '"nextweek"'
    sub['arguments'] = f"transferFlow_train/transferFlow_test.yaml transferFlow_test"

# General
sub['request_cpus'] = f"{args.ngpu*3}"
sub['request_gpus'] = f"{args.ngpu}"
if args.good_gpus:
    sub['requirements'] = 'regexp("A100", TARGET.CUDADeviceName) || regexp("V100", TARGET.CUDADeviceName)'

    
if model != "flow_evaluation_labframe" and not os.path.exists(f"{basedir}/configs/{sub['arguments'].split()[0]}"):
    print("Missing configuration file! The jobs has not been submitted")
    exit(1)

if antonio:
    sub['arguments'] = sub['arguments'] + " antonio"
    
print(sub)
if not dry:
    schedd = htcondor.Schedd()
    with schedd.transaction() as txn:
        cluster_id = sub.queue(txn)

    print(f"Submitted to {cluster_id:=}")
