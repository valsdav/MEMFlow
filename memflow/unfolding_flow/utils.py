import torch
import vector
import numpy as np
import awkward as ak
from memflow.phasespace.phasespace import PhaseSpace

M_HIGGS = 125.25
M_TOP = 172.76

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_flat_tensor(X, fields, axis=1, allow_missing=False):
    return torch.tensor(np.stack([ak.to_numpy(getattr(X,f), allow_missing=allow_missing) for f in fields], axis=axis))

class Compute_ParticlesTensor:

    def get_HttISR(cond_X, log_mean, log_std, device):

        higgs = cond_X[0].unsqueeze(dim=1)
        thad = cond_X[1].unsqueeze(dim=1)
        tlep = cond_X[2].unsqueeze(dim=1)
        Htt = torch.cat((higgs, thad, tlep), dim=1)

        unscaledlog = Htt*log_std + log_mean
        data_regressed = unscaledlog
        data_regressed[:,:,0] = torch.sign(unscaledlog[:,:,0])*(torch.exp(torch.abs(unscaledlog[:,:,0])) - 1)

        higgs = data_regressed[:,0]
        thad = data_regressed[:,1]
        tlep = data_regressed[:,2]

        higgs = vector.array(
            {
                "pt": higgs[:,0].detach().cpu().numpy(),
                "eta": higgs[:,1].detach().cpu().numpy(),
                "phi": higgs[:,2].detach().cpu().numpy(),
                "mass": M_HIGGS*np.ones(higgs.shape[0])
            }
        )

        thad = vector.array(
            {
                "pt": thad[:,0].detach().cpu().numpy(),
                "eta": thad[:,1].detach().cpu().numpy(),
                "phi": thad[:,2].detach().cpu().numpy(),
                "mass": M_TOP*np.ones(thad.shape[0])
            }
        )

        tlep = vector.array(
            {
                "pt": tlep[:,0].detach().cpu().numpy(),
                "eta": tlep[:,1].detach().cpu().numpy(),
                "phi": tlep[:,2].detach().cpu().numpy(),
                "mass": M_TOP*np.ones(tlep.shape[0])
            }
        )

        higgs = ak.with_name(higgs, name="Momentum4D")
        thad = ak.with_name(thad, name="Momentum4D")
        tlep = ak.with_name(tlep, name="Momentum4D")

        gluon_px = -(higgs.px + thad.px + tlep.px)
        gluon_py = -(higgs.py + thad.py + tlep.py)
        gluon_pz = -(higgs.pz + thad.pz + tlep.pz)
        E_gluon = np.sqrt(gluon_px**2 + gluon_py**2 + gluon_pz**2)

        gluon_px = np.expand_dims(gluon_px, axis=1)
        gluon_py = np.expand_dims(gluon_py, axis=1)
        gluon_pz = np.expand_dims(gluon_pz, axis=1)
        E_gluon = np.expand_dims(E_gluon, axis=1)

        gluon = np.concatenate((E_gluon, gluon_px, gluon_py, gluon_pz), axis=1)

        glISR = vector.array(
            {
                "E": gluon[:,0],
                "px": gluon[:,1],
                "py": gluon[:,2],
                "pz": gluon[:,3],
            }
        )

        glISR = ak.with_name(glISR, name="Momentum4D")

        higgs_tensor = to_flat_tensor(higgs, ["t", "x", "y", "z"], axis=1, allow_missing=False).unsqueeze(dim=1)
        thad_tensor = to_flat_tensor(thad, ["t", "x", "y", "z"], axis=1, allow_missing=False).unsqueeze(dim=1)
        tlep_tensor = to_flat_tensor(tlep, ["t", "x", "y", "z"], axis=1, allow_missing=False).unsqueeze(dim=1)
        gluon_tensor = torch.Tensor(gluon).unsqueeze(dim=1)

        data_regressed = torch.cat((higgs_tensor, thad_tensor, tlep_tensor, gluon_tensor), dim=1)

        boost_regressed = higgs + thad + tlep + glISR
        boost_regressed = to_flat_tensor(boost_regressed, ["t", "x", "y", "z"], axis=1, allow_missing=False)

        with torch.no_grad():
            if (device == torch.device('cuda')):
                data_regressed = data_regressed.cuda()
                boost_regressed = boost_regressed.cuda()
        
        return data_regressed, boost_regressed

    def get_PS(data_HttISR, boost):
        E_CM = 13000
        phasespace = PhaseSpace(E_CM, [21, 21], [25, 6, -6, 21], dev="cpu")
        x1 = (boost[:, 0] + boost[:, 3]) / E_CM
        x2 = (boost[:, 0] - boost[:, 3]) / E_CM

        ps_regressed, detjinv_regressed = phasespace.get_ps_from_momenta(data_HttISR, x1, x2, ensure_CM=True)

        return ps_regressed, detjinv_regressed
