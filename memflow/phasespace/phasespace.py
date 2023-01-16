from . import rambo_generator
from . import utils
import torch
import tensorflow as tf
from particle import Particle


def get_pdfxQ2(pdf, pdg, x, scale2):
    """Call the PDF and return the corresponding density."""
    if pdf is None:
        return torch.ones_like(x)

    if pdg not in [21] and abs(pdg) not in range(1, 7):
        return torch.ones_like(x)

    # Call to lhapdf API
    f = pdf.xfxQ2(
        [pdg],
        tf.convert_to_tensor(x, dtype=tf.float64),
        tf.convert_to_tensor(scale2, dtype=tf.float64),
    )
    return torch.tensor(f.numpy(), dtype=torch.double, device=x.device)

def get_pdf_weight(x1, x2, pdgs, pdf):
    q2 = torch.ones_like(x1)*91.88**2
    return torch.tensor(
        (get_pdfxQ2(pdf, pdgs[0], x1, q2)*
         get_pdfxQ2(pdf, pdgs[1], x2, q2)
         ).numpy())
    


class PhaseSpace:
    def __init__(self, E_cm, initial_pdgs, final_pdgs, pdf=None):
        self.E_cm = E_cm
        self.initial_pdgs = initial_pdgs
        self.final_pdgs = final_pdgs
        self.final_masses = torch.tensor(
            [Particle.from_pdgid(pdg).mass / 1e3 for pdg in self.final_pdgs]
        )
        self.final_state_mass = torch.sum(self.final_masses)
        self.pdf = pdf
        # init the generatoer
        self.generator = rambo_generator.FlatInvertiblePhasespace(
            [0.0, 0.0],  # initial particle mass
            self.final_masses,
            pdf=pdf,
            pdf_active=True,
            tau=False,
        )

    def generate_random_phase_space_points(self, N,
                                           pT_mincut=-1,
                                           delR_mincut=-1,
                                           rap_maxcut=-1):
        '''
        Generate N random phase space points from the CM of E_cm energy,
        representing n final state particles with final_masses mass.

        '''
        # Sampling correctly x1 and x2
        x1x2_rand = torch.rand(N, 2)
        x1_, x2_, wx1x2 = self.get_x1x2_from_uniform(x1x2_rand)
        ps_rand = torch.rand(N, self.generator.nDimPhaseSpace)
        # For rambo we need actual x1 x2
        rambo_input = torch.cat((ps_rand, x1_.unsqueeze(1), x2_.unsqueeze(1)), axis=1)
        # for the output we return the [0,1] uniform x1x2 representation
        points_out = torch.cat((ps_rand, x1x2_rand), axis=1)

        momenta, weight, x1, x2 = self.generator.generateKinematics_batch(
            self.E_cm, rambo_input, pdgs=self.initial_pdgs,
            pT_mincut=pT_mincut, delR_mincut=delR_mincut,
            rap_maxcut=rap_maxcut
        )
        # multiply x1,x2 trasformation jacobian to the weight
        weight *= wx1x2.squeeze()

        return points_out, momenta, weight, x1, x2

    def get_momenta_from_ps(self, points,
                            pT_mincut=-1,
                            delR_mincut=-1,
                            rap_maxcut=-1):
        # First of all get the transformed x1, x2 from the last two numbers
        x1_, x2_, wx1x2 = self.get_x1x2_from_uniform(points[:, -2:])

        rambo_input = torch.cat((points[:, :-2],
                                 x1_.unsqueeze(1),
                                 x2_.unsqueeze(1)), axis=1)

        momenta, weight, x1, x2 = self.generator.generateKinematics_batch(
            self.E_cm, rambo_input, pdgs=self.initial_pdgs,
            pT_mincut=pT_mincut, delR_mincut=delR_mincut,
            rap_maxcut=rap_maxcut
        )
        # multiply x1,x2 trasformation jacobian to the weight
        weight *= wx1x2.squeeze()

        return momenta, weight, x1, x2

    def get_ps_from_momenta(self, momenta, x1, x2):
        ps = self.generator.getPSpoint_batch(self.E_cm, momenta)
        # Getting x1x2 uniform space point
        r1, r2 = self.get_uniform_from_x1x2(x1, x2)
        # Concat at the end
        return torch.cat((ps, r1.unsqueeze(1), r2.unsqueeze(1)), axis=1)

    def get_x1x2_from_uniform(self, r):
        '''Transform a pair of uniformally distributed variables r (N,2),
        in x1, x2 pairs keeping into account the minimum energy constraint
        given by E_cm and finalstate total mass.
        The jacobina factor the of the transformation is returned.'''
        min_fract = (self.final_state_mass / self.E_cm).to(x1.device)
        x1, dw1 = utils.uniform_distr(r[:, 0], min_fract, 1)
        x2, dw2 = utils.uniform_distr_t(r[:, 1], min_fract / x1, torch.ones(r.shape[0], device=r.device))
        return x1, x2, dw1 * dw2

    def get_uniform_from_x1x2(self, x1, x2):
        min_fract = (self.final_state_mass / self.E_cm).to(x1.device)
        r1u = (x1 - min_fract) / (1 - min_fract)
        r2u = (x2 - (min_fract / x1)) / (1 - (min_fract / x1))
        return r1u, r2u
