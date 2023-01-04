from . import rambo_generator
from . import utils
import torch
import tensorflow as tf
from particle import Particle

def generate_x1x2(N, E_cm, final_state_mass):
    rnd1 = torch.rand(N, 1)
    rnd2 = torch.rand(N, 1)
    min_fract = final_state_mass / E_cm
    x1, dw1 = utils.uniform_distr(rnd1, min_fract, 1)
    x2, dw2 = utils.uniform_distr_t(rnd2, min_fract / x1, torch.ones_like(rnd2))
    return x1, x2, dw1 * dw2


def get_x1x2(x1_u, x2_u, E_cm, final_state_mass):
    '''Transform uniformally distributed x1 and x2 to rescaled
    x1 and x2 to account for final_state_mass
    '''
    min_fract = final_state_mass / E_cm
    x1, dw1 = utils.uniform_distr(x1_u, min_fract, 1)
    x2, dw2 = utils.uniform_distr_t(x2_u, min_fract / x1, torch.ones_like(x2_u))
    return x1, x2, dw1 * dw2


def get_pdfQ2(self, pdf, pdg, x, scale2):
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



class PhaseSpace:

    def __init__(self, E_cm, initial_pdgs, final_pdgs, pdf=None):
        self.E_cm = E_cm
        self.initial_pdgs = initial_pdgs
        self.final_pdgs = final_pdgs
        self.final_masses = torch.tensor([ Particle.from_pdgid(pdg).mass/1e3 for pdg in self.final_pdgs])
        self.final_state_mass = torch.sum(self.final_masses)
        self.pdf = pdf
        # init the generatoer
        self.generator = rambo_generator.FlatInvertiblePhasespace(
            [0.0, 0.0], #initial particle mass
            self.final_masses,
            pdf=pdf, pdf_active=True, tau=False
        )

    def generate_random_phase_space_points(self, N):
        '''
        Generate N random phase space points from the CM of E_cm energy,
        representing n final state particles with final_masses mass.
        If `pdf` is not None (but a pdfflow instance), the pdf weight is included and
        read from the `pdf` object.
        '''
        # The pdf_active flag is true so that the last two random
        # points represent the x1 and x2.
        # 
       
        # Sampling correctly x1 and x2
        x1_, x2_, wx1x2 = generate_x1x2(N, self.E_cm, self.final_state_mass)
        rnd = torch.cat((torch.rand(N, self.generator.nDimPhaseSpace()),
                         x1_, x2_), axis=1)

        momenta, weight, x1, x2 = self.generator.generateKinematics_batch(
            self.E_cm, rnd, pdgs=self.initial_pdgs
        )
        #multiply x1,x2 trasformation jacobian to the weight
        weight *= wx1x2.squeeze()

        return rnd, momenta, weight, x1, x2


    def get_momenta_from_ps(self, points):
        pass
