from . import rambo_generator
from . import utils
import torch


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


def generate_random_phase_space_points(N, E_cm, initial_pdgs, final_masses, pdf=None):
    final_state_mass = torch.sum(final_masses)

    pdf_active = True if pdf else False

    generator = rambo_generator.FlatInvertiblePhasespace(
        [0.0, 0.0], final_masses, pdf=pdf, pdf_active=pdf_active, tau=False
    )

    # Sampling correctly x1 and x2
    if pdf_active:
        x1_, x2_, wpdf = generate_x1x2(N, E_cm, final_state_mass)
        rnd = torch.cat((torch.rand(N, generator.nDimPhaseSpace()), x1_, x2_), axis=1)
    else:
        rnd = torch.rand(N, generator.nDimPhaseSpace())

    momenta, weight, x1, x2 = generator.generateKinematics_batch(
        E_cm, rnd, pdgs=initial_pdgs
    )

    return momenta, weight, x1, x2, wpdf


def get_momenta_from_ps(points, E_cm, initial_pdgs, final_masses, pdf=None):
    pdf_active = True if pdf else False
    generator = rambo_generator.FlatInvertiblePhasespace(
        [0.0, 0.0], final_masses, pdf=pdf, pdf_active=pdf_active, tau=False
    )

    x1, x2 = get_x1x2(points[:, -2], points[:, -1], E_cm)
    rnd = torch.cat((points[:, generator.nDimPhaseSpace() - 2], x1, x2), axis=1)

    momenta, weight, *x = generator.generateKinematics_batch(
        E_cm, rnd, pdgs=initial_pdgs
    )

    return momenta, weight, x1, x2
