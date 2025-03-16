import pymc as pm
import arviz as az
import numpy as np
import constants as c

from typing import List, Dict

def build_model(data_dict: Dict[str, List[int | float]], coords: Dict[str, np.array]) -> pm.Model:
    """Given the correct data dictionary and model coordinates, builds a PyMC model.

    Parameters
    ----------
    data_dict : Dict[str, List[int  |  float]]
        A data dictionary for the model.
    coords : Dict[str, np.array]
        the model coordinates

    Returns
    -------
    pm.Model
        The PyMC model.
    """

    with pm.Model(coords=coords) as model:

        # Data
        home_idx = pm.Data("home_idx", data_dict[c.HOME])
        away_idx = pm.Data("away_idx", data_dict[c.AWAY])
        at_home = pm.Data("at_home", data_dict[c.LOCATION])
        home_ppp = pm.Data("home_ppp", data_dict[c.HOME_PPP])
        away_ppp = pm.Data("away_ppp", data_dict[c.AWAY_PPP])
        poss = pm.Data("poss", data_dict[c.POSSESSIONS])

        ## Parameters 
        eta_off = pm.HalfCauchy("eta_off", 1)
        eta_def = pm.HalfCauchy("eta_def", 1)

        theta_off = pm.Normal("theta_off", mu=0, sigma=eta_off, dims="teams")
        theta_def = pm.Normal("theta_def", mu=0, sigma=eta_def, dims="teams")

        league_eff = pm.Normal("league_eff", mu=1, sigma=0.1)
        league_hfa = pm.Normal("league_hfa", mu=0.03, sigma=0.015)

        sigma_eff = pm.HalfCauchy("sigma_eff", 1)

        rho_pace = pm.HalfCauchy("rho_pace", 1)
        team_pace = pm.TruncatedNormal("team_pace", mu=1, sigma=rho_pace, lower=0, dims="teams")
        league_pace = pm.TruncatedNormal("league_pace", mu=70, sigma=5, lower=0)
        sigma_poss = pm.HalfCauchy("sigma_poss", 3)

        home_xeff = pm.Deterministic("home_xeff", league_eff + league_hfa * at_home + theta_off[home_idx] - theta_def[away_idx])
        away_xeff = pm.Deterministic("away_xeff", league_eff - league_hfa * at_home + theta_off[away_idx] - theta_def[home_idx])
        xposs = pm.Deterministic("xposs", league_pace * team_pace[home_idx] * team_pace[away_idx])
        sigma_game_eff = pm.Deterministic("sigma_game_eff", sigma_eff * league_pace / xposs)

        # Likelihoods
        home_eff = pm.Normal("home_eff", mu=home_xeff, sigma=sigma_game_eff, observed=home_ppp, shape=home_idx.shape)
        away_eff = pm.Normal("away_eff", mu=away_xeff, sigma=sigma_game_eff, observed=away_ppp, shape=home_idx.shape)
        game_poss = pm.Normal("game_poss", mu=xposs, sigma=sigma_poss, observed=poss, shape=home_idx.shape)

        # Deterministics
        home_pts = pm.Deterministic("home_pts", home_eff * game_poss)
        away_pts = pm.Deterministic("away_pts", away_eff * game_poss)
        pts_diff = pm.Deterministic("pts_diff", home_pts - away_pts)
        _ = pm.Deterministic("home_win", pts_diff > 0)

    return model

def sample_model(model: pm.Model) -> az.InferenceData:
    """Samples the supplied model.

    Parameters
    ----------
    model : pm.Model
        A PyMC model to sample.

    Returns
    -------
    az.InferenceData
        The sampled model trace.
    """

    with model:
        idata = pm.sample()

    return idata