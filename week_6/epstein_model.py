import math
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.axes_grid1 import make_axes_locatable

from mesa import Model, Agent
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid, CellAgent


class CitizenState(Enum):
    ACTIVE = 1
    QUIET = 2
    ARRESTED = 3


class CivilViolence(Model):
    """Model class for Eppstein's Civil Violence model I.

    The initial values are from Eppstein's article.
    """

    def __init__(self, height=40, width=40, citizen_density=0.7, citizen_vision=7,
                 legitimacy=0.82, activation_treshold=0.1, arrest_prob_constant=2.3,
                 cop_density=0.04, cop_vision=7, max_jail_term=15, seed=None):
        super().__init__(seed=seed)

        assert (citizen_density + cop_density) < 1

        # setup Citizen class attributes
        Citizen.vision = citizen_vision
        Citizen.legitimacy = legitimacy
        Citizen.arrest_prob_constant = arrest_prob_constant
        Citizen.activation_threshold = activation_treshold

        # setup Cop class attributes
        Cop.vision = cop_vision
        Cop.max_jail_term = max_jail_term

        self.grid = OrthogonalVonNeumannGrid((width, height), capacity=1,
                                             torus=True, random=self.random)

        # Set up agents
        for cell in self.grid.all_cells:
            klass = self.random.choices([Citizen, Cop, None],
                                        cum_weights=[citizen_density,
                                                     citizen_density + cop_density, 1])[0]
            if klass:
                agent = klass(self)
                agent.cell = cell
        # setup data collection
        self.ACTIVE = 0
        self.ARRESTED = 0
        self.QUIET = 0

        model_reporters = {'active': CitizenState.ACTIVE.name,
                           'quiet': CitizenState.QUIET.name,
                           'arrested': CitizenState.ARRESTED.name}
        self._update_counts()
        self.datacollector = DataCollector(model_reporters=model_reporters)
        self.datacollector.collect(self)

    def _update_counts(self):
        for state, count in self.agents_by_type[Citizen].groupby("state").count().items():
            setattr(self, state.name, count)

    def step(self):
        """
        Run one step of the model.
        """
        self.agents.shuffle_do("step")
        self._update_counts()
        self.datacollector.collect(self)


class BaseAgent(CellAgent):
    '''Base Agent class implementing vision and moving

    Attributes
    ----------
    moore : boolean

    '''

    def get_agents_in_vision(self):
        """
        identify cops and active citizens within vision

        Returns
        -------
        tuple with list of cops, and list of active citizens

        """
        cops = []
        active_citizens = []

        for agent in self.cell.get_neighborhood(radius=self.__class__.vision).agents:
            if isinstance(agent, Cop):
                cops.append(agent)
            elif agent.state == CitizenState.ACTIVE:
                active_citizens.append(agent)
        return cops, active_citizens

    def move(self):
        """Identify all empty cells within vision and move to a randomly selected one."""
        empty = [cell for cell in self.cell.get_neighborhood(radius=self.__class__.vision)
                 if cell.is_empty]

        if empty:
            self.cell = self.random.choice(empty)


class Citizen(BaseAgent):
    '''Citizen class

    Attributes
    ----------
    legitimacy : boolean
    vision : int
    arrest_prob_constant : float
    activation_treshold : float
    hardship : float
    risk_aversion : float
    state : {CitizenState.QUIET, CitizenState.ACTIVE, CitizenState.ARRESTED }
    jail_time_remaining  :int
    grievance : float

    '''
    legitimacy = 1
    vision = 1
    arrest_prob_constant = 1
    activation_treshold = 1

    def __init__(self, model):
        super().__init__(model)
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.state = CitizenState.QUIET
        self.jail_time_remaining = 0
        self.grievance = self.hardship * (1 - Citizen.legitimacy)

    def _check_jail_time(self):
        if (self.state == CitizenState.ARRESTED) and (self.jail_time_remaining > 0):
            self.jail_time_remaining -= 1
        return self.jail_time_remaining

    def step(self):
        """
        move and then decide whether to activate
        """

        if self._check_jail_time() > 0:
            return

        self.move()

        cops, active_citizens = self.get_agents_in_vision()
        n_cops = len(cops)
        n_active_citizens = len(active_citizens) + 1  # self is always considerd active

        arrest_p = 1 - math.exp(-1 * Citizen.arrest_prob_constant * round(n_cops / n_active_citizens))
        net_risk = self.risk_aversion * arrest_p

        if (self.grievance - net_risk) > self.activation_threshold:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET


class Cop(BaseAgent):
    '''Cop class

    Attributes
    ----------
    vision : int
    max_jail_term : int
    '''
    vision = 1
    max_jail_term = 1

    def step(self):
        self.move()
        _, active_citizens = self.get_agents_in_vision()

        if active_citizens:
            citizen = self.random.choice(active_citizens)
            citizen.state = CitizenState.ARRESTED
            citizen.jail_time_remaining = self.random.randint(0, Cop.max_jail_term)


def visualize_model(model):
    sns.set_style('white')
    colors = sns.color_palette()[0:4]

    # if the plot is not nice given your window size
    # consider changing figsize, but keep the ratio intact
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 20))
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    positions = []
    color_type = []
    grievance_level = []

    for cell in model.grid.all_cells:
        if cell.is_empty:
            continue

        positions.append(cell.coordinate)
        agent = cell.agents[0]
        if isinstance(agent, Citizen):
            color_type.append(colors[agent.state.value])
            grievance_level.append(agent.grievance)
        else:
            color_type.append(colors[0])
            grievance_level.append(np.nan)  # dirty hack for masking

    positions = np.asarray(positions)

    ax1.scatter(positions[:, 0], positions[:, 1], s=15, c=color_type)
    im = ax2.scatter(positions[:, 0], positions[:, 1], s=15, c=grievance_level)

    for ax in (ax1, ax2):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax1.set_title('Agent states')
    ax2.set_title('grievance view')