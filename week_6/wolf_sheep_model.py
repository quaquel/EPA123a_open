import mesa
from mesa.discrete_space import CellAgent, FixedAgent, OrthogonalMooreGrid

class Animal(CellAgent):
    """The base animal class."""

    def __init__(self, model, energy, p_reproduce, energy_from_food, cell):
        """Initializes an animal.

        Args:
            model: a model instance
            energy: starting amount of energy
            p_reproduce: probability of sexless reproduction
            energy_from_food: energy obtained from 1 unit of food
            cell: the cell in which the animal starts
        """
        super().__init__(model)
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.cell = cell

    def spawn_offspring(self):
        """Create offspring."""
        self.energy /= 2
        self.__class__(
            self.model,
            self.energy,
            self.p_reproduce,
            self.energy_from_food,
            self.cell,
        )

    def feed(self): ...

    def step(self):
        """One step of the agent."""
        self.cell = self.cell.neighborhood.select_random_cell()
        self.energy -= 1

        self.feed()

        if self.energy < 0:
            self.remove()
        elif self.random.random() < self.p_reproduce:
            self.spawn_offspring()


class Sheep(Animal):
    """A sheep that walks around, reproduces (asexually) and gets eaten."""

    def feed(self):
        """If possible eat the food in the current location."""
        # If there is grass available, eat it
        if self.model.grass:
            grass_patch = next(
                obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
            )
            if grass_patch.fully_grown:
                self.energy += self.energy_from_food
                grass_patch.fully_grown = False


class Wolf(Animal):
    """A wolf that walks around, reproduces (asexually) and eats sheep."""

    def feed(self):
        """If possible eat the food in the current location."""
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if len(sheep) > 0:
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food

            # Kill the sheep
            sheep_to_eat.remove()


class GrassPatch(FixedAgent):
    """
    A patch of grass that grows at a fixed rate and it is eaten by sheep
    """

    def __init__(self, model, fully_grown, countdown):
        """
        Creates a new patch of grass

        Args:
            grown: (boolean) Whether the patch of grass is fully grown or not
            countdown: Time for the patch of grass to be fully grown again
        """
        super().__init__(model)
        self.fully_grown = fully_grown
        self.countdown = countdown

    def step(self):
        if not self.fully_grown:
            if self.countdown <= 0:
                # Set as fully grown
                self.fully_grown = True
                self.countdown = self.model.grass_regrowth_time
            else:
                self.countdown -= 1

class WolfSheep(mesa.Model):
    """
    Wolf-Sheep Predation Model
    """

    height = 20
    width = 20

    initial_sheep = 100
    initial_wolves = 50

    sheep_reproduce = 0.04
    wolf_reproduce = 0.05

    wolf_gain_from_food = 20

    grass = False
    grass_regrowth_time = 30
    sheep_gain_from_food = 4

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=50,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=False,
        grass_regrowth_time=30,
        sheep_gain_from_food=4,
        seed=None,
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.

        Args:
            initial_sheep: Number of sheep to start with
            initial_wolves: Number of wolves to start with
            sheep_reproduce: Probability of each sheep reproducing each step
            wolf_reproduce: Probability of each wolf reproducing each step
            wolf_gain_from_food: Energy a wolf gains from eating a sheep
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
            sheep_gain_from_food: Energy sheep gain from grass, if enabled.
        """
        super().__init__(seed=seed)
        # Set parameters
        self.width = width
        self.height = height
        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.grass = grass
        self.grass_regrowth_time = grass_regrowth_time

        self.grid = OrthogonalMooreGrid((self.width, self.height), torus=True, random=self.random)

        collectors = {
            "Wolves": lambda m: len(m.agents_by_type[Wolf]),
            "Sheep": lambda m: len(m.agents_by_type[Sheep]),
            "Grass": lambda m: len(
                m.agents_by_type[GrassPatch].select(lambda a: a.fully_grown)
            )
            if m.grass
            else -1,
        }

        self.datacollector = mesa.DataCollector(collectors)

        # Create sheep:
        for _ in range(self.initial_sheep):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.sheep_gain_from_food)
            Sheep(
                self, energy, sheep_reproduce, sheep_gain_from_food, self.grid[(x, y)]
            )

        # Create wolves
        for _ in range(self.initial_wolves):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            energy = self.random.randrange(2 * self.wolf_gain_from_food)
            Wolf(self, energy, wolf_reproduce, wolf_gain_from_food, self.grid[(x, y)])

        # Create grass patches
        if self.grass:
            for cell in self.grid.all_cells:
                fully_grown = self.random.choice([True, False])

                if fully_grown:
                    countdown = self.grass_regrowth_time
                else:
                    countdown = self.random.randrange(self.grass_regrowth_time)

                patch = GrassPatch(self, fully_grown, countdown)
                patch.cell = cell

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.random.shuffle(self.agent_types)
        for agent_type in self.agent_types:
            self.agents_by_type[agent_type].shuffle_do("step")

        # collect data
        self.datacollector.collect(self)

