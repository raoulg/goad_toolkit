from dataclasses import dataclass
from typing import Any, Dict, List

from scipy import stats


@dataclass
class Distribution:
    """Dataclass to hold information about a statistical distribution."""

    name: str
    dist: Any  # The scipy.stats distribution object
    is_discrete: bool
    num_params: int

    def __str__(self):
        """String representation of the distribution."""
        return f"Distribution({self.name},discrete={self.is_discrete},params={self.num_params})"

    def __repr__(self):
        """Detailed representation of the distribution."""
        return f"Distribution(name='{self.name}',\n dist={self.dist.__class__.__name__},\n is_discrete={self.is_discrete},\n num_params={self.num_params})"


@dataclass
class DistributionRegistry:
    """Registry for statistical distributions with metadata."""

    _instance = None

    def __new__(cls):
        """Ensure only one instance of DistributionRegistry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the registry with common distributions."""
        if not self.initialized:
            self.distributions: Dict[str, Distribution] = {}
            self.register_distribution(
                "norm", stats.norm, is_discrete=False, num_params=2
            )
            self.register_distribution(
                "uniform", stats.uniform, is_discrete=False, num_params=2
            )
            self.register_distribution(
                "lognorm", stats.lognorm, is_discrete=False, num_params=3
            )
            self.register_distribution(
                "poisson", stats.poisson, is_discrete=True, num_params=1
            )
            self.register_distribution(
                "exponential", stats.expon, is_discrete=False, num_params=2
            )
            self.register_distribution(
                "skewnorm", stats.skewnorm, is_discrete=False, num_params=3
            )
            self.register_distribution(
                "gamma", stats.gamma, is_discrete=False, num_params=3
            )
            self.register_distribution(
                "weibull", stats.weibull_min, is_discrete=False, num_params=3
            )
            self.initialized = True

    def __repr__(self) -> str:
        """Detailed representation of the registry."""
        return f"DistributionRegistry({self.get_names()})"

    def register_distribution(
        self, name: str, dist, is_discrete: bool, num_params: int
    ):
        """Register a new distribution with metadata."""
        self.distributions[name] = Distribution(
            name=name, dist=dist, is_discrete=is_discrete, num_params=num_params
        )

    def get_distribution(self, name: str) -> Distribution:
        """Get a distribution by name."""
        if name not in self.distributions:
            raise ValueError(f"Distribution '{name}' not found in registry.")
        return self.distributions[name]

    def get_names(self) -> List[str]:
        """Get all registered distribution names."""
        return list(self.distributions.keys())

    def is_discrete(self, name: str) -> bool:
        """Check if a distribution is discrete."""
        return self.get_distribution(name).is_discrete
