"""
state.py - Global state class with method to make Herbie objects.

idea:
    Use this if global contect is needed to be passed down from the main
    to subcommands.  Way to define attributes at the top-level.

    Not used at this time.
"""
from herbie.core import Herbie


class GlobalState:
    """Container for options shared by every subcommand."""

    def __init__(
        self,
        # dates: list[str],
        # model: str,
        # product: str | None,
        # fxx: list[int],
        verbose: bool,
        # overwrite: bool,
    ) -> None:
        # self.dates = dates
        # self.model = model
        # self.product = product
        # self.fxx = fxx
        self.verbose = verbose
        # self.overwrite = overwrite

    def make_herbie(self, date: str, fxx: int) -> Herbie:
        """Build a Herbie object for one (date, fxx) pair using the global settings."""
        kwargs = {
            "date": date,
            # "model": self.model,
            "fxx": fxx,
            "verbose": self.verbose,
            # "overwrite": self.overwrite,
        }
        # if self.product:
        #     kwargs["product"] = self.product

        return Herbie(**kwargs)
