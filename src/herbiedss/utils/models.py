from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastIntervalRule:
    first_fxx: int
    last_fxx: int
    step_hours: int
    interval_hours: int

    def matches(self, fxx: int) -> bool:
        return (
            self.first_fxx <= fxx <= self.last_fxx
            and (fxx - self.first_fxx) % self.step_hours == 0
        )


# Schedule supplied for your package.
FORECAST_INTERVAL_POLICIES = {
    ("gfs", "pgrb2.0p25"): (
        ForecastIntervalRule(
            first_fxx=1,
            last_fxx=120,
            step_hours=1,
            interval_hours=1,
        ),
        ForecastIntervalRule(
            first_fxx=123,
            last_fxx=384,
            step_hours=3,
            interval_hours=3,
        ),
    ),
}
