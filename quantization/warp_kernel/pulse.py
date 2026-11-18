
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from .surrogate import BITS, EnergySurrogate


@dataclass
class PulseResult:
    bitmap: list[int]
    predicted_energy: float
    average_bitwidth: float
    total_bits: int
    bitwidth_cap: float
    bit_units_cap: int
    summed_trps: float
    best_trps: float
    trps_limit: float
    trps_tolerance: float
    visited_nodes: int


def solve_pulse(
    scores: dict[tuple[int, int], float],
    surrogate: EnergySurrogate,
    trps_tolerance: float,
    bitwidth_cap: float = 3.1,
) -> PulseResult:
    n = surrogate.n_blocks
    cap = Fraction(str(bitwidth_cap))
    tolerance = Fraction(str(trps_tolerance))
    if n < 1 or not min(BITS) <= cap <= max(BITS) or tolerance < 0:
        raise ValueError("Expected blocks, a bit cap between 2 and 4, and nonnegative TRPS tolerance")
    total_cap = int(n * cap)
    extra_cap = total_cap - n * min(BITS)

    def integer_table(values, *extras):
       
        rows = [[Fraction(str(values[l, b])) for b in BITS] for l in range(n)]
        scale = lcm(*(v.denominator for row in rows for v in row),
                    *(v.denominator for v in extras))
        return [[int(v * scale) for v in row] for row in rows], scale

    losses, loss_scale = integer_table(scores, tolerance)
    energies, _ = integer_table(surrogate.costs)

   
    min_loss = [[(0, 0)] * (extra_cap + 1) for _ in range(n + 1)]
    min_energy = [[0] * (extra_cap + 1) for _ in range(n + 1)]
    best_loss_choice = [[0] * (extra_cap + 1) for _ in range(n)]
    for l in reversed(range(n)):
        for remaining in range(extra_cap + 1):
            choices = range(min(len(BITS), remaining + 1))
            loss, energy, choice = min(
                (losses[l][b] + min_loss[l + 1][remaining - b][0],
                 energies[l][b] + min_loss[l + 1][remaining - b][1], b)
                for b in choices
            )
            min_loss[l][remaining] = (loss, energy)
            best_loss_choice[l][remaining] = choice
            min_energy[l][remaining] = min(
                energies[l][b] + min_energy[l + 1][remaining - b] for b in choices
            )

    best_loss, initial_energy = min_loss[0][extra_cap]
    limit = best_loss + int(tolerance * loss_scale)
    incumbent = (initial_energy, best_loss)
    remaining = extra_cap
    bitmap = []
    for l in range(n):
        choice = best_loss_choice[l][remaining]
        bitmap.append(BITS[choice])
        remaining -= choice

    visited = 0
    path = []

    def pulse(l, remaining, energy, loss):
        nonlocal incumbent, bitmap, visited
        visited += 1
        loss_bound = loss + min_loss[l][remaining][0]
        energy_bound = energy + min_energy[l][remaining]
        if loss_bound > limit or (energy_bound, loss_bound) >= incumbent:
            return
        if l == n:
            incumbent = (energy, loss)
            bitmap = path.copy()
            return
        choices = sorted(
            range(min(len(BITS), remaining + 1)),
            key=lambda b: energies[l][b] + min_energy[l + 1][remaining - b],
        )
        for b in choices:
            path.append(BITS[b])
            pulse(l + 1, remaining - b, energy + energies[l][b], loss + losses[l][b])
            path.pop()

    pulse(0, extra_cap, 0, 0)
    return PulseResult(
        bitmap=bitmap,
        predicted_energy=surrogate.predict(bitmap),
        average_bitwidth=sum(bitmap) / n,
        total_bits=sum(bitmap),
        bitwidth_cap=float(cap),
        bit_units_cap=total_cap,
        summed_trps=incumbent[1] / loss_scale,
        best_trps=best_loss / loss_scale,
        trps_limit=limit / loss_scale,
        trps_tolerance=float(tolerance),
        visited_nodes=visited,
    )
