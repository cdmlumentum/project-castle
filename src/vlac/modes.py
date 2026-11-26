"""Mode snapshots, overlap tracking and self-consistent coupling extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np


class InvalidModeError(RuntimeError):
    """A completed FDE solve did not yield a usable isolated mode."""


@dataclass
class ModeSnapshot:
    neff: complex
    ex: np.ndarray
    ey: np.ndarray
    te_fraction: float
    core_fraction: float
    identity: str

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray([self.neff.real, self.neff.imag]).tobytes())
        digest.update(self.ex[::4, ::4].astype(np.complex64).tobytes())
        digest.update(self.ey[::4, ::4].astype(np.complex64).tobytes())
        return digest.hexdigest()


@dataclass
class ModeReference:
    upper: ModeSnapshot
    lower: ModeSnapshot
    supermodes: tuple[ModeSnapshot, ModeSnapshot]

    def fingerprint(self) -> str:
        values = (
            self.upper.fingerprint(),
            self.lower.fingerprint(),
            self.supermodes[0].fingerprint(),
            self.supermodes[1].fingerprint(),
        )
        return hashlib.sha256("|".join(values).encode()).hexdigest()


@dataclass
class BareModeReference:
    upper: ModeSnapshot
    lower: ModeSnapshot

    def fingerprint(self) -> str:
        return hashlib.sha256(
            f"{self.upper.fingerprint()}|{self.lower.fingerprint()}".encode()
        ).hexdigest()


@dataclass
class Coefficients:
    beta_upper: complex
    beta_lower: complex
    beta_plus: complex
    beta_minus: complex
    delta_beta: complex
    delta_pm: complex
    delta_12: complex
    kappa: complex
    kappa_profile_check: float
    overlap_upper_lower: float
    min_mode_overlap: float
    upper_identity: str
    lower_identity: str
    supermode_identities: tuple[str, str]
    extraction_flag: str = "ok"


def field_overlap(a: ModeSnapshot, b: ModeSnapshot) -> float:
    """Normalized E-field overlap on the shared fixed tracking grid."""
    if a.ex.shape != b.ex.shape or a.ey.shape != b.ey.shape:
        raise ValueError("Mode snapshots must use the same field-tracking grid.")
    numerator = np.vdot(a.ex, b.ex) + np.vdot(a.ey, b.ey)
    norm_a = np.vdot(a.ex, a.ex).real + np.vdot(a.ey, a.ey).real
    norm_b = np.vdot(b.ex, b.ex).real + np.vdot(b.ey, b.ey).real
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(np.clip(abs(numerator) / np.sqrt(norm_a * norm_b), 0.0, 1.0))


def select_mode(
    candidates: list[ModeSnapshot],
    *,
    previous: tuple[ModeReference, ...] = (),
    guide: str,
    minimum_te_fraction: float,
) -> tuple[ModeSnapshot, float]:
    if not candidates:
        raise InvalidModeError(f"FDE returned no modes for the {guide} isolated guide.")
    scored: list[tuple[float, ModeSnapshot, float]] = []
    for mode in candidates:
        overlap = 0.0
        if previous:
            ref_modes = [getattr(ref, guide) for ref in previous]
            overlap = max(field_overlap(mode, ref_mode) for ref_mode in ref_modes)
            score = 0.7 * overlap + 0.2 * mode.core_fraction + 0.1 * mode.te_fraction
        else:
            score = 0.65 * mode.core_fraction + 0.35 * mode.te_fraction
        scored.append((score, mode, overlap))
    eligible = [v for v in scored if v[1].te_fraction >= minimum_te_fraction]
    if not eligible:
        eligible = scored
    score, selected, overlap = max(
        eligible,
        key=lambda item: (item[0], item[1].neff.real),
    )
    return selected, (1.0 if not previous else overlap)


def select_supermode_pair(
    candidates: list[ModeSnapshot],
    upper: ModeSnapshot,
    lower: ModeSnapshot,
    *,
    previous: tuple[ModeReference, ...] = (),
    minimum_te_fraction: float,
) -> tuple[ModeSnapshot, ModeSnapshot, float]:
    if len(candidates) < 2:
        raise RuntimeError(
            "FDE found fewer than two modes in the stacked cross-section; "
            "increase number_of_trial_modes or review the geometry/domain."
        )
    valid = [m for m in candidates if m.te_fraction >= minimum_te_fraction] or candidates
    best: tuple[float, ModeSnapshot, ModeSnapshot, float] | None = None
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            m1, m2 = valid[i], valid[j]
            bare_score = (
                field_overlap(m1, upper) ** 2
                + field_overlap(m1, lower) ** 2
                + field_overlap(m2, upper) ** 2
                + field_overlap(m2, lower) ** 2
            )
            track_score = 0.0
            track_overlap = 1.0
            if previous:
                assignments = []
                for ref in previous:
                    ov_a = field_overlap(m1, ref.supermodes[0]) + field_overlap(
                        m2, ref.supermodes[1]
                    )
                    ov_b = field_overlap(m1, ref.supermodes[1]) + field_overlap(
                        m2, ref.supermodes[0]
                    )
                    assignments.append(max(ov_a, ov_b) / 2)
                track_overlap = max(assignments)
                track_score = track_overlap
            score = (0.75 * bare_score / 4.0) + (0.25 * track_score)
            candidate = (score, m1, m2, track_overlap)
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    _, first, second, tracking_overlap = best
    # A consistent high/low beta convention defines delta_pm >= 0.
    plus, minus = sorted((first, second), key=lambda m: m.neff.real, reverse=True)
    return plus, minus, tracking_overlap


def extract_coupling(
    beta_upper: complex,
    beta_lower: complex,
    beta_plus: complex,
    beta_minus: complex,
    *,
    previous_kappa: complex | None = None,
    kappa_profile_check: float = float("nan"),
    overlap_upper_lower: float = float("nan"),
    min_mode_overlap: float = 1.0,
    identities: tuple[str, str, str, str] = ("upper", "lower", "plus", "minus"),
) -> Coefficients:
    """Extract detuning and coupling from uncoupled and coupled FDE modes.

    Sign convention: delta_beta = beta_upper - beta_lower. The upper guide is
    guide 1; the lower guide is guide 2. beta_plus/minus are ordered by real
    propagation constant, so delta_pm=(beta_plus-beta_minus)/2 is nonnegative
    in its real part. kappa is the principal square root of delta_pm^2-delta_12^2,
    phase-followed against the previous z/lambda sample when available.
    """
    delta_beta = beta_upper - beta_lower
    delta_12 = delta_beta / 2
    delta_pm = (beta_plus - beta_minus) / 2
    kappa_squared = delta_pm**2 - delta_12**2
    kappa = complex(np.sqrt(kappa_squared + 0j))
    if previous_kappa is not None and abs(-kappa - previous_kappa) < abs(
        kappa - previous_kappa
    ):
        kappa = -kappa
    elif previous_kappa is None:
        # Fix the initial gauge with nonnegative real part, then nonnegative
        # imaginary part if the real part is indistinguishable from zero.
        if kappa.real < -1e-12 or (
            abs(kappa.real) <= 1e-12 and kappa.imag < 0
        ):
            kappa = -kappa
    flag = "ok"
    scale = max(abs(delta_pm) ** 2, abs(delta_12) ** 2, 1.0)
    if kappa_squared.real < -1e-8 * scale and abs(kappa_squared.imag) < 1e-8 * scale:
        flag = "negative_kappa_squared"
    return Coefficients(
        beta_upper=beta_upper,
        beta_lower=beta_lower,
        beta_plus=beta_plus,
        beta_minus=beta_minus,
        delta_beta=delta_beta,
        delta_pm=delta_pm,
        delta_12=delta_12,
        kappa=kappa,
        kappa_profile_check=kappa_profile_check,
        overlap_upper_lower=overlap_upper_lower,
        min_mode_overlap=min_mode_overlap,
        upper_identity=identities[0],
        lower_identity=identities[1],
        supermode_identities=(identities[2], identities[3]),
        extraction_flag=flag,
    )
