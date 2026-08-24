"""Traffic-light colors for Trust Report signals."""

from __future__ import annotations

from collections.abc import Mapping

from judgetrust.config import TRUST_THRESHOLDS
from judgetrust.models import ColorName

_RANK = {"green": 0, "amber": 1, "red": 2}


def traffic_light(
    value: float | None,
    green: float,
    amber: float,
    *,
    higher_is_better: bool,
) -> ColorName:
    """Map a metric onto green / amber / red. None → gray."""

    if value is None:
        return "gray"
    if higher_is_better:
        if value >= green:
            return "green"
        if value >= amber:
            return "amber"
        return "red"
    if value <= green:
        return "green"
    if value <= amber:
        return "amber"
    return "red"


def signal_colors(
    *,
    kappa: float | None,
    position_consistency: float | None,
    length_bias_rate: float | None,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[ColorName, ColorName, ColorName, ColorName]:
    """Return (kappa, consistency, length-bias, overall) colors."""

    cuts = dict(thresholds or TRUST_THRESHOLDS)
    kappa_color = traffic_light(
        kappa,
        cuts["kappa_green"],
        cuts["kappa_amber"],
        higher_is_better=True,
    )
    consistency_color = traffic_light(
        position_consistency,
        cuts["position_consistency_green"],
        cuts["position_consistency_amber"],
        higher_is_better=True,
    )
    length_color = traffic_light(
        length_bias_rate,
        cuts["length_bias_green"],
        cuts["length_bias_amber"],
        higher_is_better=False,
    )
    return kappa_color, consistency_color, length_color, overall_color(
        kappa_color, consistency_color, length_color
    )


def overall_color(*colors: ColorName) -> ColorName:
    """Worst of the non-gray signals. All gray → gray."""

    present = [color for color in colors if color != "gray"]
    if not present:
        return "gray"
    return max(present, key=lambda color: _RANK[color])
