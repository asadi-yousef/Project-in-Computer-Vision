"""Shared plot styling, so every Stage 2 figure encodes method and T the same way.

Colour identifies the method; line style and marker identify the Euler-step
count. Keeping the maps here rather than in one plotting module means a
reader can carry the legend from one figure to the next.
"""

from typing import Optional, Tuple

# The prototype baseline is black so it reads as the reference the FM
# variants are measured against.
METHOD_COLORS = {
    "linear_probe": "tab:gray",
    "prototype": "black",
    "fm_standard": "tab:blue",
    "fm_rolled": "tab:red",
}
DEFAULT_METHOD_COLOR = "tab:green"

# (linestyle, marker) per Euler-step count. None covers the Stage 1 methods,
# which have no T.
EULER_STYLES = {
    None: ("-", "o"),
    4: ("-", "o"),
    12: ("--", "s"),
}
DEFAULT_EULER_STYLE = (":", "^")


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, DEFAULT_METHOD_COLOR)


def euler_style(num_euler_steps: Optional[int]) -> Tuple[str, str]:
    return EULER_STYLES.get(num_euler_steps, DEFAULT_EULER_STYLE)
