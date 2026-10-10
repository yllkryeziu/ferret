# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Ferret's mascot: a ferret curled around a magnifying glass whose lens holds
the trigrams of "ferret". Every shape is placed by hand; a displacement
filter gives the lines a pen wobble, and a dilated outline makes the drawing
a sticker that reads on light and dark backgrounds.

    python3 docs/assets/draw_logo.py docs/assets/ferret-logo.svg
"""

import math
import random
import sys

CHAR, TERRA, SLATE = "#2A2926", "#C2633F", "#4E5763"
CARAMEL, SABLE, DARK, CREAM, BELLY = (
    "#C79563",
    "#9A6A42",
    "#4B3324",
    "#F4E8D4",
    "#F7EEDF",
)
PAPER, PINK, GLASS = "#FBF8F1", "#E3A49A", "#DCE7EA"
C = (232.0, 250.0)  # lens centre
R = 146.0  # rim radius
T0, T1 = 290.0, 605.0  # the body runs clockwise from the neck to the tail tip
HIPS = 0.79  # body parameter where the tail begins
random.seed(7)


def pt(theta, r):
    a = math.radians(theta)
    return (C[0] + r * math.cos(a), C[1] + r * math.sin(a))


def f(p):
    return f"{p[0]:.1f},{p[1]:.1f}"


def radius(t):
    """Centre line of the body: on the rim, the tail tip lifting a little."""
    return R + 2 + 7 * max(0.0, (t - 0.9) / 0.1)


def half(t):
    """Half-width of the body: neck, trunk, hips, then a tapering tail."""
    if t < 0.06:
        return 21 + 100 * t
    if t < 0.70:
        return 27
    if t < HIPS:
        return 27 + 2.5 * math.sin((t - 0.70) / (HIPS - 0.70) * math.pi)
    u = (t - HIPS) / (1 - HIPS)
    return 17 - 9 * u**1.3


def ribbon(ta, tb, n, rad=radius, wid=half, cap=False):
    """A closed band along the body from parameter ta to tb."""
    outer, inner = [], []
    for i in range(n + 1):
        t = ta + (tb - ta) * i / n
        th = T0 + (T1 - T0) * t
        outer.append(pt(th, rad(t) + wid(t)))
        inner.append(pt(th, rad(t) - wid(t)))
    d = "M" + " L".join(f(p) for p in outer)
    if cap:
        w = wid(tb)
        d += f" A{w:.1f},{w:.1f} 0 0 1 {f(inner[-1])}"
    return d + " L" + " L".join(f(p) for p in inner[::-1]) + " Z"


def paw(theta, r, rot):
    x, y = pt(theta, r)
    return (
        f'<g transform="translate({x:.1f},{y:.1f}) rotate({rot})">'
        f'<ellipse rx="13" ry="10" fill="{DARK}" stroke="{CHAR}" stroke-width="4.5"/>'
        f'<path d="M-5,5 L-5,8 M0,6 L0,9 M5,5 L5,8" stroke="{CREAM}" stroke-width="2" opacity="0.8"/></g>'
    )


out = []
a = out.append
a(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">'
)
a("<title>Ferret</title>")
a(f'''<defs>
  <filter id="rough" x="-5%" y="-5%" width="110%" height="110%">
    <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="4" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="3.2" xChannelSelector="R" yChannelSelector="G"/>
  </filter>
  <filter id="sticker" x="-10%" y="-10%" width="120%" height="120%">
    <feMorphology in="SourceAlpha" operator="dilate" radius="9" result="fat"/>
    <feFlood flood-color="{PAPER}"/>
    <feComposite in2="fat" operator="in" result="border"/>
    <feDropShadow in="border" dx="0" dy="3" stdDeviation="4" flood-color="#1E1712" flood-opacity="0.28" result="shadowed"/>
    <feMerge><feMergeNode in="shadowed"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>''')
a(
    '<g filter="url(#sticker)"><g filter="url(#rough)" stroke-linecap="round" stroke-linejoin="round">'
)

# Handle, leaving the rim at 45 degrees, with a terracotta ferrule.
a(
    f'<g transform="translate({C[0]},{C[1]}) rotate(45)">'
    f'<rect x="{R + 2}" y="-24" width="156" height="48" rx="22" fill="{CHAR}" stroke="{CHAR}" stroke-width="5"/>'
    f'<rect x="{R + 26}" y="-26" width="32" height="52" rx="8" fill="{TERRA}" stroke="{CHAR}" stroke-width="5"/>'
    f'<path d="M{R + 70},-12 L{R + 140},-12" stroke="#5A5650" stroke-width="5" fill="none"/></g>'
)

# Glass and rim.
a(f'<circle cx="{C[0]}" cy="{C[1]}" r="{R - 6}" fill="{GLASS}"/>')
a(
    f'<circle cx="{C[0]}" cy="{C[1]}" r="{R}" fill="none" stroke="{SLATE}" stroke-width="16"/>'
)

# The trigrams of "ferret", as tiles in the lens; one is a hit.
TW, TH, G = 76, 50, 10
for text, sx, sy, hit in (
    ("fer", -1, -1, False),
    ("err", 1, -1, True),
    ("rre", -1, 1, False),
    ("ret", 1, 1, False),
):
    x = C[0] + sx * (TW + G) / 2 - TW / 2
    y = C[1] + 18 + sy * (TH + G) / 2 - TH / 2
    a(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{TW}" height="{TH}" rx="9" '
        f'fill="{TERRA if hit else PAPER}" stroke="{CHAR}" stroke-width="4"/>'
    )
    a(
        f'<text x="{x + TW / 2:.1f}" y="{y + TH / 2 + 11:.1f}" text-anchor="middle" '
        "font-family=\"ui-monospace, SFMono-Regular, Menlo, Consolas, 'DejaVu Sans Mono', monospace\" "
        f'font-size="31" font-weight="700" fill="{PAPER if hit else CHAR}" stroke="none">{text}</text>'
    )

# Glint.
a(
    f'<path d="M{f(pt(200, R - 44))} A{R - 44},{R - 44} 0 0 1 {f(pt(238, R - 44))}" stroke="#FFFFFF" '
    'stroke-width="9" fill="none" opacity="0.85"/>'
)
a(
    f'<circle cx="{pt(249, R - 44)[0]:.1f}" cy="{pt(249, R - 44)[1]:.1f}" r="4.5" fill="#FFFFFF" opacity="0.85"/>'
)

# Tail first, so the rump covers its root.
a(
    f'<path d="{ribbon(HIPS - 0.03, 1.0, 70, cap=True)}" fill="{DARK}" stroke="{CHAR}" stroke-width="5"/>'
)
for i in range(8):
    t = HIPS + 0.04 + i * 0.021
    th = T0 + (T1 - T0) * t
    a(
        f'<path d="M{f(pt(th, radius(t) + half(t) * 0.1))} L{f(pt(th + 3, radius(t) + half(t) * 0.75))}" '
        'stroke="#7A5638" stroke-width="2" opacity="0.7" fill="none"/>'
    )

# Hind paws on the inner rim, under the rump.
a(paw(157, R - 30, 72))
a(paw(171, R - 31, 86))

# Body: caramel, a darker back on the outer edge, a cream belly inside, fur.
END = HIPS + 0.015
a(f'<path d="{ribbon(0, END, 140, cap=True)}" fill="{CARAMEL}"/>')
a(
    f'<path d="{ribbon(0.02, END - 0.01, 130, lambda t: radius(t) + half(t) * 0.6, lambda t: half(t) * 0.38, cap=True)}" fill="{SABLE}"/>'
)
a(
    f'<path d="{ribbon(0.04, END - 0.04, 130, lambda t: radius(t) - half(t) * 0.6, lambda t: half(t) * 0.3, cap=True)}" fill="{BELLY}"/>'
)
for i in range(44):
    t = 0.03 + i * 0.0168
    th = T0 + (T1 - T0) * t
    r0 = radius(t) + half(t) * 0.42 + random.uniform(-1.5, 1.5)
    a(
        f'<path d="M{f(pt(th, r0))} L{f(pt(th + 2.6, r0 + 8 + random.uniform(-1.5, 1.5)))}" '
        f'stroke="{DARK}" stroke-width="2.3" opacity="0.5" fill="none"/>'
    )
a(
    f'<path d="{ribbon(0, END, 140, cap=True)}" fill="none" stroke="{CHAR}" stroke-width="5"/>'
)

# Front paws over the rim, under the chin.
a(paw(263, R - 34, 0))
a(paw(279, R - 34, 0))

# Head, facing the viewer, resting on the top of the rim.
a(f'<g transform="translate({C[0]:.1f},{C[1] - R - 10:.1f})">')
for s in (-1, 1):
    a(
        f'<circle cx="{42 * s}" cy="-40" r="16" fill="{CARAMEL}" stroke="{CHAR}" stroke-width="5"/>'
        f'<circle cx="{42 * s}" cy="-39" r="8" fill="{PINK}"/>'
    )
a(
    f'<path d="M-52,-4 C-54,-40 -28,-58 0,-58 C28,-58 54,-40 52,-4 C50,24 30,46 0,48 '
    f'C-30,46 -50,24 -52,-4 Z" fill="{CREAM}" stroke="{CHAR}" stroke-width="5"/>'
)
a(
    f'<path d="M-42,-30 C-32,-56 32,-56 42,-30 C28,-38 13,-34 0,-22 C-13,-34 -28,-38 -42,-30 Z" fill="{CARAMEL}"/>'
)
a(
    '<path d="M-48,-6 C-46,-24 -26,-26 -14,-15 C-8,-10 -4,-9 0,-9 C4,-9 8,-10 14,-15 C26,-26 46,-24 48,-6 '
    'C46,8 30,11 18,5 C10,1 4,3 0,5 C-4,3 -10,1 -18,5 C-30,11 -46,8 -48,-6 Z" fill="#4A3324"/>'
)
for s in (-1, 1):
    a(
        f'<ellipse cx="{24 * s}" cy="-8" rx="7.5" ry="8.5" fill="#15110E"/>'
        f'<circle cx="{24 * s + 2.6}" cy="-11" r="2.8" fill="#FFFFFF"/>'
        f'<ellipse cx="{33 * s}" cy="18" rx="8" ry="5" fill="{PINK}" opacity="0.45"/>'
    )
a('<ellipse cx="0" cy="22" rx="23" ry="15" fill="#FBF6EC"/>')
a(
    f'<path d="M-8,10 C-8,5 8,5 8,10 C8,15 2,18 0,18 C-2,18 -8,15 -8,10 Z" fill="#D98A86" stroke="{CHAR}" stroke-width="2.8"/>'
)
a(
    f'<path d="M0,18 L0,23 M-9,24 C-6,28 -2,27 0,23 C2,27 6,28 9,24" stroke="{CHAR}" stroke-width="2.8" fill="none"/>'
)
for s in (-1, 1):
    for y0, y1 in ((18, 10), (22, 22), (26, 34)):
        a(
            f'<path d="M{14 * s},{y0} L{50 * s},{y1}" stroke="{CHAR}" stroke-width="1.8" opacity="0.75" fill="none"/>'
        )
a("</g>")

a("</g></g></svg>")
with open(sys.argv[1] if len(sys.argv) > 1 else "ferret-logo.svg", "w") as svg:
    svg.write("\n".join(out) + "\n")
