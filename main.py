from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def inversion_point(z: complex, center: complex, radius: float) -> complex:
    """
    円 center, radius に関する点 z の反転を返す。

    反転の公式:
        z' = center + radius^2 * (z - center) / |z - center|^2

    ただし z == center のときは反転先が無限遠点になるため、
    この関数では ValueError を送出する。
    """
    offset = z - center
    denominator = abs(offset) ** 2

    if denominator == 0:
        raise ValueError("円の中心は反転できません。")

    return center + (radius**2) * offset / denominator


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * i for i in range(count)]


def make_grid_lines(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    step: float,
    samples: int = 400,
) -> list[list[complex]]:
    """複素平面上の格子線を生成する。"""
    lines: list[list[complex]] = []

    x = xmin
    while x <= xmax + 1e-12:
        lines.append([complex(x, y) for y in linspace(ymin, ymax, samples)])
        x += step

    y = ymin
    while y <= ymax + 1e-12:
        lines.append([complex(x, y) for x in linspace(xmin, xmax, samples)])
        y += step

    return lines


def split_line_at_center(line: list[complex], center: complex, eps: float = 1e-12) -> list[list[complex]]:
    """
    中心付近の点を含む線分を分割する。
    反転で中心を含む点は無限遠に飛ぶため除外する。
    """
    segments: list[list[complex]] = []
    current: list[complex] = []

    for p in line:
        if abs(p - center) < eps:
            if len(current) >= 2:
                segments.append(current)
            current = []
        else:
            current.append(p)

    if len(current) >= 2:
        segments.append(current)

    return segments


def invert_polyline(points: list[complex], center: complex, radius: float) -> list[complex]:
    """点列を円に関して反転する。"""
    return [inversion_point(p, center, radius) for p in points]


def plot_complex_line(ax, points: list[complex], **kwargs) -> None:
    """複素数列を x-y 平面の折れ線として描画する。"""
    if len(points) < 2:
        return
    xs = [p.real for p in points]
    ys = [p.imag for p in points]
    ax.plot(xs, ys, **kwargs)


def visualize_inversion_of_grid(
    center: complex = 0 + 0j,
    radius: float = 1.0,
    grid_step: float = 0.25,
    extent: float = 3.0,
) -> None:
    """円に関する格子反転を可視化する。"""
    fig, ax = plt.subplots(figsize=(8, 8))

    grid_lines = make_grid_lines(
        xmin=-extent,
        xmax=extent,
        ymin=-extent,
        ymax=extent,
        step=grid_step,
    )

    # 元の格子
    for line in grid_lines:
        plot_complex_line(ax, line, color="#9a9a9a", linewidth=0.8, alpha=0.45)

    # 反転後の格子
    for line in grid_lines:
        for segment in split_line_at_center(line, center):
            inverted = invert_polyline(segment, center, radius)
            plot_complex_line(ax, inverted, color="crimson", linewidth=1.2, alpha=0.9)

    # 反転円
    circle = Circle(
        (center.real, center.imag),
        radius,
        fill=False,
        color="black",
        linewidth=2.2,
    )
    ax.add_patch(circle)

    ax.scatter([center.real], [center.imag], color="black", s=30, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("円に関する反転による格子の変形")
    ax.grid(False)

    plt.show()


def main() -> None:
    print("Hello from indraspearl!")
    visualize_inversion_of_grid(center=0 + 0j, radius=1.0, grid_step=0.25, extent=3.0)


if __name__ == "__main__":
    main()
