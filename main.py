from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


def configure_japanese_font() -> None:
    """
    matplotlib の日本語フォントを設定する。

    利用可能性が高い Noto Serif CJK JP を優先する。
    存在しない可能性のあるフォント名は候補から外す。
    """
    matplotlib.rcParams["font.family"] = [
        "Noto Serif CJK JP",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


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


def split_line_at_center(
    line: list[complex],
    center: complex,
    eps: float = 1e-12,
) -> list[list[complex]]:
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
    configure_japanese_font()

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
        plot_complex_line(
            ax,
            line,
            color="#9a9a9a",
            linewidth=0.8,
            alpha=0.45,
            zorder=1,
        )

    # 反転後の格子
    for line in grid_lines:
        for segment in split_line_at_center(line, center):
            inverted = invert_polyline(segment, center, radius)
            plot_complex_line(
                ax,
                inverted,
                color="crimson",
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )

    # 反転円
    circle = Circle(
        (center.real, center.imag),
        radius,
        fill=False,
        color="black",
        linewidth=2.2,
        zorder=3,
    )
    ax.add_patch(circle)

    ax.scatter([center.real], [center.imag], color="black", s=30, zorder=4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("実部")
    ax.set_ylabel("虚部")
    ax.set_title("円に関する反転による格子の変形")
    ax.grid(False)

    legend_handles = [
        Line2D([0], [0], color="#9a9a9a", lw=1.2, alpha=0.7, label="元の格子"),
        Line2D([0], [0], color="crimson", lw=1.5, alpha=0.9, label="反転後の格子"),
        Line2D([0], [0], color="black", lw=2.2, label="反転の円"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    plt.show()


def main() -> None:
    print("Hello from indraspearl!")
    visualize_inversion_of_grid(center=0 + 0j, radius=1.0, grid_step=0.25, extent=3.0)


if __name__ == "__main__":
    main()
