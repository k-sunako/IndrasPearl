from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


ComplexOrInfinity = Optional[complex]


@dataclass(frozen=True)
class MobiusTransform:
    """
    1 次分数変換

        T(z) = (a z + b) / (c z + d)

    を表す。

    係数は複素数を想定する。
    """
    a: complex
    b: complex
    c: complex
    d: complex

    def determinant(self) -> complex:
        """行列式 ad - bc を返す。"""
        return self.a * self.d - self.b * self.c

    def normalized(self) -> "MobiusTransform":
        """
        行列式が 1 になるように正規化した変換を返す。

        行列式が 0 の場合は正規化できないので ValueError を送出する。
        """
        det = self.determinant()
        if det == 0:
            raise ValueError("行列式が 0 のため正規化できません。")

        scale = det ** 0.5
        return MobiusTransform(
            a=self.a / scale,
            b=self.b / scale,
            c=self.c / scale,
            d=self.d / scale,
        )


def mobius_on_point(T: MobiusTransform, z: ComplexOrInfinity) -> ComplexOrInfinity:
    """
    メビウス変換 T を点 z に作用させる。

    ルール:
    - z が None のとき、∞ を表すとみなす
    - z が有限複素数のときは (a z + b) / (c z + d)
    - 分母が 0 のときは None（∞）を返す
    - z = ∞ のときは c != 0 なら a/c、c == 0 なら ∞

    返り値の None は ∞ を表す。
    """
    if z is None:
        if T.c != 0:
            return T.a / T.c
        return None

    numerator = T.a * z + T.b
    denominator = T.c * z + T.d

    if denominator == 0:
        return None

    return numerator / denominator


def compose(T: MobiusTransform, S: MobiusTransform) -> MobiusTransform:
    """
    合成 T ∘ S を返す。
    先に S、次に T を適用する。
    """
    return MobiusTransform(
        a=T.a * S.a + T.b * S.c,
        b=T.a * S.b + T.b * S.d,
        c=T.c * S.a + T.d * S.c,
        d=T.c * S.b + T.d * S.d,
    )


def inverse(T: MobiusTransform) -> MobiusTransform:
    """
    逆変換を返す。

    det != 0 が必要。
    """
    det = T.determinant()
    if det == 0:
        raise ValueError("行列式が 0 のため逆変換を作れません。")

    return MobiusTransform(
        a=T.d / det,
        b=-T.b / det,
        c=-T.c / det,
        d=T.a / det,
    )


def fixed_points(T: MobiusTransform) -> list[ComplexOrInfinity]:
    """
    メビウス変換 T(z) = (a z + b) / (c z + d) の固定点を返す。

    固定点は方程式

        z = (a z + b) / (c z + d)

    すなわち

        c z^2 + (d - a) z - b = 0

    の解である。

    返り値は固定点のリストで、∞ は None で表す。
    """
    a, b, c, d = T.a, T.b, T.c, T.d

    if c == 0:
        if d - a == 0:
            return [None] if b == 0 else []
        return [b / (d - a)]

    B = d - a
    C = -b
    discriminant = B * B - 4 * c * C

    sqrt_discriminant = discriminant ** 0.5

    z1 = (-B + sqrt_discriminant) / (2 * c)
    z2 = (-B - sqrt_discriminant) / (2 * c)

    if z1 == z2:
        return [z1]

    return [z1, z2]


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

    for line in grid_lines:
        plot_complex_line(
            ax,
            line,
            color="#9a9a9a",
            linewidth=0.8,
            alpha=0.45,
            zorder=1,
        )

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
