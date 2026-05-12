from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fastplotlib import Figure, LineGraphic, ScatterGraphic


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
) -> list[np.ndarray]:
    """複素平面上の格子線を生成する。"""
    lines: list[np.ndarray] = []

    x = xmin
    while x <= xmax + 1e-12:
        ys = np.array(linspace(ymin, ymax, samples), dtype=float)
        xs = np.full_like(ys, x)
        lines.append(np.column_stack([xs, ys]))
        x += step

    y = ymin
    while y <= ymax + 1e-12:
        xs = np.array(linspace(xmin, xmax, samples), dtype=float)
        ys = np.full_like(xs, y)
        lines.append(np.column_stack([xs, ys]))
        y += step

    return lines


def split_line_at_center(
    line: np.ndarray,
    center: complex,
    eps: float = 1e-12,
) -> list[np.ndarray]:
    """
    中心付近の点を含む線分を分割する。
    """
    segments: list[np.ndarray] = []
    current: list[list[float]] = []

    for p in line:
        if abs(complex(p[0], p[1]) - center) < eps:
            if len(current) >= 2:
                segments.append(np.array(current, dtype=float))
            current = []
        else:
            current.append([p[0], p[1]])

    if len(current) >= 2:
        segments.append(np.array(current, dtype=float))

    return segments


def inversion_point(z: complex, center: complex, radius: float) -> complex:
    """
    円 center, radius に関する点 z の反転を返す。
    """
    offset = z - center
    denominator = abs(offset) ** 2

    if denominator == 0:
        raise ValueError("円の中心は反転できません。")

    return center + (radius**2) * offset / denominator


def invert_polyline(points: np.ndarray, center: complex, radius: float) -> np.ndarray:
    """点列を円に関して反転する。"""
    out = []
    for p in points:
        z = complex(p[0], p[1])
        w = inversion_point(z, center, radius)
        out.append([w.real, w.imag])
    return np.array(out, dtype=float)


def visualize_inversion_of_grid(
    center: complex = 0 + 0j,
    radius: float = 1.0,
    grid_step: float = 0.25,
    extent: float = 3.0,
) -> None:
    """円に関する格子反転を可視化する。fastplotlib でウィンドウ表示する。"""
    grid_lines = make_grid_lines(
        xmin=-extent,
        xmax=extent,
        ymin=-extent,
        ymax=extent,
        step=grid_step,
    )

    fig = Figure()

    # 元の格子
    for line in grid_lines:
        fig.add_graphic(
            LineGraphic(
                data=line,
                colors="gray",
                thickness=1.0,
                alpha=0.45,
            )
        )

    # 反転後の格子
    for line in grid_lines:
        for segment in split_line_at_center(line, center):
            inverted = invert_polyline(segment, center, radius)
            fig.add_graphic(
                LineGraphic(
                    data=inverted,
                    colors="crimson",
                    thickness=1.5,
                    alpha=0.9,
                )
            )

    # 円
    theta = np.linspace(0, 2 * np.pi, 512)
    circle = np.column_stack(
        [
            center.real + radius * np.cos(theta),
            center.imag + radius * np.sin(theta),
        ]
    )
    fig.add_graphic(
        LineGraphic(
            data=circle,
            colors="black",
            thickness=2.5,
            alpha=1.0,
        )
    )

    # 中心点
    fig.add_graphic(
        ScatterGraphic(
            data=np.array([[center.real, center.imag]], dtype=float),
            colors="black",
            sizes=20,
        )
    )

    fig.show()
