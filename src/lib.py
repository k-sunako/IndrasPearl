from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os

import numpy as np

# rendercanvas / fastplotlib はバックエンド選択より先に読み込まれると
# GLFW を拾って失敗することがあるため、Qt バックエンドを先に明示する。
os.environ.setdefault("RENDERCANVAS_FORCE_BACKEND", "qt")

from fastplotlib import Figure, LineGraphic, ScatterGraphic  # noqa: E402


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

    def __mul__(self, other: "MobiusTransform") -> "MobiusTransform":
        """
        積演算として合成を行う。

        self * other は「先に other、その後 self」を表す。
        """
        if not isinstance(other, MobiusTransform):
            return NotImplemented

        return MobiusTransform(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
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


def _complex_from_xy(p: np.ndarray) -> complex:
    return complex(float(p[0]), float(p[1]))


def _xy_from_complex(z: complex) -> np.ndarray:
    return np.array([z.real, z.imag], dtype=np.float32)


def _circle_from_three_points(
    z1: complex,
    z2: complex,
    z3: complex,
    eps: float = 1e-12,
) -> tuple[complex, float]:
    """
    3 点を通る円を求める。
    3 点がほぼ一直線なら ValueError を送出する。
    """
    x1, y1 = z1.real, z1.imag
    x2, y2 = z2.real, z2.imag
    x3, y3 = z3.real, z3.imag

    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3

    e = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
    f = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0

    det = a * d - b * c
    if abs(det) < eps:
        raise ValueError("円の像が直線になります。")

    center_x = (d * e - b * f) / det
    center_y = (-c * e + a * f) / det
    center = complex(center_x, center_y)

    radius = abs(center - z1)
    return center, float(radius)


def mobius_on_circle(
    T: MobiusTransform,
    center: complex,
    radius: float,
) -> tuple[complex, float]:
    """
    円 |z - center| = radius の像の中心と半径を返す。

    3 点を円周上の一定パターンで取り、その像から円を復元する。
    像が直線になる場合は ValueError を送出する。
    """
    if radius < 0:
        raise ValueError("半径は 0 以上である必要があります。")

    # 円周上の 3 点を固定のパターンで取る
    z1 = center + radius
    z2 = center + radius * 1j
    z3 = center - radius

    w1 = mobius_on_point(T, z1)
    w2 = mobius_on_point(T, z2)
    w3 = mobius_on_point(T, z3)

    if w1 is None or w2 is None or w3 is None:
        raise ValueError("円の像が直線になります。")

    return _circle_from_three_points(w1, w2, w3)


def compose(T: MobiusTransform, S: MobiusTransform) -> MobiusTransform:
    """
    合成 T ∘ S を返す。
    先に S、次に T を適用する。
    """
    return T * S


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
        if d == a:
            return [None] if b == 0 else []
        return [b / (d - a)]

    # z = (a z + b) / (c z + d)
    # => c z^2 + (d - a) z - b = 0
    B = d - a
    discriminant = B * B + 4 * b * c
    sqrt_discriminant = discriminant ** 0.5

    z1 = (-B + sqrt_discriminant) / (2 * c)
    z2 = (-B - sqrt_discriminant) / (2 * c)

    if z1 == z2:
        return [z1]

    return [z1, z2]


def fixed_point(T: MobiusTransform) -> ComplexOrInfinity:
    """
    固定点が 1 つだけのときにその値を返す。

    固定点が 0 個または 2 個の場合は ValueError を送出する。
    """
    fps = fixed_points(T)
    if len(fps) != 1:
        raise ValueError(f"固定点が 1 つではありません: {fps}")
    return fps[0]


def sample_mobius_transform() -> MobiusTransform:
    """
    例として使いやすいメビウス変換を返す。
    """
    return MobiusTransform(
        a=1 + 0j,
        b=1 + 0j,
        c=0 + 0j,
        d=1 + 0j,
    )


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
        ys = np.array(linspace(ymin, ymax, samples), dtype=np.float32)
        xs = np.full_like(ys, x, dtype=np.float32)
        lines.append(np.column_stack([xs, ys]).astype(np.float32))
        x += step

    y = ymin
    while y <= ymax + 1e-12:
        xs = np.array(linspace(xmin, xmax, samples), dtype=np.float32)
        ys = np.full_like(xs, y, dtype=np.float32)
        lines.append(np.column_stack([xs, ys]).astype(np.float32))
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
        if abs(complex(float(p[0]), float(p[1])) - center) < eps:
            if len(current) >= 2:
                segments.append(np.array(current, dtype=np.float32))
            current = []
        else:
            current.append([float(p[0]), float(p[1])])

    if len(current) >= 2:
        segments.append(np.array(current, dtype=np.float32))

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
        z = complex(float(p[0]), float(p[1]))
        w = inversion_point(z, center, radius)
        out.append([w.real, w.imag])
    return np.array(out, dtype=np.float32)


def _line_graphic_from_points(points: np.ndarray, color: str, thickness: float, alpha: float) -> LineGraphic:
    return LineGraphic(
        data=points.astype(np.float32),
        colors=color,
        thickness=thickness,
        alpha=alpha,
    )


def _scatter_graphic_from_point(point: np.ndarray, color: str, size: float) -> ScatterGraphic:
    return ScatterGraphic(
        data=point.astype(np.float32),
        colors=color,
        sizes=size,
    )


def schottky_dance(
    gens: dict[str, MobiusTransform],
    seeds: dict[str, tuple[complex, float]],
    depth: int,
) -> list[tuple[str, complex, float]]:
    """
    ショットキー円の像を世代ごとにたどる。

    gens:
        {"a": a, "b": b, "A": A, "B": B} のような生成元
    seeds:
        {"a": (center, radius), ...} の初期円
    depth:
        何世代までたどるか

    返り値:
        (word, center, radius) の列
    """
    if depth < 0:
        raise ValueError("depth は 0 以上である必要があります。")

    def inverse_name(name: str) -> str:
        if name == "a":
            return "A"
        if name == "A":
            return "a"
        if name == "b":
            return "B"
        if name == "B":
            return "b"
        raise ValueError(f"未知の生成元名です: {name}")

    out: list[tuple[str, complex, float]] = []
    current: list[tuple[str, complex, float]] = [
        (name, seeds[name][0], seeds[name][1]) for name in seeds
    ]

    for _ in range(depth):
        next_current: list[tuple[str, complex, float]] = []
        for word, center, radius in current:
            out.append((word, center, radius))
            last = word[-1]

            for gname, T in gens.items():
                if inverse_name(last) == gname:
                    continue

                if gname not in seeds:
                    continue

                c0, r0 = seeds[gname]
                try:
                    c1, r1 = mobius_on_circle(T, c0, r0)
                except ValueError:
                    continue

                next_current.append((word + gname, c1, r1))

        current = next_current

    return out


def build_inversion_figure(
    center: complex = 0 + 0j,
    radius: float = 1.0,
    grid_step: float = 0.25,
    extent: float = 3.0,
) -> Figure:
    """円に関する格子反転を可視化する Figure を構築して返す。"""
    grid_lines = make_grid_lines(
        xmin=-extent,
        xmax=extent,
        ymin=-extent,
        ymax=extent,
        step=grid_step,
    )

    fig = Figure()
    layout = fig[0, 0]

    for line in grid_lines:
        layout.add_graphic(
            _line_graphic_from_points(
                line,
                color="gray",
                thickness=1.0,
                alpha=0.45,
            )
        )

    for line in grid_lines:
        for segment in split_line_at_center(line, center):
            inverted = invert_polyline(segment, center, radius)
            layout.add_graphic(
                _line_graphic_from_points(
                    inverted,
                    color="crimson",
                    thickness=1.5,
                    alpha=0.9,
                )
            )

    theta = np.linspace(0, 2 * np.pi, 512, dtype=np.float32)
    circle = np.column_stack(
        [
            np.float32(center.real) + np.float32(radius) * np.cos(theta).astype(np.float32),
            np.float32(center.imag) + np.float32(radius) * np.sin(theta).astype(np.float32),
        ]
    ).astype(np.float32)
    layout.add_graphic(
        _line_graphic_from_points(
            circle,
            color="black",
            thickness=2.5,
            alpha=1.0,
        )
    )

    layout.add_graphic(
        _scatter_graphic_from_point(
            np.array([[center.real, center.imag]], dtype=np.float32),
            color="black",
            size=20,
        )
    )

    return fig
