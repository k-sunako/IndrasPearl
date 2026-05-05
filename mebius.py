from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
