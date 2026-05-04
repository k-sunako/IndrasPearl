def inversion_point(z, center, radius):
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

    return center + (radius ** 2) * offset / denominator


def main():
    print("Hello from indraspearl!")


if __name__ == "__main__":
    main()
