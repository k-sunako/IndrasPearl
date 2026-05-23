from __future__ import annotations

import argparse

from src.lib import MobiusTransform, plot_schottky_dance


def build_demo_gens() -> dict[str, MobiusTransform]:
    """
    デモ用の生成元。
    実際のショットキー群に合わせて置き換えて使うこと。
    """
    return {
        "a": MobiusTransform(1 + 0j, 1 + 0j, 0 + 0j, 1 + 0j),
        "b": MobiusTransform(1 + 0j, -1 + 0j, 0 + 0j, 1 + 0j),
        "A": MobiusTransform(1 + 0j, -1 + 0j, 0 + 0j, 1 + 0j),
        "B": MobiusTransform(1 + 0j, 1 + 0j, 0 + 0j, 1 + 0j),
    }


def build_demo_seeds() -> dict[str, tuple[complex, float]]:
    """
    デモ用の初期円。
    実際のショットキー円に合わせて置き換えて使うこと。
    """
    return {
        "a": (0.0 + 0.0j, 1.0),
        "b": (2.0 + 0.0j, 1.0),
        "A": (0.0 + 0.0j, 1.0),
        "B": (2.0 + 0.0j, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="schottky_dance を使ってショットキー円の図を描画する"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="何世代まで描くか（default: 3）",
    )
    args = parser.parse_args()

    gens = build_demo_gens()
    seeds = build_demo_seeds()

    fig = plot_schottky_dance(gens, seeds, depth=args.depth)
    fig.show()


if __name__ == "__main__":
    main()
