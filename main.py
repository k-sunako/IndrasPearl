from __future__ import annotations

from fastplotlib import loop

from src.lib import build_inversion_figure, fixed_point, sample_mobius_transform


def main() -> None:
    print("Hello from indraspearl!")

    T = sample_mobius_transform()
    try:
        fp = fixed_point(T)
        print(f"固定点: {fp}")
    except ValueError as exc:
        print(f"固定点を 1 つに定められません: {exc}")

    fig = build_inversion_figure(center=0 + 0j, radius=1.0, grid_step=0.25, extent=3.0)
    fig.show()

    loop.run()


if __name__ == "__main__":
    main()
