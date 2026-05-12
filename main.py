from __future__ import annotations

from fastplotlib import loop

from src.lib import visualize_inversion_of_grid


def main() -> None:
    print("Hello from indraspearl!")
    visualize_inversion_of_grid(center=0 + 0j, radius=1.0, grid_step=0.25, extent=3.0)
    loop()


if __name__ == "__main__":
    main()
