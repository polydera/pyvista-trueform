"""
accessor.triangulated() vs PyVista's PolyData.triangulate()

Both measured as a user experiences them: a fresh dataset, one call,
first touch — for the trueform side that includes the accessor's
dataset-to-Mesh conversion, paid on every fresh dataset. Medians of
five reps, alternating which method goes first each rep so neither
side keeps the process-warm-up advantage.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/pyvista-trueform
"""

import statistics
import time

import pyvista as pv

import pyvista_trueform  # registers the accessor  # noqa: F401


def _fresh_plane(i_resolution, j_resolution):
    return pv.Plane(i_resolution=i_resolution, j_resolution=j_resolution)


def _time_pyvista(i_resolution, j_resolution):
    dataset = _fresh_plane(i_resolution, j_resolution)
    start = time.perf_counter()
    dataset.triangulate()
    return time.perf_counter() - start


def _time_trueform(i_resolution, j_resolution):
    dataset = _fresh_plane(i_resolution, j_resolution)
    start = time.perf_counter()
    dataset.trueform.triangulated()
    return time.perf_counter() - start


def measure(i_resolution=1000, j_resolution=1000, reps=5):
    """Per-rep wall times of both triangulation paths, fresh dataset each.

    ``i_resolution`` x ``j_resolution`` sizes the source ``pv.Plane`` (a
    quad grid, ``i_resolution * j_resolution`` faces); the default is a
    million quads. Each rep builds its own dataset for each method, and
    the two methods alternate which goes first from rep to rep.

    Returns
    -------
    dict
        ``{"pyvista": [...], "trueform": [...]}``, ``reps`` wall times
        in seconds each.
    """
    pyvista_times = []
    trueform_times = []
    for rep in range(reps):
        if rep % 2 == 0:
            pyvista_times.append(_time_pyvista(i_resolution, j_resolution))
            trueform_times.append(_time_trueform(i_resolution, j_resolution))
        else:
            trueform_times.append(_time_trueform(i_resolution, j_resolution))
            pyvista_times.append(_time_pyvista(i_resolution, j_resolution))
    return {"pyvista": pyvista_times, "trueform": trueform_times}


def main():
    results = measure()
    pyvista_median = statistics.median(results["pyvista"])
    trueform_median = statistics.median(results["trueform"])

    print(f"{'method':<10}{'median (s)':>12}   samples (s)")
    for name, times in results.items():
        median = statistics.median(times)
        samples = ", ".join(f"{t:.4f}" for t in times)
        print(f"{name:<10}{median:>12.4f}   {samples}")
    print(f"\ntrueform is {pyvista_median / trueform_median:.2f}x "
          "pyvista's median")


if __name__ == "__main__":
    main()
