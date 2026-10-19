"""
Multi-region Green's functions: pick a 1D-model CPS database per station
based on which of several polygons the station falls inside.

Typical workflow
-----------------
    from hibasin.util.multi_region_greens import get_greens_tensors_multiregion

    # one polygon per 1D model, vertices as (lat, lon) pairs
    polygons = {
        'n_vel':  [(45.9, 15.0), (46.5, 15.0), (46.5, 16.5), (45.9, 16.5)],
        'e_vel':  [...],
        'se_vel': [...],
        'sw_vel': [...],
    }
    db_paths = {
        'n_vel':  './2D_GFs/n_vel/',
        'e_vel':  './2D_GFs/e_vel/',
        'se_vel': './2D_GFs/se_vel/',
        'sw_vel': './2D_GFs/sw_vel/',
    }

    greens = get_greens_tensors_multiregion(stations, origin, polygons, db_paths)

``greens`` is an ordinary ``mtuq.GreensTensorList``, in the same station
order as the input ``stations`` list, so it's a drop-in replacement for
``db.get_greens_tensors(stations, origin)`` -- everything downstream
(``greens.map(process_data)``, resampling, misfit, plotting) is unchanged.

A station that falls inside zero, or more than one, of the given polygons
raises a ``ValueError`` naming the station and its coordinates, rather than
silently guessing a region -- fix the polygon boundaries (or exclude the
station) and re-run.
"""

from collections import defaultdict

from mtuq import open_db
from mtuq.greens_tensor.base import GreensTensorList


def _point_in_polygon(point, polygon):
    """
    Ray-casting point-in-polygon test.

    ``point`` and each vertex of ``polygon`` are 2-tuples using the same
    (coord1, coord2) convention throughout -- e.g. both (lat, lon).
    """
    x, y = point
    inside = False
    n = len(polygon)
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x < x_intersect:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def assign_region(station, polygons):
    """
    Return the name of the single polygon in ``polygons`` (dict of
    {region_name: [(lat, lon), ...]}) that contains ``station``.

    Raises ValueError if the station falls inside zero or more than one
    polygon.
    """
    point = (station.latitude, station.longitude)
    matches = [name for name, poly in polygons.items()
               if _point_in_polygon(point, poly)]

    if len(matches) == 0:
        raise ValueError(
            f"Station {station.network}.{station.station} "
            f"(lat={station.latitude}, lon={station.longitude}) does not "
            f"fall inside any of the provided region polygons: "
            f"{sorted(polygons)}. Update the polygon boundaries, or "
            f"exclude this station, and re-run."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Station {station.network}.{station.station} "
            f"(lat={station.latitude}, lon={station.longitude}) falls "
            f"inside more than one region polygon: {matches}. Region "
            f"polygons must not overlap."
        )
    return matches[0]


def get_greens_tensors_multiregion(stations, origin, polygons, db_paths,
                                    verbose=False):
    """
    Build a single GreensTensorList for ``stations``, drawing each
    station's Green's functions from whichever CPS database in
    ``db_paths`` corresponds to the polygon (in ``polygons``) that
    contains it.

    Parameters
    ----------
    stations : list of mtuq.Station
    origin : mtuq.Origin
    polygons : dict {region_name: [(lat, lon), ...]}
    db_paths : dict {region_name: path to a CPS database}
        Must have exactly the same keys as ``polygons``.
    verbose : bool
        Print how many stations were assigned to each region.

    Returns
    -------
    mtuq.GreensTensorList, ordered identically to ``stations``.
    """
    if set(polygons.keys()) != set(db_paths.keys()):
        raise ValueError(
            f"polygons and db_paths must define the same region names.\n"
            f"  polygons: {sorted(polygons)}\n"
            f"  db_paths: {sorted(db_paths)}"
        )

    # station index -> region name (raises if unmatched/ambiguous)
    region_of = [assign_region(station, polygons) for station in stations]

    indices_by_region = defaultdict(list)
    for i, region in enumerate(region_of):
        indices_by_region[region].append(i)

    tensors = [None] * len(stations)
    for region, indices in indices_by_region.items():
        if verbose:
            print(f'  {region}: {len(indices)} station(s)')
        db = open_db(db_paths[region], format='CPS', model=region)
        sub_stations = [stations[i] for i in indices]
        sub_tensors = db.get_greens_tensors(sub_stations, origin)
        for idx, tensor in zip(indices, sub_tensors):
            tensors[idx] = tensor

    return GreensTensorList(tensors)
