"""The report's progress model: counts maps, prices the remaining ones from what it has measured."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import report_progress as rp  # noqa: E402


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _draw(tracker, clock, kind, draw_s, cached=0, fetched=0, tile_s=1.0):
    tracker.begin_map(kind)
    tracker.tiles_needed(cached + fetched, fetched)
    for _ in range(cached):
        tracker.tile_done(False)
    for _ in range(fetched):
        clock.t += tile_s
        tracker.tile_done(True, tile_s)
    clock.t += draw_s
    tracker.end_map()


def test_remaining_time_follows_measured_maps_and_tile_downloads() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"atlas": 10}, cache_is_empty=True, clock=clock)

    # two atlas tiles, each 3 s of drawing and 50 downloaded tiles at 1 s
    _draw(tracker, clock, "atlas", 3.0, fetched=50)
    _draw(tracker, clock, "atlas", 3.0, fetched=50)

    # measured: 3 s + 50 tiles x 1 s; the default prior (6 s, 40 tiles) counts as a third map
    per_map = (3.0 + 3.0 + rp.DEFAULT_DRAW_SECONDS) / 3 + (50 + 50 + rp.DEFAULT_TILES_PER_MAP) / 3 * 1.0
    assert abs(tracker.seconds_per_map("atlas") - per_map) < 0.01
    compose = 3.0 + rp.DEFAULT_COMPOSE_SECONDS_PER_MAP * 10
    assert abs(tracker.remaining_seconds() - (8 * per_map + compose)) < 0.01
    assert 19.0 < tracker.percent() < 21.5


def test_a_warm_cache_makes_the_same_maps_cheap() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"atlas": 10}, clock=clock)
    _draw(tracker, clock, "atlas", 3.0, cached=50)
    assert abs(tracker.seconds_per_map("atlas") - (3.0 + rp.DEFAULT_DRAW_SECONDS) / 2) < 0.01


def test_work_between_maps_is_part_of_the_next_maps_cost() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"area": 3}, history={"kinds": {"area": {"draw_seconds": 10.0, "tiles": 0}}}, clock=clock)
    clock.t += 8.0                       # tables read before the first map
    _draw(tracker, clock, "area", 2.0)
    assert abs(tracker.seconds_per_map("area") - (8.0 + 2.0 + 10.0) / 2) < 0.01


def test_one_quick_map_does_not_set_the_price_for_the_rest() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"atlas": 20}, history={"kinds": {"atlas": {"draw_seconds": 6.0, "tiles": 0}}}, clock=clock)
    _draw(tracker, clock, "atlas", 0.2)  # an empty tile returns at once
    assert tracker.seconds_per_map("atlas") > 3.0


def test_a_kind_not_yet_drawn_is_priced_from_the_previous_run() -> None:
    history = {"kinds": {"line": {"draw_seconds": 6.0, "tiles": 30}}, "tile_seconds": 0.5,
               "compose_seconds_per_map": 0.2}
    cold = rp.ReportProgress({"line": 4}, history=history, cache_is_empty=True, clock=Clock())
    warm = rp.ReportProgress({"line": 4}, history=history, cache_is_empty=False, clock=Clock())

    assert abs(cold.seconds_per_map("line") - (6.0 + 30 * 1.0 * 0.5)) < 0.01
    # an earlier run in the same project has already filled the cache with these tiles
    assert abs(warm.seconds_per_map("line") - (6.0 + 30 * rp.PRIOR_MISS_AFTER_EARLIER_RUN * 0.5)) < 0.01
    unknown = rp.ReportProgress({"line": 4}, cache_is_empty=False, clock=Clock())
    assert abs(unknown.seconds_per_map("line")
               - (rp.DEFAULT_DRAW_SECONDS + rp.DEFAULT_TILES_PER_MAP * rp.PRIOR_MISS_WARM_CACHE * rp.DEFAULT_TILE_SECONDS)) < 0.01
    # and the miss ratio measured on other maps in this run replaces the prior
    clock = Clock()
    mixed = rp.ReportProgress({"line": 4, "atlas": 1}, history=history, clock=clock)
    _draw(mixed, clock, "atlas", 1.0, cached=30, fetched=10, tile_s=0.5)
    assert abs(mixed.seconds_per_map("line") - (6.0 + 30 * 0.25 * 0.5)) < 0.01


def test_the_bar_never_runs_backwards_and_stops_short_of_done() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"atlas": 4}, clock=clock)
    _draw(tracker, clock, "atlas", 2.0, cached=10)
    first = tracker.percent()
    # the next map is far slower than the first suggested: the estimate grows, the bar holds
    _draw(tracker, clock, "atlas", 2.0, fetched=200)
    assert tracker.percent() >= first
    _draw(tracker, clock, "atlas", 2.0, cached=10)
    _draw(tracker, clock, "atlas", 2.0, cached=10)
    assert tracker.percent() <= 99.0
    tracker.finish()
    assert tracker.percent() == 100.0


def test_a_stage_that_drew_fewer_maps_than_planned_is_closed() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"line": 6, "atlas": 2}, clock=clock)
    _draw(tracker, clock, "line", 2.0)
    tracker.close_kind("line")
    assert tracker.planned_total() == 3 and tracker.done_total() == 1
    # more maps than planned never show as "map 4 of 3"
    for _ in range(3):
        _draw(tracker, clock, "atlas", 1.0)
    assert tracker.planned_total() == tracker.done_total() == 4


def test_status_text_names_the_map_and_the_tiles_being_downloaded() -> None:
    clock = Clock()
    seen: list[tuple[float, str]] = []
    tracker = rp.ReportProgress({"atlas": 20, "line": 6}, emit=lambda p, t: seen.append((p, t)), clock=clock)
    _draw(tracker, clock, "line", 2.0)
    tracker.begin_map("atlas")
    tracker.tiles_needed(212, 212)
    for _ in range(57):
        clock.t += 1.0
        tracker.tile_done(True, 1.0)
    text = tracker.status_text()
    assert text.startswith("Map 2 of 26 (atlas tile), downloading basemap tiles 57 of 212, about ")
    assert text.endswith(" left")
    assert seen and all(0.0 <= p <= 99.0 for p, _t in seen)

    tracker.end_map()
    tracker.begin_compose()
    assert tracker.status_text().startswith("Composing the Word document")


def test_a_map_drawn_inside_another_is_not_counted_twice() -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"group": 2}, clock=clock)
    tracker.begin_map("group")
    tracker.begin_map("group")
    tracker.end_map()
    tracker.end_map()
    assert tracker.done_total() == 1


def test_durations_are_rounded_to_what_is_known() -> None:
    assert rp.format_duration(3) == "about 5 s"
    assert rp.format_duration(42) == "about 40 s"
    assert rp.format_duration(95) == "about 2 min"
    assert rp.format_duration(47 * 60) == "about 47 min"
    assert rp.format_duration(78 * 60) == "about 1 h 20 min"


def test_history_round_trip(tmp_path: Path) -> None:
    clock = Clock()
    tracker = rp.ReportProgress({"atlas": 2}, clock=clock)
    _draw(tracker, clock, "atlas", 3.0, fetched=10, tile_s=0.5)
    _draw(tracker, clock, "atlas", 5.0, fetched=10, tile_s=0.5)
    tracker.begin_compose()
    clock.t += 7.0
    tracker.end_compose()

    path = tmp_path / "cache" / "report_timing.json"
    rp.save_history(path, tracker.to_history())
    history = rp.load_history(path)
    assert history["kinds"]["atlas"] == {"draw_seconds": 4.0, "tiles": 10.0}
    assert history["tile_seconds"] == 0.5
    assert history["compose_seconds_per_map"] == 2.0
    assert rp.load_history(tmp_path / "missing.json") == {}
