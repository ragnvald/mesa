# -*- coding: utf-8 -*-
"""Progress and time-left estimate for the report engine.

The report is a known number of maps of a few kinds (atlas tile, line, geocode
group, ...). A map costs its drawing time plus the basemap tiles it has to
download, and the second part is what makes one run take five minutes and the
next one an hour. The tracker counts maps as they finish, measures both parts,
and prices the maps still to come from those measurements - falling back on the
previous run's figures, then on defaults, until this run has its own.

No Qt and no report imports here: the model is plain Python so it can be tested
with a fake clock.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

# Used only until a run (this one or the previous) has measured the real thing.
DEFAULT_DRAW_SECONDS = 6.0
DEFAULT_TILES_PER_MAP = 40.0
DEFAULT_TILE_SECONDS = 1.0
DEFAULT_COMPOSE_SECONDS_PER_MAP = 0.4
# Share of a map's tiles expected to be missing before any tile has been looked up.
PRIOR_MISS_EMPTY_CACHE = 1.0
PRIOR_MISS_WARM_CACHE = 0.1
# A report has been made here before, so its tiles are in the cache unless the selection changed.
PRIOR_MISS_AFTER_EARLIER_RUN = 0.01

KIND_LABELS = {
    "overview": "overview map",
    "line": "line map",
    "atlas": "atlas tile",
    "group": "geocode map",
    "area": "study-area map",
}


def format_duration(seconds: float) -> str:
    """'about 40 s' / 'about 6 min' / 'about 1 h 10 min' - no false precision."""
    s = max(0.0, float(seconds))
    if s < 50:
        return f"about {max(5, int(round(s / 5.0)) * 5)} s"
    minutes = int(round(s / 60.0))
    if minutes < 60:
        return f"about {max(1, minutes)} min"
    hours, rest = divmod(int(round(minutes / 5.0)) * 5, 60)
    return f"about {hours} h {rest} min" if rest else f"about {hours} h"


class _KindStats:
    __slots__ = ("done", "draw_seconds", "tiles", "fetched")

    def __init__(self) -> None:
        self.done = 0
        self.draw_seconds = 0.0   # finished maps and the work leading up to each, tile downloads excluded
        self.tiles = 0            # tiles looked up (cached or downloaded)
        self.fetched = 0          # of those, downloaded


class ReportProgress:
    """Tracks finished maps and estimates the time left.

    plan: kind -> number of maps expected. emit(percent, text) is called on every
    change that is worth showing. history is the dict a previous run saved.
    """

    def __init__(self,
                 plan: dict[str, int],
                 emit: Optional[Callable[[float, str], None]] = None,
                 history: Optional[dict] = None,
                 cache_is_empty: bool = False,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._plan = {str(k): max(0, int(v)) for k, v in (plan or {}).items()}
        self._emit = emit
        self._history = history if isinstance(history, dict) else {}
        if cache_is_empty:
            self._prior_miss = PRIOR_MISS_EMPTY_CACHE
        elif self._history.get("kinds"):
            self._prior_miss = PRIOR_MISS_AFTER_EARLIER_RUN
        else:
            self._prior_miss = PRIOR_MISS_WARM_CACHE
        self._clock = clock
        self._t0 = clock()
        self._stats: dict[str, _KindStats] = {k: _KindStats() for k in self._plan}
        self._fetch_seconds = 0.0
        self._fetched = 0
        self._tiles = 0
        # the map being drawn
        self._depth = 0
        self._unit_kind: Optional[str] = None
        self._unit_t0 = 0.0
        self._unit_gap = 0.0
        self._last_end = self._t0
        self._unit_fetch_seconds = 0.0
        self._unit_tiles = 0
        self._unit_fetched = 0
        self._unit_missing = 0
        # composing the document
        self._composing = False
        self._compose_t0 = 0.0
        self._compose_seconds: Optional[float] = None
        self._last_pct = 0.0
        self._last_emit = -1e9
        self._finished = False

    # ------------------------------------------------------------------ plan
    def set_planned(self, kind: str, count: int) -> None:
        """Correct the plan when the real number of maps of a kind becomes known."""
        self._plan[kind] = max(0, int(count))
        self._stats.setdefault(kind, _KindStats())
        self._publish(force=True)

    def close_kind(self, kind: str) -> None:
        """A stage has ended: whatever was planned for it and not drawn will not come."""
        st = self._stats.setdefault(kind, _KindStats())
        self._plan[kind] = st.done
        self._publish(force=True)

    def planned_total(self) -> int:
        return sum(max(self._plan.get(k, 0), st.done) for k, st in self._stats.items())

    def done_total(self) -> int:
        return sum(st.done for st in self._stats.values())

    # ------------------------------------------------------------------ maps
    def begin_map(self, kind: str) -> None:
        self._depth += 1
        if self._depth > 1:      # a map drawn as part of another map is not a unit of its own
            return
        self._unit_kind = kind
        self._stats.setdefault(kind, _KindStats())
        self._plan.setdefault(kind, 0)
        self._unit_t0 = self._clock()
        # Tables read and charts drawn since the previous map belong to this one's cost.
        self._unit_gap = max(0.0, self._unit_t0 - self._last_end)
        self._unit_fetch_seconds = 0.0
        self._unit_tiles = self._unit_fetched = self._unit_missing = 0
        self._publish(force=True)

    def end_map(self) -> None:
        self._depth = max(0, self._depth - 1)
        if self._depth or self._unit_kind is None:
            return
        st = self._stats[self._unit_kind]
        st.done += 1
        self._last_end = self._clock()
        st.draw_seconds += self._unit_gap + max(0.0, (self._last_end - self._unit_t0) - self._unit_fetch_seconds)
        st.tiles += self._unit_tiles
        st.fetched += self._unit_fetched
        self._unit_kind = None
        self._publish(force=True)

    def tiles_needed(self, total: int, missing: int) -> None:
        """Called before a basemap is assembled: how many tiles, and how many must be downloaded."""
        self._unit_missing += max(0, int(missing))
        self._publish(force=True)

    def tile_done(self, fetched: bool, seconds: float = 0.0) -> None:
        self._tiles += 1
        self._unit_tiles += 1
        if fetched:
            self._fetched += 1
            self._unit_fetched += 1
            self._fetch_seconds += max(0.0, float(seconds))
            self._unit_fetch_seconds += max(0.0, float(seconds))
            self._publish()

    # ------------------------------------------------------------------ document
    def begin_compose(self) -> None:
        for kind in list(self._plan):
            self._plan[kind] = self._stats[kind].done
        self._composing = True
        self._compose_t0 = self._clock()
        self._publish(force=True)

    def end_compose(self) -> None:
        if self._composing:
            self._compose_seconds = self._clock() - self._compose_t0
        self._composing = False

    def finish(self, text: str = "Report completed.") -> None:
        self._finished = True
        self._last_pct = 100.0
        if self._emit:
            self._emit(100.0, text)

    # ------------------------------------------------------------------ estimate
    def _hist(self, kind: str, key: str) -> Optional[float]:
        try:
            value = self._history.get("kinds", {}).get(kind, {}).get(key)
            return float(value) if value is not None else None
        except Exception:
            return None

    def _tile_seconds(self) -> float:
        if self._fetched >= 5:
            return self._fetch_seconds / self._fetched
        try:
            past = float(self._history.get("tile_seconds"))
            if past > 0:
                return past
        except Exception:
            pass
        return (self._fetch_seconds / self._fetched) if self._fetched else DEFAULT_TILE_SECONDS

    def _miss_ratio(self, kind: str) -> float:
        st = self._stats[kind]
        if st.done and st.tiles:
            return st.fetched / st.tiles
        if self._tiles >= 20:
            return self._fetched / self._tiles
        return self._prior_miss

    def seconds_per_map(self, kind: str) -> float:
        st = self._stats[kind]
        prior_draw = self._hist(kind, "draw_seconds")
        prior_tiles = self._hist(kind, "tiles")
        prior_draw = DEFAULT_DRAW_SECONDS if prior_draw is None else prior_draw
        prior_tiles = DEFAULT_TILES_PER_MAP if prior_tiles is None else prior_tiles
        # The prior counts as one map, so a single unusually quick or slow map
        # does not set the price for all the others of its kind.
        draw = (st.draw_seconds + prior_draw) / (st.done + 1)
        tiles = (st.tiles + prior_tiles) / (st.done + 1)
        return draw + tiles * self._miss_ratio(kind) * self._tile_seconds()

    def _compose_estimate(self) -> float:
        if self._compose_seconds is not None:
            return 0.0
        try:
            per_map = float(self._history.get("compose_seconds_per_map"))
        except Exception:
            per_map = DEFAULT_COMPOSE_SECONDS_PER_MAP
        total = 3.0 + per_map * max(1, self.planned_total())
        if self._composing:
            return max(1.0, total - (self._clock() - self._compose_t0))
        return total

    def remaining_seconds(self) -> float:
        left = self._compose_estimate()
        for kind, st in self._stats.items():
            todo = max(0, self._plan.get(kind, 0) - st.done)
            if not todo:
                continue
            per_map = self.seconds_per_map(kind)
            if kind == self._unit_kind:
                # the map in hand: what is left of its tiles, or of its expected time
                spent = self._clock() - self._unit_t0
                tiles_left = max(0, self._unit_missing - self._unit_fetched)
                current = tiles_left * self._tile_seconds() if self._unit_missing else max(0.0, per_map - spent)
                left += current + (todo - 1) * per_map
            else:
                left += todo * per_map
        return left

    def percent(self) -> float:
        if self._finished:
            return 100.0
        elapsed = self._clock() - self._t0
        left = self.remaining_seconds()
        pct = 100.0 * elapsed / (elapsed + left) if (elapsed + left) > 0 else 0.0
        # The bar never runs backwards, and never claims to be done before it is.
        self._last_pct = max(self._last_pct, min(99.0, pct))
        return self._last_pct

    def status_text(self) -> str:
        if self._finished:
            return "Report completed."
        left = format_duration(self.remaining_seconds())
        if self._composing:
            return f"Composing the Word document, {left} left"
        total, done = self.planned_total(), self.done_total()
        if not total:
            return "Preparing …"
        if self._unit_kind is None:
            if done == 0:
                return f"Preparing, {total} maps to draw, {left} left"
            return f"{done} of {total} maps drawn, {left} left"
        label = KIND_LABELS.get(self._unit_kind, self._unit_kind)
        text = f"Map {min(done + 1, total)} of {total} ({label})"
        if self._unit_missing:
            text += f", downloading basemap tiles {min(self._unit_fetched, self._unit_missing)} of {self._unit_missing}"
        return f"{text}, {left} left"

    def _publish(self, force: bool = False) -> None:
        if self._emit is None or self._finished:
            return
        now = self._clock()
        if not force and (now - self._last_emit) < 0.5:
            return
        self._last_emit = now
        try:
            self._emit(self.percent(), self.status_text())
        except Exception:
            pass

    def tick(self) -> None:
        """Refresh the display from a place that has no event of its own."""
        self._publish(force=True)

    # ------------------------------------------------------------------ history
    def to_history(self) -> dict:
        kinds = {}
        for kind, st in self._stats.items():
            if st.done:
                kinds[kind] = {"draw_seconds": round(st.draw_seconds / st.done, 3),
                               "tiles": round(st.tiles / st.done, 1)}
            elif kind in self._history.get("kinds", {}):
                kinds[kind] = self._history["kinds"][kind]
        out: dict = {"kinds": kinds}
        if self._fetched >= 5:
            out["tile_seconds"] = round(self._fetch_seconds / self._fetched, 3)
        elif "tile_seconds" in self._history:
            out["tile_seconds"] = self._history["tile_seconds"]
        maps = self.done_total()
        if self._compose_seconds is not None and maps:
            out["compose_seconds_per_map"] = round(max(0.0, self._compose_seconds - 3.0) / maps, 3)
        elif "compose_seconds_per_map" in self._history:
            out["compose_seconds_per_map"] = self._history["compose_seconds_per_map"]
        return out


def load_history(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_history(path, history: dict) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(history, indent=1), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass
