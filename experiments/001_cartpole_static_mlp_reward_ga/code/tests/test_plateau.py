from awr.plateau import SmoothedPlateauTracker


def test_smoothed_plateau_waits_for_full_window():
    tracker = SmoothedPlateauTracker(window_size=3)

    tracker.update(0.5)
    tracker.update(0.4)

    assert tracker.stagnant_updates == 0


def test_smoothed_plateau_counts_after_window_stops_improving():
    tracker = SmoothedPlateauTracker(window_size=3)

    tracker.update(0.5)
    tracker.update(0.6)
    tracker.update(0.7)
    tracker.update(0.5)
    tracker.update(0.4)

    assert tracker.stagnant_updates == 2
