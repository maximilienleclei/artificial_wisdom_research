from pathlib import Path

from awr.reporting import (
    GenerationMetrics,
    read_generation_metrics_csv,
    write_action_prediction_svg,
    write_generation_metrics_csv,
)


def test_metrics_round_trip_csv(tmp_path: Path):
    rows = [
        GenerationMetrics(generation=0, best=0.6, mean=0.5, mutation_std=0.03, annealed=False),
        GenerationMetrics(generation=1, best=0.7, mean=0.65, mutation_std=0.015, annealed=True),
    ]

    output_path = tmp_path / "metrics.csv"
    write_generation_metrics_csv(rows, output_path)

    loaded = read_generation_metrics_csv(output_path)

    assert loaded == [
        GenerationMetrics(generation=0, best=0.6, mean=0.5, mutation_std=0.03, annealed=False),
        GenerationMetrics(generation=1, best=0.7, mean=0.65, mutation_std=0.015, annealed=True),
    ]


def test_action_prediction_svg_contains_title_and_series(tmp_path: Path):
    rows = [
        GenerationMetrics(generation=0, best=0.6, mean=0.5, mutation_std=0.03, annealed=False),
        GenerationMetrics(generation=5, best=0.72, mean=0.68, mutation_std=0.015, annealed=True),
    ]

    output_path = tmp_path / "curve.svg"
    write_action_prediction_svg(rows, output_path, title="Test Plot")

    svg = output_path.read_text(encoding="utf-8")

    assert "Test Plot" in svg
    assert "top: soft-accuracy fitness, bottom: mutation std across 2 generations" in svg
    assert "best fitness" in svg
    assert "mean fitness" in svg
    assert "mutation std" in svg
    assert "anneal g5 -&gt; 0.0150" in svg
