from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GenerationMetrics:
    generation: int
    best: float
    mean: float
    mutation_std: float
    annealed: bool = False


def write_generation_metrics_csv(
    rows: list[GenerationMetrics],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "best", "mean", "mutation_std", "annealed"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "generation": row.generation,
                    "best": f"{row.best:.6f}",
                    "mean": f"{row.mean:.6f}",
                    "mutation_std": f"{row.mutation_std:.6f}",
                    "annealed": str(row.annealed).lower(),
                }
            )


def read_generation_metrics_csv(path: str | Path) -> list[GenerationMetrics]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            GenerationMetrics(
                generation=int(row["generation"]),
                best=float(row["best"]),
                mean=float(row["mean"]),
                mutation_std=float(row["mutation_std"]),
                annealed=row.get("annealed", "false").lower() == "true",
            )
            for row in reader
        ]


def write_action_prediction_svg(
    rows: list[GenerationMetrics],
    output_path: str | Path,
    title: str,
) -> None:
    if not rows:
        raise ValueError("Cannot plot empty metrics.")

    width, height = 1200, 900
    left, right, top, bottom = 100, 40, 50, 80
    panel_gap = 70
    fitness_plot_height = 430
    sigma_plot_height = 180
    plot_width = width - left - right
    x_max = max(row.generation for row in rows)
    fitness_top = top + 40
    fitness_bottom = fitness_top + fitness_plot_height
    sigma_top = fitness_bottom + panel_gap
    sigma_bottom = sigma_top + sigma_plot_height

    value_min = min(min(row.best, row.mean) for row in rows)
    value_max = max(max(row.best, row.mean) for row in rows)
    y_ticks = _build_action_prediction_ticks(value_min, value_max)
    transformed_ticks = [_focus_transform(tick) for tick in y_ticks]
    y_min = min(transformed_ticks)
    y_max = max(transformed_ticks)
    if y_max - y_min < 1e-9:
        y_max = y_min + 1.0

    def scale_x(generation: int) -> float:
        if x_max == 0:
            return left + plot_width / 2
        return left + generation / x_max * plot_width

    def scale_fitness_y(value: float) -> float:
        transformed = _focus_transform(value)
        return fitness_top + (y_max - transformed) / (y_max - y_min) * fitness_plot_height

    sigma_values = [row.mutation_std for row in rows]
    sigma_ticks = _build_sigma_ticks(sigma_values)
    sigma_min = min(sigma_ticks)
    sigma_max = max(sigma_ticks)

    def scale_sigma_y(value: float) -> float:
        return sigma_top + (sigma_max - value) / max(1e-12, sigma_max - sigma_min) * sigma_plot_height

    def fitness_polyline_points(values: list[float]) -> str:
        return " ".join(
            f"{scale_x(row.generation):.2f},{scale_fitness_y(value):.2f}"
            for row, value in zip(rows, values, strict=True)
        )

    def sigma_polyline_points(values: list[float]) -> str:
        return " ".join(
            f"{scale_x(row.generation):.2f},{scale_sigma_y(value):.2f}"
            for row, value in zip(rows, values, strict=True)
        )

    x_ticks = _build_x_ticks(x_max)
    anneal_rows = _find_anneal_rows(rows)

    mean_values = [row.mean for row in rows]
    best_values = [row.best for row in rows]

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
        f'<text x="100" y="30" font-family="monospace" font-size="24" fill="#222">{_escape_xml(title)}</text>',
        f'<text x="100" y="52" font-family="monospace" font-size="14" fill="#555">top: soft-accuracy fitness, bottom: mutation std across {len(rows)} generations</text>',
        '<text x="100" y="72" font-family="monospace" font-size="12" fill="#777">fitness panel is log-scaled by remaining gap to 1.0, so the top end is expanded</text>',
        f'<text x="{left}" y="{fitness_top - 12}" font-family="monospace" font-size="13" fill="#444">fitness</text>',
    ]

    for y_tick in y_ticks:
        y = scale_fitness_y(y_tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd7c8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="88" y="{y + 5:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#555">{_format_tick(y_tick)}</text>'
        )

    for x_tick in x_ticks:
        x = scale_x(x_tick)
        elements.append(
            f'<line x1="{x:.2f}" y1="{fitness_top}" x2="{x:.2f}" y2="{fitness_bottom}" stroke="#eee7d8" stroke-width="1"/>'
        )
        elements.append(
            f'<line x1="{x:.2f}" y1="{sigma_top}" x2="{x:.2f}" y2="{sigma_bottom}" stroke="#eee7d8" stroke-width="1"/>'
        )

    elements.append(
        f'<rect x="{left}" y="{fitness_top}" width="{plot_width}" height="{fitness_plot_height}" fill="none" stroke="#222" stroke-width="1.5"/>'
    )
    for index, row in enumerate(anneal_rows):
        label_y = fitness_top + 18 + index * 18
        elements.append(
            f'<line x1="{scale_x(row.generation):.2f}" y1="{fitness_top}" x2="{scale_x(row.generation):.2f}" y2="{fitness_bottom}" stroke="#b85c38" stroke-width="2" stroke-dasharray="8 6"/>'
        )
        elements.append(
            f'<text x="{scale_x(row.generation) + 6:.2f}" y="{label_y:.2f}" font-family="monospace" font-size="12" fill="#b85c38">anneal g{row.generation} -&gt; {row.mutation_std:.4f}</text>'
        )
    elements.append(
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="4" points="{fitness_polyline_points(mean_values)}"/>'
    )
    elements.append(
        f'<polyline fill="none" stroke="#2a9d5b" stroke-width="4" points="{fitness_polyline_points(best_values)}"/>'
    )

    for row in rows:
        elements.append(
            f'<circle cx="{scale_x(row.generation):.2f}" cy="{scale_fitness_y(row.mean):.2f}" r="2.2" fill="#1f77b4"/>'
        )
    for row in rows:
        elements.append(
            f'<circle cx="{scale_x(row.generation):.2f}" cy="{scale_fitness_y(row.best):.2f}" r="2.2" fill="#2a9d5b"/>'
        )

    elements.append(
        f'<text x="{left}" y="{sigma_top - 12}" font-family="monospace" font-size="13" fill="#444">mutation std</text>'
    )
    for sigma_tick in sigma_ticks:
        y = scale_sigma_y(sigma_tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd7c8" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="88" y="{y + 5:.2f}" text-anchor="end" font-family="monospace" font-size="12" fill="#555">{sigma_tick:.4f}</text>'
        )
    elements.append(
        f'<rect x="{left}" y="{sigma_top}" width="{plot_width}" height="{sigma_plot_height}" fill="none" stroke="#222" stroke-width="1.5"/>'
    )
    elements.append(
        f'<polyline fill="none" stroke="#b85c38" stroke-width="4" points="{sigma_polyline_points(sigma_values)}"/>'
    )
    for row in rows:
        elements.append(
            f'<circle cx="{scale_x(row.generation):.2f}" cy="{scale_sigma_y(row.mutation_std):.2f}" r="2.2" fill="#b85c38"/>'
        )

    for x_tick in x_ticks:
        x = scale_x(x_tick)
        elements.append(
            f'<text x="{x:.2f}" y="{sigma_bottom + 20}" text-anchor="middle" font-family="monospace" font-size="12" fill="#555">{x_tick}</text>'
        )

    elements.extend(
        [
            f'<rect x="100" y="{sigma_bottom + 18}" width="290" height="78" rx="10" fill="#fffdf8" stroke="#d8d1c2"/>',
            f'<line x1="118" y1="{sigma_bottom + 46}" x2="152" y2="{sigma_bottom + 46}" stroke="#2a9d5b" stroke-width="4"/>',
            f'<text x="162" y="{sigma_bottom + 50}" font-family="monospace" font-size="13" fill="#222">best fitness</text>',
            f'<line x1="118" y1="{sigma_bottom + 70}" x2="152" y2="{sigma_bottom + 70}" stroke="#1f77b4" stroke-width="4"/>',
            f'<text x="162" y="{sigma_bottom + 74}" font-family="monospace" font-size="13" fill="#222">mean fitness</text>',
            f'<line x1="118" y1="{sigma_bottom + 94}" x2="152" y2="{sigma_bottom + 94}" stroke="#b85c38" stroke-width="4"/>',
            f'<text x="162" y="{sigma_bottom + 98}" font-family="monospace" font-size="13" fill="#222">mutation std</text>',
            f'<text x="600" y="{height - 22}" text-anchor="middle" font-family="monospace" font-size="13" fill="#444">generation</text>',
            f'<text x="26" y="{(fitness_top + fitness_bottom) / 2:.2f}" transform="rotate(-90 26 {(fitness_top + fitness_bottom) / 2:.2f})" text-anchor="middle" font-family="monospace" font-size="13" fill="#444">soft-accuracy fitness</text>',
            f'<text x="26" y="{(sigma_top + sigma_bottom) / 2:.2f}" transform="rotate(-90 26 {(sigma_top + sigma_bottom) / 2:.2f})" text-anchor="middle" font-family="monospace" font-size="13" fill="#444">mutation std</text>',
            "</svg>",
        ]
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _build_x_ticks(x_max: int) -> list[int]:
    if x_max <= 0:
        return [0]
    if x_max <= 10:
        return list(range(0, x_max + 1))

    step = max(1, x_max // 10)
    ticks = list(range(0, x_max + 1, step))
    if ticks[-1] != x_max:
        ticks.append(x_max)
    return ticks


def _build_action_prediction_ticks(value_min: float, value_max: float) -> list[float]:
    candidates = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
        0.995,
        0.999,
        0.9999,
    ]
    lower_bound = max(0.0, value_min - 0.02)
    upper_bound = min(0.9999, value_max + 0.01)
    ticks = [tick for tick in candidates if lower_bound <= tick <= upper_bound]
    if not ticks:
        ticks = [
            max(0.0, _round_down(value_min, 0.05)),
            min(0.9999, _round_up(value_max, 0.05)),
        ]
    if ticks[0] > value_min:
        ticks.insert(0, max(0.0, _round_down(value_min, 0.05)))
    if ticks[-1] < value_max:
        ticks.append(min(0.9999, _round_up(value_max, 0.05)))
    deduped: list[float] = []
    for tick in ticks:
        if not deduped or abs(deduped[-1] - tick) > 1e-9:
            deduped.append(tick)
    return deduped


def _find_anneal_rows(rows: list[GenerationMetrics]) -> list[GenerationMetrics]:
    return [row for row in rows if row.annealed]


def _build_sigma_ticks(values: list[float]) -> list[float]:
    value_min = min(values)
    value_max = max(values)
    if abs(value_max - value_min) < 1e-12:
        return [round(value_min, 4), round(value_min + max(0.0001, value_min * 0.1 + 0.0001), 4)]
    step = _round_up((value_max - value_min) / 4, 0.0005)
    ticks: list[float] = []
    current = _round_down(value_min, step)
    upper = _round_up(value_max, step)
    while current <= upper + 1e-12:
        ticks.append(round(current, 4))
        current += step
    if ticks[-1] < round(value_max, 4):
        ticks.append(round(value_max, 4))
    deduped: list[float] = []
    for tick in ticks:
        if not deduped or abs(deduped[-1] - tick) > 1e-9:
            deduped.append(tick)
    return deduped


def _round_down(value: float, step: float) -> float:
    return step * int(value / step)


def _round_up(value: float, step: float) -> float:
    scaled = value / step
    rounded = int(scaled)
    if scaled > rounded:
        rounded += 1
    return rounded * step


def _focus_transform(value: float) -> float:
    clipped = min(max(value, 0.0), 0.9999)
    return -math.log10(max(1e-12, 1.0 - clipped))


def _format_tick(value: float) -> str:
    if value >= 0.99:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.2f}"


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
