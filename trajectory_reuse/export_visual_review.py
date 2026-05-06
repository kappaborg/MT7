from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def _relative_image_path(output_path: Path, frames_root: Path, file_name: str) -> str:
    image_path = frames_root / file_name
    return os.path.relpath(
        str(image_path.resolve()),
        str(Path(output_path).resolve().parent),
    )


def _select_samples(samples: List[Dict], limit: int) -> List[Dict]:
    worst = sorted(samples, key=lambda item: item.get("fde", 0.0), reverse=True)[:limit]
    best = sorted(samples, key=lambda item: item.get("fde", 0.0))[:limit]
    seen = set()
    selected: List[Dict] = []
    for sample in worst + best:
        key = (sample.get("sequence_key"), sample.get("track_id"), sample.get("history_end_frame"))
        if key in seen:
            continue
        seen.add(key)
        selected.append(sample)
    return selected


def export_visual_review(
    evaluation_path: Path,
    output_path: Path,
    sample_limit: int = 12,
) -> Dict[str, int]:
    with evaluation_path.open() as file:
        data = json.load(file)

    summary = data.get("summary", {})
    frames_root = Path(summary.get("frames_root", ""))
    samples = _select_samples(data.get("samples", []), sample_limit)

    review_samples: List[Dict] = []
    for sample in samples:
        review_samples.append(
            {
                "sequence_key": sample["sequence_key"],
                "track_id": sample["track_id"],
                "history_end_frame": sample["history_end_frame"],
                "ade": sample["ade"],
                "fde": sample["fde"],
                "confidence": sample["confidence"],
                "intention": sample["intention"],
                "history": sample["history"],
                "target_future": sample["target_future"],
                "predicted_future": sample["predicted_future"],
                "image_src": _relative_image_path(output_path, frames_root, sample["current_file_name"]),
            }
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone Trajectory Review</title>
  <style>
    :root {{
      --bg: #08111b;
      --panel: rgba(15, 25, 38, 0.92);
      --text: #eef4ff;
      --muted: #9bb0cb;
      --history: #4cc9f0;
      --target: #80ed99;
      --predicted: #ff6b6b;
      --border: rgba(255,255,255,0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(76,201,240,0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(255,107,107,0.16), transparent 28%),
        linear-gradient(180deg, #08111b 0%, #03070c 100%);
      color: var(--text);
      padding: 24px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    .summary {{ color: var(--muted); margin-bottom: 20px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(0,0,0,0.32);
    }}
    .meta {{
      padding: 14px 16px 10px;
      font-size: 14px;
      line-height: 1.5;
    }}
    .meta strong {{ color: var(--text); }}
    .stage {{
      position: relative;
      background: #000;
    }}
    .stage img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .stage canvas {{
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
    }}
    .legend {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      padding: 0 16px 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .legend span::before {{
      content: '';
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 999px;
      vertical-align: middle;
    }}
    .history::before {{ background: var(--history); }}
    .target::before {{ background: var(--target); }}
    .predicted::before {{ background: var(--predicted); }}
  </style>
</head>
<body>
  <h1>Drone Trajectory Review</h1>
  <div class="summary">
    Samples shown: {len(review_samples)}. Each card overlays history, target future, and predicted future on the current frame.
  </div>
  <div id="grid" class="grid"></div>
  <script>
    const samples = {json.dumps(review_samples)};

    function drawPath(ctx, points, color) {{
      if (!points || !points.length) return;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(points[0][0], points[0][1]);
      for (let i = 1; i < points.length; i++) {{
        ctx.lineTo(points[i][0], points[i][1]);
      }}
      ctx.stroke();
      for (const [x, y] of points) {{
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}

    function makeCard(sample) {{
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <div class="meta">
          <div><strong>${{sample.sequence_key}}</strong></div>
          <div>track ${{sample.track_id}} | frame ${{sample.history_end_frame}}</div>
          <div>ADE ${{sample.ade.toFixed(2)}} | FDE ${{sample.fde.toFixed(2)}} | conf ${{sample.confidence.toFixed(2)}} | ${{sample.intention}}</div>
        </div>
        <div class="stage">
          <img src="${{sample.image_src}}" alt="frame">
          <canvas></canvas>
        </div>
        <div class="legend">
          <span class="history">history</span>
          <span class="target">target future</span>
          <span class="predicted">predicted future</span>
        </div>
      `;

      const img = card.querySelector('img');
      const canvas = card.querySelector('canvas');
      img.addEventListener('load', () => {{
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        drawPath(ctx, sample.history, getComputedStyle(document.documentElement).getPropertyValue('--history').trim());
        drawPath(ctx, sample.target_future, getComputedStyle(document.documentElement).getPropertyValue('--target').trim());
        drawPath(ctx, [sample.history[sample.history.length - 1], ...sample.predicted_future], getComputedStyle(document.documentElement).getPropertyValue('--predicted').trim());
      }});
      return card;
    }}

    const grid = document.getElementById('grid');
    samples.forEach(sample => grid.appendChild(makeCard(sample)));
  </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return {"sample_count": len(review_samples)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an HTML review page for predictor outputs and linked tracks."
    )
    parser.add_argument(
        "--evaluation",
        default="/Users/kappasutra/MT7/annotations/predictor_evaluation.json",
        help="Path to the predictor evaluation JSON file.",
    )
    parser.add_argument(
        "--output",
        default="/Users/kappasutra/MT7/annotations/predictor_review.html",
        help="Path to write the HTML review page.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=12,
        help="Number of best and worst samples to include.",
    )
    args = parser.parse_args()

    summary = export_visual_review(
        evaluation_path=Path(args.evaluation),
        output_path=Path(args.output),
        sample_limit=args.sample_limit,
    )
    print("Visual review export summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
