"""Put the clips of record_clip.py together: the demonstration video.

    python make_demo.py --clips <folder with the clips> [--out results/video/demo.mp4]

Three parts, as in the report: the failure of the baseline, the three
methods, and the monitor with two fallbacks. Each clip is one typical
episode. The numbers in the text come from the result files of the tests in
this repository, not from the clips.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import imageio.v2 as imageio
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
W, H, PANEL, TOP = 1440, 720, 480, 110
SURFACE, INK, MUTED, RULE = (252, 252, 251), (11, 11, 11), (82, 81, 78), (228, 227, 223)
REPO = "github.com/Shengwei-Li/microduck_control"


def font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


F_HEAD, F_SUB, F_LABEL, F_TEXT, F_SMALL = font(30, True), font(22), font(23, True), font(21), font(18)
F_TITLE = font(46, True)


def test_numbers():
    """The numbers of the captions, from the result files (median over the 3 training seeds)."""
    p0 = {}
    with open(R / "d2" / "p0_failure_curve_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            p0[(int(float(r["payload_mass_g"])), r["command"])] = float(r["fall_rate"])
    vx = defaultdict(list)
    with open(R / "d3" / "h2_h3_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            vx[(r["method"], int(r["payload_mass_g"]))].append(float(r["vx_backward"]))
    fall = defaultdict(list)
    with open(R / "d4" / "h5_h6_test.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["table"] == "fallback" and r["quantity"] == "fall_rate" and r["method"] == "P1" and r["payload_mass_g"] == "200":
                fall[r["row"]].append(float(r["value"]))
    return p0, {k: median(v) for k, v in vx.items()}, {k: median(v) for k, v in fall.items()}


def parts():
    p0, vx, fall = test_numbers()

    def falls(x):
        return [f"falls in {x:.2f} of the test episodes"]

    def speed(method):
        return ["mean velocity in the tests:", f"{vx[(method, 0)]:+.2f} m/s without payload, {vx[(method, 100)]:+.2f} m/s with 100 g"]

    return [
        {"head": "1. A policy trained without payload",
         "sub": "Command: walk forward at 0.4 m/s. The payload appears at 5 s.",
         "panels": [("s1_p0_0g", "P0, no payload", falls(p0[(0, "forward_fast")])),
                    ("s1_p0_120g", "P0, payload of 120 g", falls(p0[(120, "forward_fast")]))]},
        {"head": "2. Three policies trained with a random payload, 0 to 100 g",
         "sub": "Command: walk backward at 0.3 m/s. Payload: 100 g. None of them falls. Their velocity changes by different amounts.",
         "panels": [("s2_p1_100g", "P1: random payload only", speed("P1")),
                    ("s2_p2_100g", "P2: knows the payload mass", speed("P2")),
                    ("s2_p3_100g", "P3: history of observations", speed("P3"))]},
        {"head": "3. A payload outside the training range: 200 g",
         "sub": "Policy P1. Command: walk forward at 0.4 m/s. A learned one-step model gives the alarm.",
         "panels": [("s3_p1_none", "No fallback", falls(fall["none"]) + ["(all commands)"]),
                    ("s3_p1_stand", "At the alarm: stand", falls(fall["residual"]) + ["(all commands)"]),
                    ("s3_p1_drop", "At the alarm: drop the payload", falls(fall["residual_drop"]) + ["(all commands)"])]},
    ]


def status(info, t):
    """Text for what happens to this robot at time t (s)."""
    mass, moved = info["payload_mass_g"], info["fallback_s"]
    if info["fall_s"] is not None and t >= info["fall_s"]:
        return f"fell at {info['fall_s']:.1f} s"
    if moved is not None and t >= moved:
        action = "it drops the payload" if info["fallback"] == "drop" else "it gets the command to stand"
        return f"alarm at {moved:.1f} s: {action}"
    if mass > 0 and t >= info["payload_onset_s"]:
        return f"holds {mass:.0f} g"
    return "before the payload" if mass > 0 else "walks"


def centered(draw, x, y, text, fnt, fill):
    draw.text((x - draw.textlength(text, font=fnt) / 2, y), text, font=fnt, fill=fill)


def card(lines):
    """A text page: list of (text, font, colour)."""
    img = Image.new("RGB", (W, H), SURFACE)
    draw = ImageDraw.Draw(img)
    y = (H - sum(f.size * 1.7 for _, f, _ in lines)) / 2
    for text, fnt, fill in lines:
        centered(draw, W / 2, y, text, fnt, fill)
        y += fnt.size * 1.7
    return np.asarray(img)


def part_frames(part, clips):
    """All frames of one part: a still first frame, the clips side by side, a still last frame."""
    loaded = []
    for name, label, stat in part["panels"]:
        info = json.loads((clips / f"{name}.json").read_text())
        frames = [f for f in imageio.get_reader(clips / f"{name}.mp4")]
        loaded.append((info, frames, label, stat))
    fps, t_start = loaded[0][0]["fps"], loaded[0][0]["from_s"]
    n = max(len(frames) for _, frames, _, _ in loaded)
    gap = 0 if len(loaded) == 3 else 60
    x0 = (W - len(loaded) * PANEL - (len(loaded) - 1) * gap) // 2

    def compose(k):
        t = t_start + k / fps
        img = Image.new("RGB", (W, H), SURFACE)
        draw = ImageDraw.Draw(img)
        draw.text((30, 18), part["head"], font=F_HEAD, fill=INK)
        draw.text((30, 64), part["sub"], font=F_SUB, fill=MUTED)
        clock = f"t = {t:4.1f} s"
        draw.text((W - 30 - draw.textlength(clock, font=F_HEAD), 18), clock, font=F_HEAD, fill=INK)
        for j, (info, frames, label, stat) in enumerate(loaded):
            x = x0 + j * (PANEL + gap)
            img.paste(Image.fromarray(frames[min(k, len(frames) - 1)]).resize((PANEL, PANEL)), (x, TOP))
            if j > 0 and gap == 0:
                draw.line([(x, TOP), (x, TOP + PANEL)], fill=SURFACE, width=4)
            cx, y = x + PANEL / 2, TOP + PANEL + 8
            centered(draw, cx, y, label, F_LABEL, INK)
            centered(draw, cx, y + 31, status(info, t), F_TEXT, INK)
            for i, line in enumerate(stat):
                centered(draw, cx, y + 62 + i * 23, line, F_SMALL, MUTED)
        return np.asarray(img)

    first, last = compose(0), compose(n - 1)
    return [first] * round(1.5 * fps) + [compose(k) for k in range(n)] + [last] * round(1.5 * fps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clips", required=True)
    p.add_argument("--out", default=str(R / "video" / "demo.mp4"))
    args = p.parse_args()
    clips = Path(args.clips)
    fps = 50

    title = card([("A small biped picks up an unknown payload", F_TITLE, INK),
                  ("Randomize, adapt, or monitor?", F_HEAD, MUTED),
                  ("", F_SUB, MUTED),
                  ("Simulation of the Microduck robot (0.74 kg). The payload is the ball at the end of a rod with no mass.", F_SUB, INK),
                  ("Each clip shows one typical episode. The numbers come from 768 test episodes for each condition.", F_SUB, INK)])
    end = card([("What the tests show", F_HEAD, INK),
                ("", F_SUB, MUTED),
                ("1. Without payload in training, the velocity changes first. Then the robot falls.", F_SUB, INK),
                ("2. A random payload in training stops the falls inside the training range.", F_SUB, INK),
                ("    A policy that knows or infers the mass keeps its velocity when the payload changes.", F_SUB, INK),
                ("3. A learned model gives the alarm in time. To stand does not prevent the fall. To drop the payload does.", F_SUB, INK),
                ("", F_SUB, MUTED),
                (f"Code, data and report: {REPO}", F_SUB, MUTED)])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=16) as w:
        for frame in [title] * (4 * fps):
            w.append_data(frame)
        for part in parts():
            frames = part_frames(part, clips)
            total += len(frames)
            for frame in frames:
                w.append_data(frame)
        for frame in [end] * (7 * fps):
            w.append_data(frame)
    print(f"wrote {out}: {(total + 11 * fps) / fps:.1f} s at {fps} fps, {W} x {H}")


if __name__ == "__main__":
    main()
