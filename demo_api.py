#!/usr/bin/env python3
"""
Demo: fire 10 messy radiology exam descriptions at the *running* RadMatcher
app over HTTP (POST /api/match) and print what comes back -- the top match plus
any AI suggestion the LLM fallback produced.

Start the app first (python app.py), then in another terminal:
    python demo_api.py                 # all 10, against localhost:5000
    python demo_api.py -n 3            # first 3
    python demo_api.py --url http://localhost:5050   # custom host/port
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

# Messy, real-world order strings. We send only the raw text -- the app infers
# modality, scores, and decides on its own whether to invoke the LLM fallback,
# exactly like the web UI does.
DEMO_QUERIES = [
    "ct abd pelv w/wo dye",
    "cr chest 2v",
    "mri l-spine wo",
    "us kidney",
    "xr left knee 3 views",
    "ct head wo contrast",
    "mammo screening bilat",
    "mr brain w and wo",
    "us abdomen complete",
    "mr brain wwo",
    "ct abdomen pelvis with contrast",
    "ct abd/pelvis w contrast",
    "ct a/p wo",
    "ct a/p w",
    "ct chest abd pelvis w dye",
    "ct chest abdomen pelvis w/wo",
    "cta chest pe protocol",
    "ct angiogram chest for pe",
    "cta head neck w contrast",
    "ct cervical spine wo",
    "ct c-spine without contrast",
    "ct thoracic spine wo",
    "ct lumbar spine wo",
    "ct sinus wo contrast",
    "ct maxillofacial wo",
    "ct facial bones without",
    "ct temporal bones w/o",
    "ct soft tissue neck w contrast",
    "ct chest wo contrast",
    "ct chest with contrast",
    "ct renal stone protocol",
    "ct abdomen wo contrast",
    "ct abdomen with contrast",
    "ct pelvis with contrast",
    "ct pelvis wo contrast",
    "mri brain without contrast",
    "mri brain with contrast",
    "mri brain w/wo contrast",
    "mri pituitary wwo",
    "mri orbits w and wo",
    "mri c-spine wwo",
    "mri thoracic spine without",
    "mri t-spine w and wo",
    "mri lumbar spine w/wo",
    "mr l spine no dye",
    "mri pelvis wwo",
    "mri abdomen w contrast",
    "mri abdomen mrcp wo",
    "mri prostate wwo",
    "mri shoulder right wo",
    "mri shoulder left w/o",
    "mri elbow right wo",
    "mri wrist left wo",
    "mri hand right wo",
    "mri hip left wo",
    "mri knee right wo",
    "mri left knee without contrast",
    "mri ankle right wo",
    "mri foot left wo",
    "mra head wo contrast",
    "mra neck w contrast",
    "mra abdomen wwo",
    "us abdomen complete",
    "us abdomen limited ruq",
    "us gallbladder",
    "us liver doppler",
    "us renal complete",
    "us kidney bladder",
    "us pelvic transvaginal",
    "us pelvis complete",
    "us ob less than 14 weeks",
    "us ob 1st trimester",
    "us ob greater than 14 weeks",
    "us thyroid",
    "us scrotum testicular",
    "us breast left limited",
    "us breast right complete",
    "us venous doppler left leg",
    "venous duplex lower ext bilat",
    "us carotid bilateral",
    "arterial duplex right leg",
    "xr chest portable 1 view",
    "chest xray pa lateral",
    "cxr 2 views",
    "xr abdomen kub 1 view",
    "abdomen xray 2 views",
    "xr cervical spine 4 views",
    "xr lumbar spine 2-3 views",
    "xray rt ankle 3 views",
    "xr hand left min 3 views",
    "xr shoulder right 2 views",
    "xr hip left 2-3 views",
    "xr pelvis 1-2 views",
    "xr foot right minimum 3 views",
    "xr wrist left 3 views",
    "xr elbow right 2 views",
    "screening mammogram bilateral tomo",
    "diagnostic mammo left w tomo",
    "mammo diag bilat",
    "dexa bone density axial",
]


def post_match(base_url: str, query: str, timeout: float) -> dict:
    payload = json.dumps({"query": query}).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/match",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    default_port = os.environ.get("RADMATCHER_PORT", "5000")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--num", type=int, default=len(DEMO_QUERIES),
                    help="how many of the demo queries to send")
    ap.add_argument("--url", default=f"http://localhost:{default_port}",
                    help="base URL of the running app")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="per-request timeout in seconds (LLM fallback can be slow)")
    args = ap.parse_args()

    queries = DEMO_QUERIES[: max(0, args.num)]
    print(f"RadMatcher API demo -> {args.url}/api/match  ({len(queries)} queries)\n")

    ok = 0
    timings: list[float] = []
    t_start = time.time()
    for i, query in enumerate(queries, 1):
        t0 = time.time()
        try:
            result = post_match(args.url, query, args.timeout)
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            print(f"[{i:>2}/{len(queries)}] {query!r}")
            print(f"        ✗ could not reach the app ({reason}). Is it running? "
                  f"Start it with: python app.py\n")
            # No point hammering a dead server.
            if isinstance(reason, ConnectionRefusedError) or "refused" in str(reason).lower():
                break
            continue

        dt = time.time() - t0
        timings.append(dt)
        matches = result.get("matches") or []
        top = matches[0] if matches else None
        ai = result.get("ai_suggestion")

        print(f"[{i:>2}/{len(queries)}] {query!r}  ({result.get('modality', '?')})  ⏱ {dt:.3f}s")
        if top:
            ok += 1
            cc = top.get("calibrated_confidence")
            cc_s = f"  conf {cc:.2f}" if isinstance(cc, (int, float)) else ""
            print(f"        → {top.get('code', '?')}  {top.get('description', '')}")
            print(f"          score {top.get('score', 0)}  [{top.get('confidence', '?')}]{cc_s}")
        else:
            print(f"        (no match above threshold)")
        if ai:
            print(f"        🤖 AI suggestion: {ai.get('suggested_code', '?')}  "
                  f"{ai.get('suggested_description', '')}  (conf {ai.get('confidence', '?')})")
        print()

    total = time.time() - t_start
    print("─" * 60)
    print(f"Matched : {ok}/{len(queries)} returned a top match")
    if timings:
        print(f"Per-query: min {min(timings):.3f}s | "
              f"avg {sum(timings) / len(timings):.3f}s | "
              f"max {max(timings):.3f}s")
    print(f"Total    : {total:.3f}s  ({len(timings)} requests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
