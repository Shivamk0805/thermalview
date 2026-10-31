"""Example monitoring agent: sends readings to ThermalView's telemetry API.

This is the piece a data centre would replace with its own collector. It
uses only the Python standard library, so it can run on any host.

  # 1. In the console (Inventory), add a server with Telemetry = External,
  #    e.g. MAA1-EDGE-01. It shows "no telemetry" until readings arrive.
  # 2. Run the agent:
  python scripts/telemetry_agent.py --server MAA1-EDGE-01                 # synthetic readings
  python scripts/telemetry_agent.py --server MAA1-EDGE-01 --stop-after 10 # then go silent (stale alert)
  python scripts/telemetry_agent.py --server MAA1-EDGE-01 \\
      --redfish-thermal thermal.json --redfish-power power.json --load 45 # map saved Redfish responses

Modes
  synthetic (default)   a plausible CPU temperature that follows a slowly varying load
  --redfish-*           read Redfish Thermal / Power JSON documents (saved files or URLs)
                        and map them to ThermalView fields; see docs/API.md for the mapping.
                        Redfish has no CPU utilisation, so --load (or an OS metric) supplies it.

The console's clock runs faster than real time; one reading per tick
(--interval) keeps the server fresh at speed 1 min/s.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request


def redfish_to_reading(thermal: dict, power: dict | None, load_pct: float) -> dict:
    """Map DMTF Redfish Thermal/Power documents to one ThermalView reading."""
    temps = thermal.get("Temperatures", [])
    cpu = [t.get("ReadingCelsius") for t in temps
           if str(t.get("PhysicalContext", "")).upper() == "CPU" and t.get("ReadingCelsius") is not None]
    if not cpu:   # some BMCs only name the sensor
        cpu = [t.get("ReadingCelsius") for t in temps
               if "cpu" in str(t.get("Name", "")).lower() and t.get("ReadingCelsius") is not None]
    if not cpu:
        raise ValueError("no CPU temperature sensor in the Thermal document")
    inlet = [t.get("ReadingCelsius") for t in temps
             if str(t.get("PhysicalContext", "")).lower() == "intake" and t.get("ReadingCelsius") is not None]
    fans = [f.get("Reading") for f in thermal.get("Fans", [])
            if str(f.get("ReadingUnits", "RPM")).upper() == "RPM" and f.get("Reading") is not None]
    reading = {"cpu_temp_c": float(max(cpu)), "load_pct": float(load_pct)}
    if inlet:
        reading["inlet_c"] = float(inlet[0])
    if fans:
        reading["fan_rpm"] = float(sum(fans) / len(fans))
    if power:
        ctl = power.get("PowerControl") or []
        if ctl and ctl[0].get("PowerConsumedWatts") is not None:
            reading["power_w"] = float(ctl[0]["PowerConsumedWatts"])
    return reading


def _load_json(src: str) -> dict:
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src, timeout=10) as r:   # add auth for a real BMC
            return json.load(r)
    with open(src, encoding="utf-8") as f:
        return json.load(f)


class Synthetic:
    """A first-order heat response to a wandering load, with sensor noise."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.temp, self.t = 45.0, 0

    def reading(self) -> dict:
        self.t += 1
        load = max(0.0, min(100.0, 45 + 30 * math.sin(self.t / 25) + self.rng.gauss(0, 6)))
        target = 30 + 0.42 * load
        self.temp += 0.15 * (target - self.temp) + self.rng.gauss(0, 0.3)
        return {"cpu_temp_c": round(self.temp, 2), "load_pct": round(load, 1),
                "power_w": round(320 + 11.5 * load), "inlet_c": round(22 + self.rng.gauss(0, 0.2), 2)}


def post(url: str, token: str, readings: list[dict]) -> dict:
    req = urllib.request.Request(url, data=json.dumps({"readings": readings}).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--token", default="tv-demo-agent-token")
    ap.add_argument("--server", required=True, help="server name (or id) registered as external")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between readings")
    ap.add_argument("--stop-after", type=int, default=0, help="stop after N readings (0 = run until Ctrl+C)")
    ap.add_argument("--redfish-thermal", help="Redfish Thermal JSON (file or URL)")
    ap.add_argument("--redfish-power", help="Redfish Power JSON (file or URL)")
    ap.add_argument("--load", type=float, default=50.0, help="CPU load %% to report in Redfish mode")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    endpoint = args.url.rstrip("/") + "/api/v1/telemetry"
    synth = Synthetic(args.seed)
    n = 0
    print(f"sending to {endpoint} as {args.server} (Ctrl+C to stop)")
    try:
        while not args.stop_after or n < args.stop_after:
            if args.redfish_thermal:
                r = redfish_to_reading(_load_json(args.redfish_thermal),
                                       _load_json(args.redfish_power) if args.redfish_power else None, args.load)
            else:
                r = synth.reading()
            r["server"] = args.server
            res = post(endpoint, args.token, [r])
            n += 1
            if "error" in res or res.get("rejected"):
                print("rejected:", res.get("error") or res["rejected"])
                if "error" in res and "401" in res["error"]:
                    return 1
            else:
                print(f"{n:5d}  {r['cpu_temp_c']:6.2f} °C  load {r['load_pct']:5.1f}%  sim {res['sim_time'][11:16]}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    except urllib.error.URLError as e:
        print(f"cannot reach {args.url}: {e.reason}. Is `python -m thermalview serve` running?")
        return 1
    print(f"sent {n} readings; the server will be flagged stale shortly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
