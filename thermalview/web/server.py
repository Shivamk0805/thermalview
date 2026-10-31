"""HTTP layer of the operator console: routes, sign-in, roles, CSV export.

The fleet, recommender, alerts and audit trail live in web/live.py; this
module only checks who is asking and turns requests into calls on it.
"""
from __future__ import annotations

import asyncio
import contextlib
import csv
import hmac
import io
import json
import logging
import secrets
import time
from functools import wraps
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from thermalview.config import Config
from thermalview.recommender import Workload
from thermalview.web import apispec
from thermalview.web.live import HORIZONS, ActionError, LiveApp, SetupError

log = logging.getLogger("thermalview")
STATIC = Path(__file__).parent / "static"
COOKIE = "tv_session"
EXPORT_KINDS = ("fleet", "alerts", "audit", "jobs", "telemetry")


class Sessions:
    def __init__(self, hours: float):
        self.ttl = hours * 3600
        self.items: dict[str, dict] = {}

    def create(self, user: dict) -> str:
        token = secrets.token_urlsafe(32)
        self.items[token] = {**user, "expires": time.time() + self.ttl}
        return token

    def get(self, token: str | None) -> dict | None:
        s = self.items.get(token or "")
        if s and s["expires"] < time.time():
            self.items.pop(token, None)
            return None
        return s

    def drop(self, token: str | None) -> None:
        self.items.pop(token or "", None)

    def drop_user(self, username: str, keep: str | None = None) -> None:
        """Sign a user out everywhere (after removal or a password reset), except session `keep`."""
        for t in [t for t, u in self.items.items() if u["username"] == username and t != keep]:
            self.items.pop(t, None)


def _err(msg: str, status: int) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=status)


async def _body(request: Request) -> dict:
    raw = await request.body()
    try:
        data = json.loads(raw or b"{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"request body is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("request body must be a JSON object")
    return data


def _workload(body: dict) -> Workload:
    try:
        w = Workload(cpu_pct=float(body["cpu_pct"]), mem_gb=float(body["mem_gb"]),
                     duration_min=int(body.get("duration_min", 30)), name=str(body.get("name") or "workload")[:60])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("cpu_pct and mem_gb are required numbers") from exc
    if not (1 <= w.cpu_pct <= 100 and 0 < w.mem_gb <= 4096 and 1 <= w.duration_min <= 1440):
        raise ValueError("cpu_pct must be 1-100, mem_gb 0-4096, duration_min 1-1440")
    return w


def _csv(rows: list[dict], filename: str, fields: list[str] | None = None) -> Response:
    buf = io.StringIO()
    fields = list(rows[0].keys()) if rows else (fields or [])
    if fields:
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def create_app(cfg: Config, live: LiveApp | None = None) -> Starlette:
    live = live or LiveApp(cfg)
    sessions = Sessions(float(cfg["app"].get("session_hours", 10)))
    tokens = [str(t) for t in cfg.get("api", {}).get("ingest_tokens", []) if t]

    def requires(role: str):
        """role: 'session' (any signed-in user), 'viewer' (same), or 'operator'."""
        def deco(fn):
            @wraps(fn)
            async def wrapper(request: Request):
                user = sessions.get(request.cookies.get(COOKIE))
                if user is None:
                    return _err("sign in first", 401)
                if role == "operator" and user["role"] != "operator":
                    return _err("this needs the operator role; you are signed in as a viewer", 403)
                request.state.user = user
                try:
                    return await fn(request)
                except ActionError as exc:
                    return _err(str(exc), 409)
                except KeyError as exc:
                    return _err(str(exc.args[0]) if exc.args else "not found", 404)
                except (ValueError, TypeError) as exc:
                    return _err(str(exc), 422)
            return wrapper
        return deco

    def token_ok(request: Request) -> bool:
        auth = request.headers.get("authorization", "")
        given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        return bool(given) and any(hmac.compare_digest(given, t) for t in tokens)

    def sid_of(request: Request) -> int:
        return int(request.path_params["sid"])

    # ---- pages and session ------------------------------------------------
    async def index(request: Request):
        return FileResponse(STATIC / "index.html")

    async def login(request: Request):
        try:
            body = await _body(request)
        except ValueError as exc:
            return _err(str(exc), 422)
        username, password = str(body.get("username", ""))[:64], str(body.get("password", ""))
        user = live.store.check_user(username, password)
        if user is None:
            live.audit(username or "(blank)", "-", "login_failed", "", {"client": request.client.host if request.client else ""})
            live.store.commit()
            return _err("wrong username or password", 401)
        token = sessions.create(user)
        live.audit(user["username"], user["role"], "login", "", {})
        live.store.commit()
        resp = JSONResponse(user)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", max_age=int(sessions.ttl))
        return resp

    @requires("session")
    async def logout(request: Request):
        u = request.state.user
        sessions.drop(request.cookies.get(COOKIE))
        live.audit(u["username"], u["role"], "logout", "", {})
        live.store.commit()
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE)
        return resp

    @requires("session")
    async def me(request: Request):
        u = request.state.user
        return JSONResponse({k: u[k] for k in ("username", "display_name", "role")})

    # ---- user accounts: only operators manage them; anyone can change their own password -------
    @requires("operator")
    async def users_list(request: Request):
        return JSONResponse({"users": live.list_users()})

    @requires("operator")
    async def users_add(request: Request):
        b = await _body(request)
        return JSONResponse(live.add_user(request.state.user, str(b.get("username", "")), str(b.get("password", "")),
                                          str(b.get("role", "viewer")), str(b.get("display_name", ""))))

    @requires("operator")
    async def users_remove(request: Request):
        name = request.path_params["username"]
        out = live.delete_user(request.state.user, name)
        sessions.drop_user(name)
        return JSONResponse(out)

    @requires("operator")
    async def users_password(request: Request):
        b, name = await _body(request), request.path_params["username"]
        out = live.set_password(request.state.user, name, str(b.get("password", "")))
        sessions.drop_user(name, keep=request.cookies.get(COOKIE))
        return JSONResponse(out)

    @requires("session")
    async def my_password(request: Request):
        b, u = await _body(request), request.state.user
        if live.store.check_user(u["username"], str(b.get("current", ""))) is None:
            return _err("current password is wrong", 403)
        out = live.set_password(u, u["username"], str(b.get("password", "")))
        sessions.drop_user(u["username"], keep=request.cookies.get(COOKIE))
        return JSONResponse(out)

    # ---- read -----------------------------------------------------------------
    @requires("viewer")
    async def fleet(request: Request):
        return JSONResponse(live.fleet())

    @requires("viewer")
    async def history(request: Request):
        sid = sid_of(request)
        live.server_by_ref(sid)
        minutes = min(int(request.query_params.get("minutes", 180)), int(cfg["app"]["history_minutes"]))
        return JSONResponse({"server_id": sid, "threshold": live.threshold,
                             "rows": live.store.history(sid, max(0, live.sim.minute - minutes)),
                             "forecast": {str(h): live.forecasts.get(h, {}).get(sid) for h in HORIZONS}})

    @requires("viewer")
    async def inventory(request: Request):
        return JSONResponse({"servers": live.store.inventory(),
                             "sites": [{"code": c, "name": s["name"], "halls": [h["name"] for h in s["halls"]]}
                                       for c, s in live.sites.items()]})

    @requires("viewer")
    async def jobs(request: Request):
        return JSONResponse({"jobs": live.store.jobs(int(request.query_params.get("limit", 60)))})

    @requires("viewer")
    async def alerts(request: Request):
        st = request.query_params.get("status", "active")
        return JSONResponse({"alerts": live.store.alerts(None if st == "all" else st)})

    @requires("viewer")
    async def audit(request: Request):
        q = request.query_params
        return JSONResponse({"entries": live.store.audit_log(int(q.get("limit", 300)), q.get("user") or None,
                                                             q.get("action") or None, q.get("scheduler", "1") != "0")})

    @requires("viewer")
    async def results(request: Request):
        return JSONResponse(live.results())

    @requires("viewer")
    async def export(request: Request):
        kind = request.path_params["kind"]
        if kind not in EXPORT_KINDS:
            return _err(f"kind must be one of {', '.join(EXPORT_KINDS)}", 404)
        q = request.query_params
        if kind == "fleet":
            rows = [{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in s.items()} for s in live.server_rows()]
        elif kind == "alerts":
            rows = live.store.alerts(None, limit=100_000)
        elif kind == "audit":
            rows = live.store.audit_log(100_000)
        elif kind == "jobs":
            rows = live.store.jobs(100_000)
        else:
            minutes = int(np.clip(int(q.get("minutes", 60)), 1, int(cfg["app"]["history_minutes"])))
            sid = int(q["server_id"]) if q.get("server_id") else None
            rows = live.store.telemetry_export(max(0, live.sim.minute - minutes), sid)
        u = request.state.user
        live.audit(u["username"], u["role"], "export", kind, {"rows": len(rows)})
        live.store.commit()
        stamp = live.sim.now().strftime("%Y%m%d-%H%M")
        table = {"alerts": "alerts", "audit": "audit", "jobs": "jobs"}.get(kind)
        fields = [d[0] for d in live.store.db.execute(f"SELECT * FROM {table} LIMIT 0").description] if table else None
        return _csv(rows, f"thermalview-{kind}-{stamp}.csv", fields)

    # ---- operator actions -----------------------------------------------------
    @requires("operator")
    async def recommend_ep(request: Request):
        body = await _body(request)
        return JSONResponse(live.recommend_for(request.state.user, _workload(body), body.get("site") or None))

    @requires("operator")
    async def place(request: Request):
        body = await _body(request)
        w = _workload(body)
        if "server_id" not in body:
            raise ValueError("server_id is required")
        return JSONResponse(live.place_for(request.state.user, w, int(body["server_id"]), body.get("recommendation_id"),
                                           str(body.get("reason") or ""), site=body.get("site") or None))

    @requires("operator")
    async def add_server(request: Request):
        b = await _body(request)
        return JSONResponse(live.add_server(request.state.user, str(b.get("site", "")), str(b.get("hall", "")),
                                            str(b.get("rack", "")), bool(b.get("external", False)), b.get("name") or None))

    @requires("operator")
    async def maintenance(request: Request):
        b = await _body(request)
        return JSONResponse(live.set_maintenance(request.state.user, sid_of(request), bool(b.get("on", True)),
                                                 str(b.get("reason", ""))))

    @requires("operator")
    async def retire(request: Request):
        b = await _body(request)
        return JSONResponse(live.retire_server(request.state.user, sid_of(request), bool(b.get("force", False))))

    @requires("operator")
    async def fault(request: Request):
        b, u, sid = await _body(request), request.state.user, sid_of(request)
        if "sensor" in b:
            return JSONResponse(live.set_sensor_loss(u, sid, bool(b["sensor"])))
        if "runaway_pct" in b:
            return JSONResponse(live.set_runaway(u, sid, float(b["runaway_pct"] or 0)))
        return JSONResponse(live.set_fan_fault(u, sid, bool(b.get("fan", True))))

    @requires("viewer")
    async def whatif(request: Request):
        b = await _body(request)
        return JSONResponse(live.whatif(str(b.get("kind", "")), b.get("site"), b.get("hall"),
                                        int(b["server_id"]) if b.get("server_id") is not None else None,
                                        float(b["magnitude"]) if b.get("magnitude") is not None else None))

    @requires("viewer")
    async def resilience(request: Request):
        rise = float(np.clip(float(request.query_params.get("rise", 6)), 1, 12))
        return JSONResponse({"rise_c": rise, "halls": live.resilience(rise)})

    @requires("viewer")
    async def queue_list(request: Request):
        return JSONResponse({"mode": live.mode, "queue": live.queue_rows()})

    @requires("operator")
    async def queue_decide(request: Request):
        b = await _body(request)
        if request.path_params["decision"] not in ("approve", "reject"):
            raise KeyError("use /approve or /reject")
        approve = request.path_params["decision"] == "approve"
        sid = int(b["server_id"]) if b.get("server_id") is not None else None
        return JSONResponse(live.decide_queued(request.state.user, int(request.path_params["qid"]), approve, sid,
                                               str(b.get("reason") or "")))

    @requires("operator")
    async def cooling(request: Request):
        b = await _body(request)
        return JSONResponse(live.set_cooling(request.state.user, str(b.get("site", "")), str(b.get("hall", "")),
                                             float(b.get("inlet_rise_c", 0))))

    @requires("operator")
    async def ack(request: Request):
        b = await _body(request)
        return JSONResponse(live.ack_alert(request.state.user, int(request.path_params["aid"]), str(b.get("note", ""))))

    @requires("operator")
    async def control(request: Request):
        body = await _body(request)
        action, u = body.get("action"), request.state.user
        if action == "pause":
            live.running = False
        elif action == "resume":
            live.running = True
        elif action == "speed":
            live.speed = int(np.clip(int(body.get("value", 1)), 1, 10))
        elif action == "auto_jobs":
            live.auto_jobs = bool(body.get("value", True))
        elif action == "mode":
            if body.get("value") not in ("auto", "manual"):
                raise ValueError("mode must be auto or manual")
            live.mode = body["value"]
        elif action == "step":
            for _ in range(int(np.clip(int(body.get("value", 1)), 1, 120))):
                live.advance()
        else:
            raise ValueError(f"unknown action {action!r}")
        live.audit(u["username"], u["role"], "clock_" + action, "", {"value": body.get("value")})
        live.store.commit()
        return JSONResponse({"ok": True, "action": action})

    # ---- integration API --------------------------------------------------------
    async def metrics(request: Request):
        """Prometheus scrape endpoint. Same bearer tokens as the ingest API, or a signed-in session."""
        if not token_ok(request) and sessions.get(request.cookies.get(COOKIE)) is None:
            return _err("missing or invalid bearer token", 401)
        return Response(live.metrics_text(), media_type="text/plain; version=0.0.4")

    async def spec(request: Request):
        return JSONResponse(apispec.spec())

    async def v1_servers(request: Request):
        if not token_ok(request) and sessions.get(request.cookies.get(COOKIE)) is None:
            return _err("missing or invalid bearer token", 401)
        return JSONResponse({"servers": [{"server_id": s["server_id"], "name": s["name"], "site": s["site"],
                                          "hall": s["hall"], "rack": s["rack"], "slot": s["slot"], "kind": s["kind"]}
                                         for s in live.server_rows()]})

    async def v1_telemetry(request: Request):
        if not token_ok(request):
            return _err("missing or invalid bearer token", 401)
        try:
            body = await _body(request)
        except ValueError as exc:
            return _err(str(exc), 422)
        readings = body.get("readings")
        if not isinstance(readings, list) or not readings:
            return _err("body must be {\"readings\": [ ... ]}", 422)
        if len(readings) > 1000:
            return _err("at most 1000 readings per call", 422)
        return JSONResponse(live.ingest(readings))

    async def figure(request: Request):
        if sessions.get(request.cookies.get(COOKIE)) is None:
            return _err("sign in first", 401)
        name = Path(request.path_params["name"]).name
        p = cfg.path("results") / "figures" / name
        if not p.exists() or p.suffix != ".png":
            return _err("figure not found", 404)
        return FileResponse(p)

    async def loop() -> None:
        tick = float(cfg["app"]["tick_seconds"])
        per_tick = int(cfg["app"]["sim_minutes_per_tick"])
        while True:
            await asyncio.sleep(tick)
            if live.running:
                try:
                    for _ in range(per_tick * live.speed):
                        live.advance()
                except Exception:  # keep the console alive, but say what happened
                    log.exception("simulation step failed")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        task = asyncio.create_task(loop())
        yield
        task.cancel()

    routes = [
        Route("/", index),
        Route("/api/login", login, methods=["POST"]),
        Route("/api/logout", logout, methods=["POST"]),
        Route("/api/me", me),
        Route("/api/me/password", my_password, methods=["POST"]),
        Route("/api/users", users_list),
        Route("/api/users", users_add, methods=["POST"]),
        Route("/api/users/{username}", users_remove, methods=["DELETE"]),
        Route("/api/users/{username}/password", users_password, methods=["POST"]),
        Route("/api/fleet", fleet),
        Route("/api/servers/{sid:int}/history", history),
        Route("/api/inventory", inventory),
        Route("/api/servers", add_server, methods=["POST"]),
        Route("/api/servers/{sid:int}/maintenance", maintenance, methods=["POST"]),
        Route("/api/servers/{sid:int}/retire", retire, methods=["POST"]),
        Route("/api/servers/{sid:int}/fault", fault, methods=["POST"]),
        Route("/api/halls/cooling", cooling, methods=["POST"]),
        Route("/api/whatif", whatif, methods=["POST"]),
        Route("/api/resilience", resilience),
        Route("/api/queue", queue_list),
        Route("/api/queue/{qid:int}/{decision:str}", queue_decide, methods=["POST"]),
        Route("/api/recommend", recommend_ep, methods=["POST"]),
        Route("/api/place", place, methods=["POST"]),
        Route("/api/jobs", jobs),
        Route("/api/alerts", alerts),
        Route("/api/alerts/{aid:int}/ack", ack, methods=["POST"]),
        Route("/api/audit", audit),
        Route("/api/export/{kind}.csv", export),
        Route("/api/control", control, methods=["POST"]),
        Route("/api/results", results),
        Route("/api/v1/spec", spec),
        Route("/metrics", metrics),
        Route("/api/v1/servers", v1_servers),
        Route("/api/v1/telemetry", v1_telemetry, methods=["POST"]),
        Route("/figures/{name}", figure),
        Mount("/static", StaticFiles(directory=STATIC), name="static"),
    ]
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.live = live
    app.state.sessions = sessions
    return app


def start_public_tunnel(port: int) -> None:
    """Expose the console on a temporary public https address with a Cloudflare quick tunnel
    (no account needed). Needs the free `cloudflared` program on PATH."""
    import re
    import shutil
    import subprocess
    import threading
    import os
    from pathlib import Path as _P
    exe = shutil.which("cloudflared")
    if not exe:            # winget installs here but a terminal opened before the install has an old PATH
        for c in (r"C:\Program Files (x86)\cloudflared\cloudflared.exe", r"C:\Program Files\cloudflared\cloudflared.exe",
                  os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\cloudflared.exe"),
                  str(_P.cwd() / "cloudflared.exe")):
            if _P(c).exists():
                exe = c
                break
    if not exe:
        print("--public needs cloudflared. Install it once with:\n"
              "    winget install --id Cloudflare.cloudflared\n"
              "or download cloudflared-windows-amd64.exe from\n"
              "    https://github.com/cloudflare/cloudflared/releases/latest\n"
              "rename it to cloudflared.exe and put it in this project folder. Then run the command again.\n"
              "Continuing without a public link (this laptop only; use --lan for the same Wi-Fi).")
        return
    proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def watch():
        for line in proc.stdout:
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if m:
                print("(the link can take up to a minute to start working)", flush=True)
                print("\n" + "=" * 70 + f"\nPUBLIC ADDRESS (any laptop, any network):  {m.group(0)}\n"
                      "Anyone with this link sees the sign-in page; it stops working when you press Ctrl+C.\n"
                      + "=" * 70 + "\n", flush=True)
    threading.Thread(target=watch, daemon=True).start()
    import atexit
    atexit.register(proc.terminate)


def serve(cfg: Config, public: bool = False) -> None:
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    try:
        app = create_app(cfg)
    except SetupError as exc:
        raise SystemExit(str(exc)) from None
    users = ", ".join(f"{u['username']} ({u['role']})" for u in app.state.live.list_users()) \
        if hasattr(app.state, "live") else ", ".join(f"{u['username']} ({u['role']})" for u in cfg.get("auth", {}).get("users", []))
    host, port = cfg["app"]["host"], int(cfg["app"]["port"])
    if host in ("0.0.0.0", "::"):
        import socket
        ips = set()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("10.255.255.255", 1))          # no packet is sent; picks the LAN interface
                ips.add(s.getsockname()[0])
        except OSError:
            pass
        try:
            ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
        except OSError:
            pass
        ips = sorted(i for i in ips if not i.startswith("127."))
        print("ThermalView console, reachable from other computers on this network:")
        for ip in ips or ["<this laptop's IP address>"]:
            print(f"    http://{ip}:{port}")
        print(f"    (and http://127.0.0.1:{port} on this laptop)")
        weak = [u["username"] for u in cfg.get("auth", {}).get("users", []) if u.get("password", "").endswith("-demo")]
        if weak:
            print(f"WARNING: accounts {', '.join(weak)} still use the demo passwords from config.toml. Change them first.")
        print("If other laptops can't connect, allow the port through Windows Firewall (see README).")
    else:
        print(f"ThermalView console at http://{host}:{port}  (Ctrl+C to stop)")
    print(f"Accounts: {users}  (operators manage accounts on the Users page)")
    if public:
        start_public_tunnel(port)
    uvicorn.run(app, host=cfg["app"]["host"], port=int(cfg["app"]["port"]), log_level="warning")
