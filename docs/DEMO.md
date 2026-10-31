# ThermalView demo runbook (about 18 minutes)

A scripted walkthrough of the operator console for the faculty presentation. Each step says what
to click, what the audience should see, and the one sentence that explains why it matters.
Split it between the three presenters at the marked hand-overs.

## Before the session (the day before)

1. Real results exist: `python -m thermalview all` has finished, and `results/forecast_summary.json`
   says `"data_source": "M100"`. The console shows a yellow "Development data" banner otherwise.
   If you see that banner during the demo, say so; never present dev numbers as results.
2. `pip install -r requirements.txt` inside `.venv`, then `python -m pytest` passes.
3. Change the demo passwords in `config.toml` → `[[auth.users]]` if anyone else will use the laptop.

## Setup (5 minutes before)

```powershell
cd thermalview
python -m thermalview serve --lan
```

`--lan` lets other laptops open the console; the window prints the address to give them, e.g.
`http://192.168.1.23:8000`. On this laptop keep using http://127.0.0.1:8000.

* Browser window 1 (normal): http://127.0.0.1:8000, sign in as **operator**.
* Browser window 2 (private/incognito): same address, sign in as **viewer**.
* A second PowerShell window in the project folder (for the telemetry agent in step 7).
* Leave speed at 1 min/s for the first minutes so forecasts build up; the KPI tile
  "live 5-min forecast error" appears after ~1 simulated hour (use speed 10 briefly to get there).

**If the panel wants to follow on their own laptops:**
1. Everyone joins the same Wi-Fi (a phone hotspot is the safest: college Wi-Fi often blocks
   laptop-to-laptop traffic).
2. Start with `python -m thermalview serve --lan`. The first time, Windows asks whether to allow
   Python on networks: tick **Private** and allow. If another laptop still cannot connect, run once in
   an administrator PowerShell:
   `netsh advfirewall firewall add rule name="ThermalView 8000" dir=in action=allow protocol=TCP localport=8000`
3. They open the printed address and sign in as **viewer** (they can look, not change anything).
4. Different networks, or nothing else works: install cloudflared once
   (`winget install --id Cloudflare.cloudflared`), then `python -m thermalview serve --public`. It prints a
   temporary `https://….trycloudflare.com` link that works from any laptop or phone until you press Ctrl+C.
   Change the demo passwords in `config.toml` first, because the link is public.

Each start of `serve` is a fresh simulation: restarting resets the fleet, alerts and audit log.

---

## Presenter 1: the problem and the fleet

**1. The viewer role (1 min).** In the viewer window: the banner says viewers can look and export
only; no Pause, no placement buttons. Try nothing else here.
> "Operations teams separate people who watch from people who act, and every action is attributable."

**2. Fleet overview (2 min).** Operator window, Overview.
* Three data centres, halls, racks; each cell is one server, coloured by CPU temperature.
* Point at the legend: red ring = over the limit, red top bar = *forecast* to cross it within 5 min,
  F = fan fault, hatched = drained, dashed = no telemetry.
* Hover a cell: now, +5, +15, +30 minute forecasts.
> "Each simulated server copies a real node from the Marconi100 supercomputer: its heat behaviour
> is fitted from that node's real data, and its background load is that node's recorded load."

**3. One server (1 min).** Click a cell. The chart shows measured temperature, the 5-minute forecast
as it was issued (orange, lagging by design), and the forecast now (blue dashes).
> "The orange line lets you see how good the forecast has been on this server, live."

## Presenter 2: deciding where work goes

**4. Place a workload (2 min).** Place workload → preset **Heavy** → Rank servers.
* Read the top card's explanation aloud: now, heading to, + this job, margin to the limit.
* Open "servers excluded" and read one reason (CPU, memory, maintenance, telemetry).
* Click **Place on …** for the top server.
> "ThermalView ranks by where the server *will* be after the job lands, not where it is now."

**5. A facilities incident (3 min).** Overview → Scenarios: hall **HYD2 hall A**, inlet +10 °C,
**Apply to hall**. Set speed to 5 min/s.
* The hall gets a "cooling degraded" tag; cells darken; red top bars (forecast warnings) appear
  *before* red rings (actual breaches). The Alerts badge counts up.
* Alerts page: acknowledge one with a note ("Facilities informed").
* Place workload again with Site = **HYD2**: servers show "Close to limit" / "Would exceed limit".
  Then Site = Any: the ranking now prefers Pune and Chennai.
* Set the slider back to 0 and Apply; alerts resolve on their own after a few minutes.
> "The forecast gives the operator minutes of warning, and the recommender routes work away
> from the hot hall without anyone having to tell it."

**5b. Look before you break it (1 min).** Before applying a failure, press **Preview** in Scenarios
(or **What if?** next to a failure on a server). It lists which servers would cross the limit within
5/15/30 minutes. The **Resilience** panel answers "which hall can least afford to lose cooling?" for all
halls at once.

**5c. A second failure (1 min).** With the hall still degraded, open one of its servers and press
**What if?** for a fan failure: the prediction starts from the already degraded state. Apply it: the
server becomes ineligible at once and a critical alert with "drain and replace the fan" opens. Leave an
alert unacknowledged for 10 simulated minutes to show it escalate.

**5d. Manual control (2 min).** Place workload → switch to **Manual approval**. Arriving jobs now wait
with ThermalView's suggestion. Approve one; reject one with a reason; press **Rank it myself** on
another, pick the third server and show that it asks *why* (an override). Audit log: the override and
its reason sit next to what ThermalView would have done. Switch back to Automatic.
> "Nothing is forced on the operator: the system recommends and explains; people decide, and every
> decision is accountable."

**6. Maintenance (1 min).** Click any server → **Drain for maintenance**. Rank a workload: that
server is listed under excluded, "in maintenance (drained)". Back on the server: **Retire** is enabled
once its running jobs finish (Inventory shows the status).

## Presenter 3: running it for real

**7. Plugging in a real server (2 min).** Inventory → Add a server: site MAA1, hall A, rack A01,
name `MAA1-EDGE-01`, Telemetry **External** → Add. On the Overview it is dashed: no telemetry yet.
In the second PowerShell window:

```powershell
python scripts\telemetry_agent.py --server MAA1-EDGE-01 --stop-after 20
```

* The cell fills in and gets forecasts like any other server.
* After the agent stops, it turns stale within 3 simulated minutes, a "No telemetry" alert opens,
  and the server is excluded from placement.
* Optional: `--redfish-thermal scripts\sample_redfish\thermal.json --redfish-power scripts\sample_redfish\power.json --load 45`
  shows the same agent reading Redfish-shaped data. API tab: the field mapping.
> "This API is where a data centre would connect its own monitoring: the console does not care
> whether a reading came from our simulator or from a real BMC."

**8. Audit and reports (1 min).** Audit log: every sign-in, ranking, placement, drain, fault and
acknowledgement, with wall-clock and simulated time. Tick "Include automatic scheduler placements"
to show the background placements. **Export CSV** on Alerts and Audit.

**9. The evidence (3 min).** Results tab, top to bottom. Each section starts with "In plain words".
* Forecasting: about 9% less error than "no change" at 5 minutes, about 12% on servers it never
  trained on; the gain fades by 30 minutes, and we say why.
* How sure is the forecast: the 90% range held about 93% of the time; using its top as an alarm
  catches over 90% of limit crossings.
* A month it never saw: the frozen February model on January still beats "no change" at every horizon.
* Placement: about 15% less time over the limit than coolest-now at 1.5× load, no jobs rejected.
* Limit sensitivity: the advantage holds wherever the limit is set.
* Real job starts: temperature jumps ~8 °C while CPU load rises only ~10 points, because the GPUs heat
  the node. This is our main limitation and the next step (use the scheduler's job information).
> "Everything on the console runs on the same code that produced these numbers, and the paper is
> generated from these files, so no number is typed by hand."

---

## Honest answers to likely questions

| Question | Answer |
|---|---|
| Is this running on real servers? | No. The fleet is simulated, calibrated from real M100 telemetry. The telemetry API accepts real readings, but we have not connected real hardware. |
| Why M100 and not the internship servers? | We no longer have access to them. M100 is a public, citable dataset of a production supercomputer (CC BY 4.0). |
| Does it save energy? | We did not measure energy. The claim is fewer and shorter thermal-limit violations in simulation, at a stated cost in rejected jobs. |
| How good is the forecast? | Quote the Results tab: skill vs persistence with its confidence interval, and say where it is not better (30 minutes on February). It also held on January, a month it never saw. |
| Why not deep learning / LSTM? | We tried a neural network on the same last-hour features; it was weaker than the tree models on validation. The signal at these horizons is small. |
| Did you tune on the test set? | No. Every choice was made on validation data, and January (never used for any choice) confirms the result. |
| What if the model is wrong? | Eligibility rules (CPU, memory, faults, maintenance, stale telemetry) are hard constraints; the forecast only orders feasible servers, and every ranking is explained and logged. |
| Is the login production grade? | It is a role model for the demo: salted PBKDF2 hashes, HttpOnly session cookie, per-action audit. A deployment would sit behind the company's SSO and HTTPS. |
