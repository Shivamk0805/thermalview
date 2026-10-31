"use strict";

/* ================= helpers ================= */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html !== undefined) e.innerHTML = html; return e; };
const f1 = (x) => (x === null || x === undefined || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(1));
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "–");
const fmtWall = (iso) => (iso ? new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "–");

const STATE_LABEL = { ok: "Normal", watch: "Near limit", warning: "Forecast over limit", hot: "Over limit", fault: "Fan fault", stale: "No telemetry" };
const STATUS_LABEL = { recommended: "Recommended", caution: "Close to limit", not_recommended: "Would exceed limit" };
const MODEL_LABEL = { persistence: "Persistence", ridge: "Ridge regression", random_forest: "Random forest", hist_gb: "Gradient boosting (HistGB)", mlp: "Neural network (MLP)" };
const POLICY_LABEL = { random: "Random", least_loaded: "Least loaded", coolest_now: "Coolest now (reactive)",
  predictive_persist: "Predictive, persistence (ablation)", thermalview: "ThermalView (ML + heat model)" };
const ACTION_LABEL = { login: "Signed in", logout: "Signed out", login_failed: "Failed sign-in", recommend: "Asked for a ranking",
  place: "Placed a workload", user_add: "Added a user", user_remove: "Removed a user", user_password: "Changed a password",
  drain: "Drained", undrain: "Returned to service", server_add: "Added a server",
  server_retire: "Retired a server", fault_fan_inject: "Simulated fan failure", fault_fan_clear: "Repaired fan",
  fault_cooling: "Degraded hall cooling", fault_cooling_clear: "Restored hall cooling", alert_ack: "Acknowledged alert",
  export: "Exported CSV", start: "Console started", clock_pause: "Paused clock", clock_resume: "Resumed clock",
  clock_speed: "Changed speed", clock_auto_jobs: "Toggled scheduler", clock_step: "Stepped clock",
  clock_mode: "Changed placement mode", place_override: "Placed against the recommendation", job_held: "Held a job for a decision",
  job_rejected: "Rejected a job", job_expired: "Held job expired", fault_sensor_inject: "Cut a server's telemetry",
  fault_sensor_clear: "Restored telemetry", fault_runaway_inject: "Started a stuck process", fault_runaway_clear: "Stopped the stuck process" };
const KIND_LABEL = (k) => (k.startsWith("hall_incident") ? "Hall incident (likely cooling)" :
  { breach: "Over limit", forecast: "Forecast over limit", fan_fault: "Fan fault", stale: "No telemetry" }[k] || k);

let me = null, fleet = null, drawerId = null, lastWorkload = null, lastRecId = null, lastQueueId = null, polling = false;
const byId = () => new Map((fleet?.servers || []).map((s) => [s.server_id, s]));
const isOp = () => me && me.role === "operator";

async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && path !== "/api/login") { showLogin(); throw new Error("signed out"); }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}
function toast(text) {
  const t = $("#toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove("show"), 2800);
}
function showError(msg) {
  const b = $("#error-banner");
  b.hidden = !msg;
  b.textContent = msg ? `Can't reach the ThermalView server: ${msg}. Is \`python -m thermalview serve\` still running?` : "";
}

/* ================= sign-in ================= */
function showLogin() {
  me = null; polling = false;
  $("#app").hidden = true;
  $("#login").hidden = false;
  closeDrawer();
  setTimeout(() => $("#lg-user").focus(), 0);
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#lg-error").textContent = "";
  try {
    const u = await api("/api/login", { username: $("#lg-user").value.trim(), password: $("#lg-pass").value });
    $("#lg-pass").value = "";
    start(u);
  } catch (err) { $("#lg-error").textContent = err.message; }
});
$("#btn-logout").addEventListener("click", async () => { try { await api("/api/logout", {}); } catch (e) { /* already out */ } showLogin(); });

function start(u) {
  me = u;
  document.body.classList.toggle("viewer", u.role !== "operator");
  $("#login").hidden = true;
  $("#app").hidden = false;
  $("#user-name").textContent = u.display_name || u.username;
  const r = $("#user-role");
  r.textContent = u.role;
  r.className = `role ${u.role}`;
  $("#viewer-banner").hidden = u.role === "operator";
  route();
  if (!polling) { polling = true; pollFleet(); }
  setTimeout(loadResilience, 1500);
}

/* ================= routing ================= */
const VIEWS = ["overview", "servers", "place", "alerts", "inventory", "audit", "users", "results", "api"];
function currentView() { const v = (location.hash.replace("#/", "") || "overview").split("/")[0]; return VIEWS.includes(v) ? v : "overview"; }
function route() {
  const view = currentView();
  for (const v of VIEWS) $(`#view-${v}`).hidden = v !== view;
  $$(".tabs a").forEach((a) => (a.dataset.view === view ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (view === "alerts") loadAlerts();
  if (view === "inventory") loadInventory();
  if (view === "audit") loadAudit();
  if (view === "users") loadUsers();
  if (view === "results") loadResults();
  if (view === "api") loadApi();
  if (view === "servers" && fleet) renderServers();
  if (view === "overview") loadResilience();
}
window.addEventListener("hashchange", route);
setInterval(() => {
  if (!me || document.hidden) return;
  const v = currentView();
  if (v === "alerts" && !document.activeElement?.closest("#al-table")) loadAlerts();
  if (v === "inventory" && !document.activeElement?.closest("#add-form")) loadInventory();
  if (v === "audit") loadAudit();
}, 5000);

/* ================= heat colour (sequential, one hue) ================= */
let RAMP = [];
function readRamp() {
  const cs = getComputedStyle(document.documentElement);
  RAMP = [0, 1, 2, 3, 4, 5].map((i) => cs.getPropertyValue(`--t${i}`).trim());
}
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readRamp(); layoutSig = ""; });
readRamp();
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
function heat(t, thr) {
  const lo = thr - 30, x = Math.max(0, Math.min(1, (t - lo) / (thr - lo))) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x)), f = x - i;
  const a = hex(RAMP[i]), b = hex(RAMP[i + 1]);
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  const lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
  return { bg: `rgb(${c.join(",")})`, dark: lum < 0.5 };
}

/* ================= fleet poll ================= */
async function pollFleet() {
  if (!polling) return;
  try {
    renderFleet(await api("/api/fleet"));
    showError(null);
  } catch (e) {
    if (e.message !== "signed out") showError(e.message);
  }
  if (polling) setTimeout(pollFleet, 1000);
}

function renderFleet(data) {
  fleet = data;
  $("#org").textContent = data.org || "Heat-aware workload placement";
  $("#sim-time").textContent = fmtTime(data.sim_time);
  $("#btn-run").textContent = data.running ? "Pause" : "Resume";
  if (document.activeElement !== $("#speed")) $("#speed").value = String(data.speed);
  $("#dev-banner").hidden = data.data_source !== "dev-fixture";
  $$(".h-label").forEach((n) => (n.textContent = data.horizon));
  $("#auto-jobs").checked = data.auto_jobs;

  const al = data.alerts, badge = $("#alert-badge");
  const nActive = al.critical + al.warning;
  badge.hidden = nActive === 0;
  badge.textContent = nActive;
  badge.className = `nav-badge${al.critical ? "" : " warn"}`;

  const qb = $("#queue-badge");
  qb.hidden = !data.queue.length;
  qb.textContent = data.queue.length;
  renderKpis(data);
  renderSites(data);
  renderOvAlerts(data);
  renderQueue(data);
  fillSiteSelects(data);
  if (currentView() === "servers") renderServers();
  if (drawerId) updateDrawerStats();
}

function renderKpis(d) {
  const s = d.servers, n = (st) => s.filter((x) => x.state === st).length;
  const maint = s.filter((x) => x.maintenance).length;
  const chk = d.live_forecast_check || {};
  const fcErr = chk.n > 50 ? `${f1(chk.mae)} °C` : "–";
  const fcSub = chk.n > 50 ? `vs ${f1(chk.persist)} °C assuming no change` : "collecting…";
  const tiles = [
    [`${s.length}`, "servers in service", `${d.sites.length} sites · ${maint} drained`],
    [`${n("hot") + n("fault")}`, "over limit or faulted", `limit ${f1(d.threshold)} °C`, n("hot") + n("fault") > 0],
    [`${n("warning")}`, `forecast over limit in ${d.horizon} min`, `${n("watch")} more within ${f1(d.safety_margin)} °C`],
    [`${d.alerts.critical + d.alerts.warning}`, "active alerts", `${d.alerts.unacked} not acknowledged`, d.alerts.critical > 0],
    [fcErr, `live ${d.horizon}-min forecast error`, fcSub],
  ];
  $("#kpis").innerHTML = tiles.map(([b, l, sub, bad]) => `<div class="kpi${bad ? " alert" : ""}"><b>${esc(b)}</b><span>${esc(l)}</span><small>${esc(sub)}</small></div>`).join("");
  $("#ov-lede").textContent = `${s.length} servers in ${d.sites.length} data centres. Simulated servers copy real M100 nodes; ` +
    `forecasts come from the ${(MODEL_LABEL[d.model] || d.model).toLowerCase()} model, ${d.horizon} minutes ahead.`;
}

let layoutSig = "";
function renderSites(d) {
  const thr = d.threshold;
  $("#scale-lo").textContent = `${Math.round(thr - 30)}°`;
  $("#scale-hi").textContent = `${f1(thr)}° limit (${d.limit_source})`;
  const sig = JSON.stringify(d.sites.map((s) => s.halls.map((h) => [h.name, h.racks.map((r) => [r.rack, r.servers])])));
  const host = $("#sites");
  if (sig !== layoutSig) {
    layoutSig = sig;
    host.innerHTML = d.sites.map((site) => `
      <section class="site" data-site="${esc(site.code)}">
        <div class="site-head"><h2>${esc(site.name)}</h2><span class="code">${esc(site.code)}</span><span class="facts"></span></div>
        <div class="halls">${site.halls.map((h) => `
          <div class="hall" data-hall="${esc(site.code)}/${esc(h.name)}">
            <div class="hall-head">Hall ${esc(h.name)} <span class="tag" hidden></span></div>
            <div class="racks">${h.racks.map((r) => `
              <div class="rackcol"><div class="slots">${r.servers.map((id) => `<button type="button" class="cell" data-id="${id}"><span class="t"></span><span class="g"></span></button>`).join("")}</div>
              <div class="rlabel">${esc(r.rack)}</div></div>`).join("")}</div>
          </div>`).join("")}</div>
      </section>`).join("");
  }
  const ids = byId();
  for (const site of d.sites) {
    const sec = host.querySelector(`[data-site="${CSS.escape(site.code)}"]`);
    const st = site.states, bad = (st.hot || 0) + (st.fault || 0);
    sec.querySelector(".facts").innerHTML = `<span><b>${site.n}</b> servers</span><span>hottest <b>${f1(site.max_temp)} °C</b></span>` +
      `<span>mean load <b>${Math.round(site.mean_load)}%</b></span>` + (bad ? `<span><b>${bad}</b> over limit / faulted</span>` : "") +
      (st.warning ? `<span><b>${st.warning}</b> forecast over</span>` : "") + (site.maintenance ? `<span><b>${site.maintenance}</b> drained</span>` : "");
    for (const h of site.halls) {
      const hall = sec.querySelector(`[data-hall="${CSS.escape(site.code + "/" + h.name)}"]`);
      hall.classList.toggle("cooling", h.inlet_rise_c > 0);
      const tag = hall.querySelector(".tag");
      tag.hidden = !(h.inlet_rise_c > 0);
      tag.textContent = `cooling degraded +${h.inlet_rise_c} °C`;
    }
  }
  for (const cell of host.querySelectorAll(".cell")) {
    const s = ids.get(Number(cell.dataset.id));
    if (!s) continue;
    const fc = s.forecast[d.horizon];
    const c = s.temp === null ? { bg: "", dark: false } : heat(s.temp, thr);
    cell.style.background = c.bg;
    cell.className = "cell" + (c.dark ? " dark" : "") + (s.state === "hot" ? " hot" : "") +
      (s.state === "warning" ? " fc" : "") + (s.fan_fault ? " fault" : "") + (s.maintenance ? " maint" : "") + (s.state === "stale" ? " stale" : "") + (s.kind === "external" ? " ext" : "");
    cell.querySelector(".t").textContent = s.temp === null ? "—" : Math.round(s.temp);
    cell.querySelector(".g").textContent = s.fan_fault ? "F" : s.maintenance ? "M" : s.state === "stale" ? "?" : s.state === "hot" ? "!" : "";
    cell.setAttribute("aria-label", `${s.name}: ${s.temp === null ? "no reading" : f1(s.temp) + " °C"}, forecast ${f1(fc)} °C, ${STATE_LABEL[s.state]}${s.maintenance ? ", drained" : ""}`);
  }
}
const tip = el("div", "cell-tip");
document.body.appendChild(tip);
$("#sites").addEventListener("mousemove", (e) => {
  const cell = e.target.closest(".cell");
  if (!cell || !fleet) { tip.style.display = "none"; return; }
  const s = byId().get(Number(cell.dataset.id));
  if (!s) return;
  const f = s.forecast;
  tip.innerHTML = `<b>${esc(s.name)}</b>${s.kind === "external" ? " · external" : ""}<br>${esc(STATE_LABEL[s.state])}${s.maintenance ? " · drained" : ""}<br>` +
    `Now ${f1(s.temp)} °C → ${f1(f["5"])} / ${f1(f["15"])} / ${f1(f["30"])} °C (+5/15/30 min)<br>Load ${s.load === null ? "–" : Math.round(s.load) + "%"} · ${s.n_jobs} job${s.n_jobs === 1 ? "" : "s"}`;
  tip.style.display = "block";
  tip.style.left = `${Math.min(window.innerWidth - 290, e.clientX + 14)}px`;
  tip.style.top = `${e.clientY + 14}px`;
});
$("#sites").addEventListener("mouseleave", () => (tip.style.display = "none"));
$("#sites").addEventListener("click", (e) => { const c = e.target.closest(".cell"); if (c) openDrawer(Number(c.dataset.id)); });

function renderOvAlerts(d) {
  const ol = $("#ov-alerts");
  const list = d.alerts.active.slice(0, 10);
  ol.innerHTML = list.length ? list.map((a) => `<li class="${a.severity}"><time>${fmtTime(a.opened_ts)} · ${esc(a.status)}${a.escalated ? '<span class="esc">ESCALATED</span>' : ""}</time>` +
    `<a href="#" data-srv="${a.server_id}"><span class="sev">${a.severity}</span>${esc(a.text)}</a></li>`).join("")
    : `<li class="none">No active alerts.</li>`;
}
$("#ov-alerts").addEventListener("click", (e) => { const a = e.target.closest("[data-srv]"); if (a) { e.preventDefault(); if (byId().has(Number(a.dataset.srv))) openDrawer(Number(a.dataset.srv)); } });

let selectsFilled = "";
function fillSiteSelects(d) {
  const sig = JSON.stringify(d.sites.map((s) => [s.code, s.halls.map((h) => h.name)]));
  if (sig === selectsFilled) return;
  selectsFilled = sig;
  const opts = d.sites.map((s) => `<option value="${esc(s.code)}">${esc(s.name)} (${esc(s.code)})</option>`).join("");
  for (const id of ["#sv-site", "#wl-site"]) { const sel = $(id), v = sel.value; sel.innerHTML = `<option value="">${id === "#sv-site" ? "All sites" : "Any site"}</option>${opts}`; sel.value = v; }
  $("#sc-hall").innerHTML = d.sites.flatMap((s) => s.halls.map((h) => `<option value="${esc(s.code)}/${esc(h.name)}">${esc(s.code)} hall ${esc(h.name)}</option>`)).join("");
  syncScenario();
  const as = $("#add-site");
  as.innerHTML = opts;
  fillHalls();
}

/* ================= scenario & clock ================= */
function syncScenario() {
  if (!fleet) return;
  const [site, hall] = $("#sc-hall").value.split("/");
  const h = fleet.sites.find((s) => s.code === site)?.halls.find((x) => x.name === hall);
  $("#sc-rise").value = h ? h.inlet_rise_c : 0;
  $("#sc-out").textContent = `${$("#sc-rise").value} °C`;
}
$("#sc-hall").addEventListener("change", syncScenario);
$("#sc-rise").addEventListener("input", (e) => ($("#sc-out").textContent = `${e.target.value} °C`));
$("#sc-apply").addEventListener("click", async () => {
  const [site, hall] = $("#sc-hall").value.split("/");
  try {
    const r = await api("/api/halls/cooling", { site, hall, inlet_rise_c: Number($("#sc-rise").value) });
    toast(r.inlet_rise_c > 0 ? `${site} hall ${hall}: inlet air +${r.inlet_rise_c} °C. Watch the hall heat up.` : `${site} hall ${hall}: cooling restored.`);
  } catch (e) { toast(e.message); }
});
function whatifHtml(w) {
  const rows = w.servers.slice(0, 8).map((r) => {
    const cell = (v) => `<td class="${v >= w.threshold ? "over" : ""}">${f1(v)}</td>`;
    return `<tr><td>${esc(r.name)}</td><td>${f1(r.now)}</td>${cell(r.pred["5"])}${cell(r.pred["15"])}${cell(r.pred["30"])}</tr>`;
  }).join("");
  return `<p><b>${esc(w.summary)}</b></p><table><thead><tr><th>Server</th><th>Now</th><th>+5</th><th>+15</th><th>+30 min</th></tr></thead><tbody>${rows}</tbody></table>` +
    (w.servers.length > 8 ? `<p class="muted">${w.servers.length - 8} more not shown (coolest).</p>` : "") +
    `<p class="muted">Prediction only: current forecast + this server's fitted heat response. Nothing was applied.</p>`;
}
$("#sc-preview").addEventListener("click", async () => {
  const [site, hall] = $("#sc-hall").value.split("/");
  const rise = Number($("#sc-rise").value) || 6;
  try {
    const w = await api("/api/whatif", { kind: "cooling", site, hall, magnitude: rise });
    $("#sc-result").hidden = false;
    $("#sc-result").innerHTML = whatifHtml(w);
  } catch (e) { toast(e.message); }
});
async function loadResilience() {
  if (!me || currentView() !== "overview") return;
  try {
    const r = await api(`/api/resilience?rise=${$("#res-rise").value}`);
    $("#resilience").innerHTML = r.halls.map((h) => `<li><span>${esc(h.site)} hall ${esc(h.hall)}</span>` +
      `<span><b class="${h.would_cross ? "bad" : ""}">${h.would_cross}</b> of ${h.servers} would cross the limit</span></li>`).join("");
  } catch (e) { /* shown by the fleet poll */ }
}
$("#res-rise").addEventListener("change", loadResilience);
setInterval(loadResilience, 15000);

$("#btn-run").addEventListener("click", () => api("/api/control", { action: fleet && fleet.running ? "pause" : "resume" }).catch((e) => toast(e.message)));
$("#speed").addEventListener("change", (e) => api("/api/control", { action: "speed", value: Number(e.target.value) }).catch((er) => toast(er.message)));
$("#auto-jobs").addEventListener("change", (e) => api("/api/control", { action: "auto_jobs", value: e.target.checked }).catch((er) => toast(er.message)));

/* ================= servers table ================= */
let sortKey = "name", sortDir = 1;
const sortVal = (s, k) => ({ name: s.name, loc: `${s.site}${s.rack}${String(s.slot).padStart(2, "0")}`, temp: s.temp ?? -1,
  f5: s.forecast["5"] ?? -1, f15: s.forecast["15"] ?? -1, f30: s.forecast["30"] ?? -1, load: s.load ?? -1, power: s.power ?? -1,
  n_jobs: s.n_jobs, state: ["hot", "fault", "warning", "stale", "watch", "ok"].indexOf(s.state) }[k]);
function renderServers() {
  const q = $("#sv-q").value.trim().toLowerCase(), site = $("#sv-site").value, st = $("#sv-state").value;
  let rows = fleet.servers.filter((s) => (!q || `${s.name} ${s.rack}`.toLowerCase().includes(q)) && (!site || s.site === site));
  if (st === "attention") rows = rows.filter((s) => ["hot", "fault", "warning", "stale"].includes(s.state));
  else if (st === "maint") rows = rows.filter((s) => s.maintenance);
  else if (st) rows = rows.filter((s) => s.state === st);
  rows.sort((a, b) => { const x = sortVal(a, sortKey), y = sortVal(b, sortKey); return (x > y ? 1 : x < y ? -1 : 0) * sortDir; });
  $("#sv-count").textContent = `${rows.length} of ${fleet.servers.length}`;
  const thr = fleet.threshold;
  const tcell = (v) => `<td class="num${v !== null && v >= thr ? " worse" : ""}">${f1(v)}</td>`;
  $("#sv-table tbody").innerHTML = rows.map((s) => `<tr class="click" data-id="${s.server_id}" tabindex="0">
    <td><strong>${esc(s.name)}</strong>${s.kind === "external" ? `<span class="kind-ext">external</span>` : ""}</td>
    <td class="loc">${esc(s.site)} · hall ${esc(s.hall)} · ${esc(s.rack)}/${s.slot}</td>
    ${tcell(s.temp)}${tcell(s.forecast["5"])}${tcell(s.forecast["15"])}${tcell(s.forecast["30"])}
    <td class="num">${s.load === null ? "–" : Math.round(s.load) + "%"}</td><td class="num">${s.power === null ? "–" : s.power + " W"}</td>
    <td class="num">${s.n_jobs}</td>
    <td><span class="state ${s.state}"><span>${STATE_LABEL[s.state]}</span></span>${s.maintenance ? ` <span class="state maint"><span>Drained</span></span>` : ""}</td></tr>`).join("");
  $$("#sv-table th").forEach((th) => (th.dataset.k === sortKey ? th.setAttribute("aria-sort", sortDir > 0 ? "ascending" : "descending") : th.removeAttribute("aria-sort")));
}
$$("#sv-table th").forEach((th) => th.addEventListener("click", () => { if (sortKey === th.dataset.k) sortDir = -sortDir; else { sortKey = th.dataset.k; sortDir = th.dataset.k === "name" || th.dataset.k === "loc" ? 1 : -1; } renderServers(); }));
["#sv-q", "#sv-site", "#sv-state"].forEach((s) => $(s).addEventListener("input", () => fleet && renderServers()));
$("#sv-table").addEventListener("click", (e) => { const r = e.target.closest("tr[data-id]"); if (r) openDrawer(Number(r.dataset.id)); });
$("#sv-table").addEventListener("keydown", (e) => { const r = e.target.closest("tr[data-id]"); if (r && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openDrawer(Number(r.dataset.id)); } });

/* ================= drawer ================= */
let drawerReturn = null;
function openDrawer(id) {
  drawerReturn = document.activeElement;
  if (drawerId !== id) { $("#drawer-whatif").hidden = true; }
  drawerId = id;
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  updateDrawerStats();
  loadHistory();
  $("#drawer-close").focus();
}
function closeDrawer() {
  drawerId = null;
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  if (drawerReturn && document.body.contains(drawerReturn)) drawerReturn.focus();
}
$("#drawer-close").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && drawerId) closeDrawer(); });

function updateDrawerStats() {
  const s = byId().get(drawerId);
  if (!s) { $("#drawer-sub").textContent = "This server is no longer in service."; return; }
  $("#drawer-title").textContent = s.name;
  $("#drawer-sub").textContent = `${s.site} · hall ${s.hall} · rack ${s.rack}, slot ${s.slot} · ` +
    (s.kind === "external" ? "external: telemetry from an agent" : `copies M100 node ${s.source_node}`) + ` · ${STATE_LABEL[s.state]}${s.maintenance ? " · drained" : ""}`;
  $("#drawer-stats").innerHTML = [
    [`${f1(s.temp)} °C`, "CPU temperature"], [`${f1(s.forecast[fleet.horizon])} °C`, `in ${fleet.horizon} min`],
    [s.load === null ? "–" : `${Math.round(s.load)}%`, `CPU load · ${s.n_jobs} job${s.n_jobs === 1 ? "" : "s"}`], [`${f1(s.inlet)} °C`, "inlet air"],
  ].map(([b, l]) => `<div class="stat"><b>${esc(b)}</b><span>${esc(l)}</span></div>`).join("");
  $("#fault-box").hidden = s.kind === "external";
  const fan = $("#btn-fan");
  fan.textContent = s.fan_fault ? "Repair fan" : "Fail fan";
  fan.className = `btn small op-only ${s.fan_fault ? "ghost" : "danger"}`;
  const rw = $("#btn-runaway");
  rw.textContent = s.runaway_pct ? `Stop it (${Math.round(s.runaway_pct)}%)` : "Start it";
  rw.className = `btn small op-only ${s.runaway_pct ? "ghost" : "danger"}`;
  const sn = $("#btn-sensor");
  sn.textContent = s.sensor_lost ? "Restore telemetry" : "Cut telemetry";
  sn.className = `btn small op-only ${s.sensor_lost ? "ghost" : "danger"}`;
  $("#btn-drain").textContent = s.maintenance ? "Return to service" : "Drain for maintenance";
  $("#btn-retire").disabled = !s.maintenance;
  $("#btn-retire").title = s.maintenance ? "" : "Drain the server first";
  $("#btn-tel").href = `/api/export/telemetry.csv?minutes=720&server_id=${s.server_id}`;
  $("#drawer-note").textContent = s.maintenance ? (s.n_jobs ? `Drained: no new work. ${s.n_jobs} job(s) still finishing before it can be retired.` : "Drained and idle: safe to retire.") :
    s.kind === "external" ? `Last reading ${s.age_min === null ? "never" : s.age_min + " min ago"}. Readings arrive via POST /api/v1/telemetry.` : "";
}
$("#btn-fan").addEventListener("click", async () => {
  const s = byId().get(drawerId);
  try { await api(`/api/servers/${drawerId}/fault`, { fan: !s.fan_fault }); toast(s.fan_fault ? `${s.name}: fan repaired` : `${s.name}: fan failure simulated. Watch it heat up.`); }
  catch (e) { toast(e.message); }
});
$("#btn-runaway").addEventListener("click", async () => {
  const s = byId().get(drawerId);
  const pct = s.runaway_pct ? 0 : Number($("#rw-pct").value) || 60;
  try { await api(`/api/servers/${drawerId}/fault`, { runaway_pct: pct }); toast(pct ? `${s.name}: stuck process using ${pct}% CPU` : `${s.name}: stuck process stopped`); }
  catch (e) { toast(e.message); }
});
$("#btn-sensor").addEventListener("click", async () => {
  const s = byId().get(drawerId);
  try { await api(`/api/servers/${drawerId}/fault`, { sensor: !s.sensor_lost }); toast(s.sensor_lost ? `${s.name}: telemetry restored` : `${s.name}: telemetry cut. It turns stale in a few minutes and is excluded from placement.`); }
  catch (e) { toast(e.message); }
});
$$("[data-wi]").forEach((b) => b.addEventListener("click", async () => {
  try {
    const w = await api("/api/whatif", { kind: b.dataset.wi, server_id: drawerId, magnitude: b.dataset.wi === "runaway" ? Number($("#rw-pct").value) || 60 : null });
    $("#drawer-whatif").hidden = false;
    $("#drawer-whatif").innerHTML = whatifHtml(w);
  } catch (e) { toast(e.message); }
}));
$("#btn-drain").addEventListener("click", async () => {
  const s = byId().get(drawerId);
  try { await api(`/api/servers/${drawerId}/maintenance`, { on: !s.maintenance, reason: "from server panel" }); toast(s.maintenance ? `${s.name} is back in service` : `${s.name} drained: it gets no new placements`); }
  catch (e) { toast(e.message); }
});
$("#btn-retire").addEventListener("click", async () => {
  const s = byId().get(drawerId);
  try { await api(`/api/servers/${drawerId}/retire`, {}); toast(`${s.name} retired`); closeDrawer(); }
  catch (e) { toast(e.message); }
});

async function loadHistory() {
  if (!drawerId) return;
  const id = drawerId;
  try {
    const h = await api(`/api/servers/${id}/history?minutes=180`);
    const rows = h.rows.filter((r) => r.temp !== null);
    if (rows.length > 1 && id === drawerId) {
      const m = rows.map((r) => r.minute);
      const last = m[m.length - 1];
      const fc = Object.entries(h.forecast).filter(([, v]) => v !== null).map(([k, v]) => [last + Number(k), v]);
      lineChart($("#chart-temp"), {
        x: m, threshold: h.threshold, unit: "°C",
        series: [
          { name: "Measured", values: rows.map((r) => r.temp), color: "var(--ink)" },
          { name: "5-min forecast as issued", values: rows.map((r) => r.pred5_issued), color: "var(--warm)", width: 1.5 },
        ],
        ahead: { name: "Forecast now", points: [[last, rows[rows.length - 1].temp], ...fc], color: "var(--accent)" },
        xLabel: (v) => minuteLabel(v),
      });
      lineChart($("#chart-load"), { x: m, unit: "%", yMin: 0, yMax: 100,
        series: [{ name: "CPU load", values: rows.map((r) => r.load), color: "var(--ink-2)" }], xLabel: (v) => minuteLabel(v) });
    } else if (id === drawerId) {
      $("#chart-temp").innerHTML = `<p class="muted">No readings yet.</p>`;
      $("#chart-load").innerHTML = "";
    }
  } catch (e) { /* fleet poll reports connection errors */ }
  if (drawerId === id) setTimeout(loadHistory, 3000);
}
function minuteLabel(minute) {
  if (!fleet) return "";
  const t = new Date(new Date(fleet.sim_time).getTime() + (minute - fleet.minute) * 60000);
  return t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/* ---------- SVG line chart with hover ---------- */
function lineChart(host, o) {
  const W = host.clientWidth || 560, H = host.clientHeight || 220;
  const pad = { l: 40, r: 12, t: 10, b: 22 };
  const xs = o.x;
  const xMax = o.ahead ? Math.max(...o.ahead.points.map((p) => p[0])) : xs[xs.length - 1];
  const vals = o.series.flatMap((s) => s.values).concat(o.ahead ? o.ahead.points.map((p) => p[1]) : []).filter((v) => v !== null && v !== undefined);
  if (o.threshold) vals.push(o.threshold);
  let lo = o.yMin ?? Math.floor(Math.min(...vals) - 2), hi = o.yMax ?? Math.ceil(Math.max(...vals) + 2);
  if (hi - lo < 4) hi = lo + 4;
  const X = (v) => pad.l + ((v - xs[0]) / Math.max(1, xMax - xs[0])) * (W - pad.l - pad.r);
  const Y = (v) => H - pad.b - ((v - lo) / (hi - lo)) * (H - pad.t - pad.b);
  const path = (pts) => {
    let d = "", pen = false;
    for (const p of pts) {
      if (p[1] === null || p[1] === undefined) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  const ticks = 4;
  let g = `<g class="grid">`;
  for (let i = 0; i <= ticks; i++) { const v = lo + ((hi - lo) * i) / ticks; g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(v)}" y2="${Y(v)}"/>`; }
  g += `</g><g class="axis">`;
  for (let i = 0; i <= ticks; i++) { const v = lo + ((hi - lo) * i) / ticks; g += `<text x="${pad.l - 6}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`; }
  for (let i = 0; i <= 3; i++) { const v = xs[0] + ((xMax - xs[0]) * i) / 3; g += `<text x="${X(v)}" y="${H - 4}" text-anchor="${i === 0 ? "start" : i === 3 ? "end" : "middle"}">${o.xLabel(Math.round(v))}</text>`; }
  g += `</g>`;
  if (o.threshold) g += `<line x1="${pad.l}" x2="${W - pad.r}" y1="${Y(o.threshold)}" y2="${Y(o.threshold)}" stroke="var(--hot)" stroke-width="1.5"/>
    <text x="${W - pad.r}" y="${Y(o.threshold) - 5}" text-anchor="end" fill="var(--hot)" font-size="11">limit ${f1(o.threshold)}°</text>`;
  for (const s of o.series) g += `<path d="${path(xs.map((x, i) => [x, s.values[i]]))}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2}" stroke-linejoin="round"/>`;
  if (o.ahead) {
    g += `<path d="${path(o.ahead.points)}" fill="none" stroke="${o.ahead.color}" stroke-width="2" stroke-dasharray="5 4"/>`;
    for (const p of o.ahead.points.slice(1)) g += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="4" fill="${o.ahead.color}" stroke="var(--panel)" stroke-width="2"/>`;
    g += `<line x1="${X(xs[xs.length - 1])}" x2="${X(xs[xs.length - 1])}" y1="${pad.t}" y2="${H - pad.b}" stroke="var(--line)" stroke-dasharray="2 3"/>`;
  }
  g += `<line class="cross" y1="${pad.t}" y2="${H - pad.b}" stroke="var(--ink-3)" visibility="hidden"/>`;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.series.map((s) => s.name).join(", "))}">${g}</svg><div class="tip"></div>`;
  const svg = host.querySelector("svg"), tipEl = host.querySelector(".tip"), cross = host.querySelector(".cross");
  svg.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const v = xs[0] + ((px - pad.l) / (W - pad.l - pad.r)) * (xMax - xs[0]);
    let idx = xs.findIndex((x) => x >= v), line;
    if (idx === -1 && o.ahead) {
      const p = o.ahead.points.reduce((a, b) => (Math.abs(b[0] - v) < Math.abs(a[0] - v) ? b : a));
      line = `${o.xLabel(p[0])}  ${o.ahead.name}: ${f1(p[1])}${o.unit}`;
      cross.setAttribute("x1", X(p[0])); cross.setAttribute("x2", X(p[0]));
    } else {
      if (idx === -1) idx = xs.length - 1;
      line = `${o.xLabel(xs[idx])}  ` + o.series.map((s) => `${s.name}: ${f1(s.values[idx])}${o.unit}`).join("  ");
      cross.setAttribute("x1", X(xs[idx])); cross.setAttribute("x2", X(xs[idx]));
    }
    cross.setAttribute("visibility", "visible");
    tipEl.style.display = "block";
    tipEl.textContent = line;
    tipEl.style.left = `${Math.min(r.width - 120, Math.max(120, e.clientX - r.left))}px`;
    tipEl.style.top = `${e.clientY - r.top}px`;
  });
  svg.addEventListener("mouseleave", () => { tipEl.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
}

/* ================= placement ================= */
$$("[data-preset]").forEach((b) => b.addEventListener("click", () => {
  const [cpu, mem, dur, name] = b.dataset.preset.split(",");
  $("#wl-cpu").value = cpu; $("#wl-mem").value = mem; $("#wl-dur").value = dur; $("#wl-name").value = name;
  $("#wl-error").textContent = "";
}));
["#wl-cpu", "#wl-mem", "#wl-dur", "#wl-name"].forEach((s) => $(s).addEventListener("input", () => ($("#wl-error").textContent = "")));
function readWorkload() {
  const cpu = Number($("#wl-cpu").value), mem = Number($("#wl-mem").value), dur = Number($("#wl-dur").value);
  if (!(cpu >= 1 && cpu <= 100)) return "CPU needed must be between 1 and 100%.";
  if (!(mem >= 1 && mem <= 4096)) return "Memory needed must be between 1 and 4096 GB.";
  if (!(dur >= 1 && dur <= 1440)) return "Run time must be between 1 and 1440 minutes.";
  return { cpu_pct: cpu, mem_gb: mem, duration_min: Math.round(dur), name: $("#wl-name").value.trim() || "Workload", site: $("#wl-site").value || null };
}
$("#wl-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!isOp()) return;
  if (!e.submitter || e.submitter.dataset.keepQueue !== "1") lastQueueId = null;
  const w = readWorkload();
  if (typeof w === "string") { $("#wl-error").textContent = w; return; }
  try {
    const r = await api("/api/recommend", w);
    lastWorkload = w; lastRecId = r.recommendation_id;
    renderRanking(r);
  } catch (err) { $("#wl-error").textContent = err.message; }
});
function renderRanking(r) {
  $("#rank-empty").hidden = true;
  const eligible = r.candidates.filter((c) => c.status !== "ineligible");
  const excluded = r.candidates.filter((c) => c.status === "ineligible");
  const head = $("#rank-head");
  head.hidden = false;
  head.textContent = `${r.workload.name}: ${r.workload.cpu_pct}% CPU, ${r.workload.mem_gb} GB, ${r.workload.duration_min} min${r.site ? ", " + r.site + " only" : ""}. ` +
    `${eligible.length} of ${r.candidates.length} servers can take it. Predicted with ${(MODEL_LABEL[r.model] || r.model).toLowerCase()}, ${r.horizon} min ahead. Showing the best 10.`;
  const list = $("#rank-list");
  list.innerHTML = "";
  if (!eligible.length) list.appendChild(el("li", "empty", "No server can take this workload right now. Try less CPU or memory, another site, or wait for jobs to finish."));
  const shown = eligible.slice(0, 10);
  const temps = shown.flatMap((c) => [c.current_temp, c.predicted_temp]).concat([r.threshold]);
  const lo = Math.floor(Math.min(...temps) - 4), hi = Math.ceil(Math.max(...temps) + 3);
  const X = (v) => ((v - lo) / (hi - lo)) * 100;
  shown.forEach((c, i) => {
    const li = el("li", `rank-item${i === 0 ? " top" : ""}`);
    const b = X(c.forecast_temp), cEnd = X(c.predicted_temp), now = X(c.current_temp);
    const trendSeg = c.forecast_temp >= c.current_temp ? `<span class="budget-seg budget-trend" style="left:${now}%;width:${b - now}%"></span>` : "";
    li.innerHTML = `
      <div class="rank-no">${c.rank}</div>
      <div>
        <div class="rank-title"><strong>${esc(c.name)}</strong><span class="loc">${esc(c.site)} · hall ${esc(c.hall)} · ${esc(c.rack)}</span><span class="badge ${c.status}">${STATUS_LABEL[c.status]}</span></div>
        <div class="budget" aria-hidden="true">
          <span class="budget-track"></span>
          <span class="budget-seg budget-now" style="left:0;width:${Math.min(now, b)}%"></span>${trendSeg}
          <span class="budget-seg budget-job" style="left:${b}%;width:${Math.max(0.8, cEnd - b)}%"></span>
          <span class="budget-limit" style="left:${X(r.threshold)}%"></span>
        </div>
        <div class="budget-legend"><span><i style="background:var(--ink-3)"></i>Now <b>${f1(c.current_temp)}°</b></span><span><i style="background:var(--ink-2)"></i>Heading to <b>${f1(c.forecast_temp)}°</b></span>
          <span><i style="background:var(--accent)"></i>With this job <b>${f1(c.predicted_temp)}°</b></span><span>Margin to limit <b>${f1(c.margin_c)}°</b></span>${c.upper_temp != null ? `<span title="Calibrated on held-out data: the real temperature stayed below this 9 times in 10">90% upper <b>${f1(c.upper_temp)}°</b></span>` : ""}</div>
        <p class="rank-why">${esc(c.explanation)}</p>
      </div>
      <div class="place-col"><button class="btn ${i === 0 ? "primary" : "ghost"} op-only" type="button">Place on ${esc(c.name)}</button></div>`;
    if (i > 0 || c.status === "not_recommended") {
      const ov = el("div", "override op-only");
      ov.hidden = true;
      ov.innerHTML = `<input maxlength="200" placeholder="Why this server instead of ${esc(shown[0].name)}?" aria-label="Override reason">`;
      li.children[1].appendChild(ov);
    }
    $(".place-col button", li).addEventListener("click", () => place(c, li, i === 0 && c.status !== "not_recommended"));
    list.appendChild(li);
  });
  const ex = $("#rank-excluded");
  ex.hidden = !excluded.length;
  $("summary", ex).textContent = `${excluded.length} server${excluded.length === 1 ? "" : "s"} excluded`;
  $("ul", ex).innerHTML = excluded.map((c) => `<li><strong>${esc(c.name)}</strong>: ${esc(c.reasons.join("; "))}</li>`).join("");
}
async function place(c, li, isTop) {
  const ov = $(".override", li);
  if (!isTop && ov && ov.hidden) {
    ov.hidden = false;
    $("input", ov).focus();
    $(".place-col button", li).textContent = "Confirm override";
    return;
  }
  const reason = ov ? $("input", ov).value.trim() : "";
  if (!isTop && !reason) { toast("Give a reason for choosing a server other than ThermalView's pick."); $("input", ov).focus(); return; }
  try {
    const body = { ...lastWorkload, server_id: c.server_id, recommendation_id: lastRecId, reason };
    const r = lastQueueId ? await api(`/api/queue/${lastQueueId}/approve`, { server_id: c.server_id, reason })
      : await api("/api/place", body);
    if (lastQueueId) lastQueueId = null;
    li.classList.add("placed");
    toast(`Placed ${lastWorkload.name} on ${r.server_name}${r.override ? " (override, reason logged)" : ""}. It is in the audit log.`);
    $(".place-col button", li).disabled = true;
    $(".place-col button", li).textContent = "Placed";
    if (ov) ov.hidden = true;
  } catch (err) { toast(err.message); }
}

/* ================= manual mode queue ================= */
function renderQueue(d) {
  $$("[data-mode]").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === d.mode)));
  $("#q-count").textContent = d.queue.length ? `(${d.queue.length})` : "";
  $("#q-explain").textContent = d.mode === "manual"
    ? "Manual approval: every arriving job waits here with ThermalView's current suggestion. Approve it, rank it yourself to choose another server, or reject it. Undecided jobs expire after 15 minutes. Everything is audited."
    : "Automatic: the scheduler places each arriving job on ThermalView's pick. A job only waits here if every eligible server would exceed the limit.";
  if (document.activeElement && document.activeElement.closest("#q-list")) return;   // don't redraw while typing
  $("#q-list").innerHTML = d.queue.length ? d.queue.map((q) => `<li data-q="${q.queue_id}">
    <div><b>${esc(q.name)}</b> <span class="muted">${q.cpu_pct}% CPU · ${q.mem_gb} GB · ${q.duration_min} min · waiting ${q.waiting_min} min, expires in ${q.expires_in_min}</span>
      <div class="action-hint">${q.suggested ? `Suggested: <b>${esc(q.suggested.name)}</b> (${esc(STATUS_LABEL[q.suggested.status] || q.suggested.status)}, ${f1(q.suggested.predicted_temp)} °C in ${d.horizon} min)` : "No server can take it right now."} · held because ${esc(q.why)}</div></div>
    <div class="q-acts op-only">${q.suggested ? `<button class="btn primary small" data-qa="approve">Approve</button>` : ""}
      <button class="btn ghost small" data-qa="rank">Rank it myself</button>
      <input placeholder="reason to reject" maxlength="200" aria-label="Reason to reject"><button class="btn ghost small" data-qa="reject">Reject</button></div></li>`).join("")
    : `<li class="q-empty">Nothing waiting.</li>`;
}
$$("[data-mode]").forEach((b) => b.addEventListener("click", () => api("/api/control", { action: "mode", value: b.dataset.mode })
  .then(() => toast(b.dataset.mode === "manual" ? "Manual approval: arriving jobs now wait for you" : "Automatic placement on")).catch((e) => toast(e.message))));
$("#q-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-qa]");
  if (!b) return;
  const li = b.closest("[data-q]"), qid = Number(li.dataset.q), q = fleet.queue.find((x) => x.queue_id === qid);
  try {
    if (b.dataset.qa === "approve") {
      const r = await api(`/api/queue/${qid}/approve`, {});
      toast(`${q.name} placed on ${r.server_name}`);
    } else if (b.dataset.qa === "reject") {
      await api(`/api/queue/${qid}/reject`, { reason: $("input", li).value });
      toast(`${q.name} rejected`);
    } else {
      $("#wl-cpu").value = q.cpu_pct; $("#wl-mem").value = q.mem_gb; $("#wl-dur").value = q.duration_min; $("#wl-name").value = q.name; $("#wl-site").value = "";
      const w = readWorkload();
      const r = await api("/api/recommend", w);
      lastWorkload = w; lastRecId = r.recommendation_id; lastQueueId = qid;
      renderRanking(r);
      toast(`Ranking for ${q.name}: choose a server below. Picking anything but the top needs a reason.`);
    }
    document.activeElement?.blur();
  } catch (err) { toast(err.message); }
});

/* ================= alerts ================= */
let alertStatus = "active";
$$(".seg button").forEach((b) => b.addEventListener("click", () => {
  alertStatus = b.dataset.st;
  $$(".seg button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  loadAlerts();
}));
async function loadAlerts() {
  let r;
  try { r = await api(`/api/alerts?status=${alertStatus}`); } catch (e) { return; }

  $("#al-table tbody").innerHTML = r.alerts.length ? r.alerts.map((a) => `<tr>
    <td><span class="sev-pill ${a.severity}">${a.severity}</span>${a.escalated ? '<span class="esc">ESCALATED</span>' : ""}</td>
    <td><strong>${esc(a.server_name)}</strong><div class="loc">${esc(a.site)}</div></td>
    <td>${esc(KIND_LABEL(a.kind))}<div class="muted">${esc(a.text)}${a.peak_value !== null && !["stale"].includes(a.kind) && !a.kind.startsWith("hall") ? ` · peak ${f1(a.peak_value)} °C` : ""}</div>
      ${a.status !== "resolved" && a.action ? `<div class="action-hint">→ ${esc(a.action)}</div>` : ""}</td>
    <td class="mono">${fmtTime(a.opened_ts)}${a.resolved_ts ? `<br><span class="muted">closed ${fmtTime(a.resolved_ts)}</span>` : ""}</td>
    <td><span class="st-pill ${a.status}">${a.status}</span>${a.acked_by ? `<div class="muted">by ${esc(a.acked_by)}${a.ack_note ? `: ${esc(a.ack_note)}` : ""}</div>` : ""}</td>
    <td>${a.status === "open" && isOp() ? `<form class="ack-form" data-id="${a.alert_id}"><input placeholder="note (optional)" maxlength="200" aria-label="Acknowledgement note"><button class="btn ghost small" type="submit">Acknowledge</button></form>` : ""}</td>
  </tr>`).join("") : `<tr><td colspan="6" class="muted">No ${alertStatus === "all" ? "" : alertStatus + " "}alerts.</td></tr>`;
}
$("#al-table").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target.closest(".ack-form");
  try { await api(`/api/alerts/${f.dataset.id}/ack`, { note: $("input", f).value }); toast("Alert acknowledged"); f.closest("tr").querySelector("input")?.blur(); loadAlerts(); }
  catch (err) { toast(err.message); }
});

/* ================= inventory ================= */
let invSites = [];
function fillHalls() {
  const site = fleet?.sites.find((s) => s.code === $("#add-site").value);
  const v = $("#add-hall").value;
  $("#add-hall").innerHTML = (site?.halls || []).map((h) => `<option>${esc(h.name)}</option>`).join("");
  if ([...$("#add-hall").options].some((o) => o.value === v)) $("#add-hall").value = v;
  fillRacks();
}
function fillRacks() {
  const site = fleet?.sites.find((s) => s.code === $("#add-site").value);
  const hall = site?.halls.find((h) => h.name === $("#add-hall").value);
  $("#rack-list").innerHTML = (hall?.racks || []).map((r) => `<option value="${esc(r.rack)}">${esc(r.rack)} (${r.servers.length} servers)</option>`).join("");
  if (!$("#add-rack").value.startsWith($("#add-hall").value)) $("#add-rack").value = hall?.racks[0]?.rack || `${$("#add-hall").value}01`;
}
$("#add-site").addEventListener("change", fillHalls);
$("#add-hall").addEventListener("change", fillRacks);
$("#add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#add-error").textContent = "";
  try {
    const r = await api("/api/servers", { site: $("#add-site").value, hall: $("#add-hall").value, rack: $("#add-rack").value.trim(),
      name: $("#add-name").value.trim() || null, external: $("#add-kind").value === "external" });
    toast(r.kind === "external" ? `${r.name} registered. It will show "no telemetry" until an agent sends readings.` : `${r.name} added and in service`);
    $("#add-name").value = "";
    loadInventory();
  } catch (err) { $("#add-error").textContent = err.message; }
});
async function loadInventory() {
  let r;
  try { r = await api("/api/inventory"); } catch (e) { return; }
  invSites = r.sites;
  const live = byId();
  $("#inv-table tbody").innerHTML = r.servers.map((s) => {
    const l = live.get(s.server_id);
    const acts = s.status === "retired" || !isOp() ? "" :
      `<button class="btn ghost small" data-act="${s.status === "maintenance" ? "undrain" : "drain"}" data-id="${s.server_id}">${s.status === "maintenance" ? "Return to service" : "Drain"}</button>` +
      (s.status === "maintenance" ? ` <button class="btn ghost small" data-act="retire" data-id="${s.server_id}" ${l && l.n_jobs ? `disabled title="${l.n_jobs} job(s) still running"` : ""}>Retire</button>` : "");
    return `<tr class="${s.status}"><td><strong>${esc(s.name)}</strong></td><td>${esc(s.site)}</td><td>${esc(s.hall)}</td>
      <td class="mono">${esc(s.rack)} / ${s.slot}</td><td>${s.kind === "external" ? "external (API)" : "simulated"}</td>
      <td class="mono">${esc(s.source_node)}</td><td><span class="st-pill">${esc(s.status)}</span></td>
      <td class="mono">${fmtWall(s.status === "retired" ? s.retired_at : s.added_at)}</td><td>${acts}</td></tr>`;
  }).join("");
}
$("#inv-table").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const id = Number(b.dataset.id);
  try {
    if (b.dataset.act === "retire") await api(`/api/servers/${id}/retire`, {});
    else await api(`/api/servers/${id}/maintenance`, { on: b.dataset.act === "drain", reason: "from inventory" });
    toast(b.dataset.act === "retire" ? "Server retired" : b.dataset.act === "drain" ? "Server drained" : "Server back in service");
    loadInventory();
  } catch (err) { toast(err.message); }
});

/* ================= users ================= */
async function apiSend(method, path, body) {
  const res = await fetch(path, { method, headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401) { showLogin(); throw new Error("signed out"); }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}
async function loadUsers() {
  if (!isOp()) return;
  let r;
  try { r = await api("/api/users"); } catch (e) { return; }
  $("#us-table tbody").innerHTML = r.users.map((u) => `<tr><td><strong>${esc(u.username)}</strong>${me && u.username === me.username ? ' <span class="muted">(you)</span>' : ""}</td>
    <td>${esc(u.display_name || "")}</td><td><span class="st-pill">${esc(u.role)}</span></td><td class="muted">${esc(u.source)}</td>
    <td><button class="btn ghost small" data-uact="reset" data-u="${esc(u.username)}">Reset password</button>
    ${me && u.username === me.username ? "" : `<button class="btn ghost small" data-uact="remove" data-u="${esc(u.username)}">Remove</button>`}</td></tr>`).join("");
}
$("#user-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#nu-error").textContent = "";
  try {
    await api("/api/users", { username: $("#nu-name").value, display_name: $("#nu-display").value, role: $("#nu-role").value, password: $("#nu-pass").value });
    toast(`User ${$("#nu-name").value} added`);
    $("#user-form").reset();
    loadUsers();
  } catch (err) { $("#nu-error").textContent = err.message; }
});
$("#us-table").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-uact]");
  if (!b) return;
  const u = b.dataset.u;
  try {
    if (b.dataset.uact === "remove") {
      if (!window.confirm(`Remove user ${u}? They are signed out at once.`)) return;
      await apiSend("DELETE", `/api/users/${encodeURIComponent(u)}`);
      toast(`User ${u} removed`);
    } else {
      const pw = window.prompt(`New password for ${u} (at least 8 characters):`);
      if (!pw) return;
      await api(`/api/users/${encodeURIComponent(u)}/password`, { password: pw });
      toast(`Password for ${u} changed; their other sessions were signed out`);
    }
    loadUsers();
  } catch (err) { toast(err.message); }
});
$("#mypw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#mp-error").textContent = "";
  try {
    await api("/api/me/password", { current: $("#mp-cur").value, password: $("#mp-new").value });
    toast("Your password was changed");
    $("#mypw-form").reset();
  } catch (err) { $("#mp-error").textContent = err.message; }
});

/* ================= audit ================= */
const seenUsers = new Set(), seenActions = new Set();
async function loadAudit() {
  const q = new URLSearchParams({ limit: 300, user: $("#au-user").value, action: $("#au-action").value, scheduler: $("#au-sched").checked ? "1" : "0" });
  let r;
  try { r = await api(`/api/audit?${q}`); } catch (e) { return; }
  let changed = false;
  for (const e of r.entries) {
    if (!seenUsers.has(e.username)) { seenUsers.add(e.username); changed = true; }
    if (!seenActions.has(e.action)) { seenActions.add(e.action); changed = true; }
  }
  for (const a of Object.keys(ACTION_LABEL)) if (!seenActions.has(a)) { seenActions.add(a); changed = true; }
  if (changed) {
    const u = $("#au-user").value, a = $("#au-action").value;
    $("#au-user").innerHTML = `<option value="">All users</option>` + [...seenUsers].sort().map((x) => `<option>${esc(x)}</option>`).join("");
    $("#au-action").innerHTML = `<option value="">All actions</option>` + [...seenActions].sort().map((x) => `<option value="${esc(x)}">${esc(ACTION_LABEL[x] || x)}</option>`).join("");
    $("#au-user").value = u; $("#au-action").value = a;
  }
  $("#au-table tbody").innerHTML = r.entries.length ? r.entries.map((e) => {
    let d = {};
    try { d = JSON.parse(e.detail || "{}"); } catch (x) { d = {}; }
    const detail = Object.entries(d).filter(([, v]) => v !== null && v !== "").map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : v}`).join("  ");
    return `<tr><td class="mono">${fmtWall(e.wall_ts)}</td><td class="mono">${fmtTime(e.sim_ts)}</td><td>${esc(e.username)}${e.role && e.role !== e.username && e.role !== "-" ? ` <span class="muted">(${esc(e.role)})</span>` : ""}</td>
      <td>${esc(ACTION_LABEL[e.action] || e.action)}</td><td>${esc(e.target)}</td><td class="detail">${esc(detail)}</td></tr>`;
  }).join("") : `<tr><td colspan="6" class="muted">Nothing matches.</td></tr>`;
}
["#au-user", "#au-action", "#au-sched"].forEach((s) => $(s).addEventListener("change", loadAudit));

/* ================= results ================= */
async function loadResults() {
  const body = $("#results-body");
  body.innerHTML = `<p class="muted">Loading…</p>`;
  let r;
  try { r = await api("/api/results"); } catch (e) { body.innerHTML = `<p>${esc(e.message)}</p>`; return; }
  const parts = [];
  if (r.data_source !== "M100") parts.push(`<p class="banner">These results come from ${esc(r.data_source)} data, not M100. Run the pipeline on M100 before quoting any number.</p>`);
  else if (r.forecast_summary && r.eval_month) {
    const d = new Date(r.forecast_summary.dataset.start), m = `${String(d.getUTCFullYear()).slice(2)}-${String(d.getUTCMonth() + 1).padStart(2, "0")}`;
    if (m !== r.eval_month) parts.push(`<p class="banner">These are M100 results for month ${m}, the development month. The paper's results come from the evaluation month ${esc(r.eval_month)}: run the pipeline on it before quoting numbers.</p>`);
  }
  const fm = r.forecast_metrics, fs = r.forecast_summary;
  const ms = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
  const pc = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);
  const verdict = (lo, hi) => (lo === null || lo === undefined ? "not clearly different from" : lo > 0 ? "clearly better than" : hi < 0 ? "clearly worse than" : "not clearly different from");
  if (fm && fs) {
    const h = fs.config.primary_horizon, chosen = fs.chosen_model_by_horizon[String(h)];
    const rows = (split) => fm.filter((m) => m.split === split && m.horizon_min === h);
    const seen = rows("test_seen"), unseen = rows("test_unseen");
    const tr = seen.map((m) => {
      const u = unseen.find((x) => x.model === m.model) || {};
      const ci = m.skill_ci_low === null || m.skill_ci_low === undefined ? "" : ` <span class="muted">(${(m.skill_ci_low * 100).toFixed(1)} to ${(m.skill_ci_high * 100).toFixed(1)})</span>`;
      const cls = m.skill_vs_persistence > 0 ? "better" : m.skill_vs_persistence < 0 ? "worse" : "";
      return `<tr class="${m.model === chosen ? "chosen" : ""}"><td>${MODEL_LABEL[m.model] || esc(m.model)}</td><td class="num">${ms(m.mae, 3)}</td>
        <td class="num">${ms(u.mae, 3)}</td><td class="num">${ms(m.rmse, 3)}</td>
        <td class="num ${cls}">${(m.skill_vs_persistence * 100).toFixed(1)}%${ci}</td><td class="num">${ms(m.mae_heating_events)}</td></tr>`;
    }).join("");
    const ds = fs.dataset || {};
    const best = seen.find((m) => m.model === chosen) || {}, bestU = unseen.find((m) => m.model === chosen) || {};
    const longer = [15, 30].map((hh) => {
      const ch = fs.chosen_model_by_horizon[String(hh)], m = fm.find((x) => x.model === ch && x.horizon_min === hh && x.split === "test_seen") || {};
      return ch === "persistence" ? `at ${hh} minutes no model did better than "no change" on validation, so "no change" is used`
        : `at ${hh} minutes ${MODEL_LABEL[ch] || ch} is ${pc(m.skill_vs_persistence)} better (${verdict(m.skill_ci_low, m.skill_ci_high)} "no change")`;
    }).join("; ");
    const plainF = chosen === "persistence" ? `<p class="plain">In plain words: at ${h} minutes no model beat simply assuming the temperature stays the same, so the console uses "no change".</p>`
      : `<p class="plain">In plain words: five minutes ahead, the ${(MODEL_LABEL[chosen] || chosen).toLowerCase()} makes <b>${pc(best.skill_vs_persistence)} less error</b> than assuming the temperature stays the same
      (95% CI ${pc(best.skill_ci_low)} to ${pc(best.skill_ci_high)}), so it is <b>${verdict(best.skill_ci_low, best.skill_ci_high)}</b> that simple rule. On servers it never saw in training the gain is ${pc(bestU.skill_vs_persistence)}.
      Further ahead the gain fades: ${longer}. These nodes reach a new temperature within minutes of a load change, so beyond a few minutes the future depends on jobs that have not started yet.</p>`;
    parts.push(`<section class="res-section"><h2>Forecasting, ${h} minutes ahead</h2>${plainF}
      <p>${ds.nodes || "?"} nodes, ${ds.days || "?"} days (${esc((ds.start || "").slice(0, 16))} to ${esc((ds.end || "").slice(0, 16))}). Trained on the first ${Math.round(fs.config.train_frac * 100)}% of the time span, tuned on the next ${Math.round(fs.config.val_frac * 100)}%, tested once on the rest.
      "Unseen nodes" were never used for training. The deployed model was chosen on validation data, not on these test numbers. Errors in °C.</p>
      <table class="data"><thead><tr><th>Model</th><th class="num">MAE, seen nodes</th><th class="num">MAE, unseen nodes</th><th class="num">RMSE</th>
      <th class="num">Skill vs persistence (95% CI)</th><th class="num">MAE while heating</th></tr></thead><tbody>${tr}</tbody></table></section>`);
    const hs = [...new Set(fm.map((m) => m.horizon_min))].sort((a, b) => a - b);
    const models = [...new Set(fm.map((m) => m.model))];
    const hr = models.map((mo) => `<tr><td>${MODEL_LABEL[mo] || esc(mo)}</td>${hs.map((hh) => {
      const m = fm.find((x) => x.model === mo && x.horizon_min === hh && x.split === "test_seen");
      return `<td class="num">${m ? ms(m.mae, 3) : "–"}</td>`;
    }).join("")}</tr>`).join("");
    parts.push(`<section class="res-section"><h2>Error by horizon (seen nodes, MAE °C)</h2>
      <table class="data"><thead><tr><th>Model</th>${hs.map((hh) => `<th class="num">${hh} min</th>`).join("")}</tr></thead><tbody>${hr}</tbody></table></section>`);
  } else {
    parts.push(`<section class="res-section"><h2>Forecasting</h2><p>No results yet. Run <code>python -m thermalview forecast</code>.</p></section>`);
  }
  if (r.simulator_fidelity) {
    parts.push(`<section class="res-section"><h2>Is the simulator believable?</h2>
      <p>On the unseen test period, the heat model was given each node's real future load and predicted the temperature change it causes; that was compared with what the node actually did.</p>
      ${(() => { const f5 = r.simulator_fidelity.find((f) => f.horizon_min === 5), f30 = r.simulator_fidelity.find((f) => f.horizon_min === 30) || f5;
        return f5 ? `<p class="plain">In plain words: knowing how load changes, the heat model is off by <b>${ms(f5.sim_mae)} °C</b> after 5 minutes, against ${ms(f5.persistence_mae)} °C for assuming no change${f30 !== f5 ? `, and ${ms(f30.sim_mae)} against ${ms(f30.persistence_mae)} °C after 30 minutes` : ""}.
        ${f5.sim_mae < f5.persistence_mae ? "So the simulator reacts to extra load the way the real nodes did, which is what makes the placement comparison meaningful." : "It does not beat assuming no change, so treat the placement comparison with caution."}</p>` : ""; })()}
      <table class="data"><thead><tr><th>Horizon</th><th class="num">Heat model MAE</th><th class="num">Persistence MAE</th><th class="num">Checks</th></tr></thead><tbody>
      ${r.simulator_fidelity.map((f) => `<tr><td>${f.horizon_min} min</td><td class="num">${ms(f.sim_mae)}</td><td class="num">${ms(f.persistence_mae)}</td><td class="num">${f.n}</td></tr>`).join("")}
      </tbody></table></section>`);
  }
  const ps = r.placement_summary, pm = r.placement_meta;
  if (ps && pm) {
    const utils = [...new Set(ps.map((p) => p.load_factor ?? null))];
    const tables = utils.map((u) => {
      const sub = ps.filter((p) => (p.load_factor ?? null) === u);
      const policies = [...new Set(sub.map((p) => p.policy))];
      const get = (p, m) => sub.find((x) => x.policy === p && x.metric === m) || {};
      const rows = policies.map((p) => {
        const c = (m) => (sub.some((x) => x.metric === "caused_violation_pct") ? `caused_${m}` : m);
        const v = get(p, c("violation_pct")), ex = get(p, c("excess_degree_min")), pk = get(p, "violation_pct"), rj = get(p, "rejected");
        const diff = v.diff_vs_coolest_now === undefined || v.diff_vs_coolest_now === null ? "reference" :
          `${v.diff_vs_coolest_now > 0 ? "+" : ""}${ms(v.diff_vs_coolest_now, 3)} <span class="muted">(${ms(v.diff_ci_low, 3)} to ${ms(v.diff_ci_high, 3)}, p=${ms(v.wilcoxon_p, 3)})</span>`;
        return `<tr class="${p === "thermalview" ? "chosen" : ""}"><td>${POLICY_LABEL[p] || esc(p)}</td><td class="num">${ms(v.mean, 3)} ± ${ms(v.std, 3)}</td>
          <td class="num">${ms(ex.mean, 0)}</td><td class="num">${ms(pk.mean, 2)}</td><td class="num">${ms(rj.mean, 1)}</td><td class="num">${diff}</td></tr>`;
      }).join("");
      const vm = sub.some((x) => x.metric === "caused_violation_pct") ? "caused_violation_pct" : "violation_pct";
      const tv = get("thermalview", vm), cn = get("coolest_now", vm), tr = get("thermalview", "rejected"), cr = get("coolest_now", "rejected");
      const rel = cn.mean ? (tv.mean - cn.mean) / cn.mean : null;
      const sig = tv.diff_ci_high !== undefined && tv.diff_ci_high !== null && tv.diff_ci_high < 0 ? "and the 95% interval of the paired difference is entirely below zero"
        : tv.diff_ci_low !== undefined && tv.diff_ci_low > 0 ? "but this is worse, and the interval is entirely above zero" : "but the 95% interval includes zero, so the difference is not certain";
      const plainP = `<p class="plain">In plain words: the jobs ThermalView placed pushed servers over the limit for <b>${ms(tv.mean, 3)}%</b> of server-minutes, against ${ms(cn.mean, 3)}% when each job goes to the server that is coolest right now
        (${rel === null ? "–" : `${Math.abs(rel * 100).toFixed(0)}% ${rel < 0 ? "less" : "more"}`}), ${sig}. It turned away ${ms(tr.mean, 1)} jobs per run against ${ms(cr.mean, 1)}${(tr.mean || 0) > (cr.mean || 0) + 0.5 ? ", so part of the difference may come from turning work away" : ", so it is not buying safety by rejecting work"}.</p>`;
      return `${u === null ? "" : `<div class="res-util">Mean load ${u.toFixed(2)}× the recorded load</div>`}${plainP}
        <table class="data"><thead><tr><th>Policy</th><th class="num">% time over limit caused by placement</th><th class="num">°C·min caused</th><th class="num">Total % over limit</th>
        <th class="num">Jobs rejected</th><th class="num">Change in caused time vs coolest-now (95% CI, Wilcoxon p)</th></tr></thead><tbody>${rows}</tbody></table>`;
    }).join("");
    parts.push(`<section class="res-section"><h2>Placement in the simulated fleet</h2>
      <p>${pm.config.evaluation.seeds.length} runs of ${pm.config.evaluation.sim_days} simulated days on ${pm.fleet_nodes.length} servers at each load level. Every policy saw the same jobs, fleet and noise in a given run.
      Limit ${f1(pm.threshold_c)} °C. The main measure is time over the limit <i>caused by placement</i>: minutes when a server is over the limit but its real recording at that moment was not.
      Time the real servers already spent over the limit is the same for every policy, so it is shown only in the total. Read it together with rejected jobs: a policy can look safer by turning work away.
      This is simulation evidence, calibrated from real data.</p>${tables}</section>`);
  }
  if (fm && fs && fm.length && fm[0].interval_coverage !== undefined && fm[0].interval_coverage !== null) {
    const h = fs.config.primary_horizon, ch = fs.chosen_model_by_horizon[String(h)];
    const m = fm.find((x) => x.model === ch && x.horizon_min === h && x.split === "test_seen") || {};
    const p = fm.find((x) => x.model === "persistence" && x.horizon_min === h && x.split === "test_seen") || {};
    parts.push(`<section class="res-section"><h2>How sure is the forecast?</h2>
      <p class="plain">In plain words: each forecast comes with a range of ±${ms(m.interval_q)} °C, set so that the real temperature should land inside it 9 times in 10.
      On the test period it did <b>${pc(m.interval_coverage)}</b> of the time. The forecast itself rarely predicts a crossing of the limit (${pc(m.crossing_recall)} of crossings), because most crossings are brief.
      Alarming when the <i>top</i> of the range reaches the limit catches <b>${pc(m.upper_crossing_recall)}</b> of crossings; about ${m.upper_crossing_precision ? Math.round(1 / m.upper_crossing_precision) : "–"} alarms per real crossing, with the alarm on ${pc(m.upper_alarm_rate)} of the time
      (assuming no change with its own range: ${pc(p.upper_crossing_recall)} caught, alarm on ${pc(p.upper_alarm_rate)}).</p></section>`);
  }
  if (r.transfer_metrics && r.transfer_metrics.length) {
    const ts = r.transfer_summary || {}, td = ts.test_dataset || {};
    const grp = { all: "all nodes", trained_nodes: "nodes it trained on", new_nodes: "nodes it never saw" };
    const rows = r.transfer_metrics.map((t) => `<tr><td>${t.horizon_min} min</td><td>${grp[t.group] || esc(t.group)}</td><td class="num">${t.nodes}</td>
      <td class="num">${ms(t.persistence_mae, 3)}</td><td class="num">${ms(t.mae, 3)}</td><td class="num ${t.skill_vs_persistence > 0 ? "better" : "worse"}">${pc(t.skill_vs_persistence)} <span class="muted">(${pc(t.skill_ci_low)} to ${pc(t.skill_ci_high)})</span></td><td class="num">${pc(t.interval_coverage)}</td></tr>`).join("");
    const a5 = r.transfer_metrics.find((t) => t.horizon_min === 5 && t.group === "all") || {};
    parts.push(`<section class="res-section"><h2>A month the model never saw</h2>
      <p class="plain">In plain words: the February models were frozen and run on January 2021 (${esc(String(td.start || "").slice(0, 10))} to ${esc(String(td.end || "").slice(0, 10))}, ${ts.n_test_nodes || "?"} nodes). Five minutes ahead they still made
      <b>${pc(a5.skill_vs_persistence)} less error</b> than assuming no change, ${verdict(a5.skill_ci_low, a5.skill_ci_high)} it. Nothing was retrained.</p>
      <table class="data"><thead><tr><th>Horizon</th><th>Group</th><th class="num">Nodes</th><th class="num">Persistence MAE</th><th class="num">Model MAE</th><th class="num">Skill (95% CI)</th><th class="num">Range coverage</th></tr></thead><tbody>${rows}</tbody></table></section>`);
  }
  if (r.sensitivity_summary && r.sensitivity_summary.length) {
    const lim = r.sensitivity_summary.filter((x) => x.setting === "limit_percentile" && x.metric === "caused_violation_pct");
    const vals = [...new Set(lim.map((x) => x.value))].sort((a, b) => a - b);
    const rows = vals.map((v) => {
      const g = (p) => lim.find((x) => x.value === v && x.policy === p) || {};
      const tv = g("thermalview"), cn = g("coolest_now"), pp = g("predictive_persist");
      return `<tr><td>${v}th percentile</td><td class="num">${f1(tv.threshold_c)}</td><td class="num">${ms(cn.mean)}</td><td class="num">${ms(tv.mean)}</td>
        <td class="num ${tv.diff_ci_high < 0 ? "better" : ""}">${ms(tv.diff_vs_coolest_now)} <span class="muted">(${ms(tv.diff_ci_low)} to ${ms(tv.diff_ci_high)})</span></td>
        <td class="num">${ms(pp.diff_vs_coolest_now)} <span class="muted">(${ms(pp.diff_ci_low)} to ${ms(pp.diff_ci_high)})</span></td></tr>`;
    }).join("");
    const allBetter = vals.every((v) => (lim.find((x) => x.value === v && x.policy === "thermalview") || {}).diff_ci_high < 0);
    parts.push(`<section class="res-section"><h2>Does the result depend on where the limit is set?</h2>
      <p class="plain">In plain words: the thermal limit was moved from the ${vals[0]}th to the ${vals[vals.length - 1]}th percentile of training temperatures. ThermalView caused less time over the limit than coolest-now
      ${allBetter ? "<b>at every setting</b>, with intervals below zero" : "at some settings but not all"}. The safety margin is not varied: it changes the labels, never which server is picked.</p>
      <table class="data"><thead><tr><th>Limit</th><th class="num">°C</th><th class="num">Coolest-now caused %</th><th class="num">ThermalView caused %</th><th class="num">ThermalView change (95% CI)</th><th class="num">Ablation change (95% CI)</th></tr></thead><tbody>${rows}</tbody></table></section>`);
  }
  if (r.job_event_heat_model_transfer && r.job_event_heat_model_transfer.length) {
    const js = r.job_event_summary_transfer || {}, hm = r.job_event_heat_model_transfer;
    const h15 = hm.find((x) => x.horizon_min === 15) || hm[0];
    const fc = r.job_event_forecast_transfer || [];
    const f5a = fc.find((x) => x.horizon_min === 5 && x.minutes === "after_job_start") || {}, f5o = fc.find((x) => x.horizon_min === 5 && x.minutes === "other_minutes") || {};
    parts.push(`<section class="res-section"><h2>Real job starts (Slurm job table, January)</h2>
      <p class="plain">In plain words: across ${js.n_starts ? js.n_starts.toLocaleString() : "?"} real job starts, CPU temperature rose about <b>${ms(h15.mean_actual_change, 1)} °C in 15 minutes</b> while measured CPU load rose only ${ms(h15.mean_load_step, 0)} points.
      The heat model, which only sees CPU load, predicts ${ms(h15.mean_model_change, 1)} °C of that rise: M100 jobs heat the node through its GPUs as well, which CPU load does not show.
      ${f5a.n ? `In the 15 minutes after a start the forecast is ${pc(f5a.skill)} better than no change (MAE ${ms(f5a.model_mae)} °C), against ${pc(f5o.skill)} at other times: the scheduler knows a job is starting; the node's own history does not.` : ""}</p>
      <table class="data"><thead><tr><th>Horizon</th><th class="num">Starts</th><th class="num">Actual rise</th><th class="num">Heat model rise</th><th class="num">Persistence MAE</th><th class="num">Heat model MAE</th></tr></thead><tbody>
      ${hm.map((x) => `<tr><td>${x.horizon_min} min</td><td class="num">${x.n}</td><td class="num">${ms(x.mean_actual_change)}</td><td class="num">${ms(x.mean_model_change)}</td><td class="num">${ms(x.persistence_mae)}</td><td class="num">${ms(x.sim_mae)}</td></tr>`).join("")}
      </tbody></table></section>`);
  }
  if (r.figures && r.figures.length) {
    parts.push(`<section class="res-section"><h2>Figures for the paper</h2><div class="figs">${r.figures.map((f) =>
      `<figure><img src="/figures/${encodeURIComponent(f)}" alt="${esc(f)}" loading="lazy"><figcaption>results/figures/${esc(f)}</figcaption></figure>`).join("")}</div></section>`);
  }
  body.innerHTML = parts.join("");
}

/* ================= API docs ================= */
const mdInline = (s) => esc(s).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
function md(text) {
  const inline = mdInline;
  const out = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (/^\s*$/.test(l)) continue;
    if (l.startsWith("### ")) { out.push(`<h3>${inline(l.slice(4))}</h3>`); continue; }
    if (l.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) { rows.push(lines[i]); i++; }
      i--;
      const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
      const body = rows.slice(2).map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("");
      out.push(`<table class="data"><thead><tr>${cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table>`);
      continue;
    }
    if (/^\s*[*-] /.test(l)) {
      const items = [];
      while (i < lines.length && /^\s*([*-] |\s{2,}\S)/.test(lines[i])) {
        if (/^\s*[*-] /.test(lines[i])) items.push(lines[i].replace(/^\s*[*-] /, "")); else items[items.length - 1] += " " + lines[i].trim();
        i++;
      }
      i--;
      out.push(`<ul>${items.map((x) => `<li>${inline(x)}</li>`).join("")}</ul>`);
      continue;
    }
    const para = [l];
    while (i + 1 < lines.length && lines[i + 1].trim() && !/^(\||###|\s*[*-] )/.test(lines[i + 1])) para.push(lines[++i]);
    out.push(`<p>${inline(para.join(" "))}</p>`);
  }
  return out.join("");
}
let apiLoaded = false;
async function loadApi() {
  if (apiLoaded) return;
  let s;
  try { s = await api("/api/v1/spec"); } catch (e) { return; }
  apiLoaded = true;
  const rows = s.endpoints.map((e) => `<tr><td><span class="method ${e.method}">${e.method}</span></td><td class="mono">${esc(e.path)}</td><td>${esc(e.auth_label)}</td><td>${mdInline(e.summary)}</td></tr>`).join("");
  const ex = s.endpoints.filter((e) => e.body).map((e) => `<h3><span class="method ${e.method}">${e.method}</span> <span class="mono">${esc(e.path)}</span></h3>
    <pre>${esc(JSON.stringify(e.body, null, 2))}</pre>${typeof e.response === "object" ? `<pre>${esc(JSON.stringify(e.response, null, 2))}</pre>` : ""}`).join("");
  $("#api-body").innerHTML = md(s.intro) + `<h2>Endpoints</h2><div class="table-wrap"><table class="data"><thead><tr><th>Method</th><th>Path</th><th>Access</th><th>What it does</th></tr></thead><tbody>${rows}</tbody></table></div>` +
    md(s.ingest_notes) + `<h2>Request and response examples</h2>${ex}`;
}

/* ================= boot ================= */
(async () => {
  try { start(await api("/api/me")); } catch (e) { if (e.message !== "signed out") showLogin(); }
})();
