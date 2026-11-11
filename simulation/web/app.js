/*
 * Plant digital twin - control room client.
 * Vanilla JS, no external dependencies.  Data arrives as Server-Sent Events
 * from the monitor_dashboard node (one 'static' layout event, then JSON
 * snapshots at 5 Hz); commands go out as POST /api/command.
 * All untrusted strings are inserted with textContent.
 * Visible texts are Korean: state values / event messages of the ROS nodes
 * (English) go through i18n_ko.js (window.KO); the raw value stays in the
 * title tooltip.
 */
(function () {
  "use strict";

  // -------------------------------------------------------------- i18n
  // i18n_ko.js missing (old install): show the raw English values
  const KO = window.KO || {
    label: (kind, v, fb) => (v === undefined || v === null || v === "" ? (fb === undefined ? "" : fb) : String(v)),
    event: (t) => (t === undefined || t === null ? "" : String(t)),
    command: (target, c) => String(c),
  };
  const lbl = KO.label;

  // ------------------------------------------------------------ tokens
  const css = getComputedStyle(document.documentElement);
  const tok = (name, fallback) => (css.getPropertyValue(name) || "").trim() || fallback;
  // canvas text: same local Korean font stack as the page (no web fonts)
  const FONT = tok("--font", "'Noto Sans KR', 'Noto Sans CJK KR', 'Malgun Gothic', sans-serif");
  const C = {
    page: tok("--page", "#0d0d0d"),
    surface: tok("--surface-1", "#1a1a19"),
    text: tok("--text-primary", "#ffffff"),
    text2: tok("--text-secondary", "#c3c2b7"),
    muted: tok("--text-muted", "#898781"),
    grid: tok("--grid", "#2c2c2a"),
    baseline: tok("--baseline", "#383835"),
    s1: tok("--series-1", "#3987e5"),
    s2: tok("--series-2", "#d95926"),
    s3: tok("--series-3", "#199e70"),
    good: tok("--good", "#0ca30c"),
    warning: tok("--warning", "#fab219"),
    critical: tok("--critical", "#d03b3b"),
    friction: tok("--friction", "#8fbbe0"),
  };
  // status -> colour + glyph (shape is the secondary encoding, never colour alone)
  const STATUS = {
    normal: { color: C.good, glyph: "●", shape: "circle", label: lbl("status", "normal") },
    warning: { color: C.warning, glyph: "▲", shape: "triangle", label: lbl("status", "warning") },
    alarm: { color: C.critical, glyph: "◆", shape: "diamond", label: lbl("status", "alarm") },
  };
  const UNSET = { color: C.muted, glyph: "○", shape: "ring", label: lbl("status", "unread") };
  const statusInfo = (s) => STATUS[s] || Object.assign({}, UNSET, { label: lbl("status", s || "unknown") });

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  };
  const setText = (id, text) => { const e = $(id); if (e) e.textContent = text; };
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const fmt = (v, nd, unit) => (isNum(v) ? v.toFixed(nd === undefined ? 2 : nd) + (unit ? " " + unit : "") : "—");
  const deg = (rad) => (isNum(rad) ? (rad * 180 / Math.PI).toFixed(1) + "°" : "—");
  const fmtAge = (a) => (isNum(a) ? (a < 1 ? "방금" : a < 90 ? a.toFixed(0) + " s 전" : (a / 60).toFixed(0) + " min 전") : "없음");
  const fmtClock = (t) => {
    if (!isNum(t)) return "--";
    const s = Math.max(0, t);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    return (h ? h + ":" + String(m).padStart(2, "0") : String(m)) + ":" + sec.toFixed(1).padStart(4, "0");
  };
  const wallTime = (t) => {
    if (!isNum(t)) return "";
    const d = new Date(t * 1000);
    return d.toTimeString().slice(0, 8);
  };

  // ------------------------------------------------------------- state
  let STATIC = null;
  let SNAP = null;
  let lastMsg = 0;
  let lastEventSeq = -1;
  const layers = {
    map: { img: null, version: -1, meta: null, loading: false },
    gas: { img: null, version: -1, meta: null, loading: false },
    rssi: { img: null, version: -1, meta: null, loading: false },
  };
  const cams = {
    gauge: { img: $("imgGauge"), version: -1, t: 0, loading: false, path: "image/gauge.jpg" },
    thermal: { img: $("imgThermal"), version: -1, t: 0, loading: false, path: "image/thermal.jpg" },
  };
  // auto: the view follows the canvas size (re-fit on resize) until the
  // operator pans / zooms; the fit button (btnFit) turns it back on
  const view = { scale: 12, cx: 27, cy: 0, follow: false, fitted: false, auto: true };
  const canvas = $("map");
  const ctx = canvas.getContext("2d");
  const wrap = $("mapWrap");
  const tip = $("mapTip");
  let W = 0, H = 0;
  let hover = null;          // {x, y} canvas px
  let redrawQueued = false;

  // ======================================================== connection
  function setConn(kind, text) {
    const d = $("connDot");
    d.className = "dot " + (kind || "");
    setText("connText", text);
  }

  function connect() {
    if (!window.EventSource) { poll(); return; }
    const es = new EventSource("events");
    es.addEventListener("static", (e) => {
      try { STATIC = JSON.parse(e.data); onStatic(); } catch (err) { console.error(err); }
    });
    es.onmessage = (e) => {
      try { SNAP = JSON.parse(e.data); } catch (err) { return; }
      lastMsg = Date.now();
      setConn("ok", "실시간");
      render();
    };
    es.onopen = () => setConn("ok", "실시간");
    es.onerror = () => setConn("bad", "재연결 중…");
  }

  async function poll() {
    try {
      if (!STATIC) { STATIC = await (await fetch("api/static")).json(); onStatic(); }
      SNAP = await (await fetch("api/state", { cache: "no-store" })).json();
      lastMsg = Date.now();
      setConn("ok", "실시간 (폴링)");
      render();
    } catch (err) {
      setConn("bad", "연결 끊김");
    }
    setTimeout(poll, 600);
  }

  setInterval(() => {
    if (lastMsg && Date.now() - lastMsg > 4000) setConn("late", ((Date.now() - lastMsg) / 1000).toFixed(0) + " s 동안 데이터 없음");
  }, 1000);

  // ============================================================ commands
  let fbTimer = null;
  function feedback(ok, text) {
    const f = $("cmdFeedback");
    f.className = "feedback " + (ok ? "ok" : "err");
    f.textContent = text;
    clearTimeout(fbTimer);
    fbTimer = setTimeout(() => { f.textContent = ""; f.className = "feedback"; }, 4000);
  }

  async function sendCommand(target, command) {
    try {
      const r = await fetch("api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: target, command: command }),
      });
      let j = {};
      try { j = await r.json(); } catch (e) { /* non-JSON error */ }
      feedback(!!j.ok, j.message ? KO.event(j.message) : "HTTP " + r.status + " " + r.statusText);
    } catch (err) {
      feedback(false, "대시보드에 연결할 수 없음");
    }
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-target]");
    if (!b) return;
    const msg = b.getAttribute("data-confirm");
    if (msg && !window.confirm(msg)) return;
    sendCommand(b.getAttribute("data-target"), b.getAttribute("data-cmd"));
  });

  // ================================================================ map
  function resize() {
    const r = wrap.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    W = Math.max(50, r.width);
    H = Math.max(50, r.height);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!view.fitted || view.auto) fit();
    queueDraw();
  }

  function worldBounds() {
    if (STATIC && Array.isArray(STATIC.bounds) && STATIC.bounds.length === 4) return STATIC.bounds;
    const m = layers.map.meta;
    if (m) return [m.origin[0], m.origin[1], m.origin[0] + m.width * m.resolution, m.origin[1] + m.height * m.resolution];
    return [-10, -9, 64, 9];
  }

  function fit() {
    const b = worldBounds();
    const bw = b[2] - b[0], bh = b[3] - b[1];
    if (!(bw > 0 && bh > 0) || !W) return;
    view.scale = Math.min(W / bw, H / bh) * 0.97;
    view.cx = (b[0] + b[2]) / 2;
    view.cy = (b[1] + b[3]) / 2;
    view.fitted = true;
    queueDraw();
  }

  const w2s = (x, y) => [W / 2 + (x - view.cx) * view.scale, H / 2 - (y - view.cy) * view.scale];
  const s2w = (px, py) => [view.cx + (px - W / 2) / view.scale, view.cy - (py - H / 2) / view.scale];

  function zoomAt(px, py, factor) {
    view.auto = false;
    const [wx, wy] = s2w(px, py);
    view.scale = Math.min(400, Math.max(2, view.scale * factor));
    view.cx = wx - (px - W / 2) / view.scale;
    view.cy = wy + (py - H / 2) / view.scale;
    queueDraw();
  }

  function setFollow(on) {
    view.follow = on;
    if (on) view.auto = false;
    $("btnFollow").setAttribute("aria-pressed", on ? "true" : "false");
    if (on && view.scale < 25) view.scale = 40;
    queueDraw();
  }

  // pointer interaction: drag to pan, pinch to zoom, hover tooltips
  const pointers = new Map();
  let drag = null;
  let pinchDist = 0;
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
    if (pointers.size === 1) drag = { x: e.offsetX, y: e.offsetY, cx: view.cx, cy: view.cy, moved: false };
    if (pointers.size === 2) {
      const p = [...pointers.values()];
      pinchDist = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.offsetX, y: e.offsetY });
    if (pointers.size === 2) {
      const p = [...pointers.values()];
      const d = Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y);
      if (pinchDist > 0 && d > 0) zoomAt((p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2, d / pinchDist);
      pinchDist = d;
      return;
    }
    if (drag && pointers.size === 1) {
      const dx = e.offsetX - drag.x, dy = e.offsetY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) {
        drag.moved = true;
        view.auto = false;
        canvas.classList.add("dragging");
        if (view.follow) setFollow(false);
        view.cx = drag.cx - dx / view.scale;
        view.cy = drag.cy + dy / view.scale;
        queueDraw();
      }
      return;
    }
    hover = { x: e.offsetX, y: e.offsetY };
    queueDraw();
  });
  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinchDist = 0;
    if (pointers.size === 0) { drag = null; canvas.classList.remove("dragging"); }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);
  canvas.addEventListener("pointerleave", () => { hover = null; tip.hidden = true; queueDraw(); });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  canvas.addEventListener("dblclick", (e) => zoomAt(e.offsetX, e.offsetY, 2));
  $("btnFit").addEventListener("click", () => { setFollow(false); view.auto = true; fit(); });
  $("btnFollow").addEventListener("click", () => setFollow(!view.follow));
  const TOGGLES = ["tgGas", "tgRssi", "tgRoute", "tgLabels", "tgZones", "tgFriction"];
  TOGGLES.forEach((id) => $(id).addEventListener("change", () => {
    try { localStorage.setItem("twin." + id, $(id).checked ? "1" : "0"); } catch (e) { /* storage blocked */ }
    refreshLayers();
    queueDraw();
  }));
  TOGGLES.forEach((id) => {
    try { const v = localStorage.getItem("twin." + id); if (v !== null) $(id).checked = v === "1"; } catch (e) { /* ignore */ }
  });

  if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
  else window.addEventListener("resize", resize);

  function queueDraw() {
    if (redrawQueued) return;
    redrawQueued = true;
    requestAnimationFrame(() => { redrawQueued = false; draw(); });
  }

  // ------------------------------------------------------- layer images
  function loadLayer(name, path) {
    const L = layers[name];
    const meta = SNAP && SNAP.layers ? SNAP.layers[name] : null;
    if (!meta || L.loading || meta.version === L.version) return;
    L.loading = true;
    const img = new Image();
    img.onload = () => { L.img = img; L.meta = meta; L.version = meta.version; L.loading = false; if (name === "map" && !view.fitted) fit(); queueDraw(); };
    img.onerror = () => { L.loading = false; };
    img.src = path + "?v=" + meta.version;
  }

  function refreshLayers() {
    loadLayer("map", "map.png");
    if ($("tgGas").checked) loadLayer("gas", "gas.png");
    if ($("tgRssi").checked) loadLayer("rssi", "rssi.png");
  }

  function drawLayer(name, smooth) {
    const L = layers[name];
    if (!L.img || !L.meta) return;
    const m = L.meta;
    const w = m.width * m.resolution, h = m.height * m.resolution;
    const [sx, sy] = w2s(m.origin[0], m.origin[1] + h);
    ctx.imageSmoothingEnabled = !!smooth;
    ctx.drawImage(L.img, sx, sy, w * view.scale, h * view.scale);
    ctx.imageSmoothingEnabled = true;
  }

  // ---------------------------------------------------------- drawing
  function pathLine(pts, color, width, dash) {
    if (!pts || pts.length < 2) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    pts.forEach((p, i) => { const [x, y] = w2s(p[0], p[1]); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.stroke();
    ctx.restore();
  }

  function label(text, x, y, color, align, size) {
    ctx.save();
    ctx.font = (size || 11) + "px " + FONT;
    ctx.textAlign = align || "left";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = C.page;
    ctx.strokeText(text, x, y);
    ctx.fillStyle = color || C.text2;
    ctx.fillText(text, x, y);
    ctx.restore();
  }

  function glyph(shape, x, y, r, color) {
    ctx.save();
    ctx.beginPath();
    if (shape === "triangle") {
      ctx.moveTo(x, y - r * 1.15); ctx.lineTo(x + r * 1.05, y + r * 0.75); ctx.lineTo(x - r * 1.05, y + r * 0.75); ctx.closePath();
    } else if (shape === "diamond") {
      ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.2, y); ctx.lineTo(x, y + r * 1.2); ctx.lineTo(x - r * 1.2, y); ctx.closePath();
    } else {
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.lineWidth = 4;                  // 2 px surface ring around the mark
    ctx.strokeStyle = C.surface;
    ctx.stroke();
    if (shape === "ring") {
      ctx.lineWidth = 2;
      ctx.strokeStyle = color;
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.fill();
    }
    ctx.restore();
  }

  // ------------------------------------------------- slippery ground
  const inRect = (rc, x, y) => x >= Math.min(rc[0], rc[2]) && x <= Math.max(rc[0], rc[2]) &&
    y >= Math.min(rc[1], rc[3]) && y <= Math.max(rc[1], rc[3]);
  const validRect = (rc) => Array.isArray(rc) && rc.length === 4 && rc.every(isNum);

  // low-friction plates / zones of the layout; an older dashboard node has no
  // friction_zones -> fall back to the plate obstacles (geometry only, no mu)
  function frictionZones() {
    if (!STATIC) return [];
    const src = Array.isArray(STATIC.friction_zones) ? STATIC.friction_zones
      : (STATIC.obstacles || []).filter((o) => o && o.type === "plate").map((o) => ({ id: o.id, kind: "plate", rect: o.rect, mu: o.mu }));
    return src.filter((z) => z && validRect(z.rect));
  }

  // zones the locomotion stack is reacting to: the friction prior of
  // /terrain/info (footprint + look-ahead) plus the zone under the robot
  function activeFrictionIds(snap) {
    const ids = new Set();
    const t = (snap && snap.terrain) || {};
    if (Array.isArray(t.friction_zones)) t.friction_zones.forEach((id) => ids.add(String(id)));
    const r = snap && snap.robot;
    if (r && isNum(r.x) && isNum(r.y)) frictionZones().forEach((z) => { if (inRect(z.rect, r.x, r.y)) ids.add(String(z.id)); });
    return ids;
  }

  function drawFrictionZones(snap, showLabels) {
    const zones = frictionZones();
    if (!zones.length) return;
    const active = activeFrictionIds(snap);
    zones.forEach((z) => {
      const rc = z.rect;
      const [x0, y0] = w2s(Math.min(rc[0], rc[2]), Math.max(rc[1], rc[3]));
      const [x1, y1] = w2s(Math.max(rc[0], rc[2]), Math.min(rc[1], rc[3]));
      const w = Math.max(2, x1 - x0), h = Math.max(2, y1 - y0);
      const on = active.has(String(z.id));
      const col = on ? C.warning : C.friction;
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, y0, w, h); ctx.clip();
      ctx.fillStyle = on ? "rgba(250,178,25,0.16)" : "rgba(143,187,224,0.10)";
      ctx.fillRect(x0, y0, w, h);
      // diagonal hatch = shape encoding, the plate reads without colour too
      ctx.strokeStyle = col; ctx.globalAlpha = 0.6; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let d = -h; d < w; d += 6) { ctx.moveTo(x0 + d, y0 + h); ctx.lineTo(x0 + d + h, y0); }
      ctx.stroke();
      ctx.restore();
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = on ? 2 : 1.25; ctx.strokeRect(x0, y0, w, h); ctx.restore();
      if (showLabels && view.scale > 7) {
        const mu = isNum(z.mu) ? "μ " + z.mu.toFixed(2) : "저마찰";
        label(view.scale > 16 ? z.id + " · " + mu : mu, x0 + w / 2, y0 - 8, on ? C.warning : C.text2, "center", 10);
      }
    });
  }

  const SLIP_SOURCES = ["terrain", "detected", "both"];
  function slipText(l) {
    if (typeof l.slip_mode !== "boolean") return null;               // older controller: no slip fields
    const n = isNum(l.slip_events) ? l.slip_events : null;
    const parts = [l.slip_mode ? "켜짐" : "꺼짐"];
    if (l.slip_mode && SLIP_SOURCES.includes(l.slip_source)) parts.push(lbl("slip", l.slip_source));
    if (l.slip === true) {
      const legs = Array.isArray(l.slip_legs) ? ["FL", "FR", "RL", "RR"].filter((_, i) => l.slip_legs[i] === true) : [];
      parts.push("미끄러지는 중" + (legs.length ? " " + legs.join(" ") : "") + (isNum(l.slip_speed) ? " " + l.slip_speed.toFixed(2) + " m/s" : ""));
    }
    if (n !== null) parts.push("감지 " + n + "회");
    return parts.join(" · ");
  }

  function drawRobot(r) {
    if (!r || !isNum(r.x) || !isNum(r.y)) return;
    const body = (STATIC && STATIC.robot && STATIC.robot.body) || [0.6, 0.22];
    const feet = (STATIC && STATIC.robot && STATIC.robot.feet) || [];
    const [px, py] = w2s(r.x, r.y);
    const k = Math.max(view.scale, 14 / body[0]);       // never smaller than ~14 px long
    ctx.save();
    ctx.translate(px, py);
    ctx.rotate(-(r.yaw || 0));
    ctx.fillStyle = "rgba(57,135,229,0.16)";
    ctx.beginPath(); ctx.arc(0, 0, Math.max(body[0] * k * 0.9, 14), 0, Math.PI * 2); ctx.fill();
    feet.forEach((f) => {
      ctx.beginPath(); ctx.arc(f[0] * k, -f[1] * k, Math.max(2, 0.035 * k), 0, Math.PI * 2);
      ctx.fillStyle = C.text2; ctx.fill();
    });
    const L = body[0] * k, Wd = body[1] * k;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(-L / 2, -Wd / 2, L, Wd, Math.min(4, Wd / 3)); else ctx.rect(-L / 2, -Wd / 2, L, Wd);
    ctx.fillStyle = C.text; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = C.surface; ctx.stroke();
    ctx.beginPath();                                   // heading wedge
    ctx.moveTo(L / 2 + Math.max(6, 0.18 * k), 0); ctx.lineTo(L / 2 - 2, -Wd / 2); ctx.lineTo(L / 2 - 2, Wd / 2); ctx.closePath();
    ctx.fillStyle = C.s1; ctx.fill();
    ctx.restore();
  }

  function drawScaleBar() {
    const target = 120 / view.scale;                   // metres for ~120 px
    const nice = [0.5, 1, 2, 5, 10, 20, 50, 100].find((v) => v >= target * 0.6) || 100;
    const len = nice * view.scale;
    const x = W - len - 16, y = H - 16;
    ctx.save();
    ctx.fillStyle = "rgba(13,13,13,0.72)";            // backing so map labels never cross the bar
    ctx.fillRect(x - 8, y - 20, len + 16, 26);
    ctx.strokeStyle = C.text2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + len, y); ctx.lineTo(x + len, y - 4); ctx.stroke();
    ctx.restore();
    label(nice + " m", x + len / 2, y - 10, C.text2, "center");
    // north arrow
    ctx.save();
    ctx.translate(W - 22, 26);
    ctx.fillStyle = C.text2;
    ctx.beginPath(); ctx.moveTo(0, -11); ctx.lineTo(6, 6); ctx.lineTo(0, 2); ctx.lineTo(-6, 6); ctx.closePath(); ctx.fill();
    ctx.restore();
    label("N", W - 22, 44, C.text2, "center");
  }

  function hitList() {
    // entities that can be hovered: {x, y (world), r px, lines[]}
    const out = [];
    if (!STATIC) return out;
    const gl = {};
    (SNAP && SNAP.gauges || []).forEach((g) => { gl[g.id] = g; });
    (STATIC.gauges || []).forEach((g) => {
      const s = gl[g.id] || {};
      out.push({ x: g.x, y: g.y, title: g.id + " · " + g.tag, value: isNum(s.value) ? s.value.toFixed(3) + " " + (s.unit || g.unit) : lbl("status", "unread"),
        rows: [["상태", statusInfo(s.status).label], ["실제값", g.true_value + " " + g.unit], ["오차", isNum(s.error_pct_fs) ? s.error_pct_fs.toFixed(2) + " % FS" : "—"],
          ["신뢰도", fmt(s.confidence, 2)], ["범위", g.min + " – " + g.max + " " + g.unit]] });
    });
    const tl = {};
    (SNAP && SNAP.thermal || []).forEach((h) => { tl[h.id] = h; });
    (STATIC.hotspots || []).forEach((h) => {
      const s = tl[h.id] || {};
      out.push({ x: h.x, y: h.y, title: h.id + " · " + h.tag + " (" + lbl("kind", h.kind) + ")", value: isNum(s.max_temp_c) ? s.max_temp_c.toFixed(1) + " °C" : lbl("status", "unseen"),
        rows: [["상태", statusInfo(s.status).label], ["실제값", h.true_temp_c + " °C"], ["최고 관측", isNum(s.peak_c) ? s.peak_c.toFixed(1) + " °C" : "—"]] });
    });
    (STATIC.aps || []).forEach((a) => out.push({ x: a.x, y: a.y, title: "무선 액세스 포인트 " + a.id, value: "Wi-Fi AP", rows: [] }));
    if (STATIC.dock) out.push({ x: STATIC.dock.x, y: STATIC.dock.y, title: "충전소", value: "복귀 지점", rows: [["방향 (yaw)", deg(STATIC.dock.yaw)]] });
    const leak = STATIC.gas_leak;
    if (leak && isNum(leak.x)) out.push({ x: leak.x, y: leak.y, title: "가스 누출원 (실제 위치)", value: leak.tag || leak.id || "누출원", rows: [] });
    const gas = SNAP && SNAP.gas;
    if (gas && gas.estimate && isNum(gas.estimate[0]) && isNum(gas.estimate[1])) out.push({ x: gas.estimate[0], y: gas.estimate[1], title: "누출원 추정 위치 (추적기)", value: isNum(gas.estimate_error_m) ? "오차 " + gas.estimate_error_m.toFixed(2) + " m" : "추정", rows: [] });
    const plan = SNAP && SNAP.plan;
    if (plan && plan.viewpoints) plan.viewpoints.forEach((v) => out.push({ x: v.x, y: v.y, title: v.gauge_id + " 점검 지점", value: "순서 " + (plan.order.indexOf(v.gauge_id) + 1), rows: [["점수", fmt(v.score, 3)], ["방향 (yaw)", deg(v.yaw)]] }));
    if ($("tgFriction").checked) {
      const active = activeFrictionIds(SNAP);
      frictionZones().forEach((z) => {
        const rc = z.rect;
        out.push({ x: (rc[0] + rc[2]) / 2, y: (rc[1] + rc[3]) / 2,
          title: (z.kind === "zone" ? "저마찰 구역 " : "저마찰 바닥판 ") + z.id,
          value: isNum(z.mu) ? "μ " + z.mu.toFixed(2) : "저마찰",
          rows: [["크기", Math.abs(rc[2] - rc[0]).toFixed(1) + " × " + Math.abs(rc[3] - rc[1]).toFixed(1) + " m"],
            ["로봇", active.has(String(z.id)) ? "위 / 바로 앞" : "—"]] });
      });
    }
    const r = SNAP && SNAP.robot;
    if (r && isNum(r.x)) {
      const l = SNAP.locomotion || {};
      out.push({ x: r.x, y: r.y, title: "로봇 (spot)", value: fmt(r.x, 2) + ", " + fmt(r.y, 2) + " m",
        rows: [["방향 (yaw)", deg(r.yaw)], ["구역", lbl("zone", r.zone, "—")], ["보행 모드", lbl("state", l.mode, "—")], ["미끄럼 모드", slipText(l) || "—"]] });
    }
    return out;
  }

  function showTip(item, px, py) {
    tip.textContent = "";
    tip.appendChild(el("div", "tv", item.value));
    tip.appendChild(el("div", null, item.title));
    item.rows.forEach((r) => {
      const row = el("div", "tr");
      row.appendChild(el("span", "muted", r[0]));
      row.appendChild(el("span", null, r[1]));
      tip.appendChild(row);
    });
    tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let x = px + 14, y = py + 14;
    if (x + tw > W - 4) x = px - tw - 14;
    if (y + th > H - 4) y = py - th - 14;
    tip.style.left = Math.max(4, x) + "px";
    tip.style.top = Math.max(4, y) + "px";
  }

  function draw() {
    ctx.fillStyle = C.page;
    ctx.fillRect(0, 0, W, H);
    if (!STATIC) { label("대시보드 노드 연결 대기 중…", W / 2, H / 2, C.muted, "center", 13); return; }
    const snap = SNAP || {};
    const showLabels = $("tgLabels").checked;
    if (view.follow && snap.robot && isNum(snap.robot.x)) { view.cx = snap.robot.x; view.cy = snap.robot.y; }

    drawLayer("map", false);
    if (!layers.map.img) {
      // no map yet: outline the building shells from the layout
      [STATIC.factory, STATIC.corridor].forEach((r) => {
        if (!r) return;
        const [x0, y0] = w2s(r.x_min, r.y_max), [x1, y1] = w2s(r.x_max, r.y_min);
        ctx.strokeStyle = C.baseline; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      });
    }
    if ($("tgZones").checked) {
      (STATIC.zones || []).forEach((z) => {
        const [x0, y0] = w2s(Math.min(z.rect[0], z.rect[2]), Math.max(z.rect[1], z.rect[3]));
        const [x1, y1] = w2s(Math.max(z.rect[0], z.rect[2]), Math.min(z.rect[1], z.rect[3]));
        ctx.save(); ctx.strokeStyle = C.muted; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
        label(lbl("zone", z.name), x0 + 4, y0 + 9, C.muted, "left", 10);
      });
    }
    if ($("tgFriction").checked) drawFrictionZones(snap, showLabels);
    if ($("tgRssi").checked) {
      drawLayer("rssi", true);
      (STATIC.shadow_zones || []).forEach((z) => {
        const [x0, y0] = w2s(Math.min(z.rect[0], z.rect[2]), Math.max(z.rect[1], z.rect[3]));
        const [x1, y1] = w2s(Math.max(z.rect[0], z.rect[2]), Math.min(z.rect[1], z.rect[3]));
        ctx.save(); ctx.strokeStyle = C.text2; ctx.setLineDash([3, 3]); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
        if (showLabels) label("전파 음영", x0 + 4, y1 - 9, C.text2, "left", 10);
      });
    }
    if ($("tgGas").checked) drawLayer("gas", true);
    if (showLabels && view.scale > 9) {
      (STATIC.machines || []).forEach((m) => {
        const f = m.footprint;
        const [x, y] = w2s((f.x_min + f.x_max) / 2, (f.y_min + f.y_max) / 2);
        label(m.id, x, y, C.muted, "center", 10);
      });
    }

    const paths = snap.paths || {};
    if ($("tgRoute").checked) {
      pathLine(paths.route, C.text2, 1.5, [6, 5]);
      const plan = snap.plan;
      if (plan && plan.viewpoints) {
        plan.viewpoints.forEach((v) => {
          const [x, y] = w2s(v.x, v.y);
          const a = -(v.yaw || 0);
          ctx.save(); ctx.translate(x, y); ctx.rotate(a);
          ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(-5, -5); ctx.lineTo(-5, 5); ctx.closePath();
          ctx.lineWidth = 3; ctx.strokeStyle = C.surface; ctx.stroke();
          ctx.fillStyle = C.text2; ctx.fill(); ctx.restore();
          const idx = plan.order ? plan.order.indexOf(v.gauge_id) : -1;
          if (showLabels && idx >= 0) label(String(idx + 1), x, y - 11, C.text2, "center", 10);
        });
      }
    }
    pathLine(paths.seek, C.s3, 2);
    pathLine(paths.nav, C.s1, 2);
    pathLine(paths.mpc, C.text, 1.5, [1.5, 3.5]);

    // dock + access points
    if (STATIC.dock) {
      const d = STATIC.dock, s = d.size || [1, 1];
      const [x0, y0] = w2s(d.x - s[0] / 2, d.y + s[1] / 2);
      ctx.save(); ctx.strokeStyle = C.s1; ctx.lineWidth = 2;
      ctx.strokeRect(x0, y0, Math.max(8, s[0] * view.scale), Math.max(8, s[1] * view.scale)); ctx.restore();
      if (showLabels) label("충전소", x0 + Math.max(8, s[0] * view.scale) / 2, y0 - 9, C.text2, "center", 10);
    }
    (STATIC.aps || []).forEach((a) => {
      const [x, y] = w2s(a.x, a.y);
      ctx.save(); ctx.strokeStyle = C.text2; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, y + 6); ctx.lineTo(x, y - 4); ctx.stroke();
      [5, 9].forEach((rr) => { ctx.beginPath(); ctx.arc(x, y - 4, rr, -Math.PI * 0.8, -Math.PI * 0.2); ctx.stroke(); });
      ctx.restore();
      if (showLabels) label(a.id, x + 8, y + 2, C.text2);
    });

    // gas leak truth vs estimate
    const leak = STATIC.gas_leak;
    if (leak && isNum(leak.x)) {
      const [x, y] = w2s(leak.x, leak.y);
      ctx.save(); ctx.strokeStyle = C.text; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke();
      ctx.restore();
      if (showLabels) label("누출원 (실제)", x + 9, y - 8, C.text2);
      const est = snap.gas && snap.gas.estimate;
      if (est && isNum(est[0]) && isNum(est[1])) {
        const [ex, ey] = w2s(est[0], est[1]);
        ctx.save(); ctx.strokeStyle = C.s3; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(x, y); ctx.stroke(); ctx.restore();
        glyph("circle", ex, ey, 5, C.s3);
        const err = snap.gas.estimate_error_m;
        if (showLabels) label("추정" + (isNum(err) ? " · 오차 " + err.toFixed(2) + " m" : ""), ex + 9, ey + 9, C.text2);
      }
    }

    // hot spots (status glyph framed by a square) and gauges (status glyph)
    const tl = {};
    (snap.thermal || []).forEach((h) => { tl[h.id] = h; });
    (STATIC.hotspots || []).forEach((h) => {
      const s = tl[h.id] || {};
      const info = statusInfo(s.status);
      const [x, y] = w2s(h.x, h.y);
      ctx.save(); ctx.strokeStyle = C.text2; ctx.lineWidth = 1; ctx.strokeRect(x - 8, y - 8, 16, 16); ctx.restore();
      glyph(info.shape, x, y, 4.5, info.color);
      // label density follows the zoom: ids from 7 px/m, values from 16 px/m
      if (showLabels && view.scale > 16) label(h.id + (isNum(s.max_temp_c) ? " " + s.max_temp_c.toFixed(0) + "°C" : ""), x + 11, y + 1, C.text2);
    });
    const gl = {};
    (snap.gauges || []).forEach((g) => { gl[g.id] = g; });
    const target = snap.mission && snap.mission.target;
    (STATIC.gauges || []).forEach((g) => {
      const s = gl[g.id] || {};
      const info = statusInfo(s.status);
      const [x, y] = w2s(g.x, g.y);
      if (target === g.id) {
        ctx.save(); ctx.strokeStyle = C.s1; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 11, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
      }
      glyph(info.shape, x, y, 5.5, info.color);
      if (showLabels && view.scale > 7) label(g.id + (view.scale > 16 && isNum(s.value) ? " " + s.value.toFixed(2) + " " + (s.unit || g.unit) : ""), x + 10, y - 9, C.text);
    });

    drawRobot(snap.robot);
    drawScaleBar();

    // hover tooltip: nearest entity within 16 px
    if (hover && !drag) {
      let best = null, bd = 16;
      hitList().forEach((it) => {
        const [x, y] = w2s(it.x, it.y);
        const d = Math.hypot(x - hover.x, y - hover.y);
        if (d < bd) { bd = d; best = it; }
      });
      if (best) {
        const [x, y] = w2s(best.x, best.y);
        ctx.save(); ctx.strokeStyle = C.text; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
        showTip(best, hover.x, hover.y);
      } else {
        tip.hidden = true;
      }
    }
  }

  // ============================================================ panels
  function badgeClass(state) {
    if (["SEEK", "FALLEN"].includes(state)) return "badge s-alarm";
    if (["RTH", "MANUAL", "PAUSED", "RECOVERING", "ESTOP", "STANDING_UP", "RTH_PLANNING", "RTH_NAVIGATING", "DOCKING"].includes(state)) return "badge s-warn";
    if (["PATROL", "INSPECTING", "WALK", "STAND"].includes(state)) return "badge s-active";
    if (["DOCKED", "DONE", "CHARGING", "MONITOR"].includes(state)) return "badge s-good";
    return "badge";
  }
  // state badge: Korean label, raw value (PATROL, RTH_PLANNING ...) in the tooltip
  function setBadge(id, state, big, kind) {
    const b = $(id);
    b.className = badgeClass(state) + (big ? " big" : "");
    b.textContent = state ? lbl(kind || "state", state) : "데이터 없음";
    b.title = state || "데이터 없음";
  }
  function statusEl(status) {
    const info = statusInfo(status);
    const s = el("span", "status " + (STATUS[status] ? status : "unread"));
    s.appendChild(el("i", "ico", info.glyph));
    s.appendChild(document.createTextNode(info.label));
    if (status) s.title = String(status);
    return s;
  }
  function staleCard(id, age, limit) {
    const c = $(id);
    if (c) c.classList.toggle("stale", !isNum(age) || age > limit);
  }

  function renderRobot(s) {
    const r = s.robot || {};
    const l = s.locomotion || {};
    setBadge("locoMode", l.mode || null);
    setText("robotZone", lbl("zone", r.zone));
    $("robotZone").title = r.zone || "";
    setText("robotAge", isNum(l.age) ? "보행 상태 " + fmtAge(l.age) : "보행 상태 없음");
    setText("locoGait", lbl("gait", l.gait, "—") + (l.gait_auto ? " (자동)" : "") +
      (l.locomotion_mode ? " · " + lbl("algo", l.locomotion_mode) : ""));
    $("locoGait").title = [l.gait, l.gait_auto ? "auto" : "", l.locomotion_mode].filter(Boolean).join(" · ");
    setText("locoHeight", fmt(l.body_height, 3, "m"));
    setText("locoRP", deg(l.roll) + " / " + deg(l.pitch));
    setText("locoFalls", (isNum(l.fall_count) ? l.fall_count : "—") + " / " + (isNum(l.recovery_success) ? l.recovery_success : "—"));
    setText("locoMpc", l.locomotion_mode === "stable" ? "꺼짐 (안정 모드)" :
      (isNum(l.mpc_solve_ms) ? l.mpc_solve_ms.toFixed(1) + " ms" + (l.mpc_status ? " · " + lbl("mpc", l.mpc_status) : "") : "—"));
    $("locoMpc").title = l.mpc_status || "";
    setText("locoMargin", fmt(l.stability_margin, 3, "m"));
    setText("robotVel", isNum(r.vx) ? r.vx.toFixed(2) + " m/s · " + (r.wz || 0).toFixed(2) + " rad/s" : "—");
    const cv = s.cmd_vel || {};
    setText("cmdSource", lbl("source", cv.source, "—") + (s.flags && s.flags.teleop_active ? " (수동)" : ""));
    $("cmdSource").title = cv.source || "";
    setText("robotPose", isNum(r.x) ? r.x.toFixed(2) + ", " + r.y.toFixed(2) + " · " + deg(r.yaw) : "—");
    renderSlip(s.terrain || {}, l);
    staleCard("cardRobot", isNum(r.age) ? r.age : l.age, 5);
  }

  // slippery ground: friction prior of /terrain/info + slip state of
  // /locomotion/state (fields absent with older nodes -> dash)
  function renderSlip(t, l) {
    const cls = t["class"];
    const slippery = t.slippery === true || cls === "slippery";
    const terr = $("terrain");
    terr.textContent = "";
    if (slippery) terr.appendChild(el("span", "tag warn", "▲ 미끄러움"));
    let txt = cls && cls !== "slippery" ? lbl("terrain", cls) : "";
    if (cls && t.recommended_gait) txt += (txt ? " " : "") + "→ " + lbl("gait", t.recommended_gait);
    if (slippery && isNum(t.friction_mu)) txt += " · μ " + t.friction_mu.toFixed(2);
    if (slippery && Array.isArray(t.friction_zones) && t.friction_zones.length) txt += " · " + t.friction_zones.join(", ");
    if (slippery && isNum(t.friction_distance)) txt += t.friction_distance > 0.05 ? " · " + t.friction_distance.toFixed(1) + " m 앞" : " (밟고 있음)";
    terr.appendChild(document.createTextNode(txt || (slippery ? "" : "—")));
    terr.title = (slippery ? "미끄러움 " + txt : txt) + (cls ? "\n" + cls + (t.recommended_gait ? " → " + t.recommended_gait : "") : "");

    const mu = [l.mu_est, l.mpc_mu];
    setText("locoMu", mu.some(isNum) ? mu.map((v) => fmt(v, 2)).join(" / ") : "—");
    $("locoMu").title = isNum(l.friction_prior) ? "레이아웃 사전 μ " + l.friction_prior.toFixed(2) : "마찰 사전정보 없음";
    const st = slipText(l);
    const sm = $("slipMode");
    sm.textContent = st || "—";
    sm.className = "v" + (l.slip_mode === true ? " on" : "");
    const notes = [];
    if (isNum(l.torque_ff)) notes.push("토크 피드포워드 " + l.torque_ff.toFixed(2));
    if (isNum(l.slip_speed)) notes.push("지지발 최대 속도 " + l.slip_speed.toFixed(2) + " m/s");
    if (isNum(l.friction_prior)) notes.push("사전 μ " + l.friction_prior.toFixed(2));
    if (l.slip_source) notes.push("slip_source: " + l.slip_source);
    sm.title = (st || "미끄럼 정보 없음") + (notes.length ? "\n" + notes.join("\n") : "");
    $("slipBadge").hidden = l.slip_mode !== true;
  }

  function renderMission(s) {
    const m = s.mission || {};
    const st = m.state || null;
    setBadge("missionState", st);
    setBadge("missionBadge", st, true);
    setText("missionPhase", [m.mode ? lbl("mode", m.mode) : "", m.phase ? lbl("phase", m.phase) : ""].filter(Boolean).join(" · "));
    $("missionPhase").title = [m.mode, m.phase].filter(Boolean).join(" · ");
    setText("missionAge", isNum(m.age) ? fmtAge(m.age) : "임무 상태 없음");
    setText("missionTarget", m.target || "—");
    const pr = m.progress || [];
    setText("missionProgress", pr.length === 2 && pr[1] ? pr[0] + " / " + pr[1] : "—");
    setText("missionElapsed", isNum(m.mission_elapsed) ? fmtClock(m.mission_elapsed) : "—");
    setText("missionReplans", isNum(m.replans) ? m.replans : "—");
    setText("missionMessage", KO.event(m.message || "", "mission"));
    $("missionMessage").title = m.message || "";
    const chips = $("missionChips");
    chips.textContent = "";
    const done = new Set(m.completed || []), missed = new Set(m.missed || []), pending = new Set(m.pending || []);
    const order = (s.plan && s.plan.order && s.plan.order.length) ? s.plan.order : (STATIC ? STATIC.gauges.map((g) => g.id) : []);
    order.forEach((id) => {
      let cls = "chip", mark = "";
      if (done.has(id)) { cls += " done"; mark = "✓ "; }
      else if (missed.has(id)) { cls += " missed"; mark = "✕ "; }
      else if (m.target === id) { cls += " current"; mark = "▸ "; }
      else if (!pending.has(id) && pending.size) { mark = ""; }
      chips.appendChild(el("span", cls, mark + id));
    });
    const pb = $("btnPause");
    if (st === "PAUSED") { pb.textContent = "재개"; pb.title = "일시정지한 임무를 이어서 진행"; pb.setAttribute("data-cmd", "resume"); }
    else { pb.textContent = "일시정지"; pb.title = "임무를 일시정지하고 제자리에서 대기"; pb.setAttribute("data-cmd", "pause"); }
    staleCard("cardMission", m.age, 6);
  }

  function renderSafety(s) {
    const sf = s.safety || {};
    const b = s.battery || {};
    let pct = isNum(b.pct) ? b.pct : (isNum(sf.battery_pct) ? (sf.battery_pct <= 1 ? sf.battery_pct * 100 : sf.battery_pct) : null);
    const bp = $("battPct");
    bp.textContent = isNum(pct) ? pct.toFixed(0) : "—";
    bp.appendChild(el("small", null, "%"));
    const meter = $("battMeter");
    meter.className = "meter" + (!isNum(pct) ? " na" : pct < 20 ? " crit" : pct < 40 ? " warn" : "");
    $("battFill").style.width = (isNum(pct) ? Math.max(0, Math.min(100, pct)) : 0) + "%";
    if (isNum(pct)) meter.setAttribute("aria-valuenow", pct.toFixed(0));
    setBadge("safetyState", sf.state || null);
    const rssi = s.rssi && isNum(s.rssi.dbm) ? s.rssi.dbm : sf.rssi_dbm;
    const minR = STATIC && STATIC.comm ? STATIC.comm.rssi_min_dbm : -80;
    setText("rssi", isNum(rssi) ? rssi.toFixed(1) + " dBm" + (rssi < minR ? " ▲ 음영" : "") : "—");
    setText("rssiPred", fmt(sf.predicted_min_rssi, 1, "dBm"));
    const reason = sf.reason || (sf.rth_active ? "active" : "");
    setText("rthReason", lbl("reason", reason, "—"));
    $("rthReason").title = reason;
    setText("distHome", fmt(sf.distance_home, 1, "m"));
    setText("energyHome", isNum(sf.energy_to_home_pct) ? sf.energy_to_home_pct.toFixed(1) + " %" : "—");
    setText("charging", b.charging === true ? "충전 중" : b.charging === false ? "아니오" : (sf.docked ? "도킹됨" : "—"));
    setText("safetyAge", isNum(b.age) ? "배터리 " + fmtAge(b.age) : "");
    staleCard("cardSafety", isNum(b.age) ? b.age : null, 6);
  }

  function renderGas(s) {
    const g = s.gas || {};
    const pp = $("gasPpm");
    pp.textContent = isNum(g.ppm) ? (g.ppm < 10 ? g.ppm.toFixed(2) : g.ppm.toFixed(1)) : "—";
    pp.appendChild(el("small", null, "ppm"));
    const ab = $("gasAlarm");
    if (g.alarm) { ab.className = "badge s-alarm"; ab.textContent = "◆ 경보 > " + (g.alarm_ppm || 25) + " ppm"; }
    else if (isNum(g.ppm)) { ab.className = "badge s-good"; ab.textContent = "● 경보 기준 미만"; }
    else { ab.className = "badge"; ab.textContent = "센서 없음"; }
    const seek = g.seek || {};
    setText("seekState", lbl("seek", seek.state, "—"));
    $("seekState").title = seek.state || "";
    setText("seekErr", isNum(g.estimate_error_m) ? g.estimate_error_m.toFixed(2) + " m" : "—");
    const est = g.estimate;
    setText("seekEst", est && isNum(est[0]) && isNum(est[1]) ? est[0].toFixed(2) + ", " + est[1].toFixed(2) : "—");
    const f = g.field || {};
    setText("gasMax", fmt(f.max_ppm, 1, "ppm"));
    if (isNum(f.vis_max_ppm)) setText("gasScale", "0 – " + f.vis_max_ppm + " ppm" + (f.scale ? " (" + lbl("scale", f.scale) + ")" : ""));
    setText("gasAge", isNum(g.age) ? "센서 " + fmtAge(g.age) : "센서 데이터 없음");
    staleCard("cardGas", g.age, 5);
    drawSpark(g);
  }

  // gas sparkline (single series -> no legend; alarm reference line labelled)
  const spark = $("gasSpark");
  const sctx = spark.getContext("2d");
  let sparkHover = null;
  let sparkData = null;
  spark.addEventListener("pointermove", (e) => { sparkHover = e.offsetX; if (sparkData) drawSpark(sparkData); });
  spark.addEventListener("pointerleave", () => { sparkHover = null; $("sparkTip").hidden = true; if (sparkData) drawSpark(sparkData); });

  function drawSpark(g) {
    sparkData = g;
    const r = spark.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, r.width), h = Math.max(10, r.height);
    if (spark.width !== Math.round(w * dpr)) { spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr); }
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sctx.clearRect(0, 0, w, h);
    const hist = (g.history || []).filter((p) => isNum(p[1]));
    const alarm = g.alarm_ppm || 25;
    const pad = { l: 30, r: 6, t: 6, b: 14 };
    const span = 120;
    const vmax = Math.max(alarm * 1.3, ...hist.map((p) => p[1] * 1.1), 1);
    const X = (dt) => pad.l + (w - pad.l - pad.r) * (1 + dt / span);
    const Y = (v) => h - pad.b - (h - pad.t - pad.b) * (v / vmax);
    // recessive frame: baseline + one gridline
    sctx.strokeStyle = C.baseline; sctx.lineWidth = 1;
    sctx.beginPath(); sctx.moveTo(pad.l, Y(0) + 0.5); sctx.lineTo(w - pad.r, Y(0) + 0.5); sctx.stroke();
    sctx.font = "10px " + FONT; sctx.fillStyle = C.muted; sctx.textBaseline = "middle"; sctx.textAlign = "right";
    sctx.fillText("0", pad.l - 4, Y(0));
    sctx.fillText(vmax.toFixed(0), pad.l - 4, Y(vmax) + 4);
    sctx.textAlign = "left"; sctx.fillText("−" + span + " s", pad.l, h - 5);
    sctx.textAlign = "right"; sctx.fillText("현재", w - pad.r, h - 5);
    // alarm reference
    sctx.strokeStyle = C.critical; sctx.lineWidth = 1;
    sctx.beginPath(); sctx.moveTo(pad.l, Y(alarm) + 0.5); sctx.lineTo(w - pad.r, Y(alarm) + 0.5); sctx.stroke();
    sctx.fillStyle = C.text2; sctx.textAlign = "left"; sctx.fillText("경보 " + alarm + " ppm", pad.l + 4, Y(alarm) - 7);
    if (hist.length < 2) {
      sctx.fillStyle = C.muted; sctx.textAlign = "center"; sctx.fillText("/gas/sensor 수신 대기 중", (w + pad.l) / 2, h / 2);
      return;
    }
    sctx.fillStyle = "rgba(217,89,38,0.10)";       // 10 % area wash
    sctx.beginPath(); sctx.moveTo(X(hist[0][0]), Y(0));
    hist.forEach((p) => sctx.lineTo(X(p[0]), Y(p[1])));
    sctx.lineTo(X(hist[hist.length - 1][0]), Y(0)); sctx.closePath(); sctx.fill();
    sctx.strokeStyle = C.s2; sctx.lineWidth = 2; sctx.lineJoin = "round"; sctx.lineCap = "round";
    sctx.beginPath();
    hist.forEach((p, i) => { if (i) sctx.lineTo(X(p[0]), Y(p[1])); else sctx.moveTo(X(p[0]), Y(p[1])); });
    sctx.stroke();
    const last = hist[hist.length - 1];
    sctx.beginPath(); sctx.arc(X(last[0]), Y(last[1]), 4, 0, Math.PI * 2);
    sctx.lineWidth = 2; sctx.strokeStyle = C.surface; sctx.stroke(); sctx.fillStyle = C.s2; sctx.fill();
    // crosshair + tooltip snapped to the nearest sample
    const st = $("sparkTip");
    if (sparkHover !== null) {
      let best = hist[0];
      hist.forEach((p) => { if (Math.abs(X(p[0]) - sparkHover) < Math.abs(X(best[0]) - sparkHover)) best = p; });
      const x = X(best[0]);
      sctx.strokeStyle = C.text2; sctx.lineWidth = 1;
      sctx.beginPath(); sctx.moveTo(x + 0.5, pad.t); sctx.lineTo(x + 0.5, Y(0)); sctx.stroke();
      sctx.beginPath(); sctx.arc(x, Y(best[1]), 4, 0, Math.PI * 2); sctx.fillStyle = C.s2; sctx.fill();
      st.textContent = "";
      st.appendChild(el("div", "tv", best[1].toFixed(2) + " ppm"));
      st.appendChild(el("div", null, Math.abs(best[0]).toFixed(0) + " s 전"));
      st.hidden = false;
      st.style.left = Math.min(Math.max(4, x + 10), w - st.offsetWidth - 4) + "px";
      st.style.top = "2px";
    } else {
      st.hidden = true;
    }
  }

  function renderGauges(s) {
    const tb = $("gaugeRows");
    tb.textContent = "";
    const target = s.mission && s.mission.target;
    let nRead = 0, errs = [];
    (s.gauges || []).forEach((g) => {
      const tr = el("tr", target === g.id ? "is-target" : null);
      tr.appendChild(el("td", null, g.id));
      tr.appendChild(el("td", "muted", g.tag || ""));
      tr.appendChild(el("td", "num", isNum(g.value) ? g.value.toFixed(3) + " " + (g.unit || "") : "—"));
      const tdS = el("td"); tdS.appendChild(statusEl(g.status)); tr.appendChild(tdS);
      tr.appendChild(el("td", "num", fmt(g.confidence, 2)));
      tr.appendChild(el("td", "num muted", isNum(g.true_value) ? g.true_value + " " + (g.unit || "") : "—"));
      tr.appendChild(el("td", "num", isNum(g.error_pct_fs) ? g.error_pct_fs.toFixed(2) : "—"));
      tr.appendChild(el("td", "num muted", isNum(g.age) ? fmtAge(g.age) : "—"));
      tb.appendChild(tr);
      if (isNum(g.value)) nRead++;
      if (isNum(g.error_pct_fs)) errs.push(g.error_pct_fs);
    });
    const mean = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
    setText("gaugeSummary", "판독 " + nRead + " / " + (s.gauges || []).length + (isNum(mean) ? " · 평균 오차 " + mean.toFixed(2) + " % FS" : ""));
  }

  function renderThermal(s) {
    const tb = $("thermalRows");
    tb.textContent = "";
    let nAlert = 0;
    const rows = (s.thermal || []).concat((s.thermal_unassigned || []).map((h) => Object.assign({}, h, { id: "—", tag: h.tag || "미할당" })));
    rows.forEach((h) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, h.id));
      tr.appendChild(el("td", "muted", h.tag || ""));
      tr.appendChild(el("td", "num", fmt(h.max_temp_c, 1)));
      const tdS = el("td"); tdS.appendChild(statusEl(h.status)); tr.appendChild(tdS);
      tr.appendChild(el("td", "num muted", fmt(h.true_temp_c, 1)));
      tr.appendChild(el("td", "num muted", isNum(h.age) ? fmtAge(h.age) : "—"));
      tb.appendChild(tr);
      if (h.status === "warning" || h.status === "alarm") nAlert++;
    });
    setText("thermalSummary", "경고 이상 " + nAlert + "곳");
  }

  function renderEvents(s) {
    const evs = s.events || [];
    const lastSeq = evs.length ? evs[evs.length - 1].seq : -1;
    if (lastSeq === lastEventSeq) return;
    lastEventSeq = lastSeq;
    const ul = $("eventList");
    ul.textContent = "";
    const icon = { alarm: "◆", warn: "▲", error: "✕", info: "•" };
    evs.slice().reverse().forEach((e) => {
      const li = el("li", e.level || "info");
      li.appendChild(el("i", "ico", icon[e.level] || "•"));
      li.appendChild(el("span", "t", wallTime(e.t_wall)));
      const src = el("span", "src", lbl("event_source", e.source));
      src.title = e.source || "";
      li.appendChild(src);
      // node messages are English: Korean when a rule matches, original text in the tooltip
      const txt = el("span", "txt", KO.event(e.text || "", e.source || ""));
      txt.title = e.text || "";
      li.appendChild(txt);
      ul.appendChild(li);
    });
    setText("eventCount", "최근 " + evs.length + "건");
  }

  function renderLinks(s) {
    const ul = $("linkList");
    ul.textContent = "";
    let ok = 0, n = 0;
    Object.keys(s.topics || {}).forEach((name) => {
      const t = s.topics[name];
      n++;
      if (t.status === "ok") ok++;
      const li = el("li");
      li.appendChild(el("span", "dot " + (t.status === "ok" ? "ok" : t.status === "late" ? "late" : t.status === "stale" ? "bad" : "")));
      li.appendChild(el("span", "name", name));
      li.appendChild(el("span", "lage", t.status === "never" ? lbl("link", "never") :
        (t.status === "ok" ? "" : lbl("link", t.status) + " · ") + fmtAge(t.age)));
      li.title = name + ": " + lbl("link", t.status) + " (" + t.status + ")";
      ul.appendChild(li);
    });
    setText("linksSummary", "정상 수신 " + ok + " / " + n);
  }

  function renderCams(s) {
    const imgs = s.images || {};
    Object.keys(cams).forEach((k) => {
      const c = cams[k];
      const meta = imgs[k];
      const ageEl = $(k === "gauge" ? "imgGaugeAge" : "imgThermalAge");
      ageEl.textContent = meta && isNum(meta.t_wall) && isNum(s.t_wall) ? fmtAge(s.t_wall - meta.t_wall) : "영상 없음";
      if (!meta || c.loading || meta.version === c.version || Date.now() - c.t < 450) return;
      c.loading = true;
      c.t = Date.now();
      const v = meta.version;
      const img = new Image();
      img.onload = () => { c.img.src = img.src; c.version = v; c.loading = false; };
      img.onerror = () => { c.loading = false; };
      img.src = c.path + "?v=" + v;
    });
  }

  function onStatic() {
    view.fitted = false;
    resize();
    queueDraw();
  }

  function render() {
    const s = SNAP;
    if (!s) return;
    setText("simTime", fmtClock(s.t_sim));
    // one malformed field must not freeze every other panel
    [renderRobot, renderMission, renderSafety, renderGas, renderGauges, renderThermal, renderEvents, renderLinks,
      renderCams].forEach((fn) => {
      try { fn(s); } catch (err) { console.error(fn.name, err); }
    });
    refreshLayers();
    queueDraw();
  }

  resize();
  connect();
})();
