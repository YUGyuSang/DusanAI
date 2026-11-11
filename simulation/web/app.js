/*
 * Plant digital twin - control dashboard client.
 * Vanilla JS, no external dependencies.  Data arrives as Server-Sent Events
 * from the monitor_dashboard node (one 'static' layout event, then JSON
 * snapshots at 5 Hz); commands go out as POST /api/command.
 * All untrusted strings are inserted with textContent.
 * Visible texts are Korean: state values / event messages of the ROS nodes
 * (English) go through i18n_ko.js (window.KO); the raw value stays in the
 * title tooltip.
 *
 * Screen layout (index.html): mission strip, plant map, gauge tiles / mission
 * report (tabs) on the left; operator controls, robot + power, alarms (gas,
 * thermal) and recent events on the right.  Diagnostic values (MPC, friction,
 * slip, topic health ...) live in the "상세 정보" drawer.
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
    page: tok("--page", "#0c0f14"),
    inset: tok("--inset", "#0e1217"),
    surface: tok("--surface-1", "#141920"),
    text: tok("--text-primary", "#eef1f6"),
    text2: tok("--text-secondary", "#b6bfcd"),
    muted: tok("--text-muted", "#8a93a3"),
    baseline: tok("--baseline", "#2f3743"),
    accent: tok("--accent", "#4f8ff7"),
    gas: tok("--gas", "#e8743b"),
    seek: tok("--seek", "#2fb79a"),
    good: tok("--good", "#34c47c"),
    warning: tok("--warning", "#f5b544"),
    critical: tok("--critical", "#f05252"),
    friction: tok("--friction", "#8fbbe0"),
  };
  const LABEL_BG = "rgba(11, 14, 18, 0.86)";
  // map image palette (dashboard_render.py renders unknown/free/occupied in fixed greys;
  // they are re-tinted to the page theme in the browser): floor, walls / machines
  const MAP_FLOOR = [33, 40, 51];
  const MAP_WALL = [86, 98, 116];

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
  const fmtVal = (v) => (isNum(v) ? v.toFixed(Math.abs(v) >= 100 ? 1 : 2) : "—");
  const deg = (rad) => (isNum(rad) ? (rad * 180 / Math.PI).toFixed(1) + "°" : "—");
  const fmtAge = (a) => (isNum(a) ? (a < 1 ? "방금 갱신" : a < 60 ? Math.floor(a) + "초 전" : Math.floor(a / 60) + "분 전") : "없음");
  // m:ss.d (tenths, header clock) and m:ss (whole seconds: mission clock, event times)
  const fmtClock = (t) => {
    if (!isNum(t)) return "--";
    const d = Math.round(Math.max(0, t) * 10);
    const h = Math.floor(d / 36000), m = Math.floor((d % 36000) / 600), ds = d % 600;
    return (h ? h + ":" + String(m).padStart(2, "0") : String(m)) + ":" + (ds / 10).toFixed(1).padStart(4, "0");
  };
  const fmtMinSec = (t) => {
    if (!isNum(t)) return "—";
    const s = Math.floor(Math.max(0, t));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return (h ? h + ":" + String(m).padStart(2, "0") : String(m)) + ":" + String(s % 60).padStart(2, "0");
  };
  const fmtSec = (t) => (isNum(t) ? Math.round(t) + "초" : "—");
  const fmtDur = (t) => {
    if (!isNum(t)) return "—";
    const s = Math.max(0, Math.round(t));
    return Math.floor(s / 60) + "분 " + String(s % 60).padStart(2, "0") + "초";
  };
  const wallTime = (t) => {
    if (!isNum(t)) return "";
    const d = new Date(t * 1000);
    return d.toTimeString().slice(0, 8);
  };
  // per-gauge inspection photo (gauge.image_url = session/gauges/<file>.jpg): same-origin paths or data URIs only
  const sameOrigin = (u) => typeof u === "string" && u !== "" && !/^[a-z][a-z0-9+.-]*:/i.test(u) &&
    !u.startsWith("//") && !u.startsWith("\\");
  const safeImg = (u) => (typeof u === "string" && (/^data:image\//i.test(u) || sameOrigin(u)) ? u : null);
  const safeHref = (u) => (sameOrigin(u) ? u : null);
  // leak tag of the layout ("GASTANK-01 relief valve") -> Korean
  const leakTag = (t) => (t ? lbl("leak", t).replace(/relief valve/i, "릴리프 밸브") : "");
  // leak-source error: the mission's value (seeker, full precision) once the source is
  // found, else the aggregator's estimate error -> one number everywhere on the page
  function leakError(s) {
    const ms = s && s.mission && s.mission.seek;
    if (ms && ms.result === "FOUND" && isNum(ms.error_m)) return ms.error_m;
    const g = (s && s.gas) || {};
    return isNum(g.estimate_error_m) ? g.estimate_error_m : null;
  }
  const fmtErrM = (v) => (isNum(v) ? v.toFixed(2) + " m" : "—");

  // ------------------------------------------------------------- state
  let STATIC = null;
  let SNAP = null;
  let lastMsg = 0;
  let lastEventSeq = -1;
  let REPORT = null;         // {t_sim, s}: data of the mission report, frozen at mission completion
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
  const legend = $("legend");
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
    if (lastMsg && Date.now() - lastMsg > 4000) setConn("late", ((Date.now() - lastMsg) / 1000).toFixed(0) + "초 동안 데이터 없음");
  }, 1000);

  // ============================================================ commands
  let fbTimer = null;
  function feedback(ok, text) {
    const f = $("cmdFeedback");
    f.className = "feedback " + (ok ? "ok" : "err");
    f.textContent = text;
    f.title = text;
    clearTimeout(fbTimer);
    fbTimer = setTimeout(() => { f.textContent = ""; f.className = "feedback"; f.title = ""; }, 4000);
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

  // ===================================================== drawer + tabs
  const drawer = $("drawer");
  const backdrop = $("drawerBackdrop");
  function setDrawer(open) {
    drawer.hidden = !open;
    backdrop.hidden = !open;
    $("btnDetails").setAttribute("aria-expanded", open ? "true" : "false");
    if (open) { if (SNAP) renderDetails(SNAP); $("drawerClose").focus(); }
  }
  $("btnDetails").addEventListener("click", () => setDrawer(drawer.hidden));
  $("drawerClose").addEventListener("click", () => setDrawer(false));
  backdrop.addEventListener("click", () => setDrawer(false));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !drawer.hidden) setDrawer(false); });

  let activeTab = "gauges";
  function setTab(name) {
    activeTab = name;
    const g = name === "gauges";
    $("tabGauges").setAttribute("aria-selected", g ? "true" : "false");
    $("tabReport").setAttribute("aria-selected", g ? "false" : "true");
    $("paneGauges").hidden = !g;
    $("paneReport").hidden = g;
    $("gaugeSummary").hidden = !g;
    $("reportStamp").hidden = g;
    if (!g && SNAP) renderReportView(SNAP);
    if (!g) $("reportDot").hidden = true;
  }
  $("tabGauges").addEventListener("click", () => setTab("gauges"));
  $("tabReport").addEventListener("click", () => setTab("report"));
  $("btnReport").addEventListener("click", () => {
    setTab("report");
    const card = $("paneReport").closest(".card");
    if (card && card.getBoundingClientRect().top > window.innerHeight - 120) card.scrollIntoView({ behavior: "smooth", block: "start" });
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

  // the map panel takes the aspect ratio of the plant, so the fitted map fills it
  function applyAspect() {
    const b = worldBounds();
    const bw = b[2] - b[0], bh = b[3] - b[1];
    if (bw > 0 && bh > 0) wrap.style.aspectRatio = (bw * 1.03).toFixed(2) + " / " + (bh * 1.03).toFixed(2);
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
    syncLegend();
    refreshLayers();
    queueDraw();
  }));
  TOGGLES.forEach((id) => {
    try { const v = localStorage.getItem("twin." + id); if (v !== null) $(id).checked = v === "1"; } catch (e) { /* ignore */ }
  });
  function syncLegend() { legend.classList.toggle("show-rssi", $("tgRssi").checked); }
  syncLegend();

  if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);
  else window.addEventListener("resize", resize);

  function queueDraw() {
    if (redrawQueued) return;
    redrawQueued = true;
    requestAnimationFrame(() => { redrawQueued = false; draw(); });
  }

  // ------------------------------------------------------- layer images
  // occupancy PNG -> theme colours (unknown transparent, free = floor, occupied = wall),
  // split in a floor and a wall image: the gas plume is drawn between the two, so it
  // tints the floor only, never the walls, machines or the slab outside the building
  function themeMap(img) {
    try {
      const mk = () => { const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight; return c; };
      const c = mk(), x = c.getContext("2d");
      x.drawImage(img, 0, 0);
      const d = x.getImageData(0, 0, c.width, c.height);
      const fl = new ImageData(c.width, c.height), wl = new ImageData(c.width, c.height);
      const p = d.data, pf = fl.data, pw = wl.data;
      for (let i = 0; i < p.length; i += 4) {
        const l = (p[i] + p[i + 1] + p[i + 2]) / 3;
        if (l < 20) { p[i + 3] = 0; continue; }
        const t = Math.min(1, Math.max(0, (l - 30) / 58));
        p[i] = MAP_FLOOR[0] + (MAP_WALL[0] - MAP_FLOOR[0]) * t;
        p[i + 1] = MAP_FLOOR[1] + (MAP_WALL[1] - MAP_FLOOR[1]) * t;
        p[i + 2] = MAP_FLOOR[2] + (MAP_WALL[2] - MAP_FLOOR[2]) * t;
        p[i + 3] = 255;
        const q = t < 0.25 ? pf : pw;
        q[i] = p[i]; q[i + 1] = p[i + 1]; q[i + 2] = p[i + 2]; q[i + 3] = 255;
      }
      x.putImageData(d, 0, 0);
      const cf = mk(), cw = mk();
      cf.getContext("2d").putImageData(fl, 0, 0);
      cw.getContext("2d").putImageData(wl, 0, 0);
      return { all: c, floor: cf, walls: cw };
    } catch (e) {
      return { all: img, floor: null, walls: null };   // tainted / no canvas support: original colours
    }
  }

  function loadLayer(name, path) {
    const L = layers[name];
    const meta = SNAP && SNAP.layers ? SNAP.layers[name] : null;
    if (!meta || L.loading || meta.version === L.version) return;
    L.loading = true;
    const img = new Image();
    img.onload = () => {
      if (name === "map") {
        const t = themeMap(img);
        L.img = t.all; L.floor = t.floor; L.walls = t.walls;
      } else {
        L.img = img;
      }
      L.meta = meta; L.version = meta.version; L.loading = false;
      if (name === "map" && !view.fitted) fit();
      queueDraw();
    };
    img.onerror = () => { L.loading = false; };
    img.src = path + "?v=" + meta.version;
  }

  function refreshLayers() {
    loadLayer("map", "map.png");
    if ($("tgGas").checked) loadLayer("gas", "gas.png");
    if ($("tgRssi").checked) loadLayer("rssi", "rssi.png");
  }

  // part: "floor" / "walls" of the themed map (default: the whole image)
  function drawLayer(name, smooth, part) {
    const L = layers[name];
    const src = part ? L[part] : L.img;
    if (!src || !L.meta) return;
    const m = L.meta;
    const w = m.width * m.resolution, h = m.height * m.resolution;
    const [sx, sy] = w2s(m.origin[0], m.origin[1] + h);
    ctx.imageSmoothingEnabled = !!smooth;
    ctx.drawImage(src, sx, sy, w * view.scale, h * view.scale);
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

  function rrect(x, y, w, h, r) {
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h);
  }

  // --- label placement: every mark reserves its box, labels take the first free
  // candidate position around their anchor and are skipped when none is free
  let placed = [];
  const reserve = (x0, y0, x1, y1) => placed.push([x0, y0, x1, y1]);
  const reserveAt = (x, y, r) => placed.push([x - r, y - r, x + r, y + r]);
  const outside = (r) => r[0] < 2 || r[1] < 2 || r[2] > W - 2 || r[3] > H - 2;
  const blocked = (r) => outside(r) ||
    placed.some((p) => r[0] < p[2] && r[2] > p[0] && r[1] < p[3] && r[3] > p[1]);
  // sticky placement: a label keeps the side it had in the previous frame while that
  // side stays free, so labels do not jump around as the robot passes
  const labelPrev = new Map();

  function mapLabel(text, x, y, o) {
    o = o || {};
    const size = o.size || 12;
    ctx.save();
    ctx.font = (o.weight || 600) + " " + size + "px " + FONT;
    const tw = ctx.measureText(text).width;
    const pad = o.plain ? 1 : 5, h = size + 6, w = tw + pad * 2;
    const g = o.gap === undefined ? 9 : o.gap;
    const cands = o.center ? [[x - w / 2, y - h / 2]] : [
      [x + g, y - h / 2], [x - g - w, y - h / 2], [x - w / 2, y - g - h], [x - w / 2, y + g],
      [x + g - 3, y - g - h + 3], [x + g - 3, y + g - 3], [x - g - w + 3, y - g - h + 3], [x - g - w + 3, y + g - 3]];
    const prev = labelPrev.get(text);
    const order = cands.map((_, i) => i);
    if (isNum(prev) && prev > 0 && prev < cands.length) { order.splice(prev, 1); order.unshift(prev); }
    const rectOf = (i) => [cands[i][0], cands[i][1], cands[i][0] + w, cands[i][1] + h];
    let pick = order.find((i) => !blocked(rectOf(i)));
    // forced labels (dock, current target): never dropped, drawn over a neighbour if need be
    if (pick === undefined && o.force) pick = order.find((i) => !outside(rectOf(i)));
    if (pick !== undefined) {
      const r = rectOf(pick), lx = r[0], ly = r[1];
      placed.push(r);
      labelPrev.set(text, pick);
      if (!o.plain) { ctx.fillStyle = o.bg || LABEL_BG; rrect(lx, ly, w, h, 4); ctx.fill(); }
      ctx.textBaseline = "middle"; ctx.textAlign = "left";
      if (o.plain) { ctx.lineWidth = 3; ctx.strokeStyle = C.inset; ctx.strokeText(text, lx + pad, ly + h / 2 + 0.5); }
      ctx.fillStyle = o.color || C.text2;
      ctx.fillText(text, lx + pad, ly + h / 2 + 0.5);
    }
    ctx.restore();
    return pick !== undefined;
  }

  // plain centred text without placement (empty-state message)
  function centerText(text, x, y, color, size) {
    ctx.save();
    ctx.font = (size || 13) + "px " + FONT;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillStyle = color || C.muted;
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
    ctx.strokeStyle = C.inset;
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

  function drawFrictionZones(snap) {
    const zones = frictionZones();
    const out = [];
    if (!zones.length) return out;
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
      ctx.fillStyle = on ? "rgba(245,181,68,0.16)" : "rgba(143,187,224,0.08)";
      ctx.fillRect(x0, y0, w, h);
      // diagonal hatch = shape encoding, the plate reads without colour too
      ctx.strokeStyle = col; ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let d = -h; d < w; d += 6) { ctx.moveTo(x0 + d, y0 + h); ctx.lineTo(x0 + d + h, y0); }
      ctx.stroke();
      ctx.restore();
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = on ? 2 : 1; ctx.strokeRect(x0, y0, w, h); ctx.restore();
      out.push({ z: z, x: x0 + w / 2, y: y0, on: on });
    });
    return out;
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
    const k = Math.max(view.scale, 16 / body[0]);       // never smaller than ~16 px long
    const halo = Math.max(body[0] * k * 0.95, 17);
    reserveAt(px, py, halo);
    ctx.save();
    ctx.fillStyle = "rgba(79,143,247,0.20)";
    ctx.beginPath(); ctx.arc(px, py, halo, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "rgba(79,143,247,0.65)"; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.translate(px, py);
    ctx.rotate(-(r.yaw || 0));
    feet.forEach((f) => {
      ctx.beginPath(); ctx.arc(f[0] * k, -f[1] * k, Math.max(2, 0.035 * k), 0, Math.PI * 2);
      ctx.fillStyle = C.text2; ctx.fill();
    });
    const L = body[0] * k, Wd = body[1] * k;
    rrect(-L / 2, -Wd / 2, L, Wd, Math.min(4, Wd / 3));
    ctx.fillStyle = C.text; ctx.fill();
    ctx.lineWidth = 2; ctx.strokeStyle = C.inset; ctx.stroke();
    ctx.beginPath();                                   // heading wedge
    ctx.moveTo(L / 2 + Math.max(7, 0.18 * k), 0); ctx.lineTo(L / 2 - 2, -Wd / 2); ctx.lineTo(L / 2 - 2, Wd / 2); ctx.closePath();
    ctx.fillStyle = C.accent; ctx.fill();
    ctx.restore();
  }

  // scale bar + north arrow in the top-right corner (reserved before the
  // labels are placed, drawn last so the map image never covers it)
  function scaleBarBox() {
    const target = 110 / view.scale;                   // metres for ~110 px
    const nice = [0.5, 1, 2, 5, 10, 20, 50, 100].find((v) => v >= target * 0.6) || 100;
    const len = nice * view.scale;
    const nx = W - 20;
    const x1 = nx - 22, x0 = x1 - len;
    return { nice: nice, nx: nx, ny: 22, x0: x0, x1: x1, y: 30, rect: [x0 - 10, 8, nx + 12, 46] };
  }
  function drawScaleBar(b) {
    const nice = b.nice, nx = b.nx, ny = b.ny, x0 = b.x0, x1 = b.x1, y = b.y;
    ctx.save();
    ctx.fillStyle = "rgba(11,14,18,0.72)";
    rrect(x0 - 10, 8, (nx + 12) - (x0 - 10), 38, 6); ctx.fill();
    ctx.strokeStyle = C.text2; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x0, y - 4); ctx.lineTo(x0, y); ctx.lineTo(x1, y); ctx.lineTo(x1, y - 4); ctx.stroke();
    ctx.font = "600 11px " + FONT; ctx.fillStyle = C.text2; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(nice + " m", (x0 + x1) / 2, y - 12);
    ctx.translate(nx, ny);
    ctx.beginPath(); ctx.moveTo(0, -9); ctx.lineTo(5, 5); ctx.lineTo(0, 2); ctx.lineTo(-5, 5); ctx.closePath(); ctx.fill();
    ctx.fillText("N", 0, 16);
    ctx.restore();
  }

  function hitList() {
    // entities that can be hovered: {x, y (world), title, value, rows[]}
    const out = [];
    if (!STATIC) return out;
    const gl = {};
    (SNAP && SNAP.gauges || []).forEach((g) => { gl[g.id] = g; });
    (STATIC.gauges || []).forEach((g) => {
      const s = gl[g.id] || {};
      out.push({ x: g.x, y: g.y, title: g.id + " · " + g.tag, value: isNum(s.value) ? s.value.toFixed(3) + " " + (s.unit || g.unit) : lbl("status", "unread"),
        rows: [["상태", statusInfo(s.status).label], ["참값", g.true_value + " " + g.unit], ["오차", isNum(s.error_pct_fs) ? s.error_pct_fs.toFixed(2) + " % FS" : "—"],
          ["신뢰도", fmt(s.confidence, 2)], ["범위", g.min + " – " + g.max + " " + g.unit]] });
    });
    const tl = {};
    (SNAP && SNAP.thermal || []).forEach((h) => { tl[h.id] = h; });
    (STATIC.hotspots || []).forEach((h) => {
      const s = tl[h.id] || {};
      out.push({ x: h.x, y: h.y, title: h.id + " · " + h.tag + " (" + lbl("kind", h.kind) + ")", value: isNum(s.max_temp_c) ? s.max_temp_c.toFixed(1) + " °C" : lbl("status", "unseen"),
        rows: [["상태", statusInfo(s.status).label], ["참값", h.true_temp_c + " °C"], ["최고 관측", isNum(s.peak_c) ? s.peak_c.toFixed(1) + " °C" : "—"]] });
    });
    (STATIC.aps || []).forEach((a) => out.push({ x: a.x, y: a.y, title: "무선 액세스 포인트 " + a.id, value: "Wi-Fi AP", rows: [] }));
    if (STATIC.dock) out.push({ x: STATIC.dock.x, y: STATIC.dock.y, title: "충전소", value: "복귀 지점", rows: [["방향 (yaw)", deg(STATIC.dock.yaw)]] });
    const leak = STATIC.gas_leak;
    if (leak && isNum(leak.x)) out.push({ x: leak.x, y: leak.y, title: "가스 누출원 (실제 위치)", value: leakTag(leak.tag) || leak.id || "누출원", rows: [] });
    const gas = SNAP && SNAP.gas;
    const lerr = leakError(SNAP);
    if (gas && gas.estimate && isNum(gas.estimate[0]) && isNum(gas.estimate[1])) out.push({ x: gas.estimate[0], y: gas.estimate[1], title: "누출원 추정 위치 (추적기)", value: isNum(lerr) ? "오차 " + fmtErrM(lerr) : "추정", rows: [] });
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
    (STATIC.machines || []).forEach((m) => {
      const f = m.footprint;
      out.push({ x: (f.x_min + f.x_max) / 2, y: (f.y_min + f.y_max) / 2, title: "설비", value: m.id, rows: [] });
    });
    const r = SNAP && SNAP.robot;
    if (r && isNum(r.x)) {
      const l = SNAP.locomotion || {};
      out.push({ x: r.x, y: r.y, title: "로봇 (spot)", value: fmt(r.x, 2) + ", " + fmt(r.y, 2) + " m",
        rows: [["방향 (yaw)", deg(r.yaw)], ["구역", lbl("zone", r.zone, "—")], ["보행 상태", lbl("state", l.mode, "—")], ["미끄럼 모드", slipText(l) || "—"]] });
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

  function reserveLegend() {
    if (getComputedStyle(legend).position !== "absolute") return;
    const a = legend.getBoundingClientRect(), b = wrap.getBoundingClientRect();
    if (a.width && a.height) reserve(a.left - b.left - 4, a.top - b.top - 4, a.right - b.left + 4, a.bottom - b.top + 4);
  }

  function draw() {
    placed = [];
    ctx.fillStyle = C.inset;
    ctx.fillRect(0, 0, W, H);
    if (!STATIC) { centerText("대시보드 노드 연결 대기 중…", W / 2, H / 2, C.muted, 13); return; }
    const snap = SNAP || {};
    const showLabels = $("tgLabels").checked;
    const z = view.scale;
    if (view.follow && snap.robot && isNum(snap.robot.x)) { view.cx = snap.robot.x; view.cy = snap.robot.y; }
    reserveLegend();
    const sb = scaleBarBox();
    reserve(sb.rect[0], sb.rect[1], sb.rect[2], sb.rect[3]);

    // floor -> plates / signal / gas -> walls + machines (the plume never tints a wall)
    const split = !!(layers.map.floor && layers.map.walls);
    drawLayer("map", false, split ? "floor" : null);
    if (!layers.map.img) {
      // no map yet: outline the building shells from the layout
      [STATIC.factory, STATIC.corridor].forEach((r) => {
        if (!r) return;
        const [x0, y0] = w2s(r.x_min, r.y_max), [x1, y1] = w2s(r.x_max, r.y_min);
        ctx.strokeStyle = C.baseline; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      });
    }
    const plates = $("tgFriction").checked ? drawFrictionZones(snap) : [];
    const shadowLabels = [];
    if ($("tgRssi").checked) {
      drawLayer("rssi", true);
      (STATIC.shadow_zones || []).forEach((sz) => {
        const [x0, y0] = w2s(Math.min(sz.rect[0], sz.rect[2]), Math.max(sz.rect[1], sz.rect[3]));
        const [x1, y1] = w2s(Math.max(sz.rect[0], sz.rect[2]), Math.min(sz.rect[1], sz.rect[3]));
        ctx.save(); ctx.strokeStyle = C.text2; ctx.setLineDash([3, 3]); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
        shadowLabels.push([(x0 + x1) / 2, (y0 + y1) / 2]);
      });
    }
    if ($("tgGas").checked) drawLayer("gas", true);
    if (split) drawLayer("map", false, "walls");
    const zoneLabels = [];
    if ($("tgZones").checked) {
      (STATIC.zones || []).forEach((zn) => {
        const [x0, y0] = w2s(Math.min(zn.rect[0], zn.rect[2]), Math.max(zn.rect[1], zn.rect[3]));
        const [x1, y1] = w2s(Math.max(zn.rect[0], zn.rect[2]), Math.min(zn.rect[1], zn.rect[3]));
        ctx.save(); ctx.strokeStyle = C.muted; ctx.setLineDash([2, 3]); ctx.lineWidth = 1; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.restore();
        zoneLabels.push([lbl("zone", zn.name), x0, y0]);
      });
    }

    const paths = snap.paths || {};
    const plan = snap.plan;
    const vpLabels = [];
    if ($("tgRoute").checked) {
      pathLine(paths.route, "rgba(182,191,205,0.75)", 1.5, [6, 5]);
      if (plan && plan.viewpoints) {
        plan.viewpoints.forEach((v) => {
          const [x, y] = w2s(v.x, v.y);
          const a = -(v.yaw || 0);
          // camera viewpoint: open chevron pointing along the view direction
          // (not a filled triangle, which would read as a 'warning' status mark)
          ctx.save(); ctx.translate(x, y); ctx.rotate(a);
          ctx.beginPath(); ctx.moveTo(-3, -5); ctx.lineTo(3.5, 0); ctx.lineTo(-3, 5);
          ctx.lineJoin = "round"; ctx.lineCap = "round";
          ctx.lineWidth = 4.5; ctx.strokeStyle = C.inset; ctx.stroke();
          ctx.globalAlpha = 0.7; ctx.lineWidth = 2; ctx.strokeStyle = C.text2; ctx.stroke(); ctx.restore();
          const idx = plan.order ? plan.order.indexOf(v.gauge_id) : -1;
          if (idx >= 0) vpLabels.push([String(idx + 1), x, y]);
        });
      }
    }
    pathLine(paths.seek, C.seek, 2.5);
    pathLine(paths.nav, C.accent, 2.5);
    pathLine(paths.mpc, C.text, 1.5, [1.5, 3.5]);

    // dock + access points
    let dockAnchor = null;
    if (STATIC.dock) {
      const d = STATIC.dock, s = d.size || [1, 1];
      const [x0, y0] = w2s(d.x - s[0] / 2, d.y + s[1] / 2);
      const dw = Math.max(10, s[0] * z), dh = Math.max(10, s[1] * z);
      ctx.save(); ctx.strokeStyle = C.accent; ctx.lineWidth = 2;
      ctx.fillStyle = "rgba(79,143,247,0.12)"; ctx.fillRect(x0, y0, dw, dh); ctx.strokeRect(x0, y0, dw, dh); ctx.restore();
      reserve(x0 - 2, y0 - 2, x0 + dw + 2, y0 + dh + 2);
      dockAnchor = [x0 + dw / 2, y0 + dh / 2, dw / 2];
    }
    const apAnchors = [];
    (STATIC.aps || []).forEach((a) => {
      const [x, y] = w2s(a.x, a.y);
      ctx.save(); ctx.strokeStyle = C.muted; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, y + 5); ctx.lineTo(x, y - 3); ctx.stroke();
      [4, 7.5].forEach((rr) => { ctx.beginPath(); ctx.arc(x, y - 3, rr, -Math.PI * 0.8, -Math.PI * 0.2); ctx.stroke(); });
      ctx.restore();
      reserve(x - 8, y - 11, x + 8, y + 6);
      apAnchors.push([a.id, x, y]);
    });

    // gas leak truth vs estimate
    const leak = STATIC.gas_leak;
    let leakAnchor = null, estAnchor = null;
    if (leak && isNum(leak.x)) {
      const [x, y] = w2s(leak.x, leak.y);
      const est = snap.gas && snap.gas.estimate;
      if (est && isNum(est[0]) && isNum(est[1])) {
        const [ex, ey] = w2s(est[0], est[1]);
        ctx.save(); ctx.strokeStyle = C.seek; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(x, y); ctx.stroke(); ctx.restore();
        glyph("circle", ex, ey, 5, C.seek);
        reserveAt(ex, ey, 7);
        estAnchor = [ex, ey];
      }
      ctx.save(); ctx.strokeStyle = C.text; ctx.lineWidth = 2.5; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6); ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6); ctx.stroke();
      ctx.restore();
      reserveAt(x, y, 8);
      leakAnchor = [x, y];
    }

    // hot spots (status glyph framed by a square) and gauges (status glyph)
    const tl = {};
    (snap.thermal || []).forEach((h) => { tl[h.id] = h; });
    const hotAnchors = [];
    (STATIC.hotspots || []).forEach((h) => {
      const s = tl[h.id] || {};
      const info = statusInfo(s.status);
      const [x, y] = w2s(h.x, h.y);
      ctx.save(); ctx.fillStyle = "rgba(11,14,18,0.6)"; ctx.fillRect(x - 8, y - 8, 16, 16);
      ctx.strokeStyle = info.shape === "ring" ? C.muted : info.color; ctx.lineWidth = 1.25; ctx.strokeRect(x - 8, y - 8, 16, 16); ctx.restore();
      glyph(info.shape, x, y, 4, info.color);
      reserve(x - 9, y - 9, x + 9, y + 9);
      hotAnchors.push([h.id, x, y, s]);
    });
    const gl = {};
    (snap.gauges || []).forEach((g) => { gl[g.id] = g; });
    const target = snap.mission && snap.mission.target;
    const gaugeAnchors = [];
    (STATIC.gauges || []).forEach((g) => {
      const s = gl[g.id] || {};
      const info = statusInfo(s.status);
      const [x, y] = w2s(g.x, g.y);
      if (target === g.id) {
        ctx.save(); ctx.strokeStyle = C.accent; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
      }
      glyph(info.shape, x, y, 6, info.color);
      reserveAt(x, y, target === g.id ? 14 : 8);
      gaugeAnchors.push([g.id, x, y, target === g.id]);
    });

    drawRobot(snap.robot);

    // labels, most important first; ids only (values: tooltip and side lists)
    if (showLabels) {
      gaugeAnchors.forEach((a) => mapLabel(a[0], a[1], a[2], { color: a[3] ? "#ffffff" : C.text, size: 12, weight: 700,
        bg: a[3] ? "rgba(47,111,224,0.92)" : LABEL_BG, gap: a[3] ? 15 : 10, force: a[3] }));
      if (dockAnchor) mapLabel("충전소", dockAnchor[0], dockAnchor[1], { color: "#cfe0ff", gap: dockAnchor[2] + 5, force: true });
      if (leakAnchor) mapLabel("누출원", leakAnchor[0], leakAnchor[1], { color: C.text, gap: 10 });
      hotAnchors.forEach((a) => mapLabel(a[0], a[1], a[2], { color: C.text2, size: 11.5, gap: 11 }));
      if (estAnchor && z > 10) mapLabel("추정", estAnchor[0], estAnchor[1], { color: C.seek, size: 11, gap: 8 });
      if (z > 24) apAnchors.forEach((a) => mapLabel(a[0], a[1], a[2] - 3, { color: C.muted, size: 11, plain: true }));
      if (z > 24) plates.forEach((p) => { if (isNum(p.z.mu)) mapLabel("μ " + p.z.mu.toFixed(2), p.x, p.y - 9, { color: p.on ? C.warning : C.text2, size: 11, center: true }); });
      if (z > 28) vpLabels.forEach((v) => mapLabel(v[0], v[1], v[2], { color: C.text2, size: 10.5, gap: 7 }));
      shadowLabels.forEach((s) => mapLabel("전파 음영", s[0], s[1], { color: C.text2, size: 11, center: true }));
      zoneLabels.forEach((zl) => mapLabel(zl[0], zl[1] + 4, zl[2] + 4, { color: C.muted, size: 11, plain: true, gap: 0 }));
      if (z > 9) {
        (STATIC.machines || []).forEach((m) => {
          const f = m.footprint;
          const [x, y] = w2s((f.x_min + f.x_max) / 2, (f.y_min + f.y_max) / 2);
          mapLabel(m.id, x, y, { color: C.text2, size: 10.5, weight: 600, plain: true, center: true });
        });
      }
    }
    drawScaleBar(sb);

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
        ctx.save(); ctx.strokeStyle = C.text; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, 13, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
        showTip(best, hover.x, hover.y);
      } else {
        tip.hidden = true;
      }
    }
  }

  // ============================================================ panels
  function badgeClass(state) {
    if (["SEEK", "FALLEN", "ESTOP"].includes(state)) return "badge s-alarm";
    if (["RTH", "MANUAL", "PAUSED", "RECOVERING", "STANDING_UP", "RTH_PLANNING", "RTH_NAVIGATING", "DOCKING"].includes(state)) return "badge s-warn";
    if (["PATROL", "INSPECTING", "WALK", "STAND"].includes(state)) return "badge s-active";
    if (["DOCKED", "DONE", "CHARGING", "MONITOR"].includes(state)) return "badge s-good";
    return "badge";
  }
  // state badge: Korean label, raw value (PATROL, RTH_PLANNING ...) in the tooltip
  function setBadge(id, state, size, kind) {
    const b = $(id);
    b.className = badgeClass(state) + (size ? " " + size : "");
    b.textContent = state ? lbl(kind || "state", state) : "데이터 없음";
    b.title = state || "데이터 없음";
  }
  // o: {label, glyph, cls} overrides (e.g. "판독 중" on the current target)
  function statusEl(status, o) {
    o = o || {};
    const info = statusInfo(status);
    const s = el("span", "status " + (o.cls || (STATUS[status] ? status : "unread")));
    s.appendChild(el("i", "ico", o.glyph || info.glyph));
    s.appendChild(document.createTextNode(o.label || info.label));
    if (status) s.title = String(status);
    return s;
  }
  function stale(id, age, limit) {
    const c = $(id);
    if (c) c.classList.toggle("stale", !isNum(age) || age > limit);
  }

  // ---------------------------------------------------------- mission
  function missionFinished(m) {
    const pr = m.progress || [];
    const total = pr.length === 2 && isNum(pr[1]) ? pr[1] : 0;
    const handled = (m.completed || []).length + (m.missed || []).length;
    return ["DOCKED", "DONE"].includes(m.state) && total > 0 && handled >= total;
  }

  const RUNNING = ["PATROL", "INSPECTING", "SEEK", "RTH", "PAUSED"];
  function renderMission(s) {
    const m = s.mission || {};
    const st = m.state || null;
    const fin = missionFinished(m);
    setBadge("missionState", st, "xl");
    // the planned return after the last gauge is part of the mission, not a warning
    if (st === "RTH" && m.rth_reason === "mission_complete") $("missionState").className = "badge s-active xl";
    if (fin) {                                     // docked after the whole route: say so
      const b = $("missionState");
      b.className = "badge s-good xl";
      b.textContent = "임무 완료";
      b.title = st + " · " + lbl("state", st);
    }
    // the phase belongs to a moving mission (stale while paused / finished)
    const showPhase = st !== "PAUSED" && !fin;
    setText("missionPhase", showPhase ? [m.mode ? lbl("mode", m.mode) : "", m.phase ? lbl("phase", m.phase) : ""].filter(Boolean).join(" · ") : "");
    $("missionPhase").title = [m.mode, m.phase].filter(Boolean).join(" · ");
    const tgt = m.target || (st === "SEEK" ? "누출원" : st === "RTH" ? "충전소" : fin ? "완료" : "—");
    setText("missionTarget", tgt);
    $("missionTarget").title = m.target ? m.target + " 점검 지점" : st === "SEEK" ? "가스 누출원 추적" : st === "RTH" ? "충전소로 복귀" : "";
    const pr = m.progress || [];
    const total = pr.length === 2 && pr[1] ? pr[1] : 0;
    setText("missionProgress", total ? pr[0] + " / " + total : "—");
    $("missionBar").style.width = (total ? Math.min(100, 100 * pr[0] / total) : 0) + "%";
    setText("missionElapsed", isNum(m.mission_elapsed) ? fmtMinSec(m.mission_elapsed) : "—");
    $("missionElapsed").title = isNum(m.mission_elapsed) ? "임무 시작 후 시뮬레이션 시간 " + fmtClock(m.mission_elapsed) : "";
    $("btnStart").classList.toggle("is-running", RUNNING.includes(st));
    const msg = KO.event(m.message || "", "mission");
    setText("missionMessage", msg);
    $("missionMessage").title = m.message ? msg + "\n" + m.message : "";

    // route steps in planned order
    const chips = $("missionChips");
    chips.textContent = "";
    const done = new Set(m.completed || []), missed = new Set(m.missed || []);
    const order = (s.plan && s.plan.order && s.plan.order.length) ? s.plan.order : (STATIC ? STATIC.gauges.map((g) => g.id) : []);
    order.forEach((id, i) => {
      let cls = "step", mark = "", tipText = (i + 1) + "번째 · " + id;
      if (done.has(id)) { cls += " done"; mark = "✓"; tipText += " 판독 완료"; }
      else if (missed.has(id)) { cls += " missed"; mark = "✕"; tipText += " 판독 실패"; }
      else if (m.target === id) { cls += " current"; mark = "▸"; tipText += " 점검 중"; }
      else tipText += " 대기";
      const sp = el("span", cls);
      if (mark) sp.appendChild(el("span", "sm", mark));
      sp.appendChild(document.createTextNode(id));
      sp.title = tipText;
      chips.appendChild(sp);
    });

    const pb = $("btnPause");
    const paused = st === "PAUSED";
    setText("btnPauseLabel", paused ? "재개" : "일시정지");
    setText("btnPauseIcon", paused ? "▶" : "❚❚");
    pb.title = paused ? "일시정지한 임무를 이어서 진행" : "임무를 일시정지하고 제자리에서 대기";
    pb.setAttribute("data-cmd", paused ? "resume" : "pause");
    pb.classList.toggle("is-paused", paused);

    $("btnReport").classList.toggle("ready", fin);
    $("btnReport").textContent = fin ? "✓ 임무 보고서 보기" : "임무 보고서";
    $("reportDot").hidden = !fin || activeTab === "report";
    stale("cardMission", m.age, 6);
  }

  // ------------------------------------------------------ robot + power
  function renderRobot(s) {
    const r = s.robot || {};
    const l = s.locomotion || {};
    const t = s.terrain || {};
    setBadge("locoMode", l.mode || null, "sm");
    // the gait only means something while walking ("서 있음 트롯" reads oddly)
    setText("locoGait", l.gait && l.mode === "WALK" ? lbl("gait", l.gait) : "");
    $("locoGait").title = l.gait ? "보행 방식 " + lbl("gait", l.gait) + (l.gait_auto ? " (자동 선택)" : "") : "";
    setText("robotZone", r.zone ? "위치 · " + lbl("zone", r.zone) : "");
    $("robotZone").title = "현재 구역" + (r.zone ? " (" + r.zone + ")" : "");
    setText("robotVel", isNum(r.vx) ? Math.abs(r.vx).toFixed(2) + " m/s" : "—");
    $("robotVel").title = isNum(r.vx) ? r.vx.toFixed(2) + " m/s · " + (r.wz || 0).toFixed(2) + " rad/s" : "";
    const falls = isNum(l.fall_count) ? l.fall_count : null;
    setText("locoFalls", (falls === null ? "—" : falls + "회") + " / " + (isNum(l.recovery_success) ? l.recovery_success + "회" : "—"));
    $("locoFalls").className = "v" + (falls ? " warn-text" : "");
    const slipOn = l.slip_mode === true;
    const sb = $("slipBadge");
    sb.hidden = !(slipOn || t.slippery === true);
    sb.textContent = slipOn ? "▲ 미끄럼 대응" : "▲ 미끄러운 바닥";
    sb.title = slipText(l) || "";
    stale("cardRobot", isNum(r.age) ? r.age : l.age, 5);
  }

  function renderSafety(s) {
    const sf = s.safety || {};
    const b = s.battery || {};
    const pct = isNum(b.pct) ? b.pct : (isNum(sf.battery_pct) ? (sf.battery_pct <= 1 ? sf.battery_pct * 100 : sf.battery_pct) : null);
    const bp = $("battPct");
    bp.textContent = isNum(pct) ? pct.toFixed(0) : "—";
    bp.appendChild(el("small", null, "%"));
    const meter = $("battMeter");
    meter.className = "meter" + (!isNum(pct) ? " na" : pct < 20 ? " crit" : pct < 40 ? " warn" : "");
    $("battFill").style.width = (isNum(pct) ? Math.max(0, Math.min(100, pct)) : 0) + "%";
    if (isNum(pct)) meter.setAttribute("aria-valuenow", pct.toFixed(0));
    const ch = $("charging");
    ch.textContent = b.charging === true ? "충전 중" : sf.docked ? "충전소 도킹" : b.charging === false ? "방전 중" : "—";
    ch.className = "batt-sub" + (b.charging === true ? " is-charging" : "");
    const rssi = s.rssi && isNum(s.rssi.dbm) ? s.rssi.dbm : sf.rssi_dbm;
    const minR = STATIC && STATIC.comm ? STATIC.comm.rssi_min_dbm : -80;
    const rs = $("rssi");
    rs.textContent = isNum(rssi) ? rssi.toFixed(0) + " dBm" + (rssi < minR ? " ▲ 음영" : rssi < minR + 10 ? " 약함" : " 양호") : "—";
    rs.className = "v" + (isNum(rssi) && rssi < minR ? " warn-text" : "");
    rs.title = isNum(rssi) ? rssi.toFixed(1) + " dBm (음영 한계 " + minR + " dBm)" : "";
    const reason = sf.reason || (sf.rth_active ? "active" : "");
    const rt = $("rthState");
    if (sf.rth_active || (sf.state && /^RTH|DOCKING/.test(sf.state))) {
      rt.textContent = "작동 · " + lbl("reason", reason, lbl("state", sf.state));
      rt.className = "v warn-text";
    } else {
      rt.textContent = sf.state ? "대기" : "—";
      rt.className = "v";
    }
    rt.title = "안전 계층: " + lbl("state", sf.state, "—") + (reason ? " · " + lbl("reason", reason) : "");
    setText("distHome", fmt(sf.distance_home, 1, "m"));
    stale("cardSafety", isNum(b.age) ? b.age : null, 6);
  }

  // ---------------------------------------------------------- alarms
  function renderGas(s) {
    const g = s.gas || {};
    const pp = $("gasPpm");
    const ppm = isNum(g.ppm) ? Math.max(0, g.ppm) : null;
    pp.textContent = ppm === null ? "—" : ppm.toFixed(1);    // one decimal: no 5 Hz flicker in the 2nd one
    pp.appendChild(el("small", null, "ppm"));
    pp.title = isNum(g.ppm) ? "센서 원시값 " + g.ppm.toFixed(3) + " ppm" : "";
    const ab = $("gasAlarm");
    const lim = g.alarm_ppm || 25;
    if (g.alarm) { ab.className = "badge sm s-alarm"; ab.textContent = "경보 · " + lim + " ppm 초과"; }
    else if (isNum(g.ppm)) { ab.className = "badge sm s-good"; ab.textContent = "정상 · 기준 " + lim + " ppm"; }
    else { ab.className = "badge sm"; ab.textContent = "센서 없음"; }
    const card = $("cardGas").closest(".card");
    if (card) card.classList.toggle("is-alarm", !!g.alarm);
    const seek = g.seek || {};
    const ms = (s.mission && s.mission.seek) || null;
    let seekTxt = lbl("seek", seek.state, "—");
    if ((!seek.state || seek.state === "IDLE") && ms && ms.result) seekTxt = lbl("seek", ms.result);
    setText("seekState", seekTxt);
    $("seekState").title = [seek.state, ms && ms.result].filter(Boolean).join(" · ");
    setText("seekErr", fmtErrM(leakError(s)));
    const f = g.field || {};
    if (isNum(f.vis_max_ppm)) setText("gasScale", "0 – " + f.vis_max_ppm + " ppm" + (f.scale && f.scale !== "linear" ? " (" + lbl("scale", f.scale) + ")" : ""));
    setText("gasAge", isNum(g.age) ? "가스 센서 " + fmtAge(g.age) : "가스 센서 데이터 없음");
    stale("cardGas", g.age, 5);
    drawSpark(g);
  }

  // gas sparkline (single series -> no legend; alarm reference line labelled)
  const spark = $("gasSpark");
  const sctx = spark.getContext("2d");
  let sparkHover = null;
  let sparkData = null;
  spark.addEventListener("pointermove", (e) => { sparkHover = e.offsetX; if (sparkData) drawSpark(sparkData); });
  spark.addEventListener("pointerleave", () => { sparkHover = null; $("sparkTip").hidden = true; if (sparkData) drawSpark(sparkData); });

  function niceCeil(v) {
    const p = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-9))));
    const n = v / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }

  function drawSpark(g) {
    sparkData = g;
    const r = spark.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, r.width), h = Math.max(10, r.height);
    if (spark.width !== Math.round(w * dpr) || spark.height !== Math.round(h * dpr)) { spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr); }
    sctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sctx.clearRect(0, 0, w, h);
    const hist = (g.history || []).filter((p) => isNum(p[0]) && isNum(p[1])).map((p) => [p[0], Math.max(0, p[1])]);
    const alarm = g.alarm_ppm || 25;
    const pad = { l: 26, r: 4, t: 6, b: 15 };
    const span = 120;
    // 1-2-5 rounded top, at least twice the alarm level: the alarm line sits mid-chart,
    // clear of the top label
    const vmax = niceCeil(Math.max(alarm * 2, ...hist.map((p) => p[1] * 1.1), 1));
    const X = (dt) => pad.l + (w - pad.l - pad.r) * (1 + dt / span);
    const Y = (v) => h - pad.b - (h - pad.t - pad.b) * (v / vmax);
    sctx.strokeStyle = C.baseline; sctx.lineWidth = 1;
    sctx.beginPath(); sctx.moveTo(pad.l, Y(0) + 0.5); sctx.lineTo(w - pad.r, Y(0) + 0.5); sctx.stroke();
    sctx.font = "10.5px " + FONT; sctx.fillStyle = C.muted; sctx.textBaseline = "middle"; sctx.textAlign = "right";
    sctx.fillText("0", pad.l - 5, Y(0));
    sctx.fillText(vmax.toFixed(0), pad.l - 5, Y(vmax) + 4);
    sctx.textAlign = "left"; sctx.fillText("−" + span + "초", pad.l, h - 5);
    sctx.textAlign = "right"; sctx.fillText("현재", w - pad.r, h - 5);
    // alarm reference
    sctx.strokeStyle = C.critical; sctx.lineWidth = 1; sctx.setLineDash([4, 3]);
    sctx.beginPath(); sctx.moveTo(pad.l, Y(alarm) + 0.5); sctx.lineTo(w - pad.r, Y(alarm) + 0.5); sctx.stroke();
    sctx.setLineDash([]);
    sctx.fillStyle = C.text2; sctx.textAlign = "left"; sctx.fillText("경보 기준 " + alarm + " ppm", pad.l + 4, Y(alarm) - 7);
    if (hist.length < 2) {
      sctx.fillStyle = C.muted; sctx.textAlign = "center"; sctx.fillText("/gas/sensor 수신 대기 중", (w + pad.l) / 2, h / 2);
      return;
    }
    sctx.fillStyle = "rgba(232,116,59,0.12)";
    sctx.beginPath(); sctx.moveTo(X(hist[0][0]), Y(0));
    hist.forEach((p) => sctx.lineTo(X(p[0]), Y(p[1])));
    sctx.lineTo(X(hist[hist.length - 1][0]), Y(0)); sctx.closePath(); sctx.fill();
    sctx.strokeStyle = C.gas; sctx.lineWidth = 2; sctx.lineJoin = "round"; sctx.lineCap = "round";
    sctx.beginPath();
    hist.forEach((p, i) => { if (i) sctx.lineTo(X(p[0]), Y(p[1])); else sctx.moveTo(X(p[0]), Y(p[1])); });
    sctx.stroke();
    const last = hist[hist.length - 1];
    sctx.beginPath(); sctx.arc(X(last[0]), Y(last[1]), 3.5, 0, Math.PI * 2);
    sctx.lineWidth = 2; sctx.strokeStyle = C.surface; sctx.stroke(); sctx.fillStyle = C.gas; sctx.fill();
    // crosshair + tooltip snapped to the nearest sample
    const st = $("sparkTip");
    if (sparkHover !== null) {
      let best = hist[0];
      hist.forEach((p) => { if (Math.abs(X(p[0]) - sparkHover) < Math.abs(X(best[0]) - sparkHover)) best = p; });
      const x = X(best[0]);
      sctx.strokeStyle = C.text2; sctx.lineWidth = 1;
      sctx.beginPath(); sctx.moveTo(x + 0.5, pad.t); sctx.lineTo(x + 0.5, Y(0)); sctx.stroke();
      sctx.beginPath(); sctx.arc(x, Y(best[1]), 4, 0, Math.PI * 2); sctx.fillStyle = C.gas; sctx.fill();
      st.textContent = "";
      st.appendChild(el("div", "tv", best[1].toFixed(2) + " ppm"));
      st.appendChild(el("div", null, Math.abs(best[0]) < 1 ? "현재" : Math.round(Math.abs(best[0])) + "초 전"));
      st.hidden = false;
      st.style.left = Math.min(Math.max(4, x + 10), w - st.offsetWidth - 4) + "px";
      st.style.top = "2px";
    } else {
      st.hidden = true;
    }
  }

  function renderThermal(s) {
    const tb = $("thermalRows");
    tb.textContent = "";
    let nAlarm = 0, nWarn = 0, nSeen = 0;
    // after the whole route was inspected, a spot that never showed up was looked at
    // and is simply not hot: "검출 없음" instead of "미관측"
    const fin = missionFinished(s.mission || {});
    const known = s.thermal || [];
    const rows = known.concat((s.thermal_unassigned || []).map((h) => Object.assign({}, h, { id: "—", tag: h.tag || "미할당" })));
    rows.forEach((h) => {
      const tr = el("tr");
      tr.appendChild(el("td", null, h.id));
      const tg = el("td", "tagc", h.tag || "");
      tg.title = h.tag || "";
      tr.appendChild(tg);
      tr.appendChild(el("td", "num temp", isNum(h.max_temp_c) ? h.max_temp_c.toFixed(1) + " °C" : "—"));
      tr.appendChild(el("td", "num muted", isNum(h.true_temp_c) ? "참값 " + h.true_temp_c.toFixed(0) + " °C" : ""));
      const tdS = el("td");
      tdS.appendChild(fin && h.status === "unseen" ? statusEl(h.status, { label: "검출 없음", glyph: "–" }) : statusEl(h.status));
      tr.appendChild(tdS);
      tr.title = (h.tag || "") + (isNum(h.peak_c) ? " · 최고 관측 " + h.peak_c.toFixed(1) + " °C" : "") + (isNum(h.age) ? " · " + fmtAge(h.age) : "");
      tb.appendChild(tr);
      if (h.status === "alarm") nAlarm++;
      if (h.status === "warning") nWarn++;
      if (isNum(h.max_temp_c) && h.id !== "—") nSeen++;
    });
    setText("thermalSummary", !rows.some((h) => isNum(h.max_temp_c)) ? (fin ? "검출 없음" : "관측 전")
      : "관측 " + nSeen + " / " + known.length + " · 경보 " + nAlarm + " · 경고 " + nWarn);
  }

  // ---------------------------------------------------------- gauges
  const tiles = {};
  function tileFor(id) {
    if (tiles[id]) return tiles[id];
    const root = el("div", "tile");
    const head = el("div", "t-head");
    const idEl = el("span", "t-id", id);
    const stWrap = el("span");
    head.appendChild(idEl); head.appendChild(stWrap);
    const tag = el("div", "t-tag");
    const body = el("div", "t-body");
    const val = el("div", "t-val");
    body.appendChild(val);
    const range = el("div", "t-range");
    const track = el("div", "t-track");
    const mark = el("i", "t-mark");
    range.appendChild(track); range.appendChild(mark);
    const meta = el("div", "t-meta");
    const foot = el("div", "t-foot");
    [head, tag, body, range, meta, foot].forEach((x) => root.appendChild(x));
    const t = { root: root, stWrap: stWrap, status: undefined, tag: tag, body: body, val: val, range: range, track: track, mark: mark,
      zones: false, meta: meta, foot: foot, thumb: null, thumbLink: null, thumbSrc: null };
    tiles[id] = t;
    return t;
  }

  // mini scale of the dial range: alarm / warning / normal bands + reading marker
  const pctOf = (v, lo, hi) => Math.max(0, Math.min(100, 100 * (v - lo) / (hi - lo)));
  function updateRange(t, g) {
    const st = STATIC && (STATIC.gauges || []).find((x) => x.id === g.id);
    const ok = st && isNum(st.min) && isNum(st.max) && st.max > st.min;
    t.range.hidden = !ok;
    if (!ok) return;
    if (!t.zones) {
      const nr = Array.isArray(st.normal_range) ? st.normal_range : null;
      const ar = Array.isArray(st.alarm_range) ? st.alarm_range : null;
      const seg = (a, b, cls) => {
        if (!isNum(a) || !isNum(b) || b <= a) return;
        const d = el("i", "t-seg " + cls);
        d.style.left = pctOf(a, st.min, st.max) + "%";
        d.style.width = (pctOf(b, st.min, st.max) - pctOf(a, st.min, st.max)) + "%";
        t.track.appendChild(d);
      };
      if (ar) seg(ar[0], ar[1], "warn");
      if (nr) seg(nr[0], nr[1], "ok");
      t.range.title = "범위 " + st.min + " – " + st.max + " " + st.unit +
        (nr ? " · 정상 " + nr[0] + " – " + nr[1] : "") + (ar ? " · 경보 기준 " + ar[0] + " 미만 / " + ar[1] + " 초과" : "");
      t.zones = true;
    }
    t.mark.hidden = !isNum(g.value);
    if (isNum(g.value)) t.mark.style.left = pctOf(g.value, st.min, st.max) + "%";
  }

  function renderGauges(s) {
    const box = $("gaugeTiles");
    const target = s.mission && s.mission.target;
    const mstate = s.mission && s.mission.state;
    const results = (s.mission && s.mission.results) || {};
    const order = (s.plan && s.plan.order) || [];
    let nRead = 0, nWarn = 0, nAlarm = 0;
    const errs = [];
    (s.gauges || []).forEach((g, i) => {
      const t = tileFor(g.id);
      if (t.root.parentNode !== box || box.children[i] !== t.root) box.insertBefore(t.root, box.children[i] || null);
      const read = isNum(g.value);
      const isTarget = target === g.id;
      t.root.className = "tile " + (STATUS[g.status] ? g.status : "unread") + (isTarget ? " is-target" : "");
      // current target without a reading yet: "판독 중" at the gauge (INSPECTING), "다음 목표"
      // on the way there - not "미판독"
      const tgtLabel = isTarget && !read ? (mstate === "INSPECTING" ? "판독 중" : "다음 목표") : "";
      const stKey = tgtLabel + ":" + g.status;
      if (t.status !== stKey) {
        t.stWrap.textContent = "";
        t.stWrap.appendChild(tgtLabel ? statusEl(g.status, { label: tgtLabel, glyph: "▸", cls: "reading" }) : statusEl(g.status));
        t.status = stKey;
      }
      t.tag.textContent = g.tag || "";
      if (isTarget && read) t.tag.appendChild(el("span", "t-now", "점검 중"));
      t.tag.title = g.tag || "";
      t.val.textContent = read ? fmtVal(g.value) : "—";
      if (read) t.val.appendChild(el("small", null, g.unit || ""));
      t.val.title = read ? g.value.toFixed(3) + " " + (g.unit || "") : "";
      const unit = g.unit ? " " + g.unit : "";
      updateRange(t, g);
      // unread: the normal band only (the true value is not shown before the robot reads it);
      // read: true value + error, two unbreakable halves so a narrow tile wraps at the "·"
      const st = STATIC && (STATIC.gauges || []).find((x) => x.id === g.id);
      const nr = st && Array.isArray(st.normal_range) ? st.normal_range : null;
      t.meta.textContent = "";
      if (read) {
        if (isNum(g.true_value)) t.meta.appendChild(el("span", "nw", "참값 " + g.true_value + unit));
        if (isNum(g.error_pct_fs)) {
          if (t.meta.childNodes.length) t.meta.appendChild(document.createTextNode(" · "));
          t.meta.appendChild(el("span", "nw", "오차 " + g.error_pct_fs.toFixed(2) + " % FS"));
        }
      } else if (nr) {
        t.meta.appendChild(el("span", "nw", "정상 범위 " + nr[0] + " – " + nr[1] + unit));
      }
      t.meta.title = t.meta.textContent;
      const pos = order.indexOf(g.id);
      t.foot.textContent = "";
      if (read) {
        t.foot.appendChild(el("span", "f-conf", "신뢰도 " + fmt(g.confidence, 2) + " · "));
        t.foot.appendChild(document.createTextNode((isNum(g.n_updates) ? "판독 " + g.n_updates + "회 · " : "") + fmtAge(g.age)));
      } else {
        t.foot.textContent = pos >= 0 ? "순찰 순서 " + (pos + 1) + "번째" : "판독 대기";
      }
      t.foot.title = read ? "신뢰도 " + fmt(g.confidence, 2) + (isNum(g.n_updates) ? " · 판독 " + g.n_updates + "회" : "") + " · " + fmtAge(g.age) : "";
      if (read) t.val.title += "\n" + t.foot.title;          // the footer is hidden on short screens
      // inspection photo of the accepted reading (metrics session): thumbnail, click = full camera frame
      const src = safeImg(g.image_url) || safeImg((results[g.id] || {}).image_url);
      if (src) {
        if (!t.thumb) {
          t.thumbLink = el("a", "t-thumb-link");
          t.thumbLink.target = "_blank"; t.thumbLink.rel = "noopener";
          t.thumb = el("img", "t-thumb"); t.thumb.alt = g.id + " 점검 사진"; t.thumb.loading = "lazy";
          t.thumbLink.appendChild(t.thumb); t.body.appendChild(t.thumbLink);
        }
        if (t.thumbSrc !== src) {
          t.thumb.src = src; t.thumbSrc = src;
          const full = safeHref(g.image_full_url) || safeHref(src);
          if (full) t.thumbLink.href = full; else t.thumbLink.removeAttribute("href");
          t.thumbLink.title = g.id + " 점검 사진 (누르면 전체 화면 사진)";
        }
      } else if (t.thumb) {
        t.thumbLink.remove(); t.thumb = null; t.thumbLink = null; t.thumbSrc = null;
      }
      if (read) nRead++;
      if (g.status === "warning") nWarn++;
      if (g.status === "alarm") nAlarm++;
      if (isNum(g.error_pct_fs)) errs.push(g.error_pct_fs);
    });
    const mean = errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : null;
    setText("gaugeSummary", "판독 " + nRead + " / " + (s.gauges || []).length +
      (isNum(mean) ? " · 평균 오차 " + mean.toFixed(2) + " % FS" : "") +
      (nWarn + nAlarm ? " · 이상 " + (nWarn + nAlarm) : ""));
  }

  // ---------------------------------------------------------- report
  function reportKpi(label, value, sub, cls) {
    const d = el("div", "rk");
    d.appendChild(el("div", "k", label));
    const v = el("div", "v" + (cls ? " " + cls : ""), value);
    d.appendChild(v);
    d.appendChild(el("div", "s", sub || " "));
    return d;
  }
  function issue(level, glyph, text, sub) {
    const li = el("li", level);
    li.appendChild(el("i", "ico", glyph));
    li.appendChild(el("span", "main", text));
    li.appendChild(el("span", "sub", sub || ""));
    li.title = text + (sub ? " · " + sub : "");
    return li;
  }

  function renderReport(s) {
    const m = s.mission || {};
    const l = s.locomotion || {};
    const g = s.gas || {};
    const gauges = s.gauges || [];
    const fin = missionFinished(m);
    const statics = {};
    (STATIC && STATIC.gauges || []).forEach((x) => { statics[x.id] = x; });
    const read = gauges.filter((x) => isNum(x.value));
    const errs = gauges.filter((x) => isNum(x.error_pct_fs));
    const mean = errs.length ? errs.reduce((a, x) => a + x.error_pct_fs, 0) / errs.length : null;
    const worst = errs.reduce((a, x) => (!a || x.error_pct_fs > a.error_pct_fs ? x : a), null);
    const gWarn = gauges.filter((x) => x.status === "warning"), gAlarm = gauges.filter((x) => x.status === "alarm");
    const th = (s.thermal || []);
    const tWarn = th.filter((x) => x.status === "warning"), tAlarm = th.filter((x) => x.status === "alarm");
    const seek = m.seek || null;
    const missed = m.missed || [];

    const k = $("reportKpis");
    k.textContent = "";
    const stateTxt = fin ? "완료" : m.state && m.state !== "IDLE" ? "진행 중" : "대기";
    k.appendChild(reportKpi("임무 결과", stateTxt, lbl("state", m.state, "—") + " · 경과 " + fmtDur(m.mission_elapsed)));
    k.appendChild(reportKpi("계기 판독", read.length + " / " + gauges.length, "판독 실패 " + missed.length + "곳"));
    k.appendChild(reportKpi("평균 판독 오차", isNum(mean) ? mean.toFixed(2) + " % FS" : "—",
      worst ? "최대 " + worst.error_pct_fs.toFixed(2) + " % FS (" + worst.id + ")" : ""));
    k.appendChild(reportKpi("이상 계기", String(gWarn.length + gAlarm.length) + "개", "경보 " + gAlarm.length + " · 경고 " + gWarn.length));
    k.appendChild(reportKpi("열화상 이상", String(tWarn.length + tAlarm.length) + "곳", "경보 " + tAlarm.length + " · 경고 " + tWarn.length));
    const seekVal = seek && seek.result ? lbl("seek", seek.result) : (g.seek && g.seek.state && g.seek.state !== "IDLE" ? "추적 중" : "추적 안 함");
    k.appendChild(reportKpi("가스 누출원", seekVal,
      seek && isNum(seek.error_m) ? "위치 오차 " + fmtErrM(seek.error_m) + (isNum(seek.elapsed) ? " · 추적 " + fmtSec(seek.elapsed) : "")
        : "넘어짐 " + (isNum(l.fall_count) ? l.fall_count : "—") + "회"));

    // issues: the found leak source first (headline finding), then alarms, warnings, information
    const ul = $("reportIssues");
    ul.textContent = "";
    const items = [];
    const rangeTxt = (id) => {
      const st = statics[id];
      return st && Array.isArray(st.normal_range) ? "정상 범위 " + st.normal_range[0] + " – " + st.normal_range[1] + " " + st.unit : "";
    };
    gAlarm.forEach((x) => items.push([0, "alarm", "◆", x.id + " " + (x.tag || "") + " — " + fmtVal(x.value) + " " + (x.unit || ""), "계기 경보 · " + rangeTxt(x.id)]));
    tAlarm.forEach((x) => items.push([0, "alarm", "◆", x.id + " " + (x.tag || "") + " — " + fmt(x.max_temp_c, 1, "°C"),
      "열화상 경보 · 기준 " + (STATIC && STATIC.thermal ? STATIC.thermal.alarm_c : 80) + " °C 초과"]));
    if (g.alarm) items.push([0, "alarm", "◆", "가스 농도 " + fmt(g.ppm, 1, "ppm"), "가스 경보 · 기준 " + (g.alarm_ppm || 25) + " ppm 초과"]);
    gWarn.forEach((x) => items.push([1, "warning", "▲", x.id + " " + (x.tag || "") + " — " + fmtVal(x.value) + " " + (x.unit || ""), "계기 경고 · " + rangeTxt(x.id)]));
    tWarn.forEach((x) => items.push([1, "warning", "▲", x.id + " " + (x.tag || "") + " — " + fmt(x.max_temp_c, 1, "°C"),
      "열화상 경고 · 기준 " + (STATIC && STATIC.thermal ? STATIC.thermal.warning_c : 60) + " °C 초과"]));
    missed.forEach((id) => items.push([1, "warning", "✕", id + " 판독 실패", "재점검 필요"]));
    if (seek && seek.result === "FOUND" && Array.isArray(seek.estimate)) {
      const leak = STATIC && STATIC.gas_leak;
      const est = seek.estimate.every(isNum) ? " (" + seek.estimate.map((v) => v.toFixed(2)).join(", ") + ")" : "";
      items.push([-1, "alarm", "✕", "가스 누출원 발견" + est,
        [leak && leak.tag ? leakTag(leak.tag) : "", isNum(seek.error_m) ? "위치 오차 " + fmtErrM(seek.error_m) : ""].filter(Boolean).join(" · ")]);
    }
    items.sort((a, b) => a[0] - b[0]);
    items.forEach((it) => ul.appendChild(issue(it[1], it[2], it[3], it[4])));
    if (!items.length) ul.appendChild(issue("ok", "✓", read.length ? "조치가 필요한 항목 없음" : "아직 판독 결과 없음", ""));

    const tb = $("reportGauges");
    tb.textContent = "";
    gauges.forEach((x) => {
      const tr = el("tr");
      const td0 = el("td", null, x.id);
      td0.title = x.tag || "";
      tr.appendChild(td0);
      tr.appendChild(el("td", "num", isNum(x.value) ? fmtVal(x.value) + " " + (x.unit || "") : "—"));
      tr.appendChild(el("td", "num muted", isNum(x.true_value) ? x.true_value + " " + (x.unit || "") : "—"));
      tr.appendChild(el("td", "num", isNum(x.error_pct_fs) ? x.error_pct_fs.toFixed(2) : "—"));
      const tdS = el("td"); tdS.appendChild(statusEl(x.status)); tr.appendChild(tdS);
      tb.appendChild(tr);
    });
  }

  // link to the inspection report written by metrics_logger (session/inspection_report.html)
  function renderReportLink(s) {
    const a = $("reportLink");
    const r = s.report || null;
    const href = r ? safeHref(r.url) : null;
    a.hidden = !href;
    if (!href) return;
    if (a.getAttribute("href") !== href) a.setAttribute("href", href);
    a.classList.toggle("final", !!r.final);
    a.textContent = r.final ? "보고서 열기" : "보고서 열기 (중간)";
    a.title = "점검 보고서를 새 탭에서 열기 · 계기별 사진·판독값·참값·오차, 열화상·가스 결과" +
      (r.final ? "" : " · 임무 진행 중 중간 보고") + (r.written ? " · 작성 " + r.written : "");
  }

  // the frozen report: first finished snapshot of the current mission (the live
  // snapshot keeps changing after docking - battery, gas, events ...)
  function captureReport(s) {
    const m = s.mission || {};
    if (["PATROL", "INSPECTING", "SEEK"].includes(m.state)) { REPORT = null; return; }   // a mission is running again
    if (REPORT || !missionFinished(m)) return;
    const g = s.gas || {};
    REPORT = { t_sim: s.t_sim, s: JSON.parse(JSON.stringify({
      mission: m, gauges: s.gauges, thermal: s.thermal,
      gas: { alarm: !!g.alarm, ppm: g.ppm, alarm_ppm: g.alarm_ppm, seek: g.seek },
      locomotion: { fall_count: (s.locomotion || {}).fall_count } })) };
  }
  function renderReportView(s) {
    renderReport(REPORT ? REPORT.s : s);
    setText("reportStamp", REPORT ? "임무 종료 시점 기준 · 시뮬레이션 " + fmtMinSec(REPORT.t_sim) : "진행 중인 임무 · 실시간");
  }

  // ---------------------------------------------------------- events
  // routine patrol <-> inspect transitions are left out of the list (each gauge already
  // gives 'inspecting', its reading and 'next viewpoint'); SEEK / RTH / PAUSED stay
  const ROUTINE = /^state (PATROL|INSPECTING) -> (PATROL|INSPECTING)$/;
  const EV_ICON = { alarm: "◆", warn: "▲", error: "✕", info: "●" };
  function renderEvents(s) {
    const evs = s.events || [];
    const lastSeq = evs.length ? evs[evs.length - 1].seq : -1;
    if (lastSeq === lastEventSeq) return;
    lastEventSeq = lastSeq;
    const ul = $("eventList");
    ul.textContent = "";
    // the planned return after a completed mission is not a warning
    const doneAt = evs.filter((e) => /\(mission_complete\)/.test(e.text || "")).map((e) => e.t_sim).filter(isNum);
    const levelOf = (e) => {
      const t = e.text || "";
      if (/\(mission_complete\)/.test(t)) return "info";
      if (e.source === "mission" && /^state \S+ -> RTH$/.test(t) && isNum(e.t_sim) &&
        doneAt.some((d) => Math.abs(d - e.t_sim) <= 1)) return "info";
      return e.level || "info";
    };
    evs.slice().reverse().forEach((e) => {
      if (e.source === "mission" && ROUTINE.test(e.text || "")) return;
      const lv = levelOf(e);
      const li = el("li", lv);
      li.appendChild(el("i", "ico", EV_ICON[lv] || "●"));
      // simulation time, like the header clock and the mission clock (the wall time
      // runs faster / slower than the simulation): wall time in the tooltip
      // (events from before the first simulation clock message carry no t_sim: shown as 0:00)
      li.appendChild(el("span", "t", fmtMinSec(isNum(e.t_sim) ? e.t_sim : 0)));
      li.title = (isNum(e.t_sim) ? "시뮬레이션 " + fmtClock(e.t_sim) : "시뮬레이션 시작 전") + (isNum(e.t_wall) ? " · 시각 " + wallTime(e.t_wall) : "");
      const src = el("span", "src", lbl("event_source", e.source));
      src.title = e.source || "";
      li.appendChild(src);
      // node messages are English: Korean when a rule matches, original text in the tooltip
      const ko = KO.event(e.text || "", e.source || "");
      const txt = el("span", "txt", ko);
      txt.title = ko + (e.text && e.text !== ko ? "\n" + e.text : "");
      li.appendChild(txt);
      ul.appendChild(li);
    });
    setText("eventCount", "최근 " + evs.length + "건");
    fitEvents();
  }

  // one-screen mode: the list shows whole rows only (no half-cut last row)
  function fitEvents() {
    const ul = $("eventList"), card = ul.closest(".card"), head = card.querySelector(".card-head");
    if (getComputedStyle(document.body).overflow !== "hidden") { ul.style.maxHeight = ""; return; }
    const cs = getComputedStyle(card);
    const row = ul.firstElementChild ? ul.firstElementChild.offsetHeight : 23;
    if (!row) return;
    const avail = card.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom) -
      head.offsetHeight - parseFloat(getComputedStyle(head).marginBottom);
    ul.style.maxHeight = Math.max(row, Math.floor(avail / row) * row) + "px";
  }
  if (window.ResizeObserver) new ResizeObserver(fitEvents).observe($("eventList").closest(".card"));
  else window.addEventListener("resize", fitEvents);

  // ---------------------------------------------------------- details drawer
  function renderDetails(s) {
    const r = s.robot || {};
    const l = s.locomotion || {};
    setText("robotAge", isNum(l.age) ? "보행 상태 " + fmtAge(l.age) : "보행 상태 없음");
    setText("locoHeight", fmt(l.body_height, 3, "m"));
    setText("locoGaitFull", lbl("gait", l.gait, "—") + (l.gait_auto ? " (자동 선택)" : "") +
      (l.locomotion_mode ? " · " + lbl("algo", l.locomotion_mode) : ""));
    setText("locoRP", deg(l.roll) + " / " + deg(l.pitch));
    setText("locoMpc", l.locomotion_mode === "stable" ? "꺼짐 (안정 모드)" :
      (isNum(l.mpc_solve_ms) ? l.mpc_solve_ms.toFixed(1) + " ms" + (l.mpc_status ? " · " + lbl("mpc", l.mpc_status) : "") : "—"));
    $("locoMpc").title = l.mpc_status || "";
    setText("locoMargin", fmt(l.stability_margin, 3, "m"));
    const cv = s.cmd_vel || {};
    setText("cmdSource", lbl("source", cv.source, "—") + (s.flags && s.flags.teleop_active ? " (수동)" : ""));
    $("cmdSource").title = cv.source || "";
    setText("robotPose", isNum(r.x) ? r.x.toFixed(2) + ", " + r.y.toFixed(2) + " m · " + deg(r.yaw) +
      (isNum(r.vx) ? " · " + r.vx.toFixed(2) + " m/s, " + (r.wz || 0).toFixed(2) + " rad/s" : "") : "—");
    renderSlip(s.terrain || {}, l);

    const m = s.mission || {};
    setText("missionAge", isNum(m.age) ? "임무 상태 " + fmtAge(m.age) : "임무 상태 없음");
    setText("missionMode", [m.mode ? lbl("mode", m.mode) : "", m.phase ? lbl("phase", m.phase) : ""].filter(Boolean).join(" · ") || "—");
    setText("missionReplans", isNum(m.replans) ? m.replans + "회" : "—");
    setText("missionRemain", fmt(m.path_remaining, 1, "m"));
    setText("planSource", m.plan_source || "—");

    const sf = s.safety || {};
    const b = s.battery || {};
    setBadge("safetyState", sf.state || null, "sm");
    const reason = sf.reason || (sf.rth_active ? "active" : "");
    setText("rthReason", lbl("reason", reason, "—"));
    $("rthReason").title = reason;
    setText("rssiPred", fmt(sf.predicted_min_rssi, 1, "dBm"));
    setText("energyHome", isNum(sf.energy_to_home_pct) ? sf.energy_to_home_pct.toFixed(1) + " %" : "—");
    setText("battVI", (isNum(b.voltage) ? b.voltage.toFixed(1) + " V" : "—") + " · " + (isNum(b.current) ? b.current.toFixed(1) + " A" : "—"));
    setText("safetyAge", isNum(b.age) ? "배터리 " + fmtAge(b.age) : "");

    const g = s.gas || {};
    const est = g.estimate;
    setText("seekEst", est && isNum(est[0]) && isNum(est[1]) ? est[0].toFixed(2) + ", " + est[1].toFixed(2) : "—");
    const f = g.field || {};
    setText("gasMax", fmt(f.max_ppm, 1, "ppm"));
    setText("gasLimit", (g.alarm_ppm || 25) + " ppm");
    const ms = m.seek || {};
    setText("seekTime", isNum(ms.elapsed) ? ms.elapsed.toFixed(1) + "초" + (isNum(ms.path_length) ? " · " + ms.path_length.toFixed(1) + " m" : "") : "—");
    renderLinks(s);
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
      ageEl.textContent = meta && isNum(meta.t_wall) && isNum(s.t_wall) ? fmtAge(s.t_wall - meta.t_wall) : "";
      // no frame loaded yet: a Korean waiting note instead of the server's "no signal"
      // placeholder (the class goes when the first real frame has loaded)
      const fig = c.img.closest("figure");
      if (fig) fig.classList.toggle("cam-empty", !meta || c.version < 0);
      if (!meta || c.loading || meta.version === c.version || Date.now() - c.t < 450) return;
      c.loading = true;
      c.t = Date.now();
      const v = meta.version;
      const img = new Image();
      img.onload = () => {
        c.img.src = img.src; c.version = v; c.loading = false;
        if (fig) fig.classList.remove("cam-empty");
      };
      img.onerror = () => { c.loading = false; };
      img.src = c.path + "?v=" + v;
    });
  }

  function onStatic() {
    view.fitted = false;
    applyAspect();
    resize();
    queueDraw();
  }

  function render() {
    const s = SNAP;
    if (!s) return;
    setText("simTime", fmtClock(s.t_sim));
    // on every snapshot, whichever tab is open: the report freezes at the real completion moment
    try { captureReport(s); } catch (err) { console.error("captureReport", err); }
    // one malformed field must not freeze every other panel
    const fns = [renderMission, renderRobot, renderSafety, renderGas, renderThermal, renderGauges, renderReportLink,
      renderEvents, renderCams];
    if (!drawer.hidden) fns.push(renderDetails);
    if (activeTab === "report") fns.push(renderReportView);
    fns.forEach((fn) => {
      try { fn(s); } catch (err) { console.error(fn.name, err); }
    });
    refreshLayers();
    queueDraw();
  }

  resize();
  connect();
})();
