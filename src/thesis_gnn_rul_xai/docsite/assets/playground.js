/* The playground: set up a run, follow it, read its results, explain it.
   README.md in this folder explains what each part is for and why every run
   started here goes through scripts/train.py and lands in reports/.

   How the page is put together: every view is a string of HTML built from
   the page's state, S, and written into the page. Nothing keeps a handler on
   an element; instead the whole page has one click listener, one change
   listener and one input listener, and each element says what it wants
   through a data-action or data-field attribute. Redrawing a view therefore
   never loses a handler, and every button can be found by reading the
   markup. */

(function () {
  "use strict";

  var ROUTE = "#/explore/playground";
  var STORAGE = "thesis-playground-settings";
  var SECTIONS = "thesis-playground-sections";
  var PALETTE = ["var(--fig-accent)", "var(--fig-warn)", "var(--fig-lpt)", "var(--fig-lpc)",
    "var(--fig-core)", "var(--fig-cool)", "var(--fig-burn)", "var(--fig-fan)"];
  // The one vocabulary for temporal stages, job states and methods: the same
  // words in every label, chip, filter, toast and table cell.
  var TEMPORAL = { lstm: "LSTM", gru: "GRU", last_cycle: "w/o LSTM", mean_window: "window average",
    window_summary: "window summary", raw_window: "whole window" };
  var STATUS = { queued: "Queued", running: "Running", cancelling: "Stopping", done: "Done",
    failed: "Failed", cancelled: "Stopped", interrupted: "Interrupted" };
  var METHODS = { shap: "Exact Shapley values", occlusion: "Occlusion" };
  var ENGINE_SOURCE = { validation: "held-back", training: "training", test: "test" };
  var MEASURES = [["rmse", "RMSE", 3], ["score", "Score", 1], ["r2", "R²", 3], ["asue", "ASUE", 3]];
  // The paper's own precision for each measure, read from Tables 7 to 10: RMSE
  // to two places, R² and ASUE to three, and the score to one place or none
  // ("278.6", "1438").
  var PAPER_PLACES = { rmse: 2, r2: 3, asue: 3 };
  // The engine the paper's Fig. 7 draws for each sub-dataset, from its caption:
  // "(a) FD001-Unit38, (b) FD002-Unit48, (c) FD003-Unit30, and (d) FD004-Unit58".
  var FIG7_UNIT = { FD001: 38, FD002: 48, FD003: 30, FD004: 58 };
  // What each source tag means, shown when the pointer rests on the tag.
  var SOURCE_NAMES = { paper: "Paper", assumption: "Our choice", project: "Project rule", extension: "Not in the paper" };
  var SOURCE_TITLES = { paper: "The paper prints this value", assumption: "The paper is silent, so this project chose",
    project: "A rule of this repository", extension: "An option offered for comparison" };
  // Choices short enough to sit side by side in one control. Every other
  // choice is a list of options, each with the source of its value.
  var SEGMENTED = { subset: true, device: true };
  var SUBSETS = ["FD001", "FD002", "FD003", "FD004"];
  // Groups of settings that are not drawn as folding sections: the sub-dataset
  // stands alone at the top of the form, and the procedure cards hold theirs.
  var UNSECTIONED = { data: true, procedure: true };
  var CHEVRON = '<svg class="nav-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 L8 6 L4 9.5" ' +
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  var T;          // the helpers app.js hands over
  var root;       // the element the page is drawn into
  var S = {};     // everything the page is showing

  // ---------------------------------------------------------------- basics

  function post(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body)
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) { throw new Error(data.error || response.statusText); }
        return data;
      });
    });
  }

  function esc(text) { return T.escapeHtml(text === null || text === undefined ? "" : text); }
  function fixed(value, places) { return T.fixed(value, places); }
  function $(selector) { return root.querySelector(selector); }
  function $all(selector) { return Array.prototype.slice.call(root.querySelectorAll(selector)); }
  function onPage() { return window.location.hash.indexOf(ROUTE) === 0; }
  function copy(value) { return JSON.parse(JSON.stringify(value)); }
  function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
  function plural(n, word) { return count(n) + " " + word + (n === 1 ? "" : "s"); }

  // A whole number with a space between thousands, "61 729", as the figures print it.
  function count(n) {
    return typeof n === "number" && isFinite(n) ? Number(n).toLocaleString("en").replace(/,/g, " ") : String(n);
  }

  // A measured number to a fixed number of places, thousands spaced the same way.
  function number(value, places) {
    var text = fixed(value, places);
    return /^-?\d{4,}/.test(text) ? text.replace(/^(-?)(\d+)/, function (all, sign, whole) {
      return sign + whole.replace(/\B(?=(\d{3})+(?!\d))/g, " ");
    }) : text;
  }

  // A published figure at the precision the paper prints it.
  function paperNumber(key, value) {
    var places = PAPER_PLACES[key] !== undefined ? PAPER_PLACES[key] : (Number.isInteger(value) ? 0 : 1);
    return { text: number(value, places), places: places };
  }

  // Every time the page shows: "2026-09-14 23:39 UTC".
  function when(stamp) {
    var m = /^(\d{4})-?(\d\d)-?(\d\d)T(\d\d):?(\d\d)/.exec(stamp || "");
    return m ? m[1] + "-" + m[2] + "-" + m[3] + " " + m[4] + ":" + m[5] + " UTC" : (stamp || "—");
  }

  function whenSeconds(seconds) {
    return seconds ? when(new Date(seconds * 1000).toISOString()) : "";
  }

  function duration(seconds) {
    if (seconds === null || seconds === undefined) { return ""; }
    seconds = Math.round(seconds);
    var h = Math.floor(seconds / 3600), m = Math.floor(seconds % 3600 / 60), s = seconds % 60;
    return (h ? h + " h " : "") + (h || m ? m + " min " : "") + s + " s";
  }

  function refLink(ref) {
    if (!ref) { return ""; }
    if (/^[AF]-\d+$/.test(ref)) { return '<a href="#/entry/' + ref + '">' + ref + "</a>"; }
    if (/^D-\d+$/.test(ref)) {
      return '<a href="#/doc/docs/AUDIT_2026-09-08_scientific_validity.md">' + ref + "</a>";
    }
    return esc(ref);
  }

  // Where a default comes from: the paper, a numbered choice, a project rule,
  // or an option the paper does not have. README.md lists the four.
  function sourceTag(source) {
    if (!source || !SOURCE_NAMES[source.kind]) { return ""; }
    return '<span class="src src--' + source.kind + '" title="' + esc(SOURCE_TITLES[source.kind]) + '">' +
      SOURCE_NAMES[source.kind] + (source.ref ? " · " + refLink(source.ref) : "") + "</span>";
  }

  // A source's tag followed by its words, or nothing when the JSON gives neither.
  function sourceNote(source) {
    if (!source) { return ""; }
    return [sourceTag(source), source.text ? esc(source.text) : ""].filter(Boolean).join(" ");
  }

  function temporalName(value) {
    return value === null || value === undefined ? "not recorded" : (TEMPORAL[value] || value);
  }

  // The stage with its size where it has one: "LSTM 32", "w/o LSTM".
  function stageName(temporal, hidden) {
    return temporalName(temporal) + ((temporal === "lstm" || temporal === "gru") && hidden ? " " + hidden : "");
  }

  // The graph with its rescaling, when that is not the default: "physics", "physics · symmetric".
  // A run that recorded no graph gets nothing.
  function graphName(graph, rescaling) {
    return graph ? graph + (rescaling && rescaling !== "row" ? " · " + rescaling : "") : "";
  }

  function statusWord(status) { return STATUS[status] || status; }

  // A setting two runs differ in, named as the form names it, with its value in
  // the form's own words. A record the form has no field for keeps its name.
  var RECORD_LABELS = { seeds: "Seeds", split: "Held-back engines", commit: "Commit" };
  var RECORD_WORDS = { varies_with_seed: "a different draw for every seed", fixed: "the same for every seed" };

  function settingLabel(setting) {
    var field = S.fields && S.fields[setting.split(".").pop()];
    return field ? field.label : (RECORD_LABELS[setting] || setting);
  }

  function settingWords(setting, value) {
    var field = S.fields && S.fields[setting.split(".").pop()];
    var choice = field && (field.choices || []).filter(function (c) { return c.value === value; })[0];
    if (choice) { return choice.label; }
    if (value === true || value === false) { return value ? "on" : "off"; }
    if (value === null || value === undefined) { return "not recorded"; }
    if (Array.isArray(value)) { return value.join(" · "); }
    return RECORD_WORDS[value] || String(value);
  }

  // Whether a caution belongs in a view: "run" for a stored run's page, "design" beside Train.
  function cautionFor(caution, where) {
    var w = caution.where;
    if (w === undefined || w === null) { return where === "run"; }
    return Array.isArray(w) ? w.indexOf(where) >= 0 : String(w).split(/[\s,]+/).indexOf(where) >= 0;
  }

  function errorBox(message) { return '<div class="error-box">' + esc(message) + "</div>"; }

  function remember(key, value) {
    try { window.localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* private window */ }
  }

  function recall(key) {
    try { return JSON.parse(window.localStorage.getItem(key) || "null"); } catch (e) { return null; }
  }

  function saveSettings() { remember(STORAGE, S.settings); }

  function loadSettings(defaults) {
    var stored = recall(STORAGE);
    var settings = copy(defaults);
    if (stored) {
      Object.keys(settings).forEach(function (key) {
        if (stored[key] !== undefined) { settings[key] = stored[key]; }
      });
    }
    return settings;
  }

  // ---------------------------------------------------------------- charts

  function niceTicks(low, high, count) {
    if (!isFinite(low) || !isFinite(high)) { return [0, 1]; }
    if (low === high) { high = low + 1; }
    var step = Math.pow(10, Math.floor(Math.log10((high - low) / count)));
    [1, 2, 5, 10].some(function (m) {
      if ((high - low) / (step * m) <= count) { step = step * m; return true; }
      return false;
    });
    var ticks = [];
    for (var v = Math.ceil(low / step) * step; v <= high + step * 1e-9; v += step) {
      ticks.push(Math.round(v / step) * step);
    }
    return ticks;
  }

  function tickLabel(value) {
    var a = Math.abs(value);
    if (a !== 0 && (a < 0.01 || a >= 100000)) { return value.toExponential(0); }
    return String(Math.round(value * 1000) / 1000);
  }

  // A plot of one or more series of points. Every number drawn is one given.
  // An area is a series filled down to zero, the way the paper's Fig. 7
  // shades the error between the predicted and the true remaining life.
  function plot(spec) {
    var W = spec.width || 640, H = spec.height || 260;
    var left = 54, right = 14, top = 14, bottom = 40;
    var xs = [], ys = [];
    spec.series.forEach(function (s) {
      s.points.forEach(function (p) {
        if (p[1] !== null && isFinite(p[1])) { xs.push(p[0]); ys.push(p[1]); }
      });
    });
    (spec.areas || []).forEach(function (a) {
      ys.push(0);
      a.points.forEach(function (p) {
        if (p[1] !== null && isFinite(p[1])) { xs.push(p[0]); ys.push(p[1]); }
      });
    });
    (spec.hlines || []).forEach(function (h) { ys.push(h.y); });
    if (!xs.length) {
      return '<div class="pg-empty">Nothing to draw.</div>';
    }
    var xlo = spec.xmin !== undefined ? spec.xmin : Math.min.apply(null, xs);
    var xhi = spec.xmax !== undefined ? spec.xmax : Math.max.apply(null, xs);
    var ylo = spec.ymin !== undefined ? spec.ymin : Math.min.apply(null, ys);
    var yhi = Math.max.apply(null, ys);
    var pad = (yhi - ylo) * 0.06 || 1;
    if (spec.ymin === undefined) { ylo -= pad; }
    yhi += pad;
    if (xlo === xhi) { xlo -= 1; xhi += 1; }
    var X = function (v) { return left + (v - xlo) / (xhi - xlo) * (W - left - right); };
    var Y = function (v) { return top + (1 - (v - ylo) / (yhi - ylo)) * (H - top - bottom); };

    var out = '<svg class="pg-plot" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' +
      esc(spec.label || "") + '">';
    niceTicks(ylo, yhi, 5).forEach(function (t) {
      out += '<line x1="' + left + '" x2="' + (W - right) + '" y1="' + Y(t) + '" y2="' + Y(t) +
        '" stroke="var(--line-soft)"/><text x="' + (left - 6) + '" y="' + (Y(t) + 4) +
        '" text-anchor="end" class="pg-tick">' + tickLabel(t) + "</text>";
    });
    niceTicks(xlo, xhi, 7).forEach(function (t) {
      out += '<text x="' + X(t) + '" y="' + (H - bottom + 16) + '" text-anchor="middle" class="pg-tick">' +
        tickLabel(t) + "</text>";
    });
    out += '<line x1="' + left + '" x2="' + (W - right) + '" y1="' + (H - bottom) + '" y2="' + (H - bottom) +
      '" stroke="var(--fig-axis)"/>';
    (spec.areas || []).forEach(function (a) {
      var pts = a.points.filter(function (p) { return p[1] !== null && isFinite(p[1]); });
      if (pts.length < 2) { return; }
      var zero = Y(0).toFixed(1);
      out += '<polygon fill="' + a.colour + '" fill-opacity="' + (a.opacity || 0.35) + '" points="' +
        X(pts[0][0]).toFixed(1) + "," + zero + " " +
        pts.map(function (p) { return X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join(" ") + " " +
        X(pts[pts.length - 1][0]).toFixed(1) + "," + zero + '"/>' +
        '<line x1="' + left + '" x2="' + (W - right) + '" y1="' + zero + '" y2="' + zero + '" stroke="var(--fig-axis)" stroke-width="0.8"/>';
    });
    (spec.hlines || []).forEach(function (h) {
      out += '<line x1="' + left + '" x2="' + (W - right) + '" y1="' + Y(h.y) + '" y2="' + Y(h.y) +
        '" stroke="' + h.colour + '" stroke-dasharray="5 4" stroke-width="1.5"/>' +
        '<text x="' + (W - right - 4) + '" y="' + (Y(h.y) - 5) + '" text-anchor="end" class="pg-tick" fill="' +
        h.colour + '">' + esc(h.label) + "</text>";
    });
    spec.series.forEach(function (s) {
      var pts = s.points.filter(function (p) { return p[1] !== null && isFinite(p[1]); });
      if (s.line !== false && pts.length > 1) {
        out += '<polyline fill="none" stroke="' + s.colour + '" stroke-width="' + (s.width || 1.7) +
          '" points="' + pts.map(function (p) { return X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1); }).join(" ") +
          '"' + (s.dash ? ' stroke-dasharray="' + s.dash + '"' : "") + "/>";
      }
      if (s.dots) {
        pts.forEach(function (p) {
          out += '<circle cx="' + X(p[0]).toFixed(1) + '" cy="' + Y(p[1]).toFixed(1) + '" r="' + (s.radius || 3) +
            '" fill="' + s.colour + '" fill-opacity="0.85"><title>' + esc((s.name ? s.name + ": " : "") +
            tickLabel(p[0]) + ", " + tickLabel(p[1])) + "</title></circle>";
        });
      }
      (s.marks || []).forEach(function (p) {
        out += '<circle cx="' + X(p[0]) + '" cy="' + Y(p[1]) + '" r="5" fill="none" stroke="' + s.colour +
          '" stroke-width="2"><title>' + esc(p[2] || "") + "</title></circle>";
      });
    });
    out += '<text x="' + ((left + W - right) / 2) + '" y="' + (H - 6) + '" text-anchor="middle" class="pg-axis">' +
      esc(spec.xlabel || "") + "</text>";
    out += '<text transform="translate(13,' + ((top + H - bottom) / 2) + ') rotate(-90)" text-anchor="middle" class="pg-axis">' +
      esc(spec.ylabel || "") + "</text></svg>";
    var legend = (spec.areas || []).concat(spec.series).filter(function (s) { return s.name && !s.nolegend; }).map(function (s) {
      return '<span><i style="background:' + s.colour + (s.points && s.opacity ? ";opacity:" + s.opacity : "") + '"></i>' + esc(s.name) + "</span>";
    }).join("");
    return out + (legend ? '<div class="pg-legend">' + legend + "</div>" : "");
  }

  // The blue-to-red scale the paper's Fig. 9 colours its dots with: a low
  // reading blue, a high one red. Fixed colours, because the mix is computed
  // here rather than by the stylesheet.
  var SWARM_LOW = [47, 127, 209], SWARM_HIGH = [179, 38, 30];

  function mixColour(share) {
    var t = Math.max(0, Math.min(1, share));
    return "rgb(" + SWARM_LOW.map(function (low, i) { return Math.round(low + (SWARM_HIGH[i] - low) * t); }).join(",") + ")";
  }

  // Every value of every sensor as one dot, the way the paper's Fig. 9 (left)
  // draws it: the sensors ranked by the average size of their values, a red
  // frame round the four largest and a blue one round the rest, and each dot
  // coloured by the sensor's own reading at that window. An explanation
  // stored before the readings were kept colours by the window's place in
  // the sequence instead, and the legend says which.
  function beeswarm(rows, ranked) {
    var hasReadings = rows.length > 0 && rows[0]["reading_" + ranked[0]] !== undefined;
    var rowH = 20, left = 64, right = 18, top = 10, bottom = 34, W = 640, dot = 2.4, step = 5.2;
    var H = top + rowH * ranked.length + bottom;
    var xs = [];
    rows.forEach(function (r) { ranked.forEach(function (s) { xs.push(r[s]); }); });
    var lo = Math.min(0, Math.min.apply(null, xs)), hi = Math.max(0, Math.max.apply(null, xs));
    var pad = (hi - lo) * 0.04 || 1;
    lo -= pad; hi += pad;
    var X = function (v) { return left + (v - lo) / (hi - lo) * (W - left - right); };
    var frame = function (from, to, colour) {
      return '<rect x="' + (left - 6) + '" y="' + (top + from * rowH + 1) + '" width="' + (W - left - right + 12) +
        '" height="' + ((to - from) * rowH - 2) + '" rx="4" fill="none" stroke="' + colour + '" stroke-width="1.4"/>';
    };
    var out = '<svg class="pg-plot" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Every value of every sensor">';
    out += frame(0, Math.min(4, ranked.length), "var(--fig-burn)");
    if (ranked.length > 4) { out += frame(4, ranked.length, "var(--fig-accent)"); }
    out += '<line x1="' + X(0).toFixed(1) + '" x2="' + X(0).toFixed(1) + '" y1="' + top + '" y2="' + (H - bottom) + '" stroke="var(--fig-axis)" stroke-dasharray="3 3"/>';
    niceTicks(lo, hi, 7).forEach(function (t) {
      out += '<text x="' + X(t).toFixed(1) + '" y="' + (H - bottom + 14) + '" text-anchor="middle" class="pg-tick">' + tickLabel(t) + "</text>";
    });
    out += '<text x="' + ((left + W - right) / 2) + '" y="' + (H - 4) + '" text-anchor="middle" class="pg-axis">value, in cycles of remaining life</text>';
    ranked.forEach(function (s, j) {
      var cy = top + j * rowH + rowH / 2;
      out += '<text x="' + (left - 10) + '" y="' + (cy + 3.5) + '" text-anchor="end" class="pg-tick">' + esc(s) + "</text>";
      var pts = rows.map(function (r, i) {
        return { x: r[s], shade: hasReadings ? r["reading_" + s] : i, label: "cycle " + r.end_cycle + (r.unit !== undefined ? ", engine " + r.unit : "") };
      }).filter(function (p) { return p.x !== null && isFinite(p.x); }).sort(function (a, b) { return a.x - b.x; });
      var shades = pts.map(function (p) { return p.shade; });
      var sLo = Math.min.apply(null, shades), sHi = Math.max.apply(null, shades);
      var bins = {};
      pts.forEach(function (p) {
        var key = Math.round(X(p.x) / step);
        var n = bins[key] = (bins[key] || 0) + 1;
        var offset = n === 1 ? 0 : (n % 2 === 0 ? 1 : -1) * Math.floor(n / 2) * step * 0.85;
        offset = Math.max(-rowH / 2 + dot, Math.min(rowH / 2 - dot, offset));
        out += '<circle cx="' + X(p.x).toFixed(1) + '" cy="' + (cy + offset).toFixed(1) + '" r="' + dot + '" fill="' +
          mixColour(sHi > sLo ? (p.shade - sLo) / (sHi - sLo) : 0.5) + '" fill-opacity="0.8"><title>' +
          esc(s + " · " + p.label + ": " + (p.x >= 0 ? "+" : "") + p.x.toFixed(2) + " cycles" +
            (hasReadings ? " · reading, smoothed and rescaled: " + Number(p.shade).toFixed(2) : "")) + "</title></circle>";
      });
    });
    out += "</svg>";
    var scale = '<span><i style="background:linear-gradient(to right,' + mixColour(0) + "," + mixColour(1) + ')"></i>' +
      (hasReadings ? "the sensor's reading as the model was given it, low to high"
        : "the window's place in the sequence, first to last") + "</span>";
    return out + '<div class="pg-legend"><span><i style="background:transparent;border:1.5px solid var(--fig-burn)"></i>the four largest</span>' +
      '<span><i style="background:transparent;border:1.5px solid var(--fig-accent)"></i>the other ten</span>' + scale + "</div>";
  }

  // The sensors in rank order, the way the paper's Fig. 9 (middle) draws
  // them: the four with the largest average values red, deeper for larger;
  // the other ten blue, from deep to light as they matter less. Each circle
  // has the average it is ranked by written under it.
  function rankedStrip(meanAbs, ranked) {
    var st = fig10Style(meanAbs, ranked);
    // Each row is tall enough for a circle and the average written under it,
    // so the frame round the four largest encloses both.
    var perRow = Math.ceil(ranked.length / 2), gap = 88, r = 19, W = 640, rowH = 68;
    var x0 = (W - (perRow - 1) * gap) / 2;
    var H = 2 * rowH + 4;
    var out = '<svg class="pg-plot pg-strip" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="The sensors ranked">';
    var box = function (from, to, colour) {
      var row = Math.floor(from / perRow), col = from % perRow, cols = to - from;
      return '<rect x="' + (x0 + col * gap - r - 12) + '" y="' + (row * rowH + 4) + '" width="' + ((cols - 1) * gap + 2 * r + 24) +
        '" height="' + (2 * r + 24) + '" rx="8" fill="none" stroke="' + colour + '" stroke-width="1.4" stroke-dasharray="4 3"/>';
    };
    out += box(0, Math.min(4, perRow), "var(--fig-burn)");
    ranked.forEach(function (s, rank) {
      var row = Math.floor(rank / perRow), col = rank % perRow;
      var cx = x0 + col * gap, cy = row * rowH + 30, k = st.style[s];
      out += '<g><title>' + esc(s + ": rank " + (rank + 1) + " of " + ranked.length) + "</title>" +
        '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/>' +
        '<text x="' + cx + '" y="' + (cy + 3) + '" text-anchor="middle" class="pg-strip-label">' + esc(s) + "</text>" +
        '<text x="' + cx + '" y="' + (cy + r + 12) + '" text-anchor="middle" class="pg-tick">' + fixed(meanAbs[s], 2) + "</text></g>";
    });
    return out + "</svg>";
  }

  // -------------------------------------------------------------- controls
  // Each returns the HTML of one control. `data` is the attribute string that
  // tells the page's one change listener what the control sets.

  function numberField(id, spec, value, data, unit) {
    var input = '<input class="field" type="number" id="' + id + '" min="' + spec.min + '" max="' + spec.max +
      '" step="' + (spec.step || 1) + '" value="' + esc(value) + '" ' + data + ">";
    return unit ? '<span class="field-unit">' + input + "<span>" + esc(unit) + "</span></span>" : input;
  }

  function switchControl(id, checked, data) {
    return '<span class="switch"><input type="checkbox" role="switch" id="' + id + '"' +
      (checked ? " checked" : "") + " " + data + '><span class="switch-track"></span></span>';
  }

  function segmented(name, choices, current, data, labelledBy) {
    return '<div class="seg" role="radiogroup"' + (labelledBy ? ' aria-labelledby="' + labelledBy + '"' : "") + ">" +
      choices.map(function (c) {
        return '<label><input type="radio" name="' + name + '" value="' + esc(c.value) + '"' +
          (String(c.value) === String(current) ? " checked" : "") + " " + data + "><span>" + esc(c.label) + "</span></label>";
      }).join("") + "</div>";
  }

  function optionList(name, choices, current, data, noteOf, labelledBy) {
    return '<div class="opts" role="radiogroup"' + (labelledBy ? ' aria-labelledby="' + labelledBy + '"' : "") + ">" +
      choices.map(function (c) {
        var note = noteOf(c);
        return '<label class="opt' + (c.value === current ? " is-active" : "") + '"><input type="radio" name="' + name +
          '" value="' + esc(c.value) + '"' + (c.value === current ? " checked" : "") + " " + data + ">" +
          '<span class="opt-label">' + esc(c.label) + "</span>" +
          (note ? '<span class="opt-note">' + note + "</span>" : "") + "</label>";
      }).join("") + "</div>";
  }

  function selectField(id, choices, current, data) {
    return '<select class="field" id="' + id + '" ' + data + ">" + choices.map(function (c) {
      return '<option value="' + esc(c.value) + '"' + (String(c.value) === String(current) ? " selected" : "") + ">" +
        esc(c.label) + "</option>";
    }).join("") + "</select>";
  }

  // One setting: its name and where it comes from on the left, its control on
  // the right. A list of options is too wide to sit beside its name, so it
  // goes underneath instead.
  // A setting whose list of options is named by the section head above it
  // has no label of its own, and then no name column either.
  function row(spec) {
    var name = !spec.label ? "" : spec.id
      ? '<label for="' + spec.id + '" id="' + spec.id + '-label">' + esc(spec.label) + "</label>"
      : '<span id="' + spec.key + '-label">' + esc(spec.label) + "</span>";
    var nameColumn = name || spec.hint
      ? '<div class="pg-row-name">' + (name ? '<div class="pg-row-label">' + name + "</div>" : "") +
        '<div class="pg-row-hint"' + (spec.hint ? "" : " hidden") + ">" + (spec.hint || "") + "</div></div>"
      : "";
    return '<div class="pg-row' + (spec.stacked ? " pg-row--stack" : "") + '"' +
      (spec.key ? ' data-row="' + spec.key + '"' : "") + (spec.hidden ? " hidden" : "") + ">" +
      nameColumn + '<div class="pg-row-control">' + spec.control + (spec.reset || "") + "</div></div>";
  }

  function chip(kind, label) { return T.chip(kind, label); }

  // ------------------------------------------------------------------ page

  function start(target, tools) {
    T = tools;
    root = target;
    if (window.__thesisPlaygroundTimer) { clearTimeout(window.__thesisPlaygroundTimer); }
    // seen: the last state each job was polled in. mine: the jobs this page
    // started, whose end is announced even when the first poll finds them ended.
    S = { tab: S.tab || "design", jobs: [], seen: {}, mine: {}, runs: null, selected: null, detail: null,
      compare: [], filter: { subset: "all", results: false, weights: false },
      jobFilter: { kind: "all", status: "all" },
      sections: recall(SECTIONS) || {},
      explain: { run: null, detail: null, seed: null, method: "shap", scope: "engine", engine: "",
        every: 1, shown: null, values: null, index: 0 } };

    // The browser tab is named by app.js, from the menu, as every page is.
    return tools.get("/api/playground/options").then(function (options) {
      S.options = options;
      S.fields = {};
      options.fields.forEach(function (f) { S.fields[f.key] = f; });
      S.settings = loadSettings(options.defaults);
      frame();
      showTab(S.tab);
      pollJobs();
      T.buildToc([]);
    }).catch(function (error) {
      root.innerHTML = errorBox("The playground could not load: " + error.message);
    });
  }

  // The page's name is a heading for screen readers only; the tab strip is
  // the first thing seen, and the menu already names the page.
  function frame() {
    root.innerHTML = '<div class="pg"><h1 class="visually-hidden">Playground</h1>' +
      '<div id="pg-current" class="pg-current" aria-live="polite" hidden></div>' +
      '<div class="pg-steps" role="tablist" aria-label="The three steps, and the jobs">' +
      tabButton("design", 1, "Design a run") + tabButton("results", 2, "Results") + tabButton("explain", 3, "Explain") +
      '<button type="button" role="tab" class="pg-step-tab pg-step-tab--aside" data-tab="jobs" aria-controls="pg-panel">Jobs ' +
      '<span class="pg-jobs-badge" id="pg-jobs-badge"></span></button>' +
      "</div>" +
      '<div id="pg-panel" class="pg-panel" role="tabpanel"></div></div>';
    root.addEventListener("click", onClick);
    root.addEventListener("change", onChange);
    root.addEventListener("input", onInput);
  }

  function tabButton(key, step, label) {
    return '<button type="button" role="tab" class="pg-step-tab" data-tab="' + key + '" aria-controls="pg-panel">' +
      '<span class="pg-step-n">' + step + "</span>" + label + "</button>";
  }

  function showTab(tab) {
    S.tab = tab;
    $all(".pg-step-tab").forEach(function (button) {
      button.setAttribute("aria-selected", String(button.getAttribute("data-tab") === tab));
    });
    // The banner is hidden while the Jobs tab lists the same job, so it is
    // checked again whenever the tab changes, not only when jobs are polled.
    paintBanner();
    if (tab === "design") { drawDesign(); }
    else if (tab === "results") { drawResults(); }
    else if (tab === "jobs") { drawJobsTab(); }
    else { drawExplain(); }
  }

  // ------------------------------------------------------------ dispatch
  // The page's three listeners. An element names what it wants with
  // data-action, data-tab, data-field and the like; the handler is found here.

  function onClick(event) {
    var target = event.target.closest("[data-tab], [data-action], [data-ablation], [data-procedure], " +
      "[data-section], [data-reset], tr[data-run], tr[data-explanation]");
    if (!target || !root.contains(target)) { return; }
    if (target.hasAttribute("data-tab")) { showTab(target.getAttribute("data-tab")); return; }
    if (target.hasAttribute("data-ablation")) { applyPreset(S.options.ablations, target.getAttribute("data-ablation")); return; }
    if (target.hasAttribute("data-procedure")) { chooseProcedure(target.getAttribute("data-procedure")); return; }
    if (target.hasAttribute("data-section")) { toggleSection(target.getAttribute("data-section")); return; }
    if (target.hasAttribute("data-reset")) { resetField(target.getAttribute("data-reset")); return; }
    // The named action is looked for first. A button in the job list carries
    // both an action and the name of the run it opens, and the two branches
    // below would otherwise take it and open that run instead.
    var action = target.getAttribute("data-action");
    if (action) {
      if (ACTIONS[action]) { ACTIONS[action](target); }
      return;
    }
    if (target.matches("tr[data-run]")) {
      if (event.target.tagName !== "INPUT") { openRun(target.getAttribute("data-run")); }
      return;
    }
    if (target.matches("tr[data-explanation]")) {
      var e = S.explain;
      e.shown = target.getAttribute("data-explanation"); e.values = null; e.index = 0;
      drawExplain();
    }
  }

  var ACTIONS = {
    // One button that opens every folded section, or folds them all once none is folded.
    "toggle-sections": function () { setAllSections(!allSectionsOpen()); },
    "reset-all": function () { S.settings = copy(S.options.defaults); saveSettings(); drawDesign(); },
    "run": function () { submit(S.settings); },
    "cancel": function (button) {
      post("/api/playground/cancel", { id: button.getAttribute("data-id") }).then(repoll)
        .catch(function (e) { T.toast(e.message); });
    },
    // Removing a job deletes its own folder under reports/.playground: what it
    // printed and its progress. The run or explanation it produced is kept.
    "forget": function (button) {
      if (!window.confirm("Remove this job from the list? Its log is deleted; anything it wrote to reports/ stays.")) { return; }
      post("/api/playground/forget", { id: button.getAttribute("data-id") }).then(repoll)
        .catch(function (e) { T.toast(e.message); });
    },
    "forget-finished": function () {
      if (!window.confirm("Remove every done, failed, stopped and interrupted job from the list? Their logs are deleted; the runs and explanations they wrote stay.")) { return; }
      post("/api/playground/forget", { all_finished: true }).then(repoll)
        .catch(function (e) { T.toast(e.message); });
    },
    "open-run": function (button) {
      S.selected = button.getAttribute("data-run"); S.detail = null; S.runs = null; showTab("results");
    },
    "open-explanation": function (button) {
      S.explain.run = button.getAttribute("data-run"); S.explain.detail = null;
      S.explain.shown = button.getAttribute("data-name"); S.explain.values = null; showTab("explain");
    },
    "back": function () { S.selected = null; drawResults(); },
    "again": function () { S.settings = settingsFromRun(S.detail); saveSettings(); showTab("design"); },
    "explain-this": function () {
      S.explain.run = S.detail.name; S.explain.detail = null; S.explain.shown = null; showTab("explain");
    },
    // The Explain tab has no run picker: another run is chosen in Results,
    // among the runs that kept their weights.
    "choose-run": function () {
      S.selected = null; S.filter.weights = true; showTab("results");
    },
    "explain": function () { submitExplanation(); }
  };

  function onChange(event) {
    var input = event.target;
    if (input.hasAttribute("data-field")) { fieldChanged(input); return; }
    // The slider redraws when it is let go, not while it is being dragged.
    if (input.hasAttribute("data-window-index")) {
      S.explain.index = Number(input.value);
      var y = window.scrollY;
      drawExplanation();
      window.scrollTo(0, y);
      return;
    }
    if (input.hasAttribute("data-filter")) {
      var f = S.filter, which = input.getAttribute("data-filter");
      f[which] = which === "subset" ? input.value : input.checked;
      drawResults();
      return;
    }
    if (input.hasAttribute("data-compare")) { compareTicked(input); return; }
    if (input.hasAttribute("data-job-filter")) {
      S.jobFilter[input.getAttribute("data-job-filter")] = input.value;
      drawJobsTab();
      paintBanner();
      return;
    }
    if (input.hasAttribute("data-pred-seed")) { S.predSeed = input.value; drawDetail(); return; }
    if (input.hasAttribute("data-pred-order")) { S.predOrder = input.value; drawDetail(); return; }
    if (input.hasAttribute("data-explain")) { explainChanged(input); }
  }

  // While the slider is being dragged only its caption follows it. Redrawing
  // the figure here would replace the slider itself, and the browser ends the
  // drag when the element under the pointer disappears.
  function onInput(event) {
    var input = event.target;
    if (!input.hasAttribute("data-window-index")) { return; }
    S.explain.index = Number(input.value);
    var rows = S.explain.values && S.explain.values.values;
    var caption = $("[data-window-caption]");
    var row = rows && rows[S.explain.index];
    if (row && caption) { caption.textContent = windowCaption(row); }
    if (historyCursor) { historyCursor(S.explain.index); }
  }

  function windowCaption(row) {
    return "cycle " + row.end_cycle + ", true remaining life " + fixed(row.rul, 0) +
      ", predicted " + fixed(row.predicted, 1);
  }

  // ------------------------------------------------------------------ jobs

  // Ask again now. The pending timer is cleared first, or every call would
  // leave a second chain of polling running beside the first.
  function repoll() {
    clearTimeout(window.__thesisPlaygroundTimer);
    return pollJobs();
  }

  function isLive(job) { return job.status === "running" || job.status === "cancelling" || job.status === "queued"; }

  // A job that has ended is announced once, by name, unless the Jobs tab is
  // open, where its card changes in front of the reader. A job is announced
  // when a poll finds it ended after an earlier poll saw it live, or, for a
  // job this page started, when the very first poll already finds it ended:
  // a short explanation can finish between the request and the next poll.
  function pollJobs() {
    if (!onPage()) { return Promise.resolve(); }
    return T.get("/api/playground/jobs").then(function (data) {
      S.torch = data.torch;
      var busy = false;
      data.jobs.forEach(function (job) {
        var before = S.seen[job.id];
        if (isLive(job)) { busy = true; }
        var wasLive = before ? isLive({ status: before }) : !!S.mine[job.id];
        if (!isLive(job) && wasLive) { announceEnd(job); }
        S.seen[job.id] = job.status;
      });
      S.jobs = data.jobs;
      drawJobs();
      window.__thesisPlaygroundTimer = setTimeout(pollJobs, busy ? 2000 : 10000);
    }).catch(function () {
      window.__thesisPlaygroundTimer = setTimeout(pollJobs, 10000);
    });
  }

  function announceEnd(job) {
    if (S.tab !== "jobs") { T.toast(statusWord(job.status) + ": " + job.label); }
    if (job.status !== "done") { return; }
    S.runs = null;
    if (S.tab === "results") { drawResults(); }
    if (S.tab === "explain" && job.kind === "explain") { S.explain.detail = null; drawExplain(); }
  }

  // What the status word cannot say: while a run trains, which seed it is on
  // and its latest held-back error; once a job was stopped, that nothing was
  // kept. The pass count is left to the bar, and a finished job to its button.
  function jobLine(job) {
    var p = job.progress || {};
    if (job.status === "cancelled") { return "Nothing was written to reports/."; }
    if (job.status === "interrupted") { return "Its server stopped; nothing was written to reports/."; }
    if (job.status !== "running") { return ""; }
    if (job.kind === "train") {
      if (p.seeds === undefined) { return "Preparing the data and model…"; }
      var text = "Seed " + ((p.seed_index || 0) + 1);
      var last = p.curve && p.curve.length ? p.curve[p.curve.length - 1] : null;
      if (last && last.validation_rmse !== null && last.validation_rmse !== undefined) {
        text += " · held-back error " + fixed(last.validation_rmse, 2) + " cycles";
      }
      return text;
    }
    return p.total ? "" : "Preparing the data and model…";
  }

  function jobFraction(job) {
    var p = job.progress || {};
    if (job.kind === "train" && p.seeds) {
      return Math.min(1, ((p.seed_index || 0) + (p.epoch || 0) / (p.epochs || 1)) / p.seeds);
    }
    return p.total ? p.done / p.total : 0;
  }

  // The bar only moves while a job runs or is being stopped.
  function jobBar(job) {
    return job.status === "running" || job.status === "cancelling"
      ? '<div class="pg-bar"><span style="width:' + Math.round(jobFraction(job) * 100) + '%"></span></div>' : "";
  }

  function statusPill(job) {
    return '<span class="pg-status">' + esc(statusWord(job.status)) + "</span>";
  }

  // A running job can be stopped and a queued one cancelled; one already
  // stopping has nothing left to press.
  function stopButton(job) {
    if (job.status !== "running" && job.status !== "queued") { return ""; }
    return '<button type="button" class="button button--small" data-action="cancel" data-id="' + esc(job.id) + '">' +
      (job.status === "queued" ? "Cancel" : "Stop") + "</button>";
  }

  // What is happening now first, then what will happen next in the order it
  // will happen, then what has finished, newest first.
  function orderedJobs() {
    var rank = { running: 0, cancelling: 0, queued: 1 };
    return S.jobs.slice().sort(function (a, b) {
      var ra = rank[a.status] === undefined ? 2 : rank[a.status];
      var rb = rank[b.status] === undefined ? 2 : rank[b.status];
      if (ra !== rb) { return ra - rb; }
      return ra === 1 ? a.created - b.created : b.created - a.created;
    });
  }

  function jobShown(job) {
    var f = S.jobFilter;
    var stateGroup = isLive(job) ? "live" : job.status === "done" ? "done" : job.status === "failed" ? "failed" : "stopped";
    return (f.kind === "all" || job.kind === f.kind) && (f.status === "all" || stateGroup === f.status);
  }

  // The banner above the tabs shows the one job running now, and the Jobs
  // tab's badge counts the jobs queued behind it. The Jobs tab holds
  // everything, and is redrawn if it is open.
  function drawJobs() {
    paintBanner();
    var badge = $("#pg-jobs-badge");
    if (badge) {
      var current = orderedJobs().filter(isLive)[0];
      var waiting = S.jobs.filter(function (job) { return job.status === "queued" && job !== current; }).length;
      badge.textContent = waiting ? count(waiting) : "";
      badge.title = waiting ? plural(waiting, "job") + " waiting" : "";
    }
    if (S.tab === "jobs") { drawJobsTab(); }
  }

  // Hidden while the Jobs tab is open with filters that list the same job.
  function paintBanner() {
    var box = $("#pg-current");
    if (!box) { return; }
    var current = orderedJobs().filter(isLive)[0];
    var listed = current && S.tab === "jobs" && jobShown(current);
    box.hidden = !current || listed;
    box.className = "pg-current" + (current ? " pg-job--" + current.status : "");
    if (!current || listed) { box.innerHTML = ""; return; }
    var line = jobLine(current);
    box.innerHTML = '<div class="pg-job-main"><div class="pg-job-head">' + statusPill(current) + "<strong>" +
      esc(current.label) + "</strong>" + (current.elapsed ? '<span class="pg-faint">' + duration(current.elapsed) + "</span>" : "") + "</div>" +
      jobBar(current) + (line ? '<div class="pg-job-line">' + esc(line) + "</div>" : "") + "</div>" +
      '<div class="pg-current-side">' + stopButton(current) + "</div>";
  }

  // Whether the banner can be seen without scrolling. Queuing a job raises a
  // toast only when it cannot, since otherwise the banner already says so.
  function bannerInView() {
    var box = $("#pg-current");
    if (!box || box.hidden) { return false; }
    var r = box.getBoundingClientRect();
    var bar = document.querySelector(".topbar");
    var top = bar ? bar.getBoundingClientRect().bottom : 0;
    return r.bottom > top && r.top < window.innerHeight;
  }

  // The Jobs tab: every job this server ran and every one an earlier server
  // left on disk, filtered by kind and by state, each removable once it has ended.
  function drawJobsTab() {
    var panel = $("#pg-panel");
    if (!panel) { return; }
    var f = S.jobFilter;
    var all = orderedJobs();
    var shown = all.filter(jobShown);
    var ended = all.filter(function (job) { return !isLive(job); }).length;
    panel.innerHTML =
      '<div class="pg-toolbar">' +
      segmented("pg-jf-kind", [{ value: "all", label: "Any kind" }, { value: "train", label: "Training runs" }, { value: "explain", label: "Explanations" }],
        f.kind, 'data-job-filter="kind"') +
      segmented("pg-jf-status", [{ value: "all", label: "Any state" }, { value: "live", label: "Running or queued" }, { value: "done", label: "Done" },
        { value: "failed", label: "Failed" }, { value: "stopped", label: "Stopped or interrupted" }], f.status, 'data-job-filter="status"') +
      (shown.length && shown.length < all.length ? '<span class="pg-faint">' + count(shown.length) + " of " + plural(all.length, "job") + "</span>" : "") +
      (ended ? '<button type="button" class="button button--small" data-action="forget-finished">Remove ended jobs</button>' : "") +
      "</div>" +
      (shown.length ? '<div class="pg-jobs">' + shown.map(jobHtml).join("") + "</div>"
        : '<div class="pg-empty">' + (all.length ? "No job matches these filters." : "No jobs yet.") + "</div>");
  }

  function jobHtml(job) {
    var p = job.progress || {};
    var open = "";
    if (job.status === "done" && p.result) {
      open = job.kind === "train"
        ? '<button type="button" class="button button--small" data-action="open-run" data-run="' + esc(p.result.directory) + '">Open result</button>'
        : '<button type="button" class="button button--small" data-action="open-explanation" data-run="' + esc(p.result.run) +
          '" data-name="' + esc(p.result.directory) + '">Open explanation</button>';
    }
    var forget = !isLive(job)
      ? '<button type="button" class="button button--small button--quiet" data-action="forget" data-id="' + esc(job.id) + '">Remove</button>' : "";
    var line = jobLine(job);
    return '<div class="pg-job pg-job--' + job.status + '">' +
      '<div class="pg-job-main"><div class="pg-job-head">' + statusPill(job) +
      "<strong>" + esc(job.label) + "</strong>" +
      (job.created ? '<span class="pg-faint">' + whenSeconds(job.created) + "</span>" : "") +
      (job.elapsed ? '<span class="pg-faint">' + duration(job.elapsed) + "</span>" : "") + "</div>" +
      jobBar(job) + (line ? '<div class="pg-job-line">' + esc(line) + "</div>" : "") +
      (job.log ? '<pre class="pg-log">' + esc(job.log) + "</pre>" : "") +
      (job.command ? '<details class="pg-details"><summary>Command</summary><code class="pg-code">' + esc(job.command) + "</code></details>" : "") +
      '</div><div class="pg-job-actions">' + open + stopButton(job) + forget + "</div></div>";
  }

  // ---------------------------------------------------------------- design

  function fieldsOf(group) {
    return S.options.fields.filter(function (f) { return f.group === group; });
  }

  // The folding sections: every group except those drawn somewhere else.
  function sectionGroups() {
    return S.options.groups.filter(function (g) { return !UNSECTIONED[g.key]; });
  }

  function groupLabel(key) {
    var found = S.options.groups.filter(function (g) { return g.key === key; })[0];
    return found ? found.label : "";
  }

  function visible(key) {
    var s = S.settings;
    if (key === "temporal_hidden" || key === "temporal_out_features" || key === "shared_across_sensors") {
      return s.temporal === "lstm" || s.temporal === "gru";
    }
    if (key === "smoothing_alpha") { return !!s.smoothing; }
    if (key === "early_stopping_patience") { return !!s.early_stopping; }
    return true;
  }

  function isChanged(f) { return !same(S.settings[f.key], f.default); }

  function changedIn(group) {
    return fieldsOf(group).filter(function (f) { return visible(f.key) && isChanged(f); }).length;
  }

  function matches(settings) {
    return Object.keys(settings).every(function (k) { return S.settings[k] === settings[k]; });
  }

  // The procedure card the settings follow. A card decides two things,
  // whether any engines are held back and whether training stops early, so a
  // held-back fraction other than the card's own still counts as that card.
  // Settings that fit neither card (read back from an older run, say) count
  // as the card that holds engines back, whose inputs then show them.
  function chosenProcedure() {
    var list = S.options.procedures;
    var found = list.filter(function (p) {
      return Object.keys(p.settings).every(function (k) {
        var want = p.settings[k], have = S.settings[k];
        return typeof want === "number" ? (Number(have) > 0) === (want > 0) : have === want;
      });
    })[0];
    return found || list.filter(function (p) { return p.settings.validation_fraction > 0; })[0] || list[0];
  }

  function choiceLabel(key, value) {
    var found = (S.fields[key].choices || []).filter(function (c) { return c.value === value; })[0];
    return found ? found.label : String(value);
  }

  // A label such as "Window length (cycles)" is a name and a unit.
  function splitUnit(label) {
    var m = /^(.*?)\s*\(([^)]+)\)$/.exec(label);
    return m ? { label: m[1], unit: m[2] } : { label: label, unit: "" };
  }

  // What a folded section holds right now, in one line. It is shown only
  // while the section is folded; an open one shows the controls themselves.
  function groupSummary(group) {
    var s = S.settings;
    switch (group) {
      case "preprocessing":
        return s.window_size + "-cycle windows, stride " + s.window_stride + " · ceiling " + s.rul_max +
          " · " + choiceLabel("normalization", s.normalization).toLowerCase() +
          " · " + (s.smoothing ? "smoothed" : "not smoothed") + " · " +
          choiceLabel("graph", s.graph).toLowerCase() + " · " + choiceLabel("adjacency_normalization", s.adjacency_normalization).toLowerCase();
      case "temporal":
        if (s.temporal === "lstm" || s.temporal === "gru") {
          return choiceLabel("temporal", s.temporal) + " · hidden " + s.temporal_hidden + " · " + s.temporal_out_features +
            " per sensor" + (s.shared_across_sensors ? " · shared" : "");
        }
        return choiceLabel("temporal", s.temporal);
      case "graph":
        return "GCN " + s.gcn_hidden.join(" · ") + " → FC " + s.fc_hidden + (s.gcn_bias ? " · with bias" : "");
      case "training":
        return s.epochs + " passes · " + s.batch_size + " per update · " + s.learning_rate + " → " + s.final_learning_rate +
          ", " + choiceLabel("lr_schedule", s.lr_schedule).toLowerCase();
      case "repeats":
        return "from " + s.seed + (s.fixed_split ? " · same split for every seed" : "") + " · " + s.device;
      default:
        return "";
    }
  }

  function sectionOpen(group) {
    return !!S.sections[group];
  }

  function allSectionsOpen() {
    return sectionGroups().every(function (g) { return sectionOpen(g.key); });
  }

  function drawDesign() {
    $("#pg-panel").innerHTML =
      '<div class="pg-design"><div class="pg-form">' + dataHtml() + presetsHtml() +
      '<div class="pg-sections-bar"><h2 class="pg-block-title">Every setting</h2>' +
      '<button type="button" class="button button--quiet button--small" data-action="toggle-sections" id="pg-toggle-sections">' +
      (allSectionsOpen() ? "Collapse all" : "Expand all") + "</button>" +
      '<button type="button" class="pg-reset" data-action="reset-all">Reset all</button></div>' +
      sectionGroups().map(sectionHtml).join("") +
      '</div><aside class="pg-summary pg-block" id="pg-summary"></aside></div>';
    drawSummary($("#pg-summary"), null, null);
    refreshPreview();
  }

  // The sub-dataset stands at the top of the form on its own: one setting
  // needs no section around it.
  function dataHtml() {
    return '<section class="pg-block pg-data">' + fieldsOf("data").map(fieldRow).join("") + "</section>";
  }

  // The variants of Table 11 as one-click presets, each with the RMSE the
  // paper prints for it on the chosen sub-dataset.
  function variantsHtml() {
    var o = S.options, subset = S.settings.subset;
    return o.ablations.map(function (a) {
      var paperRow = (o.paper[subset] || {})[a.key];
      var active = !!(a.settings && matches(a.settings));
      return '<button type="button" class="pg-variant' + (active ? " is-active" : "") + '"' +
        (a.settings ? ' data-ablation="' + esc(a.key) + '" aria-pressed="' + active + '"' : " disabled") +
        (a.reason ? ' title="' + esc(a.reason) + '"' : "") + ">" +
        "<strong>" + esc(a.label) + "</strong>" +
        (paperRow ? "<span>paper RMSE " + fixed(paperRow.rmse, 2) + "</span>" : "") +
        (a.settings ? "" : "<em>not buildable here</em>") + "</button>";
    }).join("");
  }

  // The two procedures are the only control for holding engines back. The
  // one that holds them back carries the inputs for how many, and for how
  // long training may go without improving, while it is the one chosen.
  function presetsHtml() {
    var chosen = chosenProcedure();
    var procedures = S.options.procedures.map(function (p) {
      var active = p === chosen;
      var inputs = p.settings.validation_fraction > 0
        ? '<div class="pg-procedure-fields" id="pg-procedure-fields"' + (active ? "" : " hidden") + ">" +
          fieldsOf("procedure").filter(function (f) { return f.kind !== "bool"; }).map(fieldRow).join("") + "</div>"
        : "";
      return '<div class="pg-procedure">' +
        '<button type="button" class="pg-variant' + (active ? " is-active" : "") + '" data-procedure="' + esc(p.key) +
        '" aria-pressed="' + active + '"><strong>' + esc(p.label) + "</strong><span>" + esc(p.text) + "</span></button>" +
        inputs + "</div>";
    }).join("");
    return '<section class="pg-block" id="pg-presets">' +
      '<h2 class="pg-block-title">Which model · the variants of Table 11</h2>' +
      '<div class="pg-variants" id="pg-variants">' + variantsHtml() + "</div>" +
      '<p class="pg-faint pg-no-row" id="pg-no-row" hidden></p>' +
      '<h2 class="pg-block-title">How it is trained</h2>' +
      '<div class="pg-variants pg-variants--two">' + procedures + "</div></section>";
  }

  function sectionHtml(g) {
    var open = sectionOpen(g.key);
    var changed = changedIn(g.key);
    return '<section class="pg-section" data-group="' + g.key + '">' +
      '<button type="button" class="pg-section-head" data-section="' + g.key + '" aria-expanded="' + open +
      '" aria-controls="pg-section-' + g.key + '">' +
      '<span class="pg-section-title" id="pg-section-title-' + g.key + '">' + esc(g.label) + "</span>" +
      '<span class="pg-section-summary">' + esc(groupSummary(g.key)) + "</span>" +
      '<span class="pg-section-changed"' + (changed ? "" : " hidden") + ">" + changed + " changed</span>" +
      CHEVRON + "</button>" +
      '<div class="pg-section-body" id="pg-section-' + g.key + '"' + (open ? "" : " hidden") + ">" +
      fieldsOf(g.key).map(fieldRow).join("") + "</div></section>";
  }

  // The words under a setting's name: where its default comes from and, for
  // the sub-dataset, what Table 5 says the chosen one holds.
  function hintFor(f) {
    var parts = [sourceNote(f.source)];
    if (f.key === "subset") {
      var t5 = (S.options.dataset || {})[S.settings.subset];
      if (t5) {
        parts.push(count(t5.train_engines) + " training and " + count(t5.test_engines) + " test engines, " +
          plural(t5.conditions, "operating condition") + ", " + plural(t5.fault_modes, "fault mode"));
      }
    }
    return parts.filter(Boolean).join(" ");
  }

  function fieldRow(f) {
    var value = S.settings[f.key];
    var id = "pg-" + f.key;
    var data = 'data-field="' + f.key + '"';
    var parts = splitUnit(f.label);
    // A list of options whose name is its section's name is labelled by the section head.
    var bare = f.kind === "choice" && !SEGMENTED[f.key] && f.label === groupLabel(f.group);
    var stacked = false, control;
    if (f.kind === "bool") {
      control = switchControl(id, !!value, data);
    } else if (f.kind === "choice" && SEGMENTED[f.key]) {
      control = segmented(id, f.choices, value, data, id + "-label");
      id = null;
    } else if (f.kind === "choice") {
      stacked = true;
      control = optionList(id, f.choices, value, data, function (c) { return sourceNote(c.source); },
        bare ? "pg-section-title-" + f.group : id + "-label");
      id = null;
    } else if (f.kind === "widths") {
      control = '<div class="pg-widths">' + value.map(function (w, i) {
        return "<label><span>GCN-" + (i + 1) + '</span><input class="field" type="number" min="' + f.min + '" max="' + f.max +
          '" step="1" value="' + w + '" ' + data + ' data-index="' + i + '"></label>';
      }).join("") + "</div>";
      id = null;
    } else {
      control = numberField(id, f, value, data, parts.unit);
    }
    return row({
      key: f.key, id: id, label: bare ? "" : parts.label, stacked: stacked, hidden: !visible(f.key),
      hint: hintFor(f),
      control: control,
      reset: resetHtml(f)
    });
  }

  function resetHtml(f) {
    return '<span class="pg-reset-slot">' + (isChanged(f)
      ? '<button type="button" class="pg-reset" data-reset="' + f.key + '" title="Back to ' + esc(defaultText(f)) + '">Reset</button>'
      : "") + "</span>";
  }

  function defaultText(f) {
    if (f.kind === "choice") { return choiceLabel(f.key, f.default); }
    if (f.kind === "bool") { return f.default ? "on" : "off"; }
    if (f.kind === "widths") { return f.default.join(", "); }
    return String(f.default);
  }

  function applyPreset(list, key) {
    var preset = list.filter(function (x) { return x.key === key; })[0];
    Object.keys(preset.settings).forEach(function (k) { S.settings[k] = preset.settings[k]; });
    saveSettings();
    drawDesign();
  }

  // Choosing the card already chosen leaves its inputs as they are.
  function chooseProcedure(key) {
    if (chosenProcedure().key === key) { return; }
    applyPreset(S.options.procedures, key);
  }

  function resetField(key) {
    S.settings[key] = copy(S.fields[key].default);
    saveSettings();
    drawDesign();
  }

  function toggleSection(group) {
    S.sections[group] = !sectionOpen(group);
    remember(SECTIONS, S.sections);
    paintSection(group);
    paintSectionsToggle();
  }

  function setAllSections(open) {
    sectionGroups().forEach(function (g) { S.sections[g.key] = open; paintSection(g.key); });
    remember(SECTIONS, S.sections);
    paintSectionsToggle();
  }

  function paintSectionsToggle() {
    var button = $("#pg-toggle-sections");
    if (button) { button.textContent = allSectionsOpen() ? "Collapse all" : "Expand all"; }
  }

  function paintSection(group) {
    var head = $('[data-section="' + group + '"]');
    var body = $("#pg-section-" + group);
    if (!head || !body) { return; }
    var open = sectionOpen(group);
    head.setAttribute("aria-expanded", String(open));
    body.hidden = !open;
    head.querySelector(".pg-section-summary").textContent = groupSummary(group);
    var badge = head.querySelector(".pg-section-changed");
    var changed = changedIn(group);
    badge.hidden = !changed;
    badge.textContent = changed + " changed";
  }

  // A control changed. The setting is read from it, then only what depends
  // on it is redrawn: its row, the rows it can show or hide, the preset
  // cards, the section heads and the run card. The control itself is left
  // alone, so a number being typed keeps its focus.
  function fieldChanged(input) {
    var key = input.getAttribute("data-field");
    var f = S.fields[key];
    if (f.kind === "bool") { S.settings[key] = input.checked; }
    else if (f.kind === "choice") { S.settings[key] = input.value; }
    else if (f.kind === "widths") {
      var widths = S.settings[key].slice();
      widths[+input.getAttribute("data-index")] = input.value === "" ? input.value : Number(input.value);
      S.settings[key] = widths;
    } else { S.settings[key] = input.value === "" ? input.value : Number(input.value); }
    saveSettings();
    paintRow(key);
    S.options.fields.forEach(function (other) {
      var box = $('[data-row="' + other.key + '"]');
      if (box) { box.hidden = !visible(other.key); }
    });
    paintPresets();
    sectionGroups().forEach(function (g) { paintSection(g.key); });
    refreshPreview();
  }

  function paintRow(key) {
    var f = S.fields[key];
    var box = $('[data-row="' + key + '"]');
    if (!box) { return; }
    box.querySelector(".pg-reset-slot").outerHTML = resetHtml(f);
    Array.prototype.forEach.call(box.querySelectorAll(".opt"), function (option) {
      option.classList.toggle("is-active", option.querySelector("input").checked);
    });
    var hint = box.querySelector(".pg-row-hint");
    if (hint) {
      var text = hintFor(f);
      hint.innerHTML = text;
      hint.hidden = !text;
    }
  }

  // The cards are repainted in place rather than rebuilt: the procedure
  // card holds inputs, and rebuilding it would take the focus from one.
  function paintPresets() {
    var variants = $("#pg-variants");
    if (variants) { variants.innerHTML = variantsHtml(); }
    var chosen = chosenProcedure();
    $all("[data-procedure]").forEach(function (button) {
      var active = button.getAttribute("data-procedure") === chosen.key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    var inputs = $("#pg-procedure-fields");
    if (inputs) { inputs.hidden = !(chosen.settings.validation_fraction > 0); }
    paintNoRow();
  }

  // When no variant card fits the settings, the reason the paper has no row
  // for them is said once, under the cards. It comes with the preview, so it
  // is shown only once the preview describes the settings on screen.
  function paintNoRow() {
    var slot = $("#pg-no-row");
    if (!slot) { return; }
    var fits = S.options.ablations.some(function (a) { return a.settings && matches(a.settings); });
    var current = S.preview && S.previewFor === JSON.stringify(S.settings);
    var reason = !fits && current && !S.preview.paper_row ? S.preview.no_row_reason : "";
    slot.textContent = reason || "";
    slot.hidden = !reason;
  }

  function refreshPreview() {
    clearTimeout(S.previewTimer);
    S.previewTimer = setTimeout(function () {
      var box = $("#pg-summary");
      if (!box) { return; }
      var asked = JSON.stringify(S.settings);
      post("/api/playground/preview", S.settings).then(function (preview) {
        if (asked !== JSON.stringify(S.settings)) { return; }  // a newer request is on its way
        S.preview = preview;
        S.previewFor = asked;
        drawSummary(box, preview, null);
        paintNoRow();
      }).catch(function (error) {
        if (asked !== JSON.stringify(S.settings)) { return; }
        S.preview = null;
        drawSummary(box, null, error.message);
        paintNoRow();
      });
    }, 220);
  }

  // A list of cautions, each said first in words, its code after it as a pointer.
  function caveatList(caveats) {
    if (!caveats || !caveats.length) { return ""; }
    return '<ul class="pg-caveats">' + caveats.map(function (c) {
      return '<li class="pg-caveat pg-caveat--' + (c.level || "info") + '">' + (c.html || esc(c.text)) +
        (c.ref ? ' <span class="pg-ref">(' + refLink(c.ref) + ")</span>" : "") + "</li>";
    }).join("") + "</ul>";
  }

  // layer_summary() lists the temporal stage, then Table 4's "Input" row. Both
  // describe the same tensor: Table 4's Input row is the stage's output, with
  // no parameters of its own. Shown as one row, as Table 4 has, and named by
  // the stage only when the stage learns numbers of its own.
  function table4Rows(layers) {
    if (layers.length > 1 && layers[0].layer === "Temporal" && layers[1].layer === "Input") {
      var temporal = S.settings.temporal;
      var from = temporal === "lstm" || temporal === "gru" ? TEMPORAL[temporal] : "";
      return [{ layer: "Input", from: from, out: layers[0].out, parameters: layers[0].parameters }].concat(layers.slice(2));
    }
    return layers;
  }

  // The right-hand card: the one action, then only what stops it or
  // qualifies it, then the network these settings build.
  function drawSummary(box, preview, error) {
    if (!box) { return; }
    var counted = preview && preview.dataset ? preview.dataset.counted : null;
    var torch = S.torch !== false && !(preview && preview.torch === false);
    var canRun = !!preview && torch && !preview.network_error && !!counted;
    var notes = [];
    if (error) { notes.push({ level: "warn", text: error }); }
    if (!torch) { notes.push({ level: "warn", text: "PyTorch is not installed, so nothing can be trained." }); }
    if (preview) {
      if (!counted) {
        notes.push({ level: "warn", text: "The NASA files are not in data/raw/, so nothing can be counted or trained." });
      }
      if (torch && preview.network_error) { notes.push({ level: "warn", text: preview.network_error }); }
      if (counted && counted.left_out > 0) {
        var one = counted.left_out === 1;
        notes.push({ level: "warn", ref: "A-23", text: count(counted.left_out) + " of " + count(counted.test_engines) +
          " test engines " + (one ? "is" : "are") + " shorter than the window and will be left out, " +
          "so the score cannot be set beside Table 7." });
      }
      notes = notes.concat(preview.design_cautions || []);
    }

    var network = "";
    if (!preview && !error) {
      network = '<div class="loading">Building the network…</div>';
    } else if (preview && torch && !preview.network_error && preview.layers) {
      network = '<div class="pg-network"><h2 class="pg-block-title">Network</h2>' +
        '<div class="pg-big">' + count(preview.parameters) + " <span>learned numbers</span></div>" +
        '<details class="pg-details"><summary>Layer by layer</summary>' +
        '<table class="pg-mini"><thead><tr><th>Layer</th><th>Out</th><th class="num">Learned numbers</th></tr></thead><tbody>' +
        table4Rows(preview.layers).map(function (l) {
          return "<tr><td>" + esc(l.layer) + (l.from ? ' <span class="pg-faint">from ' + esc(l.from) + "</span>" : "") +
            "</td><td><code>" + esc(l.out) + '</code></td><td class="num">' + count(Number(l.parameters)) + "</td></tr>";
        }).join("") + "</tbody></table>" +
        '<p class="pg-faint">B is the number of windows in a batch. Table 4 prints no counts; each is read from the built model.</p>' +
        "</details></div>";
    }

    box.innerHTML = '<button type="button" class="button button--primary pg-train" data-action="run"' + (canRun ? "" : " disabled") +
      ">Train " + plural(S.settings.seeds, "seed") + "</button>" + caveatList(notes) + network;
  }

  // Queue a run. The server names the job. A toast confirms it only when the
  // banner that shows the queue is scrolled out of sight.
  function submit(settings) {
    post("/api/playground/train", { settings: settings }).then(function (job) {
      if (job && job.id) { S.mine[job.id] = true; }
      return repoll();
    }).then(function () {
      if (!bannerInView()) { T.toast("Run queued."); }
    }).catch(function (error) { T.toast(error.message, 8000); });
  }

  // --------------------------------------------------------------- results

  var DIRTY = "-dirty: uncommitted changes, so no committed state reproduces this run";

  function loadRuns() {
    if (S.runs) { return Promise.resolve(S.runs); }
    return T.get("/api/playground/runs").then(function (data) { S.runs = data.runs; return S.runs; });
  }

  // The list of stored runs, or, once one is chosen, that run on its own with
  // a way back to the list.
  function drawResults() {
    var panel = $("#pg-panel");
    panel.innerHTML = '<div class="loading">Reading reports/…</div>';
    loadRuns().then(function (runs) {
      if (S.tab !== "results") { return; }
      if (S.selected) { drawDetail(); return; }
      var f = S.filter;
      var shown = runs.filter(function (r) {
        return (f.subset === "all" || r.subset === f.subset) && (!f.results || !r.exploratory) && (!f.weights || r.has_weights);
      });
      // The count is said only when a filter hides some runs and shows some.
      var counted = shown.length && shown.length < runs.length ? count(shown.length) + " of " + plural(runs.length, "run") + " · " : "";
      panel.innerHTML =
        '<div class="pg-toolbar">' +
        segmented("pg-f-subset", [{ value: "all", label: "All" }].concat(SUBSETS.map(function (s) { return { value: s, label: s }; })),
          f.subset, 'data-filter="subset"') +
        "<label>" + switchControl("pg-f-results", f.results, 'data-filter="results"') + "Five seeds or more</label>" +
        "<label>" + switchControl("pg-f-weights", f.weights, 'data-filter="weights"') + "Kept their weights</label>" +
        (shown.length ? '<span class="pg-faint">' + counted + "click a row to open it</span>" : "") + "</div>" +
        '<div id="pg-compare"></div>' +
        (shown.length ? runsTable(shown) : '<div class="pg-empty">No stored run matches these filters.</div>');
      drawComparison();
    }).catch(function (error) {
      panel.innerHTML = errorBox(error.message);
    });
  }

  function openRun(name) {
    S.selected = name; S.detail = null;
    drawDetail();
  }

  // One row per run. The sub-dataset has a column only while the list holds
  // more than one; ticked rows are marked, and a commit with uncommitted
  // changes is coloured, its meaning given once on the column's head.
  function runsTable(runs) {
    var subsets = {};
    runs.forEach(function (r) { subsets[r.subset] = true; });
    var showData = S.filter.subset === "all" && Object.keys(subsets).length > 1;
    return '<div class="pg-runs"><table class="data-table"><thead><tr><th><span class="visually-hidden">Compare</span></th><th>Finished</th>' +
      (showData ? "<th>Data</th>" : "") +
      '<th>Temporal stage</th><th>Graph</th><th>Seeds</th><th>Weights</th><th class="num">RMSE</th><th class="num">Paper</th>' +
      '<th title="' + esc(DIRTY) + '">Commit</th></tr></thead><tbody>' +
      runs.map(function (r) {
        var m = r.mean || {}, sd = r.std || {};
        var ticked = S.compare.indexOf(r.name) >= 0;
        return '<tr data-run="' + esc(r.name) + '"' + (ticked ? ' class="is-selected"' : "") + '><td><input type="checkbox" data-compare="' +
          esc(r.name) + '"' + (ticked ? " checked" : "") + ' aria-label="Compare this run"></td>' +
          "<td>" + when(r.finished) + "</td>" + (showData ? "<td>" + esc(r.subset) + "</td>" : "") +
          "<td>" + esc(stageName(r.temporal, r.temporal_hidden)) + "</td><td>" + esc(graphName(r.graph, r.adjacency_normalization)) + "</td>" +
          '<td><span class="chip' + (r.exploratory ? " chip--warn" : "") + '">' + count(r.seeds) + "</span></td>" +
          "<td>" + (r.reported_weights === "final_epoch" ? "final pass" : "best pass") + "</td>" +
          '<td class="num">' + number(m.rmse, 3) + (sd.rmse !== null && sd.rmse !== undefined ? ' <span class="sd">± ' + number(sd.rmse, 3) + "</span>" : "") + "</td>" +
          '<td class="num" title="' + esc(r.paper_row ? r.paper_row.label + ", " + r.paper_row.source : r.no_row_reason) + '">' +
          (r.paper_row ? fixed(r.paper_row.rmse, 2) : "—") + "</td>" +
          '<td><code class="pg-commit' + (r.dirty ? " pg-commit--dirty" : "") + '">' + esc(r.commit) + "</code>" +
          (r.has_weights ? ' <span class="chip chip--good">files</span>' : "") + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function compareTicked(box) {
    var name = box.getAttribute("data-compare");
    S.compare = S.compare.filter(function (n) { return n !== name; });
    if (box.checked) { S.compare.push(name); }
    if (S.compare.length > 2) { S.compare = S.compare.slice(-2); }
    $all("input[data-compare]").forEach(function (b) { b.checked = S.compare.indexOf(b.getAttribute("data-compare")) >= 0; });
    $all("tr[data-run]").forEach(function (tr) { tr.classList.toggle("is-selected", S.compare.indexOf(tr.getAttribute("data-run")) >= 0); });
    drawComparison();
  }

  // A run named by when it finished, as the runs table prints it.
  function runTime(name) {
    var run = (S.runs || []).filter(function (r) { return r.name === name; })[0];
    if (run && run.finished) { return when(run.finished); }
    var m = /_(\d{8}T\d{6}Z)$/.exec(name || "");
    return m ? when(m[1]) : name;
  }

  function drawComparison() {
    var box = $("#pg-compare");
    if (!box) { return; }
    if (S.compare.length !== 2) {
      box.innerHTML = S.compare.length === 1 ? '<p class="pg-hint">Tick one more run to compare.</p>' : "";
      return;
    }
    box.innerHTML = '<div class="loading">Pairing the seeds…</div>';
    T.get("/api/playground/compare?first=" + encodeURIComponent(S.compare[0]) + "&second=" + encodeURIComponent(S.compare[1]))
      .then(function (c) {
        var real = c.differences.filter(function (d) { return d.setting !== "commit"; });
        var cautions = [];
        if (real.length === 0 && c.differences.length) {
          cautions.push({ level: "warn", text: "Only the commit differs, so any difference comes from the code or the machine." });
        }
        if (real.length > 1) {
          cautions.push({ level: "warn", text: real.length + " settings differ, so the difference below cannot be put down to any one of them. Change one setting at a time." });
        }
        if (!c.shared_seeds.length) {
          cautions.push({ level: "warn", text: "The two runs share no seed, so they cannot be paired." });
        }
        var rows = MEASURES.map(function (m) {
          var e = c.measures[m[0]];
          return "<tr><td>" + m[1] + '</td><td class="num"><strong>' + (e.mean_difference > 0 ? "+" : "") +
            number(e.mean_difference, m[2]) + '</strong></td><td class="num">' + fixed(e.t, 2) + '</td><td class="num">' +
            (e.p === null || e.p === undefined ? "—" : e.p < 0.001 ? e.p.toExponential(1) : fixed(e.p, 3)) + "</td></tr>";
        }).join("");
        box.innerHTML = '<section class="pg-card pg-compare"><h2 class="pg-block-title">Comparison</h2>' + caveatList(cautions) +
          '<table class="pg-mini"><thead><tr><th>Setting that differs</th><th>First, ' + esc(runTime(c.first.name)) +
          "</th><th>Second, " + esc(runTime(c.second.name)) + "</th></tr></thead><tbody>" +
          (c.differences.length ? c.differences.map(function (d) {
            return "<tr><td>" + esc(settingLabel(d.setting)) + "</td><td>" + esc(settingWords(d.setting, d.first)) +
              "</td><td>" + esc(settingWords(d.setting, d.second)) + "</td></tr>";
          }).join("") : '<tr><td colspan="3">Nothing, not even the commit.</td></tr>') + "</tbody></table>" +
          (c.shared_seeds.length ? '<table class="pg-mini"><thead><tr><th>Measure</th>' +
            '<th class="num">Second − first</th><th class="num">paired t</th><th class="num">p</th></tr></thead><tbody>' + rows + "</tbody></table>" +
            '<p class="pg-faint">Paired over seeds ' + c.shared_seeds.join(", ") + ". For RMSE, ASUE and score lower is better; for R² higher is. " +
            "A paired p answers only whether the difference could be luck of the seed: every seed is scored on the same test engines, so it says nothing about engines not seen " +
            "(" + refLink("D-8") + ").</p>" : "") + "</section>";
      }).catch(function (error) {
        box.innerHTML = errorBox(error.message);
      });
  }

  function drawDetail() {
    var panel = $("#pg-panel");
    if (!panel || !S.selected) { return; }
    panel.innerHTML = '<div class="loading">Reading the run…</div>';
    var load = S.detail && S.detail.name === S.selected ? Promise.resolve(S.detail)
      : T.get("/api/playground/run?name=" + encodeURIComponent(S.selected));
    load.then(function (d) {
      if (S.tab !== "results") { return; }
      S.detail = d;
      var seeds = Object.keys(d.per_seed);
      var paperRow = d.paper_row;
      // Each measure with its spread, and the paper's figure at the paper's
      // own precision with the gap to it; nothing where the paper prints none.
      var cards = MEASURES.map(function (m) {
        var mean = (d.mean || {})[m[0]], sd = (d.std || {})[m[0]];
        var paper = paperRow ? paperRow[m[0]] : null;
        var published = paper !== null && paper !== undefined;
        var shown = published ? paperNumber(m[0], paper) : null;
        var gap = published && mean !== null && mean !== undefined ? mean - paper : null;
        return '<div class="pg-metric"><div class="pg-metric-name">' + m[1] + '</div><div class="pg-metric-value">' + number(mean, m[2]) +
          (sd !== null && sd !== undefined ? "<span> ± " + number(sd, m[2]) + "</span>" : "") + "</div>" +
          (published ? '<div class="pg-faint">paper ' + shown.text +
            (gap !== null ? " · gap " + (gap > 0 ? "+" : "") + number(gap, shown.places) : "") + "</div>" : "") + "</div>";
      }).join("");
      var compared = paperRow
        ? '<p class="pg-faint">Compared with ' + esc(paperRow.label) + ", " + esc(paperRow.source) +
          (paperRow.score === null || paperRow.score === undefined ? ", which prints RMSE only." : ".") + "</p>"
        : '<p class="pg-faint">' + esc(d.no_row_reason) + "</p>";
      var cautions = (d.caveats || []).filter(function (c) { return cautionFor(c, "run"); });

      var perSeedPlot = plot({
        width: 420, height: 220, xlabel: "seed", ylabel: "test RMSE (cycles)", label: "test RMSE per seed",
        series: [{ points: d.results.map(function (r) { return [r.seed, r.rmse]; }), colour: "var(--fig-accent)", dots: true, line: false, radius: 5 }],
        hlines: paperRow ? [{ y: paperRow.rmse, label: "paper", colour: "var(--fig-lpc)" }] : []
      });
      var curves = seeds.length && d.per_seed[seeds[0]].history.length;
      var hasValidation = curves && d.per_seed[seeds[0]].history[0].validation_rmse_cycles !== undefined &&
        d.per_seed[seeds[0]].history[0].validation_rmse_cycles !== null;
      var curvePlot = curves ? plot({
        width: 640, height: 260, xlabel: "pass over the training data", label: "training curves",
        ylabel: hasValidation ? "held-back RMSE (cycles)" : "training loss (scaled MSE)",
        series: seeds.map(function (s, i) {
          var h = d.per_seed[s].history;
          var result = d.results.filter(function (r) { return String(r.seed) === s; })[0] || {};
          var key = hasValidation ? "validation_rmse_cycles" : "train_loss";
          var best = h.filter(function (x) { return x.epoch === result.best_epoch; })[0];
          return { name: "seed " + s, colour: PALETTE[i % PALETTE.length],
            points: h.map(function (x) { return [x.epoch, x[key]]; }),
            marks: hasValidation && best ? [[best.epoch, best[key], "pass " + best.epoch]] : [] };
        })
      }) : "";
      var seedChoice = S.predSeed && d.per_seed[S.predSeed] ? S.predSeed : seeds[0];
      // The paper's Fig. 8 puts the test engines in unit order and joins the
      // points of each series; ordering them by true remaining life instead
      // shows how the error changes along the life, so both are offered.
      var byUnit = S.predOrder !== "rul";
      var predictions = seeds.length ? d.per_seed[seedChoice].predictions.slice().sort(byUnit
        ? function (a, b) { return a.unit - b.unit; } : function (a, b) { return a.rul - b.rul; }) : [];
      var predPlot = predictions.length ? plot({
        width: 640, height: 260, ylabel: "remaining life (cycles)", label: "predictions against truth", xlabel: "test engine",
        series: byUnit ? [
          { name: "true remaining life", colour: "var(--fig-burn)", dots: true, radius: 2.8, width: 1.3, points: predictions.map(function (p) { return [p.unit, p.rul]; }) },
          { name: "predicted", colour: "var(--fig-accent)", dots: true, radius: 2.8, width: 1.3, points: predictions.map(function (p) { return [p.unit, p.predicted]; }) }
        ] : [
          { name: "true remaining life", colour: "var(--fig-ink)", points: predictions.map(function (p, i) { return [i + 1, p.rul]; }), width: 2 },
          { name: "predicted", colour: "var(--fig-warn)", dots: true, line: false, points: predictions.map(function (p, i) { return [i + 1, p.predicted]; }) }
        ]
      }) : "";

      panel.innerHTML = '<button type="button" class="button button--small pg-back" data-action="back">← All runs</button>' +
        '<section class="pg-detail"><div class="pg-detail-head"><div><div class="chips">' +
        chip(d.dirty ? "warn" : "", "commit " + d.commit) + chip("", plural(d.seeds, "seed")) + chip("", d.subset) +
        (d.temporal ? chip("", stageName(d.temporal, d.temporal_hidden)) : "") +
        (d.graph ? chip("", graphName(d.graph, d.adjacency_normalization) + " graph") : "") +
        chip("", d.reported_weights === "final_epoch" ? "final pass reported" : "best pass restored") + "</div>" +
        '<details class="pg-details"><summary>Folder</summary><code class="pg-code">' + esc(d.name) + "</code></details></div>" +
        '<div class="pg-actions"><button type="button" class="button" data-action="again">Set these settings up again</button>' +
        (d.has_weights ? '<button type="button" class="button button--primary" data-action="explain-this">Explain this run</button>' : "") + "</div></div>" +
        '<div class="pg-metrics">' + cards + "</div>" + compared + caveatList(cautions) +
        '<div class="pg-charts"><figure class="pg-figure">' + perSeedPlot + "</figure>" +
        (curves ? '<figure class="pg-figure pg-figure--wide">' +
          (hasValidation ? "<figcaption><strong>Not in the paper.</strong> A ring marks the reported pass.</figcaption>" : "") +
          curvePlot + "</figure>"
          : '<div class="pg-empty pg-figure--wide">This run kept no per-seed files, so it has no curves, predictions or weights.</div>') +
        (predPlot ? '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 8.</strong></figcaption>' +
          '<div class="pg-toolbar pg-figure-toolbar"><label for="pg-pred-seed">Seed</label>' +
          selectField("pg-pred-seed", seeds.map(function (s) { return { value: s, label: s }; }), seedChoice, "data-pred-seed") +
          segmented("pg-pred-order", [{ value: "unit", label: "by unit number" }, { value: "rul", label: "by true remaining life" }],
            byUnit ? "unit" : "rul", 'data-pred-order') + "</div>" + predPlot + "</figure>" : "") +
        "</div></section>";
    }).catch(function (error) {
      panel.innerHTML = '<button type="button" class="button button--small pg-back" data-action="back">← All runs</button>' +
        errorBox(error.message);
    });
  }

  function settingsFromRun(d) {
    var s = copy(S.options.defaults);
    var p = d.preprocessing || {};
    ["subset", "window_size", "window_stride", "rul_max", "clip_test_labels", "normalization", "validation_fraction",
      "graph", "adjacency_normalization", "seed"].forEach(function (k) { if (p[k] !== undefined) { s[k] = p[k]; } });
    s.smoothing = p.smoothing_alpha !== null && p.smoothing_alpha !== undefined;
    if (s.smoothing) { s.smoothing_alpha = p.smoothing_alpha; }
    if (d.model) {
      Object.keys(d.model).forEach(function (k) { if (s[k] !== undefined) { s[k] = d.model[k]; } });
    } else if (d.temporal) { s.temporal = d.temporal; if (d.temporal_hidden) { s.temporal_hidden = d.temporal_hidden; } }
    if (d.training) {
      ["epochs", "batch_size", "learning_rate", "final_learning_rate"].forEach(function (k) { s[k] = d.training[k]; });
      s.early_stopping = d.training.early_stopping_patience !== null;
      if (s.early_stopping) { s.early_stopping_patience = d.training.early_stopping_patience; }
    } else {
      T.toast("This run did not record its training settings, so the defaults are used for them.", 7000);
    }
    if (d.lr_schedule) { s.lr_schedule = d.lr_schedule; }
    s.seeds = d.seeds || s.seeds;
    s.fixed_split = d.split === "fixed";
    return s;
  }

  // --------------------------------------------------------------- explain

  // The Explain tab explains one run: the one chosen in Results with
  // "Explain this run", or, when none is chosen, the newest run that kept its
  // weights. It names that run in one line; another is chosen in Results.
  function drawExplain() {
    var panel = $("#pg-panel");
    var e = S.explain;
    panel.innerHTML = '<div class="loading">Reading reports/…</div>';
    loadRuns().then(function (runs) {
      if (S.tab !== "explain") { return; }
      var usable = runs.filter(function (r) { return r.has_weights; });
      if (!usable.length) {
        panel.innerHTML = '<div class="pg-empty">No stored run has kept its weights. Runs made from 2026-09-14 on keep them.</div>';
        return;
      }
      if (!e.run || !usable.some(function (r) { return r.name === e.run; })) {
        e.run = usable[0].name; e.detail = null; e.shown = null;
      }
      var run = usable.filter(function (r) { return r.name === e.run; })[0];
      var load = e.detail && e.detail.name === e.run ? Promise.resolve(e.detail)
        : T.get("/api/playground/run?name=" + encodeURIComponent(e.run));
      return load.then(function (d) {
        if (S.tab !== "explain") { return; }
        e.detail = d;
        var seeds = Object.keys(d.per_seed).filter(function (s) { return d.per_seed[s].has_weights; });
        if (seeds.indexOf(String(e.seed)) < 0) { e.seed = seeds[0]; }
        panel.innerHTML = '<div class="pg-explain">' + explainForm(run, seeds) +
          '<section class="pg-block"><div class="pg-block-head"><h2 class="pg-block-title">Explanations stored for this run</h2>' +
          (d.explanations.length ? '<span class="pg-faint">click a row to open it</span>' : "") + "</div>" +
          (d.explanations.length ? explanationsTable(d.explanations) : '<div class="pg-empty">None yet.</div>') +
          '</section><div id="pg-e-view"></div></div>';
        if (e.shown) { drawExplanation(); }
      });
    }).catch(function (error) {
      panel.innerHTML = errorBox(error.message);
    });
  }

  function explainForm(run, seeds) {
    var e = S.explain;
    var graph = graphName(run.graph, run.adjacency_normalization);
    var named = [when(run.finished), run.subset, stageName(run.temporal, run.temporal_hidden), graph ? graph + " graph" : ""]
      .filter(Boolean).join(" · ");
    var seedChoices = seeds.map(function (s) { return { value: s, label: s }; });
    var forEngine = e.scope === "engine";
    return '<section class="pg-block">' +
      row({ key: "pg-e-run", label: "Run",
        control: '<span class="pg-run-name">' + esc(named) + "</span>" +
          '<button type="button" class="pg-link" data-action="choose-run">Choose another run</button>' }) +
      row({ id: "pg-e-seed", label: "Seed whose weights are explained",
        control: seeds.length <= 6 ? segmented("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"') : selectField("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"') }) +
      row({ key: "pg-e-method", label: "Method", stacked: true,
        control: optionList("pg-e-method", [
          { value: "shap", label: METHODS.shap, note: sourceTag({ kind: "paper", ref: "Eq. 14–15" }) +
            " every one of the " + count(16384) + " subsets of sensors, for every window. An absent sensor is replaced by its average training window (" + refLink("A-20") + ")." },
          { value: "occlusion", label: "Occlusion, one sensor at a time", note: sourceTag({ kind: "extension" }) +
            " 15 evaluations per window. The Shapley term in which the sensor arrives last; the two agree only when sensors do not interact." }
        ], e.method, 'data-explain="method"', function (c) { return c.note; }, "pg-e-method-label") }) +
      row({ key: "pg-e-scope", label: "What is explained", stacked: true,
        control: optionList("pg-e-scope", [
          { value: "engine", label: "One engine, at every cycle", note: sourceTag({ kind: "paper", ref: "Section 4.4" }) +
            " the dynamic explanation of Table 12 and Fig. 10. The test set keeps only each engine's last window, so the engine comes from the held-back engines, or the training engines when none were held back." },
          { value: "test", label: "The last window of every test engine", note: sourceTag({ kind: "extension" }) +
            " one explanation per test engine, for a ranking across engines." }
        ], e.scope, 'data-explain="scope"', function (c) { return c.note; }, "pg-e-scope-label") }) +
      row({ id: "pg-e-engine", key: "pg-e-engine-row", label: "Engine number", hidden: !forEngine,
        hint: "Blank: the engine with the longest history.",
        control: '<input class="field" type="number" id="pg-e-engine" min="1" value="' + esc(e.engine) + '" data-explain="engine">' }) +
      row({ id: "pg-e-every", key: "pg-e-every-row", label: "Explain every n-th window", hidden: !forEngine,
        hint: "Larger is quicker and coarser.",
        control: '<input class="field" type="number" id="pg-e-every" min="1" max="100" value="' + e.every + '" data-explain="every">' }) +
      '<div class="pg-row"><div class="pg-row-name"></div><div class="pg-row-control"><button type="button" class="button button--primary" data-action="explain">Explain</button></div></div>' +
      "</section>";
  }

  // What one stored explanation covers, in the page's words: "held-back engine 5, one window in 20".
  function explainedWhat(x) {
    if (x.scope !== "engine") { return "test engines"; }
    var source = ENGINE_SOURCE[x.source] || x.source || "";
    return (source ? source + " " : "") + "engine " + x.unit + (x.every > 1 ? ", one window in " + x.every : "");
  }

  function explanationsTable(explanations) {
    var e = S.explain;
    return '<div class="pg-runs"><table class="data-table"><thead><tr><th>Method</th><th>What</th><th>Seed</th>' +
      '<th class="num">Windows</th><th>Finished</th></tr></thead><tbody>' +
      explanations.map(function (x) {
        return '<tr data-explanation="' + esc(x.name) + '"' + (x.name === e.shown ? ' class="is-selected"' : "") + "><td>" +
          esc(METHODS[x.method] || x.method) + "</td><td>" + esc(explainedWhat(x)) + "</td><td>" + esc(x.seed) +
          '</td><td class="num">' + count(x.windows) + "</td><td>" + when(x.finished_utc) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function explainChanged(input) {
    var e = S.explain, which = input.getAttribute("data-explain");
    if (which === "seed") { e.seed = input.value; return; }
    if (which === "engine") { e.engine = input.value; return; }
    if (which === "every") { e.every = Number(input.value) || 1; return; }
    if (which === "method") { e.method = input.value; }
    if (which === "scope") { e.scope = input.value; }
    // The engine rows apply to one scope only; the option lists show their pick.
    var forEngine = e.scope === "engine";
    ["pg-e-engine-row", "pg-e-every-row"].forEach(function (key) {
      var box = $('[data-row="' + key + '"]');
      if (box) { box.hidden = !forEngine; }
    });
    $all(".pg-explain .opt").forEach(function (option) {
      option.classList.toggle("is-active", option.querySelector("input").checked);
    });
  }

  // Queue an explanation. The server names the job; a toast confirms it only
  // when the banner that shows the queue is scrolled out of sight.
  function submitExplanation() {
    var e = S.explain;
    var request = { run: e.run, seed: Number(e.seed), method: e.method, scope: e.scope,
      engine: e.scope === "engine" && e.engine !== "" ? Number(e.engine) : null,
      every: e.scope === "engine" ? Number(e.every) || 1 : 1 };
    post("/api/playground/explain", request).then(function (job) {
      if (job && job.id) { S.mine[job.id] = true; }
      return repoll();
    }).then(function () {
      if (!bannerInView()) { T.toast("Explanation queued."); }
    }).catch(function (error) { T.toast(error.message, 8000); });
  }

  function drawExplanation() {
    var e = S.explain, box = $("#pg-e-view");
    if (!box) { return; }
    var load = e.values && e.values.name === e.shown ? Promise.resolve(e.values)
      : T.get("/api/playground/explanation?run=" + encodeURIComponent(e.run) + "&name=" + encodeURIComponent(e.shown))
        .then(function (data) { data.name = e.shown; return data; });
    box.innerHTML = '<div class="loading">Reading the explanation…</div>';
    load.then(function (data) {
      e.values = data;
      var sum = data.summary, rows = data.values;
      if (!rows || !rows.length) {
        box.innerHTML = '<div class="pg-empty">This explanation stored no windows.</div>';
        return;
      }
      var symbols = Object.keys(sum.mean_abs);
      var sensorOrder = Object.keys(rows[0]).filter(function (k) { return symbols.indexOf(k) >= 0; });
      if (e.index >= rows.length) { e.index = 0; }
      var exact = sum.method === "shap";
      var forEngine = sum.scope === "engine";

      // The head says once what a value is, its sign and its unit; the
      // cautions after it say what the numbers can and cannot be read as.
      var cautions = exact
        ? [{ level: "info", ref: "A-24", text: "Equation 14 says a window's values add up to its prediction minus one constant. " +
            "The paper's constant is the average label; here it is the prediction at the average training window instead. " +
            "Checked here: the largest gap is " + Number(sum.sum_gap_max).toExponential(1) + " cycles." },
          { level: "info", ref: "F-08", text: "The paper's Table 12 cannot hold the signed values its own Equation 14 defines, " +
            "so only its ranking by size, not its numbers, can be set beside these." }]
        : [{ level: "warn", text: "Occlusion values miss the prediction's distance from the baseline by up to " +
            fixed(sum.sum_gap_max, 2) + " cycles." }];
      cautions.push({ level: "info", text: "A large value is not evidence that the sensor causes the engine's wear." });
      var head = '<section class="pg-detail"><p class="pg-lead">Each value is how many cycles a sensor moved the prediction away from the baseline of ' +
        fixed(sum.baseline_prediction, 2) + " cycles: positive raised the predicted remaining life, negative lowered it. Computed in " +
        duration(sum.seconds) + ".</p>" + caveatList(cautions);

      var mean = {};
      sensorOrder.forEach(function (s) { mean[s] = sum.mean_abs[s]; });

      // The paper's figures, in the order its Section 4.4 introduces them, each
      // drawn from this explanation. Every caption opens by saying which of the
      // paper's figures it follows, or that the paper has none like it.
      var swarm = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, left.</strong> Every value of every sensor, one dot per ' +
        (forEngine ? "window" : "engine") + ", the sensors ranked by the average size of their values.</figcaption>" + beeswarm(rows, symbols) + "</figure>";
      var strip = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, the two boxes above its graph.</strong> ' +
        "Deeper for larger, lighter for smaller.</figcaption>" + rankedStrip(mean, symbols) + "</figure>";

      var body;
      if (forEngine) {
        var current = rows[e.index];
        var subset = e.detail ? e.detail.subset : "";
        var lifePlot = plot({
          width: 640, height: 250, xlabel: "cycle at the end of the window", ylabel: "remaining life, and error (cycles)",
          label: "the prediction along the engine's life",
          areas: [{ name: "predicted minus true: the paper's \"new error area\"", colour: "var(--fig-lpc)", opacity: 0.35,
            points: rows.map(function (r) { return [r.end_cycle, r.predicted - r.rul]; }) }],
          series: [
            { name: "true remaining life (capped)", colour: "var(--fig-accent)", dash: "7 4", width: 2, points: rows.map(function (r) { return [r.end_cycle, r.rul]; }) },
            { name: "predicted", colour: "var(--fig-lpt)", width: 1.8, points: rows.map(function (r) { return [r.end_cycle, r.predicted]; }) },
            { name: "the chosen window", colour: "var(--fig-ink)", line: false, dots: true, radius: 5.5, points: [[current.end_cycle, current.predicted]] }
          ]
        });
        body = head + '<div class="pg-charts">' +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 7.</strong>' +
          (FIG7_UNIT[subset] ? " The paper's " + esc(subset) + " engine is unit " + FIG7_UNIT[subset] + "." : "") +
          "</figcaption>" + lifePlot + "</figure>" +
          swarm + strip +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, right.</strong> One window on the sensor graph, ' +
          'coloured by the rank of this window\'s values. <label class="pg-slider">Window ' +
          '<input type="range" min="0" max="' + (rows.length - 1) + '" value="' + e.index + '" data-window-index> <span data-window-caption>' +
          esc(windowCaption(current)) + "</span></label></figcaption>" +
          sensorGraph(current, sensorOrder) +
          '<p class="pg-faint">The paper does not say whether influence is the size of a value or the value itself; here it is the size, ignoring the sign.</p></figure>' +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>Not in the paper.</strong> ' +
          "Every sensor, across the whole life. Point at any panel to read every sensor at that cycle.</figcaption>" +
          sensorHistory(rows, symbols, e.index) + "</figure>" +
          (exact ? paperComparison(rows, sensorOrder, sum) : "") + "</div></section>";
      } else {
        // With no single window to show, Fig. 9 (right) would repeat Fig. 9
        // (middle) on the graph; the heat map carries the colour key instead
        // of the history chart, which needs one engine's life.
        var heat = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>Not in the paper.</strong> Every sensor at every ' +
          "test engine (columns, by engine number). Blue raised the predicted remaining life, orange lowered it; the stronger the colour, the more cycles.</figcaption>" +
          heatmap(rows, sensorOrder) + "</figure>";
        body = head + '<div class="pg-charts">' + swarm + strip + heat + "</div></section>";
      }
      box.innerHTML = body;
      wireHistory(box, forEngine ? rows : []);
    }).catch(function (error) {
      box.innerHTML = errorBox(error.message);
    });
  }

  // One panel per sensor, holding that sensor's value at every window of the
  // engine's life, every panel on ONE vertical scale so that a tall panel
  // means a large value and not a large axis. This is the counterpart of
  // plot_contributions_over_time in explain/plots.py, which draws the same
  // thing for the notebooks. A cursor marks one window in every panel at once
  // and each caption reads that sensor's value there; wireHistory moves it.
  var HISTORY_W = 260, HISTORY_H = 104, HISTORY_PAD = 5;
  var historyCursor = null;   // moves the cursor to a window, while the chart is on screen

  function historyScale(rows) {
    var cycles = rows.map(function (r) { return r.end_cycle; });
    var x0 = Math.min.apply(null, cycles), x1 = Math.max.apply(null, cycles);
    return {
      x0: x0, x1: x1,
      X: function (c) { return HISTORY_PAD + (c - x0) / ((x1 - x0) || 1) * (HISTORY_W - 2 * HISTORY_PAD); }
    };
  }

  function signedValue(v) {
    return (v > 0 ? "+" : "") + fixed(v, 2);
  }

  function sensorHistory(rows, symbols, current) {
    var scale = historyScale(rows), X = scale.X;
    var limit = 0;
    symbols.forEach(function (s) {
      rows.forEach(function (r) { if (isFinite(r[s])) { limit = Math.max(limit, Math.abs(r[s])); } });
    });
    var largest = limit;
    limit = (limit || 1) * 1.08;

    // The cycle at which the true remaining life first falls below its
    // ceiling: before it the engine is flat, after it the label is falling.
    var ceiling = Math.max.apply(null, rows.map(function (r) { return r.rul; }));
    var knee = null;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].rul < ceiling) { knee = rows[i].end_cycle; break; }
    }

    var W = HISTORY_W, H = HISTORY_H, pad = HISTORY_PAD, mid = H / 2;
    var Y = function (v) { return mid - (v / limit) * (mid - pad); };
    var marked = rows[current] || rows[0];
    var cursorX = X(marked.end_cycle).toFixed(1);

    var panels = symbols.map(function (s, index) {
      var pts = rows.map(function (r) { return X(r.end_cycle).toFixed(1) + "," + Y(r[s]).toFixed(1); }).join(" ");
      var area = X(rows[0].end_cycle).toFixed(1) + "," + mid + " " + pts + " " +
                 X(rows[rows.length - 1].end_cycle).toFixed(1) + "," + mid;
      var up = "pg-up" + index, down = "pg-down" + index;
      return '<figure class="pg-sm" data-sensor="' + esc(s) + '">' +
        '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(s + " over the engine's life") + '">' +
        "<defs>" +
        '<clipPath id="' + up + '"><rect x="0" y="0" width="' + W + '" height="' + mid + '"/></clipPath>' +
        '<clipPath id="' + down + '"><rect x="0" y="' + mid + '" width="' + W + '" height="' + mid + '"/></clipPath>' +
        "</defs>" +
        (knee !== null ? '<line x1="' + X(knee).toFixed(1) + '" x2="' + X(knee).toFixed(1) +
          '" y1="' + pad + '" y2="' + (H - pad) + '" stroke="var(--fig-edge)" stroke-dasharray="3 3"/>' : "") +
        '<polygon points="' + area + '" fill="var(--fig-accent)" fill-opacity="0.28" clip-path="url(#' + up + ')"/>' +
        '<polygon points="' + area + '" fill="var(--fig-warn)" fill-opacity="0.28" clip-path="url(#' + down + ')"/>' +
        '<line x1="' + pad + '" x2="' + (W - pad) + '" y1="' + mid + '" y2="' + mid + '" stroke="var(--fig-axis)" stroke-width="0.8"/>' +
        '<polyline points="' + pts + '" fill="none" stroke="var(--fig-ink)" stroke-width="1.2"/>' +
        '<line class="pg-sm-cursor" x1="' + cursorX + '" x2="' + cursorX + '" y1="0" y2="' + H + '"/>' +
        "</svg>" +
        '<figcaption><b>' + esc(s) + '</b> <span class="pg-sm-value">' + signedValue(marked[s]) + "</span></figcaption></figure>";
    }).join("");

    // The readout names the cycle only while the pointer is on a panel; the
    // rest of the time the slider's caption already says which window is marked.
    return '<div class="pg-sm-readout is-idle" aria-live="polite">Values at cycle <b>' + marked.end_cycle + "</b></div>" +
      '<div class="pg-smalls">' + panels + "</div>" +
      '<div class="pg-faint">Blue raised the predicted remaining life, orange lowered it. ' +
      "Every panel shares one scale; the largest value is " + fixed(largest, 2) + " cycles." +
      (knee !== null ? " The dashed line is cycle " + knee + ", where the true remaining life starts falling." : "") +
      "</div>";
  }

  // Pointing at any panel moves the cursor in all of them to the nearest
  // window; leaving puts it back on the window the slider has chosen.
  function wireHistory(scope, rows) {
    historyCursor = null;
    var grid = scope.querySelector(".pg-smalls");
    if (!grid || !rows.length) { return; }
    var scale = historyScale(rows);
    var readoutBox = scope.querySelector(".pg-sm-readout");
    var readout = readoutBox && readoutBox.querySelector("b");
    var panels = Array.prototype.map.call(grid.querySelectorAll(".pg-sm"), function (panel) {
      return {
        sensor: panel.getAttribute("data-sensor"),
        cursor: panel.querySelector(".pg-sm-cursor"),
        value: panel.querySelector(".pg-sm-value")
      };
    });

    function show(index) {
      var row = rows[index];
      if (!row) { return; }
      var x = scale.X(row.end_cycle).toFixed(1);
      panels.forEach(function (p) {
        p.cursor.setAttribute("x1", x);
        p.cursor.setAttribute("x2", x);
        p.value.textContent = signedValue(row[p.sensor]);
      });
      if (readout) { readout.textContent = row.end_cycle; }
    }

    grid.addEventListener("pointermove", function (event) {
      var svg = event.target.closest && event.target.closest("svg");
      if (!svg) { return; }
      var box = svg.getBoundingClientRect();
      var inView = (event.clientX - box.left) / (box.width || 1) * HISTORY_W;
      var cycle = scale.x0 + (inView - HISTORY_PAD) / (HISTORY_W - 2 * HISTORY_PAD) * (scale.x1 - scale.x0);
      var best = 0;
      rows.forEach(function (r, i) {
        if (Math.abs(r.end_cycle - cycle) < Math.abs(rows[best].end_cycle - cycle)) { best = i; }
      });
      show(best);
      if (readoutBox) { readoutBox.classList.remove("is-idle"); }
    });
    grid.addEventListener("pointerleave", function () {
      show(S.explain.index);
      if (readoutBox) { readoutBox.classList.add("is-idle"); }
    });
    historyCursor = show;
  }

  function heatmap(rows, sensors) {
    var cell = Math.max(2, Math.min(18, Math.floor(640 / rows.length)));
    var W = 60 + cell * rows.length, H = 14 * 16 + 10;
    var peak = 0;
    rows.forEach(function (r) { sensors.forEach(function (s) { peak = Math.max(peak, Math.abs(r[s])); }); });
    peak = peak || 1;
    var out = '<div class="pg-heat-wrap"><svg class="pg-heat" viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '">';
    sensors.forEach(function (s, j) {
      // This drawing is not scaled to fit, so its labels are a size on the screen and keep to the 11-pixel floor.
      out += '<text x="54" y="' + (j * 16 + 12) + '" text-anchor="end" class="pg-heat-label">' + esc(s) + "</text>";
      rows.forEach(function (r, i) {
        var v = r[s];
        out += '<rect x="' + (60 + i * cell) + '" y="' + (j * 16) + '" width="' + cell + '" height="15" fill="' + (v >= 0 ? "var(--fig-accent)" : "var(--fig-warn)") +
          '" fill-opacity="' + (Math.abs(v) / peak).toFixed(3) + '"><title>' + esc(s + " · engine " + r.unit + ": " + signedValue(v) + " cycles") + "</title></rect>";
      });
    });
    return out + '</svg></div><div class="pg-faint">Strongest colour: ' + peak.toFixed(2) + " cycles.</div>";
  }

  // Fig. 9 and Fig. 10's scheme: the four sensors with the largest values in
  // red, deeper for larger; the other ten in blue, from deep to light as they
  // matter less. Ranked by size, ignoring the sign.
  function fig10Style(values, sensors) {
    var ranked = sensors.slice().sort(function (a, b) { return Math.abs(values[b]) - Math.abs(values[a]); });
    var top = Math.abs(values[ranked[0]]) || 1;
    var style = {};
    ranked.forEach(function (s, rank) {
      style[s] = { rank: rank, colour: rank < 4 ? "var(--fig-burn)" : "var(--fig-fan)",
        opacity: rank < 4 ? 0.45 + 0.55 * Math.abs(values[s]) / top : 1 - (rank - 4) / 10 * 0.8 };
    });
    return { ranked: ranked, style: style };
  }

  function sensorGraph(values, sensors) {
    var g = S.options.graph, st = fig10Style(values, sensors);
    var scale = 0.62;
    var out = '<svg class="pg-graph" viewBox="0 0 ' + g.width * scale + " " + (g.height - 30) * scale + '">';
    g.edges.forEach(function (edge) {
      var a = g.positions[edge[0]], b = g.positions[edge[1]];
      out += '<line x1="' + a[0] * scale + '" y1="' + a[1] * scale + '" x2="' + b[0] * scale + '" y2="' + b[1] * scale + '" stroke="var(--fig-edge)" stroke-width="1.2"/>';
    });
    st.ranked.forEach(function (s, rank) {
      var p = g.positions[s], k = st.style[s];
      out += '<g><title>' + esc(s + ": rank " + (rank + 1) + ", " + signedValue(Number(values[s])) + " cycles") + "</title>" +
        '<circle cx="' + p[0] * scale + '" cy="' + p[1] * scale + '" r="17" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/>' +
        '<text x="' + p[0] * scale + '" y="' + (p[1] * scale + 30) + '" text-anchor="middle" class="pg-node">' + esc(s) + "</text></g>";
    });
    return out + "</svg>";
  }

  // How many times a sensor joins the red four, going from each graph in the
  // list to the next.
  function redFourEntries(list, sensors) {
    var entries = 0;
    for (var i = 1; i < list.length; i++) {
      var before = fig10Style(list[i - 1], sensors).ranked.slice(0, 4);
      entries += fig10Style(list[i], sensors).ranked.slice(0, 4).filter(function (s) { return before.indexOf(s) < 0; }).length;
    }
    return entries;
  }

  // Where each sensor sits round a small ring, in Fig. 10's order.
  function ringPositions() {
    var ring = S.options.table12.ring, n = ring.length, c = 50, R = 40;
    var at = {};
    ring.forEach(function (s, i) {
      var a = (-90 - 180 / n + i * 360 / n) * Math.PI / 180;
      at[s] = [c + R * Math.cos(a), c + R * Math.sin(a)];
    });
    return at;
  }

  function ringEdges(at) {
    return S.options.graph.edges.map(function (edge) {
      var a = at[edge[0]], b = at[edge[1]];
      return '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) +
        '" stroke="var(--fig-edge)" stroke-width="0.45"/>';
    }).join("");
  }

  function ringLabel(p, s) {
    return '<text x="' + p[0].toFixed(1) + '" y="' + (p[1] + 1.9).toFixed(1) + '" text-anchor="middle" class="pg-ring-label">' + esc(s) + "</text>";
  }

  // One small graph with the sensors round a circle in Fig. 10's order and
  // Table 3's edges between them. Only the red four are named, since which
  // four they are is what changes from ring to ring; the key ring names the
  // rest. unit is "" for the paper's unitless values.
  function ringGraph(values, sensors, label, unit) {
    var ring = S.options.table12.ring, n = ring.length, at = ringPositions();
    var out = '<figure class="pg-ring"><svg viewBox="0 0 100 100" role="img" aria-label="' + esc(label) + '">';
    if (!values) {
      return out + '<text x="50" y="52" text-anchor="middle" class="pg-ring-missing">not explained</text></svg><figcaption>' + esc(label) + "</figcaption></figure>";
    }
    out += ringEdges(at);
    var st = fig10Style(values, sensors);
    ring.forEach(function (s) {
      var p = at[s], k = st.style[s], v = Number(values[s]);
      out += "<g><title>" + esc(label + " · " + s + ": rank " + (k.rank + 1) + " of " + n + ", " + (unit ? signedValue(v) + " " + unit : fixed(v, 2))) + "</title>" +
        '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="8.4" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-ink)" stroke-width="0.35"/>' +
        (k.rank < 4 ? ringLabel(p, s) : "") + "</g>";
    });
    return out + "</svg><figcaption>" + esc(label) + "</figcaption></figure>";
  }

  // The ring once more with every sensor named and nothing coloured: the key to where each one sits.
  function keyRing() {
    var at = ringPositions();
    var out = '<figure class="pg-ring pg-ring--key"><svg viewBox="0 0 100 100" role="img" aria-label="Where each sensor sits">' + ringEdges(at);
    S.options.table12.ring.forEach(function (s) {
      var p = at[s];
      out += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="8.4" fill="var(--fig-bg)" stroke="var(--fig-axis)" stroke-width="0.5"/>' +
        ringLabel(p, s);
    });
    return out + "</svg><figcaption>key</figcaption></figure>";
  }

  // The paper's Fig. 10, redrawn from Table 12, above this engine at the same
  // graph numbers. Graph g is the g-th window, so with every n-th window
  // explained only graphs 1, 1 + n, 1 + 2n ... exist here.
  function paperComparison(rows, sensors, sum) {
    var t = S.options.table12, every = Number(sum.every) || 1;
    if (!t || !t.graphs) {
      return '<div class="pg-empty pg-figure--wide">The paper\'s Table 12 did not arrive with this page.</div>';
    }
    var names = ["Early", "Middle", "Late"];
    var paperSteps = 0, oursSteps = 0, oursAll = true;
    var blocks = [0, 5, 10].map(function (first, k) {
      var graphs = t.graphs.slice(first, first + 5);
      var paper = graphs.map(function (g, j) {
        var v = {};
        sensors.forEach(function (s) { v[s] = t.values[s][first + j]; });
        return v;
      });
      var ours = graphs.map(function (g) {
        var i = (g - 1) / every;
        return i === Math.floor(i) && i < rows.length ? rows[i] : null;
      });
      paperSteps += redFourEntries(paper, sensors);
      if (ours.every(Boolean)) { oursSteps += redFourEntries(ours, sensors); } else { oursAll = false; }
      // The two rows are named once, in the first stage; the stages below keep the same order.
      return '<div class="pg-fig10-stage"><div class="pg-fig10-head">' + names[k] + "</div>" +
        '<div class="pg-fig10-row">' + (k === 0 ? '<span class="pg-fig10-who">The paper</span>' : "") +
        paper.map(function (v, j) { return ringGraph(v, sensors, "graph " + graphs[j], ""); }).join("") + "</div>" +
        '<div class="pg-fig10-row">' + (k === 0 ? '<span class="pg-fig10-who">This engine</span>' : "") +
        ours.map(function (r, j) { return ringGraph(r, sensors, r ? "cycle " + r.end_cycle : "graph " + graphs[j], "cycles"); }).join("") + "</div></div>";
    }).join("");
    var allSteps = every === 1 && rows.length > 1 ? redFourEntries(rows, sensors) : null;
    return '<figure class="pg-figure pg-figure--wide"><figcaption class="pg-fig10-caption"><span><strong>As the paper\'s Fig. 10.</strong> ' +
      "Each small graph is one window, its sensors round the circle in the order the paper draws them, the red four named. " +
      "The top row of each stage is the paper's, redrawn from the numbers of its Table 12; " +
      "the row below is this engine at the same graph numbers, where graph 1 is the first window.</span>" + keyRing() + "</figcaption>" + blocks +
      '<p class="pg-fig10-count"><strong>How often the red four changes.</strong> Inside the three stages there are 12 steps from one graph to the next. ' +
      "Over those steps a sensor joins the red four " + plural(paperSteps, "time") + " in the paper's table" +
      (oursAll ? ", and " + plural(oursSteps, "time") + " for this engine." : "; this engine is missing some of those graphs, so it is not counted.") +
      (allSteps !== null ? " Over all " + count(rows.length - 1) + " steps between this engine's neighbouring windows: " + plural(allSteps, "time") + "." : "") + "</p>" +
      '<p class="pg-faint">The paper\'s printed figure agrees with Table 12 on the red four in 13 of the 15 graphs; ' +
      "in graph 81 it colours T30 red in place of phi, and in graph 82 T30 and Nc in place of P30 and phi (" + refLink("F-17") + "). " +
      "Table 12 describes one engine the paper does not name, so the engine below it is a different one.</p></figure>";
  }

  window.thesisPlayground = start;
})();
