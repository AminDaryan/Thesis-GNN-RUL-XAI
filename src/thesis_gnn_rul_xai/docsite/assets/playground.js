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
  var TEMPORAL = { lstm: "LSTM", gru: "GRU", last_cycle: "removed", mean_window: "window mean",
    window_summary: "window summary", raw_window: "raw window" };
  var MEASURES = [["rmse", "RMSE", 3], ["score", "Score", 1], ["r2", "R²", 3], ["asue", "ASUE", 3]];
  // Choices short enough to sit side by side in one control. Every other
  // choice is a list of options, each with the source of its value.
  var SEGMENTED = { subset: true, device: true };
  var SUBSETS = ["FD001", "FD002", "FD003", "FD004"];
  var CHEVRON = '<svg class="nav-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 L8 6 L4 9.5" ' +
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var ABOUT = "Choose a sub-dataset, the preprocessing, the temporal stage and the graph layers' sizes, " +
    "then train. Remove a part of the model to see what it was worth, and explain what a trained model " +
    "relies on. Nothing is computed in the browser. Every run started here is <strong>scripts/train.py</strong> " +
    "with the command shown, stored in <code>reports/</code> under the commit that produced it, exactly as " +
    "if it had been typed; an explanation is <strong>scripts/explain.py</strong> in the same way. Beside every " +
    "setting is where its value comes from: the paper, a numbered choice of ours, or neither.";

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
  function plural(n, word) { return n + " " + word + (n === 1 ? "" : "s"); }

  function when(stamp) {
    var m = /^(\d{4})(\d\d)(\d\d)T(\d\d)(\d\d)/.exec(stamp || "");
    return m ? m[1] + "-" + m[2] + "-" + m[3] + " " + m[4] + ":" + m[5] : (stamp || "—");
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
    if (!source) { return ""; }
    var names = { paper: "Paper", assumption: "Our choice", project: "Project rule",
      extension: "Not in the paper" };
    return '<span class="src src--' + source.kind + '">' + names[source.kind] +
      (source.ref ? " · " + refLink(source.ref) : "") + "</span>";
  }

  function temporalName(value) {
    return value === null || value === undefined ? "not recorded" : (TEMPORAL[value] || value);
  }

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
    return out + '<div class="pg-legend"><span><i style="background:transparent;border:1.5px solid var(--fig-burn)"></i>the four largest on average</span>' +
      '<span><i style="background:transparent;border:1.5px solid var(--fig-accent)"></i>the other ten</span>' + scale + "</div>";
  }

  // The sensors in rank order, the way the paper's Fig. 9 (middle) draws
  // them: the four with the largest average values red, deeper for larger;
  // the other ten blue, from deep to light as they matter less.
  function rankedStrip(meanAbs, ranked) {
    var st = fig10Style(meanAbs, ranked);
    var perRow = Math.ceil(ranked.length / 2), gap = 88, r = 19, W = 640;
    var x0 = (W - (perRow - 1) * gap) / 2;
    var H = 2 * 56 + 10;
    var out = '<svg class="pg-plot pg-strip" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="The sensors ranked">';
    var box = function (from, to, colour) {
      var row = Math.floor(from / perRow), col = from % perRow, cols = to - from;
      return '<rect x="' + (x0 + col * gap - r - 7) + '" y="' + (row * 56 + 4) + '" width="' + ((cols - 1) * gap + 2 * r + 14) +
        '" height="' + (2 * r + 14) + '" rx="8" fill="none" stroke="' + colour + '" stroke-width="1.4" stroke-dasharray="4 3"/>';
    };
    out += box(0, Math.min(4, perRow), "var(--fig-burn)");
    ranked.forEach(function (s, rank) {
      var row = Math.floor(rank / perRow), col = rank % perRow;
      var cx = x0 + col * gap, cy = row * 56 + 30, k = st.style[s];
      out += '<g><title>' + esc((rank + 1) + ". " + s + ": average size " + Number(meanAbs[s]).toFixed(2) + " cycles") + "</title>" +
        '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/>' +
        '<text x="' + cx + '" y="' + (cy + 3) + '" text-anchor="middle" class="pg-strip-label">' + esc(s) + "</text>" +
        '<text x="' + cx + '" y="' + (cy + r + 12) + '" text-anchor="middle" class="pg-tick">' + (rank + 1) + "</text></g>";
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
        return '<label class="opt' + (c.value === current ? " is-active" : "") + '"><input type="radio" name="' + name +
          '" value="' + esc(c.value) + '"' + (c.value === current ? " checked" : "") + " " + data + ">" +
          '<span class="opt-label">' + esc(c.label) + '</span><span class="opt-note">' + noteOf(c) + "</span></label>";
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
  function row(spec) {
    var name = spec.id
      ? '<label for="' + spec.id + '" id="' + spec.id + '-label">' + esc(spec.label) + "</label>"
      : '<span id="' + spec.key + '-label">' + esc(spec.label) + "</span>";
    return '<div class="pg-row' + (spec.stacked ? " pg-row--stack" : "") + (spec.changed ? " is-changed" : "") + '"' +
      (spec.key ? ' data-row="' + spec.key + '"' : "") + (spec.hidden ? " hidden" : "") + ">" +
      '<div class="pg-row-name"><div class="pg-row-label">' + name + "</div>" +
      (spec.hint ? '<div class="pg-row-hint">' + spec.hint + "</div>" : "") + "</div>" +
      '<div class="pg-row-control">' + spec.control + (spec.reset || "") + "</div></div>";
  }

  function chip(kind, label) { return T.chip(kind, label); }

  // ------------------------------------------------------------------ page

  function start(target, tools) {
    T = tools;
    root = target;
    if (window.__thesisPlaygroundTimer) { clearTimeout(window.__thesisPlaygroundTimer); }
    S = { tab: S.tab || "design", jobs: [], seen: {}, runs: null, selected: null, detail: null,
      compare: [], comparison: null, filter: { subset: "all", results: false, weights: false },
      jobFilter: { kind: "all", status: "all" },
      sections: recall(SECTIONS) || {},
      explain: { run: null, detail: null, seed: null, method: "shap", scope: "engine", engine: "",
        every: 1, shown: null, values: null, index: 0 } };

    return tools.get("/api/playground/options").then(function (options) {
      S.options = options;
      S.fields = {};
      options.fields.forEach(function (f) { S.fields[f.key] = f; });
      S.settings = loadSettings(options.defaults);
      frame();
      showTab(S.tab);
      pollJobs();
      document.title = "Playground — Thesis documentation";
      T.buildToc([]);
    }).catch(function (error) {
      root.innerHTML = '<div class="error-box"><strong>The playground could not load</strong>' +
        esc(error.message) + "</div>";
    });
  }

  function frame() {
    root.innerHTML = '<div class="pg">' +
      T.crumbs([{ label: "Home", href: "#/" }, { label: "Explore" }, { label: "Playground" }]) +
      T.pageHead("Explore", "Playground",
        "Design a run, train it with the project's own script, read the stored result, and explain " +
        "what the trained network relies on.",
        '<details class="pg-about"><summary>How this page works</summary><p>' + ABOUT + "</p></details>") +
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

  function tabButton(key, number, label) {
    return '<button type="button" role="tab" class="pg-step-tab" data-tab="' + key + '" aria-controls="pg-panel">' +
      '<span class="pg-step-n">' + number + "</span>" + label + "</button>";
  }

  function showTab(tab) {
    S.tab = tab;
    $all(".pg-step-tab").forEach(function (button) {
      button.setAttribute("aria-selected", String(button.getAttribute("data-tab") === tab));
    });
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
    if (target.hasAttribute("data-procedure")) { applyPreset(S.options.procedures, target.getAttribute("data-procedure")); return; }
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
    "expand": function () { setAllSections(true); },
    "collapse": function () { setAllSections(false); },
    "reset-all": function () { S.settings = copy(S.options.defaults); saveSettings(); drawDesign(); },
    "run": function () { submit([S.settings]); },
    "twin": function () {
      var ablated = copy(S.settings);
      ablated.temporal = "last_cycle";
      submit([S.settings, ablated]);
    },
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
      if (!window.confirm("Remove every finished, failed, stopped and interrupted job from the list? Their logs are deleted; the runs and explanations they wrote stay.")) { return; }
      post("/api/playground/forget", { all_finished: true }).then(function (answer) {
        T.toast(plural(answer.forgotten.length, "job") + " removed.");
        repoll();
      }).catch(function (e) { T.toast(e.message); });
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
  }

  function windowCaption(row) {
    return "cycle " + row.end_cycle + ", true life " + fixed(row.rul, 0) +
      ", predicted " + fixed(row.predicted, 1);
  }

  // ------------------------------------------------------------------ jobs

  // Ask again now. The pending timer is cleared first, or every call would
  // leave a second chain of polling running beside the first.
  function repoll() {
    clearTimeout(window.__thesisPlaygroundTimer);
    pollJobs();
  }

  function pollJobs() {
    if (!onPage()) { return; }
    T.get("/api/playground/jobs").then(function (data) {
      S.torch = data.torch;
      var busy = false;
      data.jobs.forEach(function (job) {
        var before = S.seen[job.id];
        if ((job.status === "running" || job.status === "queued" || job.status === "cancelling")) { busy = true; }
        if (before && before !== job.status && job.status === "done") {
          T.toast(job.label + " finished.");
          S.runs = null;
          if (S.tab === "results") { drawResults(); }
          if (S.tab === "explain" && job.kind === "explain") { S.explain.detail = null; drawExplain(); }
        }
        S.seen[job.id] = job.status;
      });
      S.jobs = data.jobs;
      drawJobs();
      window.__thesisPlaygroundTimer = setTimeout(pollJobs, busy ? 2000 : 10000);
    }).catch(function () {
      window.__thesisPlaygroundTimer = setTimeout(pollJobs, 10000);
    });
  }

  function jobLine(job) {
    var p = job.progress || {};
    if (job.status === "failed") { return "The command failed. Its last lines are below."; }
    if (job.status === "cancelled") { return "Stopped. Nothing was written to reports/."; }
    if (job.status === "interrupted") { return "The server it ran on stopped before it finished, so nothing was written to reports/."; }
    if (job.status === "queued") { return "Waiting for the job before it to finish."; }
    if (job.kind === "train") {
      if (p.seeds === undefined) { return "Preparing the data…"; }
      var text = "Seed " + ((p.seed_index || 0) + 1) + " of " + p.seeds;
      if (p.epoch) { text += " · pass " + p.epoch + " of " + p.epochs; }
      var last = p.curve && p.curve.length ? p.curve[p.curve.length - 1] : null;
      if (last && last.validation_rmse !== null && last.validation_rmse !== undefined) {
        text += " · held-back error " + fixed(last.validation_rmse, 2) + " cycles";
      }
      if (p.seeds_done && p.seeds_done.length) {
        text += " · test RMSE so far: " + p.seeds_done.map(function (d) { return fixed(d.rmse, 2); }).join(", ");
      }
      return text;
    }
    if (p.total) { return p.done + " of " + p.total + " windows explained"; }
    return "Preparing the data and model…";
  }

  function jobFraction(job) {
    var p = job.progress || {};
    if (job.status === "done") { return 1; }
    if (job.kind === "train" && p.seeds) {
      return Math.min(1, ((p.seed_index || 0) + (p.epoch || 0) / (p.epochs || 1)) / p.seeds);
    }
    return p.total ? p.done / p.total : 0;
  }

  function isLive(job) { return job.status === "running" || job.status === "cancelling" || job.status === "queued"; }

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

  // The strip above the tabs shows the one job running now and how many wait
  // behind it; the Jobs tab holds everything, and is redrawn if it is open.
  function drawJobs() {
    var box = $("#pg-current"), badge = $("#pg-jobs-badge");
    if (!box) { return; }
    var live = orderedJobs().filter(isLive);
    var current = live[0];
    box.hidden = !current;
    if (current) {
      var p = current.progress || {};
      var waiting = live.length - 1;
      box.innerHTML = '<div class="pg-job-main"><div class="pg-job-head"><span class="pg-status">' + esc(current.status) + "</span><strong>" +
        esc(current.label) + "</strong>" + (current.elapsed ? '<span class="pg-faint">' + duration(current.elapsed) + "</span>" : "") + "</div>" +
        '<div class="pg-bar"><span style="width:' + Math.round(jobFraction(current) * 100) + '%"></span></div>' +
        '<div class="pg-job-line">' + esc(jobLine(current)) +
        (p.seeds_done && p.seeds_done.length ? "" : "") + "</div></div>" +
        '<div class="pg-current-side">' +
        (waiting ? '<span class="pg-faint">' + plural(waiting, "more job") + " queued</span>" : "") +
        '<button type="button" class="button button--small" data-action="cancel" data-id="' + esc(current.id) + '">' +
        (current.status === "queued" ? "Remove" : "Stop") + "</button>" +
        '<button type="button" class="button button--small button--quiet" data-tab="jobs">All jobs</button></div>';
    } else {
      box.innerHTML = "";
    }
    if (badge) {
      badge.textContent = live.length ? String(live.length) : (S.jobs.length ? String(S.jobs.length) : "");
      badge.setAttribute("data-live", String(live.length > 0));
      badge.title = live.length ? plural(live.length, "job") + " running or queued" : plural(S.jobs.length, "job") + " kept";
    }
    if (S.tab === "jobs") { drawJobsTab(); }
  }

  // The Jobs tab: every job this server ran and every one an earlier server
  // left on disk, filtered by kind and by state, each removable once it has ended.
  function drawJobsTab() {
    var panel = $("#pg-panel");
    if (!panel) { return; }
    var f = S.jobFilter;
    var all = orderedJobs();
    var shown = all.filter(function (job) {
      var stateGroup = isLive(job) ? "live" : job.status === "done" ? "done" : job.status === "failed" ? "failed" : "stopped";
      return (f.kind === "all" || job.kind === f.kind) && (f.status === "all" || stateGroup === f.status);
    });
    var ended = all.filter(function (job) { return !isLive(job); }).length;
    panel.innerHTML =
      '<div class="pg-toolbar">' +
      segmented("pg-jf-kind", [{ value: "all", label: "All" }, { value: "train", label: "Training runs" }, { value: "explain", label: "Explanations" }],
        f.kind, 'data-job-filter="kind"') +
      segmented("pg-jf-status", [{ value: "all", label: "Any state" }, { value: "live", label: "Running or queued" }, { value: "done", label: "Finished" },
        { value: "failed", label: "Failed" }, { value: "stopped", label: "Stopped or interrupted" }], f.status, 'data-job-filter="status"') +
      '<span class="pg-faint">' + shown.length + " of " + plural(all.length, "job") + "</span>" +
      (ended ? '<button type="button" class="button button--small" data-action="forget-finished">Remove all ended</button>' : "") +
      "</div>" +
      (shown.length ? '<div class="pg-jobs">' + shown.map(jobHtml).join("") + "</div>"
        : '<div class="pg-empty">' + (all.length ? "No job matches these filters." :
          "No jobs yet. Runs and explanations you start appear here, one at a time, in the order you asked for them.") + "</div>") +
      '<p class="pg-faint">Every job has a folder in reports/.playground holding what it printed and its progress. ' +
      "A job from an earlier server is read back from that folder. Removing a job deletes its folder and nothing else: " +
      "the run or explanation it wrote into reports/ stays.</p>";
  }

  function jobHtml(job) {
    var p = job.progress || {};
    var curve = (p.curve || []).map(function (c) {
      return [c.epoch, c.validation_rmse !== null && c.validation_rmse !== undefined ? c.validation_rmse : c.train_loss];
    });
    var spark = job.kind === "train" && job.status === "running" && curve.length > 1
      ? '<div class="pg-spark">' + plot({ width: 200, height: 60, series: [{ points: curve, colour: "var(--fig-accent)" }] })
        .replace(/<text[^>]*>[^<]*<\/text>/g, "") + "</div>" : "";
    var open = "";
    if (job.status === "done" && p.result) {
      open = job.kind === "train"
        ? '<button type="button" class="button button--small" data-action="open-run" data-run="' + esc(p.result.directory) + '">Open result</button>'
        : '<button type="button" class="button button--small" data-action="open-explanation" data-run="' + esc(p.result.run) +
          '" data-name="' + esc(p.result.directory) + '">Open explanation</button>';
    }
    var stop = job.status === "running" || job.status === "queued"
      ? '<button type="button" class="button button--small" data-action="cancel" data-id="' + esc(job.id) + '">' +
        (job.status === "queued" ? "Remove" : "Stop") + "</button>" : "";
    var forget = !isLive(job)
      ? '<button type="button" class="button button--small button--quiet" data-action="forget" data-id="' + esc(job.id) + '" title="Delete this job\'s log and progress; what it wrote to reports/ stays">Remove</button>' : "";
    var when_ = job.created ? new Date(job.created * 1000).toISOString().slice(0, 16).replace("T", " ") : "";
    return '<div class="pg-job pg-job--' + job.status + '">' +
      '<div class="pg-job-main"><div class="pg-job-head"><span class="pg-status">' + esc(job.status) +
      "</span><strong>" + esc(job.label) + "</strong>" +
      (when_ ? '<span class="pg-faint">' + when_ + " UTC</span>" : "") +
      (job.elapsed ? '<span class="pg-faint">' + duration(job.elapsed) + "</span>" : "") +
      (job.from_disk ? '<span class="chip" title="Read back from its folder; the server that ran it has stopped">earlier server</span>' : "") + "</div>" +
      '<div class="pg-bar"><span style="width:' + Math.round(jobFraction(job) * 100) + '%"></span></div>' +
      '<div class="pg-job-line">' + esc(jobLine(job)) + "</div>" +
      (job.log ? '<pre class="pg-log">' + esc(job.log) + "</pre>" : "") +
      (job.command ? '<details class="pg-command"><summary>Command</summary><code>' + esc(job.command) + "</code></details>" : "") +
      '</div><div class="pg-job-side">' + spark + '<div class="pg-job-actions">' + open + stop + forget + "</div></div></div>";
  }

  // ---------------------------------------------------------------- design

  function fieldsOf(group) {
    return S.options.fields.filter(function (f) { return f.group === group; });
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

  function choiceLabel(key, value) {
    var found = (S.fields[key].choices || []).filter(function (c) { return c.value === value; })[0];
    return found ? found.label : String(value);
  }

  // A label such as "Window length (cycles)" is a name and a unit.
  function splitUnit(label) {
    var m = /^(.*?)\s*\(([^)]+)\)$/.exec(label);
    return m ? { label: m[1], unit: m[2] } : { label: label, unit: "" };
  }

  // What a folded group holds right now, in one line.
  function groupSummary(group) {
    var s = S.settings;
    switch (group) {
      case "data":
        return s.subset;
      case "preprocessing":
        return s.window_size + "-cycle windows, stride " + s.window_stride + " · ceiling " + s.rul_max +
          " · " + choiceLabel("normalization", s.normalization).toLowerCase() +
          " · " + (s.smoothing ? "smoothed" : "not smoothed") +
          " · " + Math.round(s.validation_fraction * 100) + " % held back · " +
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
          ", " + choiceLabel("lr_schedule", s.lr_schedule).toLowerCase() +
          (s.early_stopping ? " · stops after " + s.early_stopping_patience + " passes without improvement" : "");
      case "repeats":
        return plural(s.seeds, "seed") + " from " + s.seed + (s.fixed_split ? " · same split for every seed" : "") + " · " + s.device;
      default:
        return "";
    }
  }

  function sectionOpen(group) {
    return S.sections[group] === undefined ? group === "data" : !!S.sections[group];
  }

  function drawDesign() {
    var o = S.options;
    $("#pg-panel").innerHTML =
      '<div class="pg-design"><div class="pg-form">' + presetsHtml() +
      '<div class="pg-sections-bar"><h2 class="pg-block-title">Every setting</h2>' +
      '<button type="button" class="button button--quiet button--small" data-action="expand">Expand all</button>' +
      '<button type="button" class="button button--quiet button--small" data-action="collapse">Collapse all</button>' +
      '<button type="button" class="button button--quiet button--small" data-action="reset-all">Reset all</button></div>' +
      o.groups.map(sectionHtml).join("") +
      '</div><aside class="pg-summary" id="pg-summary"><div class="loading">Building the network…</div></aside></div>';
    refreshPreview();
  }

  function presetsHtml() {
    var o = S.options, subset = S.settings.subset;
    var variants = o.ablations.map(function (a) {
      var paperRow = o.paper[subset][a.key];
      return '<button type="button" class="pg-variant' + (a.settings && matches(a.settings) ? " is-active" : "") + '"' +
        (a.settings ? ' data-ablation="' + a.key + '"' : " disabled") + ' title="' + esc(a.reason || "") + '">' +
        "<strong>" + esc(a.label) + "</strong>" +
        (paperRow ? "<span>paper RMSE " + fixed(paperRow.rmse, 2) + " on " + esc(subset) + "</span>" : "") +
        (a.settings ? "" : "<em>not buildable here</em>") + "</button>";
    }).join("");
    var procedures = o.procedures.map(function (p) {
      return '<button type="button" class="pg-variant' + (matches(p.settings) ? " is-active" : "") + '" data-procedure="' + p.key + '">' +
        "<strong>" + esc(p.label) + "</strong><span>" + esc(p.text) + "</span></button>";
    }).join("");
    return '<section class="pg-block" id="pg-presets">' +
      '<h2 class="pg-block-title">Which model · the variants of Table 11</h2>' +
      '<div class="pg-variants">' + variants + "</div>" +
      '<h2 class="pg-block-title">How it is trained · the choice that moved what the LSTM was worth ' +
      '<span class="pg-ref">(' + refLink("F-16") + ")</span></h2>" +
      '<div class="pg-variants pg-variants--two">' + procedures + "</div></section>";
  }

  function sectionHtml(g, index) {
    var open = sectionOpen(g.key);
    return '<section class="pg-section" data-group="' + g.key + '">' +
      '<button type="button" class="pg-section-head" data-section="' + g.key + '" aria-expanded="' + open +
      '" aria-controls="pg-section-' + g.key + '">' +
      '<span class="pg-step-n">' + (index + 1) + "</span>" +
      '<span class="pg-section-title">' + esc(g.label) + "</span>" +
      '<span class="pg-section-summary">' + esc(groupSummary(g.key)) + "</span>" +
      '<span class="pg-section-changed"' + (changedIn(g.key) ? "" : " hidden") + ">" + changedIn(g.key) + " changed</span>" +
      CHEVRON + "</button>" +
      '<div class="pg-section-body" id="pg-section-' + g.key + '"' + (open ? "" : " hidden") + ">" +
      fieldsOf(g.key).map(fieldRow).join("") + "</div></section>";
  }

  function fieldRow(f) {
    var value = S.settings[f.key];
    var id = "pg-" + f.key;
    var data = 'data-field="' + f.key + '"';
    var parts = splitUnit(f.label);
    var stacked = false, control;
    if (f.kind === "bool") {
      control = switchControl(id, !!value, data);
    } else if (f.kind === "choice" && SEGMENTED[f.key]) {
      control = segmented(id, f.choices, value, data, id + "-label");
      id = null;
    } else if (f.kind === "choice") {
      stacked = true;
      control = optionList(id, f.choices, value, data, function (c) {
        return sourceTag(c.source) + " " + esc(c.source.text);
      }, id + "-label");
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
      key: f.key, id: id, label: parts.label, stacked: stacked, changed: isChanged(f), hidden: !visible(f.key),
      hint: sourceTag(f.source) + " " + esc(f.source.text),
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

  function resetField(key) {
    S.settings[key] = copy(S.fields[key].default);
    saveSettings();
    drawDesign();
  }

  function toggleSection(group) {
    S.sections[group] = !sectionOpen(group);
    remember(SECTIONS, S.sections);
    paintSection(group);
  }

  function setAllSections(open) {
    S.options.groups.forEach(function (g) { S.sections[g.key] = open; paintSection(g.key); });
    remember(SECTIONS, S.sections);
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
    var count = changedIn(group);
    badge.hidden = !count;
    badge.textContent = count + " changed";
  }

  // A control changed. The setting is read from it, then only what depends
  // on it is redrawn: its row, the rows it can show or hide, the preset
  // cards, the group heads and the summary column. The control itself is
  // left alone, so a number being typed keeps its focus.
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
    S.options.groups.forEach(function (g) { paintSection(g.key); });
    refreshPreview();
  }

  function paintRow(key) {
    var f = S.fields[key];
    var box = $('[data-row="' + key + '"]');
    if (!box) { return; }
    box.classList.toggle("is-changed", isChanged(f));
    box.querySelector(".pg-reset-slot").outerHTML = resetHtml(f);
    Array.prototype.forEach.call(box.querySelectorAll(".opt"), function (option) {
      option.classList.toggle("is-active", option.querySelector("input").checked);
    });
  }

  function paintPresets() {
    var box = $("#pg-presets");
    if (box) { box.outerHTML = presetsHtml(); }
  }

  function refreshPreview() {
    clearTimeout(S.previewTimer);
    S.previewTimer = setTimeout(function () {
      var box = $("#pg-summary");
      if (!box) { return; }
      post("/api/playground/preview", S.settings).then(function (preview) {
        S.preview = preview;
        drawSummary(box, preview, null);
      }).catch(function (error) {
        S.preview = null;
        drawSummary(box, null, error.message);
      });
    }, 220);
  }

  function paperTable(paperRow, measured) {
    if (!paperRow) { return ""; }
    return '<table class="pg-mini"><thead><tr><th></th>' + MEASURES.map(function (m) { return '<th class="num">' + m[1] + "</th>"; }).join("") +
      "</tr></thead><tbody><tr><td>" + esc(paperRow.label) + "</td>" + MEASURES.map(function (m) {
        return '<td class="num">' + (paperRow[m[0]] === null || paperRow[m[0]] === undefined ? "—" : fixed(paperRow[m[0]], m[2])) + "</td>";
      }).join("") + "</tr>" + (measured ? measured : "") + "</tbody></table>" +
      '<div class="pg-faint">' + esc(paperRow.source) + (paperRow.score === null ? " prints RMSE only." : ".") + "</div>";
  }

  function caveatList(caveats) {
    if (!caveats || !caveats.length) { return ""; }
    return '<ul class="pg-caveats">' + caveats.map(function (c) {
      // The thing is said first; its code follows as a pointer to where it is argued.
      return '<li class="pg-caveat pg-caveat--' + c.level + '">' + esc(c.text) +
        (c.ref ? ' <span class="pg-ref">(' + refLink(c.ref) + ")</span>" : "") + "</li>";
    }).join("") + "</ul>";
  }

  function jobLabel(settings) {
    return settings.subset + " · " + (settings.temporal === "last_cycle" ? "w/o LSTM" : temporalName(settings.temporal) +
      (settings.temporal === "lstm" || settings.temporal === "gru" ? " " + settings.temporal_hidden : "")) +
      " · " + (settings.graph || "physics") + " graph" + " · " +
      (settings.validation_fraction > 0 ? "best pass" : "final pass") + " · " + plural(settings.seeds, "seed");
  }

  // layer_summary() lists the temporal stage, then Table 4's "Input" row. Both
  // describe the same (B, 14, 1) tensor: Table 4's Input row is the stage's
  // output, with no parameters of its own. Shown as one row, as Table 4 has.
  function table4Rows(layers) {
    if (layers.length > 1 && layers[0].layer === "Temporal" && layers[1].layer === "Input") {
      var stage = layers[0];
      return { stage: stage.type, rows: [{ layer: "Input", stage: stage.type, out: stage.out, parameters: stage.parameters }].concat(layers.slice(2)) };
    }
    return { stage: null, rows: layers };
  }

  function drawSummary(box, preview, error) {
    if (error) {
      box.innerHTML = '<div class="pg-card pg-card--error"><strong>These settings cannot be run</strong><p>' +
        esc(error) + "</p></div>";
      return;
    }
    var d = preview.dataset, t5 = d.table5, counted = d.counted;
    var canRun = S.torch !== false && !preview.network_error && counted;
    var twin = S.settings.temporal !== "last_cycle";

    var run = '<div class="pg-card pg-card--run">' +
      '<div class="pg-actions"><button type="button" class="button button--primary" data-action="run"' + (canRun ? "" : " disabled") +
      ">Train " + plural(S.settings.seeds, "seed") + "</button>" +
      (twin ? '<button type="button" class="button" data-action="twin"' + (canRun ? "" : " disabled") +
        ' title="The same settings twice, once with the temporal stage removed, so the two can be paired seed by seed">Train it and its w/o LSTM twin</button>' : "") +
      "</div>" +
      (S.torch === false ? '<p class="pg-faint">PyTorch is not installed, so nothing can be trained here.</p>' : "") +
      caveatList(preview.caveats) +
      '<details class="pg-command pg-details"><summary>The command this runs</summary><code>' + esc(preview.command) + "</code></details></div>";

    var compare = '<div class="pg-card"><h3>A result may be set beside</h3>' +
      (preview.paper_row ? paperTable(preview.paper_row) : "<p>" + esc(preview.no_row_reason) + "</p>") + "</div>";

    var network;
    if (preview.network_error) {
      network = '<div class="pg-card pg-card--error"><h3>The network</h3><p>' + esc(preview.network_error) + "</p></div>";
    } else {
      var layers = table4Rows(preview.layers);
      network = '<div class="pg-card"><h3>The network these settings build</h3><div class="pg-big">' +
        preview.parameters.toLocaleString("en") + " <span>learned numbers</span></div>" +
        '<details class="pg-details"><summary>Layer by layer</summary>' +
        '<table class="pg-mini"><thead><tr><th>Layer</th><th>Out</th><th class="num">Learned</th></tr></thead><tbody>' +
        layers.rows.map(function (l) {
          return "<tr><td>" + esc(l.layer) + (l.stage ? ' <span class="pg-faint">from ' + esc(l.stage) + "</span>" : "") +
            "</td><td><code>" + esc(l.out) + '</code></td><td class="num">' + Number(l.parameters).toLocaleString("en") + "</td></tr>";
        }).join("") + "</tbody></table>" +
        '<div class="pg-faint">B is the number of windows in a batch. ' +
        (layers.stage ? "The Input row is where Table 4 starts: one number per sensor, which the temporal stage, " + esc(layers.stage) +
          ", makes from each sensor's 30-cycle window and hands to the first graph layer. Table 4 gives that stage no row of its own, so its learned numbers are counted here. " : "") +
        "Table 4 prints no parameter counts; every count here is read from the model.</div></details></div>";
    }

    var data = '<div class="pg-card"><h3>The data</h3><p>' + esc(S.settings.subset) + ": " + t5.train_engines +
      " training and " + t5.test_engines + " test engines, " + plural(t5.conditions, "operating condition") + ", " +
      plural(t5.fault_modes, "fault mode") + ' <span class="pg-faint">(Table 5)</span>.</p>' +
      (counted ? "<p>Counted from the files: the shortest test history is " + counted.shortest_test_history +
        " cycles, so <strong>" + counted.left_out + "</strong> of " + counted.test_engines +
        " test engines are shorter than a " + S.settings.window_size + "-cycle window and are left out " +
        '<span class="pg-faint">(' + refLink("A-23") + ")</span>.</p>"
        : '<p class="pg-faint">The NASA files are not in data/raw/, so nothing can be counted or trained.</p>') + "</div>";

    box.innerHTML = run + compare + network + data;
  }

  function submit(list) {
    var chain = Promise.resolve();
    list.forEach(function (settings) {
      chain = chain.then(function () {
        return post("/api/playground/train", { settings: settings, label: jobLabel(settings) });
      });
    });
    chain.then(function () {
      T.toast(list.length > 1 ? "Two runs queued. They train one after the other." : "Run queued.");
      clearTimeout(window.__thesisPlaygroundTimer);
      pollJobs();
    }).catch(function (error) { T.toast(error.message, 8000); });
  }

  // --------------------------------------------------------------- results

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
      if (S.selected) { drawDetail(); return; }
      var f = S.filter;
      var shown = runs.filter(function (r) {
        return (f.subset === "all" || r.subset === f.subset) && (!f.results || !r.exploratory) && (!f.weights || r.has_weights);
      });
      panel.innerHTML =
        '<div class="pg-toolbar">' +
        segmented("pg-f-subset", [{ value: "all", label: "All" }].concat(SUBSETS.map(function (s) { return { value: s, label: s }; })),
          f.subset, 'data-filter="subset"') +
        "<label>" + switchControl("pg-f-results", f.results, 'data-filter="results"') + "Results only: five seeds or more</label>" +
        "<label>" + switchControl("pg-f-weights", f.weights, 'data-filter="weights"') + "Kept their weights</label>" +
        '<span class="pg-faint">' + shown.length + " of " + plural(runs.length, "run") + " · click one to open it, tick two to compare them</span></div>" +
        '<div id="pg-compare"></div>' +
        (shown.length ? runsTable(shown) : '<div class="pg-empty">No stored run matches. Design one in the first tab.</div>');
      drawComparison();
    }).catch(function (error) {
      panel.innerHTML = '<div class="error-box"><strong>The runs could not be read</strong>' + esc(error.message) + "</div>";
    });
  }

  function openRun(name) {
    S.selected = name; S.detail = null;
    drawDetail();
  }

  function runsTable(runs) {
    return '<div class="pg-runs"><table class="data-table"><thead><tr><th></th><th>Finished (UTC)</th><th>Data</th>' +
      "<th>Temporal</th><th>Graph</th><th>Weights</th><th>Seeds</th><th class=\"num\">RMSE</th>" +
      "<th class=\"num\">Paper</th><th class=\"num\">Score</th><th class=\"num\">R²</th><th>Commit</th></tr></thead><tbody>" +
      runs.map(function (r) {
        var m = r.mean || {}, sd = r.std || {};
        return '<tr data-run="' + esc(r.name) + '"><td><input type="checkbox" data-compare="' + esc(r.name) + '"' +
          (S.compare.indexOf(r.name) >= 0 ? " checked" : "") + ' aria-label="Compare this run"></td>' +
          "<td>" + when(r.finished) + "</td><td>" + esc(r.subset) + "</td><td>" + esc(temporalName(r.temporal)) +
          (r.temporal_hidden ? " " + r.temporal_hidden : "") + "</td><td>" + esc(r.graph || "") + "</td><td>" +
          (r.reported_weights === "final_epoch" ? "final pass" : "best pass") + "</td><td>" +
          '<span class="chip' + (r.exploratory ? " chip--warn" : "") + '">' + r.seeds + "</span></td>" +
          '<td class="num">' + fixed(m.rmse, 3) + (sd.rmse !== null && sd.rmse !== undefined ? ' <span class="sd">± ' + fixed(sd.rmse, 3) + "</span>" : "") + "</td>" +
          '<td class="num" title="' + esc(r.paper_row ? r.paper_row.label + ", " + r.paper_row.source : r.no_row_reason) + '">' +
          (r.paper_row ? fixed(r.paper_row.rmse, 2) : "—") + "</td>" +
          '<td class="num">' + fixed(m.score, 1) + '</td><td class="num">' + fixed(m.r2, 3) + "</td>" +
          "<td><code>" + esc(r.commit) + "</code>" + (r.has_weights ? ' <span class="chip chip--good" title="This run kept its weights, curves and predictions">files</span>' : "") +
          "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function compareTicked(box) {
    var name = box.getAttribute("data-compare");
    S.compare = S.compare.filter(function (n) { return n !== name; });
    if (box.checked) { S.compare.push(name); }
    if (S.compare.length > 2) { S.compare = S.compare.slice(-2); }
    $all("input[data-compare]").forEach(function (b) { b.checked = S.compare.indexOf(b.getAttribute("data-compare")) >= 0; });
    drawComparison();
  }

  function drawComparison() {
    var box = $("#pg-compare");
    if (!box) { return; }
    if (S.compare.length !== 2) {
      box.innerHTML = S.compare.length === 1 ? '<div class="pg-hint">Tick one more run to pair the two seed by seed.</div>' : "";
      return;
    }
    box.innerHTML = '<div class="loading">Pairing the seeds…</div>';
    T.get("/api/playground/compare?first=" + encodeURIComponent(S.compare[0]) + "&second=" + encodeURIComponent(S.compare[1]))
      .then(function (c) {
        var real = c.differences.filter(function (d) { return d.setting !== "commit"; });
        var rows = MEASURES.map(function (m) {
          var e = c.measures[m[0]];
          return "<tr><td>" + m[1] + '</td><td class="num">' + fixed(c.first.mean[m[0]], m[2]) + '</td><td class="num">' +
            fixed(c.second.mean[m[0]], m[2]) + '</td><td class="num"><strong>' + (e.mean_difference > 0 ? "+" : "") +
            fixed(e.mean_difference, m[2]) + '</strong></td><td class="num">' + fixed(e.t, 2) + '</td><td class="num">' +
            (e.p === null ? "—" : e.p < 0.001 ? e.p.toExponential(1) : fixed(e.p, 3)) + "</td></tr>";
        }).join("");
        box.innerHTML = '<section class="pg-card pg-compare"><h3>First against second, seed by seed</h3>' +
          '<p class="pg-faint">First: <code>' + esc(c.first.name) + "</code><br>Second: <code>" + esc(c.second.name) + "</code></p>" +
          (real.length === 0 ? '<p class="pg-caveat pg-caveat--warn">No setting differs, so any difference comes from the code or the machine: the commits are ' +
            esc(c.first.name.split("_").slice(-2, -1)[0]) + " and " + esc(c.second.name.split("_").slice(-2, -1)[0]) + ".</p>" : "") +
          (real.length > 1 ? '<p class="pg-caveat pg-caveat--warn">' + real.length + " settings differ, so the difference below cannot be put down to any one of them. An ablation changes one thing.</p>" : "") +
          '<table class="pg-mini"><thead><tr><th>Setting that differs</th><th>First</th><th>Second</th></tr></thead><tbody>' +
          c.differences.map(function (d) {
            return "<tr><td>" + esc(d.setting) + "</td><td><code>" + esc(JSON.stringify(d.first)) + "</code></td><td><code>" +
              esc(JSON.stringify(d.second)) + "</code></td></tr>";
          }).join("") + "</tbody></table>" +
          (c.shared_seeds.length ? '<table class="pg-mini"><thead><tr><th>Measure</th><th class="num">First</th><th class="num">Second</th>' +
            '<th class="num">Second − first</th><th class="num">paired t</th><th class="num">p</th></tr></thead><tbody>' + rows + "</tbody></table>" +
            '<p class="pg-faint">Paired over seeds ' + c.shared_seeds.join(", ") + ". For RMSE, ASUE and score lower is better; for R² higher is. " +
            "A paired p answers only whether the difference could be an accident of initialisation: every seed is scored on the same test engines, so it says nothing about engines not seen " +
            "(" + refLink("D-8") + ").</p>"
            : '<p class="pg-caveat pg-caveat--warn">The two runs share no seed, so they cannot be paired.</p>') + "</section>";
      }).catch(function (error) {
        box.innerHTML = '<div class="error-box">' + esc(error.message) + "</div>";
      });
  }

  function drawDetail() {
    var panel = $("#pg-panel");
    if (!panel || !S.selected) { return; }
    panel.innerHTML = '<div class="loading">Reading the run…</div>';
    var load = S.detail && S.detail.name === S.selected ? Promise.resolve(S.detail)
      : T.get("/api/playground/run?name=" + encodeURIComponent(S.selected));
    load.then(function (d) {
      S.detail = d;
      var seeds = Object.keys(d.per_seed);
      var paperRow = d.paper_row;
      var cards = MEASURES.map(function (m) {
        var mean = (d.mean || {})[m[0]], sd = (d.std || {})[m[0]];
        var paper = paperRow ? paperRow[m[0]] : null;
        return '<div class="pg-metric"><div class="pg-metric-name">' + m[1] + '</div><div class="pg-metric-value">' + fixed(mean, m[2]) +
          (sd !== null && sd !== undefined ? '<span> ± ' + fixed(sd, m[2]) + "</span>" : "") + "</div>" +
          '<div class="pg-faint">' + (paper !== null && paper !== undefined ? "paper " + fixed(paper, m[2]) + " · gap " +
            (mean - paper > 0 ? "+" : "") + fixed(mean - paper, m[2]) : "no published figure") + "</div></div>";
      }).join("");
      var perSeedPlot = plot({
        width: 420, height: 220, xlabel: "seed", ylabel: "test RMSE (cycles)", label: "test RMSE per seed",
        series: [{ name: "one seed", points: d.results.map(function (r) { return [r.seed, r.rmse]; }), colour: "var(--fig-accent)", dots: true, line: false, radius: 5 }],
        hlines: paperRow ? [{ y: paperRow.rmse, label: paperRow.label + " " + paperRow.rmse, colour: "var(--fig-lpc)" }] : []
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
            marks: hasValidation && best ? [[best.epoch, best[key], "best pass " + best.epoch + ", whose weights are reported"]] : [] };
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
        width: 640, height: 260, ylabel: "remaining life (cycles)", label: "predictions against truth",
        xlabel: byUnit ? "test engine, by unit number" : "test engines, ordered by their true remaining life",
        series: byUnit ? [
          { name: "true remaining life", colour: "var(--fig-burn)", dots: true, radius: 2.8, width: 1.3, points: predictions.map(function (p) { return [p.unit, p.rul]; }) },
          { name: "predicted", colour: "var(--fig-accent)", dots: true, radius: 2.8, width: 1.3, points: predictions.map(function (p) { return [p.unit, p.predicted]; }) }
        ] : [
          { name: "true remaining life", colour: "var(--fig-ink)", points: predictions.map(function (p, i) { return [i + 1, p.rul]; }), width: 2 },
          { name: "predicted", colour: "var(--fig-warn)", dots: true, line: false, points: predictions.map(function (p, i) { return [i + 1, p.predicted]; }) }
        ]
      }) : "";

      panel.innerHTML = '<button type="button" class="button button--small pg-back" data-action="back">← All runs</button>' +
        '<section class="pg-detail"><div class="pg-detail-head"><div><div class="eyebrow">Stored run</div><code class="pg-name">' +
        esc(d.name) + '</code><div class="chips">' + chip(d.dirty ? "warn" : "", "commit " + d.commit) +
        chip(d.exploratory ? "warn" : "good", plural(d.seeds, "seed") + (d.exploratory ? " · exploratory" : "")) +
        chip("", d.subset) + chip("", "temporal: " + temporalName(d.temporal)) + chip("", (d.graph || "") + " graph") +
        chip("", d.reported_weights === "final_epoch" ? "final pass reported (the paper's procedure)" : "best pass restored") +
        '</div></div><div class="pg-actions"><button type="button" class="button" data-action="again">Set these settings up again</button>' +
        (d.has_weights ? '<button type="button" class="button button--primary" data-action="explain-this">Explain this run</button>' : "") + "</div></div>" +
        '<div class="pg-metrics">' + cards + "</div>" +
        (paperRow ? '<p class="pg-faint">Compared with ' + esc(paperRow.label) + " (" + esc(paperRow.source) + ") on " + esc(d.subset) + ".</p>" :
          '<p class="pg-faint">' + esc(d.no_row_reason) + "</p>") + caveatList(d.caveats) +
        '<div class="pg-charts"><figure class="pg-figure"><figcaption>Test RMSE of every seed' + (paperRow ? ", with the paper's figure dashed" : "") + "</figcaption>" + perSeedPlot + "</figure>" +
        (curves ? '<figure class="pg-figure pg-figure--wide"><figcaption>' + (hasValidation
          ? "Error on the held-back engines after each pass. A ring marks the pass whose weights were kept."
          : "Training loss after each pass. Nothing was held back, so the weights of the last pass are the ones reported.") +
          "</figcaption>" + curvePlot + "</figure>" : '<div class="pg-empty">This run was made before per-seed files were kept (2026-09-14), so it has no curves, predictions or weights.</div>') +
        (predPlot ? '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 8.</strong> Every test engine\'s true and predicted remaining life, for seed ' +
          selectField("pg-pred-seed", seeds.map(function (s) { return { value: s, label: s }; }), seedChoice, "data-pred-seed") + " " +
          segmented("pg-pred-order", [{ value: "unit", label: "by unit number, as the paper" }, { value: "rul", label: "ordered by true remaining life" }],
            byUnit ? "unit" : "rul", 'data-pred-order') +
          " A prediction can never exceed the label ceiling, because the output layer is a sigmoid scaled to it.</figcaption>" + predPlot + "</figure>" : "") +
        "</div></section>";
    }).catch(function (error) {
      panel.innerHTML = '<button type="button" class="button button--small pg-back" data-action="back">← All runs</button>' +
        '<div class="error-box">' + esc(error.message) + "</div>";
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

  function drawExplain() {
    var panel = $("#pg-panel");
    var e = S.explain;
    panel.innerHTML = '<div class="loading">Reading reports/…</div>';
    loadRuns().then(function (runs) {
      var usable = runs.filter(function (r) { return r.has_weights; });
      if (!usable.length) {
        panel.innerHTML = '<div class="pg-empty">No stored run has kept its weights yet. Runs made from 2026-09-14 on keep them; ' +
          "train one in the first tab and it will be explainable here.</div>";
        return;
      }
      if (!e.run || !usable.some(function (r) { return r.name === e.run; })) { e.run = usable[0].name; e.detail = null; }
      var load = e.detail && e.detail.name === e.run ? Promise.resolve(e.detail)
        : T.get("/api/playground/run?name=" + encodeURIComponent(e.run));
      return load.then(function (d) {
        e.detail = d;
        var seeds = Object.keys(d.per_seed).filter(function (s) { return d.per_seed[s].has_weights; });
        if (seeds.indexOf(String(e.seed)) < 0) { e.seed = seeds[0]; }
        panel.innerHTML = '<div class="pg-explain">' + explainForm(usable, seeds) +
          '<section class="pg-block"><h2 class="pg-block-title">Explanations stored for this run</h2>' +
          (d.explanations.length ? explanationsTable(d.explanations) : '<div class="pg-empty">None yet. Choose a method above and press Explain.</div>') +
          '</section><div id="pg-e-view"></div></div>';
        if (e.shown) { drawExplanation(); }
      });
    }).catch(function (error) {
      panel.innerHTML = '<div class="error-box"><strong>Could not load</strong>' + esc(error.message) + "</div>";
    });
  }

  function explainForm(usable, seeds) {
    var e = S.explain, d = e.detail;
    var runChoices = usable.map(function (r) {
      return { value: r.name, label: when(r.finished) + " · " + r.subset + " · " + temporalName(r.temporal) + " · " + (r.graph || "") +
        " · RMSE " + fixed((r.mean || {}).rmse, 2) + " · " + plural(r.seeds, "seed") };
    });
    var seedChoices = seeds.map(function (s) { return { value: s, label: s }; });
    var forEngine = e.scope === "engine";
    return '<section class="pg-block"><h2 class="pg-block-title">Explain a stored run</h2>' +
      row({ id: "pg-e-run", label: "Run", control: selectField("pg-e-run", runChoices, e.run, 'data-explain="run"'),
        hint: "Only runs that kept their weights can be explained." }) +
      row({ id: "pg-e-seed", label: "Seed whose weights are explained",
        control: seeds.length <= 6 ? segmented("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"') : selectField("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"') }) +
      row({ key: "pg-e-method", label: "Method", stacked: true,
        control: optionList("pg-e-method", [
          { value: "shap", label: "Exact Shapley values", note: sourceTag({ kind: "paper", ref: "Eq. 14–15" }) +
            " every one of the 16,384 subsets of sensors, for every window. An absent sensor is replaced by its average training window (" + refLink("A-20") + ")." },
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
        hint: "Left blank, the engine with the longest history is explained.",
        control: '<input class="field" type="number" id="pg-e-engine" min="1" placeholder="longest" value="' + esc(e.engine) + '" data-explain="engine">' }) +
      row({ id: "pg-e-every", key: "pg-e-every-row", label: "Explain every n-th window", hidden: !forEngine,
        hint: "1 explains every window of the engine's life; a larger step is quicker and coarser.",
        control: '<input class="field" type="number" id="pg-e-every" min="1" max="100" value="' + e.every + '" data-explain="every">' }) +
      '<div class="pg-row"><div class="pg-row-name"></div><div class="pg-row-control"><button type="button" class="button button--primary" data-action="explain">Explain</button></div></div>' +
      "</section>";
  }

  function explanationsTable(explanations) {
    var e = S.explain;
    return '<table class="pg-mini pg-clickable"><thead><tr><th>Finished</th><th>Method</th><th>What</th><th>Seed</th><th class="num">Windows</th><th class="num">Largest sum gap</th><th>Commit</th></tr></thead><tbody>' +
      explanations.map(function (x) {
        return '<tr data-explanation="' + esc(x.name) + '"' + (x.name === e.shown ? ' class="is-selected"' : "") + "><td>" + when(x.finished_utc) + "</td><td>" + esc(x.method) +
          "</td><td>" + (x.scope === "engine" ? "engine " + x.unit + " (" + esc(x.source) + ")" + (x.every > 1 ? ", every " + x.every : "") : "test engines") +
          "</td><td>" + x.seed + '</td><td class="num">' + x.windows + '</td><td class="num">' + Number(x.sum_gap_max).toExponential(1) +
          "</td><td><code>" + esc(x.commit) + "</code></td></tr>";
      }).join("") + "</tbody></table>";
  }

  function explainChanged(input) {
    var e = S.explain, which = input.getAttribute("data-explain");
    if (which === "run") { e.run = input.value; e.detail = null; e.shown = null; drawExplain(); return; }
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

  function submitExplanation() {
    var e = S.explain, d = e.detail;
    var request = { run: e.run, seed: Number(e.seed), method: e.method, scope: e.scope,
      engine: e.scope === "engine" && e.engine !== "" ? Number(e.engine) : null,
      every: e.scope === "engine" ? Number(e.every) || 1 : 1,
      label: (e.method === "shap" ? "SHAP" : "Occlusion") + " · " + (e.scope === "engine" ? "engine " + (e.engine || "longest") : "test engines") +
        " · seed " + e.seed + " · " + d.subset + " " + temporalName(d.temporal) };
    post("/api/playground/explain", request).then(function () {
      T.toast("Explanation queued."); clearTimeout(window.__thesisPlaygroundTimer); pollJobs();
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
      var head = '<section class="pg-detail"><div class="eyebrow">' + (exact ? "Exact Shapley values" : "Occlusion") + " · seed " + sum.seed + " · commit " + esc(sum.commit) + "</div>" +
        "<p>" + plural(sum.windows, "window") + (sum.scope === "engine" ? " of engine " + sum.unit + ", one of the " + esc(sum.source) + " engines" : ", the last of each test engine") +
        ", computed in " + duration(sum.seconds) + ". At the baseline window the model predicts " + fixed(sum.baseline_prediction, 2) + " cycles; each value is how many cycles a sensor moved the prediction away from that.</p>" +
        '<p class="pg-caveat pg-caveat--' + (exact ? "info" : "warn") + '">' + (exact
          ? "Equation 14 says a window's values add up to its prediction minus the baseline prediction. Checked here: the largest gap is " + Number(sum.sum_gap_max).toExponential(1) + " cycles."
          : "Occlusion values need not add up to anything, and here they miss the prediction's distance from the baseline by up to " + fixed(sum.sum_gap_max, 2) + " cycles. That gap is the part the sensors do together.") + "</p>" +
        caveatList([
          { level: "info", ref: "F-08", text: "These values are signed and in cycles. The paper's Table 12 cannot hold signed Shapley values satisfying its own Equation 14, so do not set the two side by side." },
          { level: "info", ref: "", text: "A large value says the model leaned on that sensor for this prediction. It is not evidence that the sensor causes the engine's wear." }
        ]);

      // symbols come from the summary already ranked by average size.
      var ranking = symbols.map(function (s) { return [s, sum.mean_abs[s]]; });
      var maxRank = ranking[0][1] || 1;
      var bars = '<div class="pg-bars">' + ranking.map(function (r) {
        return '<div class="pg-bar-row"><span>' + esc(r[0]) + '</span><div><i style="width:' + (r[1] / maxRank * 100).toFixed(1) + '%"></i></div><b>' + fixed(r[1], 2) + "</b></div>";
      }).join("") + "</div>";
      var mean = {};
      sensorOrder.forEach(function (s) { mean[s] = sum.mean_abs[s]; });

      // The paper's figures, in the order its Section 4.4 introduces them,
      // each drawn from this explanation and captioned with the figure it
      // follows; the one chart the paper does not have says so.
      var swarm = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, left.</strong> Every value of every sensor, one dot per ' +
        (sum.scope === "engine" ? "window" : "engine") + ". The sensors are ranked by the average size of their values, top to bottom; the red frame holds the four largest. " +
        "A dot right of the dashed line raised the predicted remaining life, one left of it lowered it.</figcaption>" + beeswarm(rows, symbols) + "</figure>";
      var strip = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, middle.</strong> The same ranking as a row of sensors: ' +
        "the four largest red, deeper for larger; the rest blue, from deep to light. The bars underneath are the averages the ranking is made from.</figcaption>" +
        rankedStrip(mean, symbols) + bars + "</figure>";
      var heat = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>Not in the paper.</strong> Every sensor at every ' +
        (sum.scope === "engine" ? "window" : "test engine (columns, by engine number)") +
        ". Blue raised the predicted remaining life, orange lowered it; the stronger the colour, the more cycles.</figcaption>" + heatmap(rows, sensorOrder) + "</figure>";

      var body;
      if (sum.scope === "engine") {
        var current = rows[e.index];
        var lifePlot = plot({
          width: 640, height: 250, xlabel: "cycle at the end of the window", ylabel: "remaining life, and error (cycles)",
          label: "the prediction along the engine's life",
          areas: [{ name: "predicted minus true: the paper's \"new error area\"", colour: "var(--fig-lpc)", opacity: 0.35,
            points: rows.map(function (r) { return [r.end_cycle, r.predicted - r.rul]; }) }],
          series: [
            { name: "true remaining life (capped)", colour: "var(--fig-accent)", dash: "7 4", width: 2, points: rows.map(function (r) { return [r.end_cycle, r.rul]; }) },
            { name: "predicted", colour: "var(--fig-lpt)", width: 1.8, points: rows.map(function (r) { return [r.end_cycle, r.predicted]; }) },
            { name: "the window drawn on the graph below", colour: "var(--fig-ink)", line: false, dots: true, radius: 5.5, points: [[current.end_cycle, current.predicted]] }
          ]
        });
        body = head + '<div class="pg-charts">' +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 7.</strong> The prediction along the engine\'s life against the true remaining life, ' +
          "with the difference between them shaded about zero, the way the paper shades what it calls the new error area. The paper's engine is FD001 unit 38; this is engine " +
          esc(sum.unit) + " of the " + esc(sum.source) + " engines of seed " + esc(sum.seed) + ".</figcaption>" + lifePlot + "</figure>" +
          swarm + strip +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, right.</strong> One window on the sensor graph, the four largest values red and the rest blue. <label class="pg-slider">Window ' +
          '<input type="range" min="0" max="' + (rows.length - 1) + '" value="' + e.index + '" data-window-index> <span data-window-caption>' +
          esc(windowCaption(current)) + "</span></label></figcaption>" +
          '<div class="pg-split">' + sensorGraph(current, sensorOrder) + signedBars(current, sensorOrder) + "</div>" +
          '<p class="pg-faint">The paper colours its four most influential sensors red, with intensity by contribution, and the rest blue from deep to light. It does not say whether influence is the size of the value or the value itself; here it is the size, ignoring the sign.</p></figure>' +
          (exact ? paperComparison(rows, sensorOrder, sum) : "") +
          heat + "</div></section>";
      } else {
        body = head + '<div class="pg-charts">' + swarm + strip +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, right.</strong> The average size of each sensor\'s value over the test engines, on the sensor graph: the four largest red, the rest blue.</figcaption>' + sensorGraph(mean, sensorOrder) + "</figure>" +
          heat + "</div></section>";
      }
      box.innerHTML = body;
    }).catch(function (error) {
      box.innerHTML = '<div class="error-box">' + esc(error.message) + "</div>";
    });
  }

  function heatmap(rows, sensors) {
    var cell = Math.max(2, Math.min(18, Math.floor(640 / rows.length)));
    var W = 60 + cell * rows.length, H = 14 * 16 + 10;
    var peak = 0;
    rows.forEach(function (r) { sensors.forEach(function (s) { peak = Math.max(peak, Math.abs(r[s])); }); });
    peak = peak || 1;
    var out = '<div class="pg-heat-wrap"><svg class="pg-heat" viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '">';
    sensors.forEach(function (s, j) {
      out += '<text x="54" y="' + (j * 16 + 12) + '" text-anchor="end" class="pg-tick">' + esc(s) + "</text>";
      rows.forEach(function (r, i) {
        var v = r[s];
        out += '<rect x="' + (60 + i * cell) + '" y="' + (j * 16) + '" width="' + cell + '" height="15" fill="' + (v >= 0 ? "var(--fig-accent)" : "var(--fig-warn)") +
          '" fill-opacity="' + (Math.abs(v) / peak).toFixed(3) + '"><title>' + esc(s + " · cycle " + r.end_cycle + " · engine " + r.unit + ": " + (v >= 0 ? "+" : "") + v.toFixed(3) + " cycles") + "</title></rect>";
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
      out += '<g><title>' + esc(s + ": rank " + (rank + 1) + ", " + (values[s] >= 0 ? "+" : "") + Number(values[s]).toFixed(3) + " cycles") + "</title>" +
        '<circle cx="' + p[0] * scale + '" cy="' + p[1] * scale + '" r="17" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/>' +
        '<text x="' + p[0] * scale + '" y="' + (p[1] * scale + 30) + '" text-anchor="middle" class="pg-node">' + esc(s) + "</text></g>";
    });
    return out + "</svg>";
  }

  // How many times a sensor joins the red four, going from each graph in the
  // list to the next.
  function redFourEntries(list, sensors) {
    var count = 0;
    for (var i = 1; i < list.length; i++) {
      var before = fig10Style(list[i - 1], sensors).ranked.slice(0, 4);
      count += fig10Style(list[i], sensors).ranked.slice(0, 4).filter(function (s) { return before.indexOf(s) < 0; }).length;
    }
    return count;
  }

  // One small graph with the sensors round a circle in Fig. 10's order and
  // Table 3's edges between them. unit is "" for the paper's unitless values.
  function ringGraph(values, sensors, label, unit) {
    var ring = S.options.table12.ring, n = ring.length, c = 50, R = 40;
    var at = {};
    ring.forEach(function (s, i) {
      var a = (-90 - 180 / n + i * 360 / n) * Math.PI / 180;
      at[s] = [c + R * Math.cos(a), c + R * Math.sin(a)];
    });
    var out = '<figure class="pg-ring"><svg viewBox="0 0 100 100" role="img" aria-label="' + esc(label) + '">';
    if (!values) {
      return out + '<text x="50" y="52" text-anchor="middle" class="pg-ring-missing">not explained</text></svg><figcaption>' + esc(label) + "</figcaption></figure>";
    }
    S.options.graph.edges.forEach(function (edge) {
      var a = at[edge[0]], b = at[edge[1]];
      out += '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) + '" stroke="var(--fig-edge)" stroke-width="0.45"/>';
    });
    var st = fig10Style(values, sensors);
    ring.forEach(function (s) {
      var p = at[s], k = st.style[s], v = Number(values[s]);
      out += "<g><title>" + esc(label + " · " + s + ": rank " + (k.rank + 1) + " of " + n + ", " + (unit && v >= 0 ? "+" : "") + v.toFixed(2) + (unit ? " " + unit : "")) + "</title>" +
        '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="8.4" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-ink)" stroke-width="0.35"/>' +
        '<text x="' + p[0].toFixed(1) + '" y="' + (p[1] + 1.9).toFixed(1) + '" text-anchor="middle" class="pg-ring-label">' + esc(s) + "</text></g>";
    });
    return out + "</svg><figcaption>" + esc(label) + "</figcaption></figure>";
  }

  // The paper's Fig. 10, redrawn from Table 12, above this engine at the same
  // graph numbers. Graph g is the g-th window, so with every n-th window
  // explained only graphs 1, 1 + n, 1 + 2n ... exist here.
  function paperComparison(rows, sensors, sum) {
    var t = S.options.table12, every = Number(sum.every) || 1;
    if (!t || !t.graphs) {
      return '<div class="pg-empty">The paper\'s Table 12 did not arrive with this page\'s settings, which happens when the ' +
        "documentation server is older than the page it serves. Stop the server and start scripts/docs_ui.py again.</div>";
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
      return '<div class="pg-fig10-stage"><div class="pg-fig10-head">' + names[k] + ": graphs " + graphs[0] + " to " + graphs[4] + "</div>" +
        '<div class="pg-fig10-row"><span class="pg-fig10-who">Paper, Table 12</span>' +
        paper.map(function (v, j) { return ringGraph(v, sensors, "graph " + graphs[j], ""); }).join("") + "</div>" +
        '<div class="pg-fig10-row"><span class="pg-fig10-who">Engine ' + esc(sum.unit) + ", seed " + esc(sum.seed) + "</span>" +
        ours.map(function (r, j) { return ringGraph(r, sensors, r ? "cycle " + r.end_cycle : "graph " + graphs[j], "cycles"); }).join("") + "</div></div>";
    }).join("");
    var allSteps = every === 1 && rows.length > 1 ? redFourEntries(rows, sensors) : null;
    return '<figure class="pg-figure pg-figure--wide"><figcaption>Beside the paper\'s Fig. 10. Each small graph is one window: the four sensors with the largest values red, the rest blue, ' +
      "each shade by rank, round the circle in the order the paper draws them. The top row of each stage is the paper's, redrawn from the numbers of its Table 12; " +
      "the row below is this engine at the same graph numbers, where graph 1 is the first window.</figcaption>" + blocks +
      '<p class="pg-fig10-count"><strong>How often the red four changes.</strong> Inside the three stages there are 12 steps from one graph to the next. ' +
      "Over those steps a sensor joins the red four " + plural(paperSteps, "time") + " in the paper's table" +
      (oursAll ? ", and " + plural(oursSteps, "time") + " for this engine." : "; this engine is missing some of those graphs, so it is not counted.") +
      (allSteps !== null ? " Over all " + (rows.length - 1) + " steps between this engine's neighbouring windows: " + allSteps + " times." : "") + "</p>" +
      '<p class="pg-faint">The paper\'s row is drawn from Table 12, not copied from its printed figure. The two agree on the red four in 13 of the 15 graphs; ' +
      "in graph 81 the printed figure colours T30 red in place of phi, and in graph 82 T30 and Nc in place of P30 and phi (" + refLink("F-17") + "). " +
      "Table 12 describes one engine the paper does not name, so the engine below it is a different one. " +
      "Every value in Table 12 is positive, which signed Shapley values cannot all be (" + refLink("F-08") + "), " +
      "so here a sensor that pulled the prediction down by many cycles is red as well: the ranking uses the size of the value.</p></figure>";
  }

  function signedBars(row, sensors) {
    var peak = Math.max.apply(null, sensors.map(function (s) { return Math.abs(row[s]); })) || 1;
    return '<div class="pg-signed">' + sensors.slice().sort(function (a, b) { return Math.abs(row[b]) - Math.abs(row[a]); }).map(function (s) {
      var v = row[s], width = Math.abs(v) / peak * 50;
      return '<div class="pg-signed-row"><span>' + esc(s) + '</span><div class="pg-signed-track"><i class="' + (v >= 0 ? "up" : "down") + '" style="width:' + width.toFixed(1) +
        "%;" + (v >= 0 ? "left:50%" : "right:50%") + '"></i></div><b>' + (v >= 0 ? "+" : "") + v.toFixed(2) + "</b></div>";
    }).join("") + "</div>";
  }

  window.thesisPlayground = start;
})();
