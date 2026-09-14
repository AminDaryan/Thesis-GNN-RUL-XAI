/* The playground: set up a run, follow it, read its results, explain it.
   README.md in this folder explains what each part is for and why every run
   started here goes through scripts/train.py and lands in reports/. */

(function () {
  "use strict";

  var ROUTE = "#/explore/playground";
  var STORAGE = "thesis-playground-settings";
  var PALETTE = ["var(--fig-accent)", "var(--fig-warn)", "var(--fig-lpt)", "var(--fig-lpc)",
    "var(--fig-core)", "var(--fig-cool)", "var(--fig-burn)", "var(--fig-fan)"];
  var TEMPORAL = { lstm: "LSTM", gru: "GRU", last_cycle: "removed", mean_window: "window mean",
    raw_window: "raw window" };
  var MEASURES = [["rmse", "RMSE", 3], ["score", "Score", 1], ["r2", "R²", 3], ["asue", "ASUE", 3]];

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

  function sourceTag(source) {
    if (!source) { return ""; }
    var names = { paper: "Paper", assumption: "Our choice", project: "Project rule",
      extension: "Not in the paper" };
    return '<span class="pg-src pg-src--' + source.kind + '">' + names[source.kind] +
      (source.ref ? " · " + refLink(source.ref) : "") + "</span>";
  }

  function temporalName(value) {
    return value === null || value === undefined ? "not recorded" : (TEMPORAL[value] || value);
  }

  function saveSettings() {
    try { window.localStorage.setItem(STORAGE, JSON.stringify(S.settings)); } catch (e) { /* private window */ }
  }

  function loadSettings(defaults) {
    var stored = null;
    try { stored = JSON.parse(window.localStorage.getItem(STORAGE) || "null"); } catch (e) { stored = null; }
    var settings = JSON.parse(JSON.stringify(defaults));
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
  function plot(spec) {
    var W = spec.width || 640, H = spec.height || 260;
    var left = 54, right = 14, top = 14, bottom = 40;
    var xs = [], ys = [];
    spec.series.forEach(function (s) {
      s.points.forEach(function (p) {
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
    var legend = spec.series.filter(function (s) { return s.name && !s.nolegend; }).map(function (s) {
      return '<span><i style="background:' + s.colour + '"></i>' + esc(s.name) + "</span>";
    }).join("");
    return out + (legend ? '<div class="pg-legend">' + legend + "</div>" : "");
  }

  // ------------------------------------------------------------------ page

  function start(target, tools) {
    T = tools;
    root = target;
    if (window.__thesisPlaygroundTimer) { clearTimeout(window.__thesisPlaygroundTimer); }
    S = { tab: S.tab || "design", jobs: [], seen: {}, runs: null, selected: null, detail: null,
      compare: [], comparison: null, filter: { subset: "all", results: false, weights: false },
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
    root.innerHTML =
      T.crumbs([{ label: "Home", href: "#/" }, { label: "Explore" }, { label: "Playground" }]) +
      '<div class="page-head pg-head"><div class="eyebrow">Runs the real scripts · writes to reports/</div>' +
      '<h1 class="pg-title">Playground</h1>' +
      "<p>Choose a sub-dataset, the preprocessing, the temporal stage and the graph layers' sizes, " +
      "then train. Remove a part of the model to see what it was worth, and explain what a trained " +
      "model relies on. Every run started here is <strong>scripts/train.py</strong> with the command " +
      "shown, stored in <code>reports/</code> under the commit that produced it, exactly as if it " +
      "had been typed. Beside every setting is where its value comes from: the paper, a numbered " +
      "choice of ours, or neither.</p></div>" +
      '<section class="pg-jobs" id="pg-jobs" aria-live="polite"></section>' +
      '<div class="pg-tabs" role="tablist">' +
      tabButton("design", "1 · Design a run") + tabButton("results", "2 · Results and comparisons") +
      tabButton("explain", "3 · Explanations") + "</div>" +
      '<div id="pg-panel" class="pg-panel"></div>';
    $all(".pg-tab").forEach(function (button) {
      button.addEventListener("click", function () { showTab(button.getAttribute("data-tab")); });
    });
  }

  function tabButton(key, label) {
    return '<button type="button" role="tab" class="pg-tab" data-tab="' + key + '">' + label + "</button>";
  }

  function showTab(tab) {
    S.tab = tab;
    $all(".pg-tab").forEach(function (button) {
      button.setAttribute("aria-selected", String(button.getAttribute("data-tab") === tab));
    });
    if (tab === "design") { drawDesign(); }
    else if (tab === "results") { drawResults(); }
    else { drawExplain(); }
  }

  // ---------------------------------------------------------------- jobs

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
    if (job.kind === "train") {
      if (job.status === "queued") { return "Waiting for the job before it to finish."; }
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
    return job.status === "queued" ? "Waiting for the job before it to finish." : "Preparing the data and model…";
  }

  function jobFraction(job) {
    var p = job.progress || {};
    if (job.status === "done") { return 1; }
    if (job.kind === "train" && p.seeds) {
      return Math.min(1, ((p.seed_index || 0) + (p.epoch || 0) / (p.epochs || 1)) / p.seeds);
    }
    return p.total ? p.done / p.total : 0;
  }

  function drawJobs() {
    var box = $("#pg-jobs");
    if (!box) { return; }
    // What is happening now first, then what will happen next in the order it
    // will happen, then what has finished, newest first.
    var rank = { running: 0, cancelling: 0, queued: 1 };
    var shown = S.jobs.slice().sort(function (a, b) {
      var ra = rank[a.status] === undefined ? 2 : rank[a.status];
      var rb = rank[b.status] === undefined ? 2 : rank[b.status];
      if (ra !== rb) { return ra - rb; }
      return ra === 1 ? a.created - b.created : b.created - a.created;
    }).slice(0, 6);
    if (!shown.length) {
      box.innerHTML = '<div class="pg-jobs-empty">No jobs yet. Runs and explanations you start ' +
        "appear here, one at a time, in the order you asked for them.</div>";
      return;
    }
    box.innerHTML = '<div class="pg-jobs-title">Jobs</div>' + shown.map(function (job) {
      var p = job.progress || {};
      var curve = (p.curve || []).map(function (c) {
        return [c.epoch, c.validation_rmse !== null && c.validation_rmse !== undefined ? c.validation_rmse : c.train_loss];
      });
      var spark = job.kind === "train" && job.status === "running" && curve.length > 1
        ? '<div class="pg-spark">' + plot({ width: 220, height: 70, series: [{ points: curve, colour: "var(--fig-accent)" }] })
          .replace(/<text[^>]*>[^<]*<\/text>/g, "") + "</div>" : "";
      var open = "";
      if (job.status === "done" && p.result) {
        open = job.kind === "train"
          ? '<button class="button pg-open-run" data-run="' + esc(p.result.directory) + '">Open result</button>'
          : '<button class="button pg-open-explanation" data-run="' + esc(p.result.run) + '" data-name="' +
            esc(p.result.directory) + '">Open explanation</button>';
      }
      var stop = job.status === "running" || job.status === "queued"
        ? '<button class="button pg-cancel" data-id="' + esc(job.id) + '">' +
          (job.status === "queued" ? "Remove" : "Stop") + "</button>" : "";
      return '<div class="pg-job pg-job--' + job.status + '">' +
        '<div class="pg-job-main"><div class="pg-job-head"><span class="pg-status">' + esc(job.status) +
        "</span><strong>" + esc(job.label) + "</strong>" +
        (job.elapsed ? '<span class="pg-faint">' + duration(job.elapsed) + "</span>" : "") + "</div>" +
        '<div class="pg-job-line">' + esc(job.status === "failed" ? "The command failed. Its last lines are below." :
          job.status === "cancelled" ? "Stopped. Nothing was written to reports/." : jobLine(job)) + "</div>" +
        '<div class="pg-bar"><span style="width:' + Math.round(jobFraction(job) * 100) + '%"></span></div>' +
        (job.log ? '<pre class="pg-log">' + esc(job.log) + "</pre>" : "") +
        '<details class="pg-command"><summary>Command</summary><code>' + esc(job.command) + "</code></details>" +
        "</div>" + spark + '<div class="pg-job-actions">' + open + stop + "</div></div>";
    }).join("");
    $all(".pg-cancel").forEach(function (b) {
      b.addEventListener("click", function () {
        post("/api/playground/cancel", { id: b.getAttribute("data-id") }).then(pollJobs)
          .catch(function (e) { T.toast(e.message); });
      });
    });
    $all(".pg-open-run").forEach(function (b) {
      b.addEventListener("click", function () {
        S.selected = b.getAttribute("data-run"); S.detail = null; S.runs = null; showTab("results");
      });
    });
    $all(".pg-open-explanation").forEach(function (b) {
      b.addEventListener("click", function () {
        S.explain.run = b.getAttribute("data-run"); S.explain.detail = null;
        S.explain.shown = b.getAttribute("data-name"); S.explain.values = null; showTab("explain");
      });
    });
  }

  // -------------------------------------------------------------- design

  function visible(key) {
    var s = S.settings;
    if (key === "temporal_hidden" || key === "temporal_out_features" || key === "shared_across_sensors") {
      return s.temporal === "lstm" || s.temporal === "gru";
    }
    if (key === "smoothing_alpha") { return !!s.smoothing; }
    if (key === "early_stopping_patience") { return !!s.early_stopping; }
    return true;
  }

  function same(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

  function drawDesign() {
    var o = S.options;
    var panel = $("#pg-panel");
    var subset = S.settings.subset;
    var ablations = o.ablations.map(function (a) {
      var row = o.paper[subset][a.key];
      var active = a.settings && Object.keys(a.settings).every(function (k) { return S.settings[k] === a.settings[k]; });
      return '<button type="button" class="pg-variant' + (active ? " is-active" : "") + '"' +
        (a.settings ? ' data-ablation="' + a.key + '"' : " disabled") + ' title="' + esc(a.reason || "") + '">' +
        "<strong>" + esc(a.label) + "</strong><span>" + (row ? "paper RMSE " + fixed(row.rmse, 2) + " on " + subset : "") +
        "</span>" + (a.settings ? "" : "<em>not buildable here</em>") + "</button>";
    }).join("");
    var procedures = o.procedures.map(function (p) {
      var active = Object.keys(p.settings).every(function (k) { return S.settings[k] === p.settings[k]; });
      return '<button type="button" class="pg-variant' + (active ? " is-active" : "") + '" data-procedure="' + p.key + '">' +
        "<strong>" + esc(p.label) + "</strong><span>" + esc(p.text) + "</span></button>";
    }).join("");

    var groups = o.groups.map(function (g, index) {
      var fields = o.fields.filter(function (f) { return f.group === g.key; });
      return '<fieldset class="pg-group"><legend><span class="pg-step">' + (index + 1) + "</span>" + esc(g.label) +
        "</legend>" + fields.map(fieldHtml).join("") + "</fieldset>";
    }).join("");

    panel.innerHTML =
      '<div class="pg-design"><div class="pg-form">' +
      '<div class="pg-group pg-presets"><div class="pg-preset-title">Which model — the variants of Table 11</div>' +
      '<div class="pg-variants">' + ablations + "</div>" +
      '<div class="pg-preset-title">How it is trained — the choice F-16 showed matters</div>' +
      '<div class="pg-variants pg-variants--two">' + procedures + "</div></div>" +
      groups + '</div><aside class="pg-summary" id="pg-summary"><div class="loading">Building the network…</div></aside></div>';

    $all("[data-ablation]").forEach(function (b) {
      b.addEventListener("click", function () {
        var a = o.ablations.filter(function (x) { return x.key === b.getAttribute("data-ablation"); })[0];
        Object.keys(a.settings).forEach(function (k) { S.settings[k] = a.settings[k]; });
        changed(true);
      });
    });
    $all("[data-procedure]").forEach(function (b) {
      b.addEventListener("click", function () {
        var p = o.procedures.filter(function (x) { return x.key === b.getAttribute("data-procedure"); })[0];
        Object.keys(p.settings).forEach(function (k) { S.settings[k] = p.settings[k]; });
        changed(true);
      });
    });
    $all("[data-field]").forEach(bindField);
    $all(".pg-reset").forEach(function (b) {
      b.addEventListener("click", function () {
        var key = b.getAttribute("data-reset");
        S.settings[key] = JSON.parse(JSON.stringify(S.fields[key].default));
        changed(true);
      });
    });
    refreshPreview();
  }

  function fieldHtml(f) {
    var value = S.settings[f.key];
    var id = "pg-" + f.key;
    var control;
    if (f.kind === "bool") {
      control = '<label class="pg-switch"><input type="checkbox" id="' + id + '" data-field="' + f.key + '"' +
        (value ? " checked" : "") + "><span>" + (value ? "Yes" : "No") + "</span></label>";
    } else if (f.kind === "choice" && f.choices.length <= 5 && f.key !== "device") {
      control = '<div class="pg-choices" role="radiogroup" aria-labelledby="' + id + '-label">' +
        f.choices.map(function (c) {
          return '<label class="pg-choice' + (c.value === value ? " is-active" : "") + '">' +
            '<input type="radio" name="' + id + '" value="' + esc(c.value) + '" data-field="' + f.key + '"' +
            (c.value === value ? " checked" : "") + ">" +
            '<span class="pg-choice-label">' + esc(c.label) + "</span>" +
            '<span class="pg-choice-source">' + sourceTag(c.source) + " " + esc(choiceNote(f.key, c)) + "</span></label>";
        }).join("") + "</div>";
    } else if (f.kind === "choice") {
      control = '<select id="' + id + '" data-field="' + f.key + '">' + f.choices.map(function (c) {
        return '<option value="' + esc(c.value) + '"' + (c.value === value ? " selected" : "") + ">" + esc(c.label) + "</option>";
      }).join("") + "</select>";
    } else if (f.kind === "widths") {
      control = '<div class="pg-widths">' + value.map(function (w, i) {
        return '<label><span>GCN-' + (i + 1) + '</span><input type="number" min="' + f.min + '" max="' + f.max +
          '" step="1" value="' + w + '" data-field="' + f.key + '" data-index="' + i + '"></label>';
      }).join("") + "</div>";
    } else {
      control = '<input type="number" id="' + id + '" data-field="' + f.key + '" min="' + f.min + '" max="' + f.max +
        '" step="' + (f.step || 1) + '" value="' + value + '">';
    }
    var isChanged = !same(value, f.default);
    var singleSource = f.kind === "choice" && f.choices.length <= 5 && f.key !== "device" ? "" :
      '<div class="pg-field-source">' + sourceTag(f.source) + " " + esc(f.source.text) + "</div>";
    return '<div class="pg-field' + (isChanged ? " is-changed" : "") + '"' + (visible(f.key) ? "" : " hidden") + ">" +
      '<div class="pg-field-head"><span class="pg-label" id="' + id + '-label">' + esc(f.label) + "</span>" +
      (isChanged ? '<button type="button" class="pg-reset" data-reset="' + f.key + '" title="Back to ' +
        esc(JSON.stringify(f.default)) + '">reset</button>' : "") + "</div>" +
      control + singleSource + "</div>";
  }

  function choiceNote(key, choice) {
    var d = key === "subset" ? S.options.dataset[choice.value] : null;
    if (!d) { return choice.source.text; }
    return d.train_engines + " training and " + d.test_engines + " test engines, " + d.conditions +
      " operating condition" + (d.conditions > 1 ? "s" : "") + ", " + d.fault_modes + " fault mode" +
      (d.fault_modes > 1 ? "s" : "");
  }

  function bindField(input) {
    var key = input.getAttribute("data-field");
    var f = S.fields[key];
    input.addEventListener("change", function () {
      if (f.kind === "bool") { S.settings[key] = input.checked; }
      else if (f.kind === "choice") { S.settings[key] = input.value; }
      else if (f.kind === "widths") {
        var copy = S.settings[key].slice();
        copy[+input.getAttribute("data-index")] = input.value === "" ? input.value : Number(input.value);
        S.settings[key] = copy;
      } else { S.settings[key] = input.value === "" ? input.value : Number(input.value); }
      changed(f.kind === "bool" || f.kind === "choice");
    });
  }

  // A choice can show or hide other fields, so the form is redrawn for those;
  // a typed number only needs the summary refreshed.
  function changed(redraw) {
    saveSettings();
    // A typed number keeps its focus, so only the summary is redrawn for it.
    if (redraw) { drawDesign(); } else { refreshPreview(); }
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

  function paperTable(row, measured) {
    if (!row) { return ""; }
    return '<table class="pg-mini"><thead><tr><th></th>' + MEASURES.map(function (m) { return "<th>" + m[1] + "</th>"; }).join("") +
      "</tr></thead><tbody><tr><td>" + esc(row.label) + "</td>" + MEASURES.map(function (m) {
        return '<td class="num">' + (row[m[0]] === null || row[m[0]] === undefined ? "—" : fixed(row[m[0]], m[2])) + "</td>";
      }).join("") + "</tr>" + (measured ? measured : "") + "</tbody></table>" +
      '<div class="pg-faint">' + esc(row.source) + (row.score === null ? " prints RMSE only." : ".") + "</div>";
  }

  function caveatList(caveats) {
    if (!caveats || !caveats.length) { return ""; }
    return '<ul class="pg-caveats">' + caveats.map(function (c) {
      return '<li class="pg-caveat pg-caveat--' + c.level + '">' + (c.ref ? '<span class="pg-ref">' + refLink(c.ref) + "</span> " : "") +
        esc(c.text) + "</li>";
    }).join("") + "</ul>";
  }

  function jobLabel(settings) {
    return settings.subset + " · " + (settings.temporal === "last_cycle" ? "w/o LSTM" : temporalName(settings.temporal) +
      (settings.temporal === "lstm" || settings.temporal === "gru" ? " " + settings.temporal_hidden : "")) +
      " · " + (settings.graph || "physics") + " graph" + " · " +
      (settings.validation_fraction > 0 ? "best pass" : "final pass") + " · " + settings.seeds + " seed" +
      (settings.seeds === 1 ? "" : "s");
  }

  function drawSummary(box, preview, error) {
    if (error) {
      box.innerHTML = '<div class="pg-card pg-card--error"><strong>These settings cannot be run</strong><p>' +
        esc(error) + "</p></div>";
      return;
    }
    var d = preview.dataset, t5 = d.table5, counted = d.counted;
    var data = '<div class="pg-card"><h3>The data</h3><p>' + esc(S.settings.subset) + ": " + t5.train_engines +
      " training and " + t5.test_engines + " test engines, " + t5.conditions + " operating condition" +
      (t5.conditions > 1 ? "s" : "") + ", " + t5.fault_modes + " fault mode" + (t5.fault_modes > 1 ? "s" : "") +
      ' <span class="pg-faint">(Table 5)</span>.</p>' +
      (counted ? "<p>Counted from the files: the shortest test history is " + counted.shortest_test_history +
        " cycles, so <strong>" + counted.left_out + "</strong> of " + counted.test_engines +
        " test engines are shorter than a " + S.settings.window_size + "-cycle window and are left out " +
        '<span class="pg-faint">(' + refLink("A-23") + ")</span>.</p>" : "<p class=\"pg-faint\">The NASA files are not in data/raw/, so nothing can be counted or trained.</p>") + "</div>";

    var compare = '<div class="pg-card"><h3>What a result may be set beside</h3>' +
      (preview.paper_row ? paperTable(preview.paper_row) : "<p>" + esc(preview.no_row_reason) + "</p>") + "</div>";

    var network;
    if (preview.network_error) {
      network = '<div class="pg-card pg-card--error"><h3>The network</h3><p>' + esc(preview.network_error) + "</p></div>";
    } else {
      network = '<div class="pg-card"><h3>The network, built now</h3><div class="pg-big">' +
        preview.parameters.toLocaleString("en") + ' <span>learned numbers</span></div>' +
        '<table class="pg-mini"><thead><tr><th>Layer</th><th>Out</th><th class="num">Learned</th></tr></thead><tbody>' +
        preview.layers.map(function (l) {
          return "<tr><td>" + esc(l.layer) + (l.layer === "Temporal" ? ' <span class="pg-faint">' + esc(l.type) + "</span>" : "") +
            "</td><td><code>" + esc(l.out) + '</code></td><td class="num">' + Number(l.parameters).toLocaleString("en") + "</td></tr>";
        }).join("") + "</tbody></table>" +
        '<div class="pg-faint">B is the number of windows in a batch. Table 4 prints no parameter counts; these are counted from the model.</div></div>';
    }

    var canRun = S.torch !== false && !preview.network_error && counted;
    var twin = S.settings.temporal !== "last_cycle";
    var actions = '<div class="pg-card pg-run"><h3>Run it</h3>' + caveatList(preview.caveats) +
      '<details class="pg-command" open><summary>The command this runs</summary><code>' + esc(preview.command) + "</code></details>" +
      '<div class="pg-actions"><button type="button" class="button pg-primary" id="pg-run"' + (canRun ? "" : " disabled") +
      ">Train " + S.settings.seeds + " seed" + (S.settings.seeds === 1 ? "" : "s") + "</button>" +
      (twin ? '<button type="button" class="button" id="pg-run-twin"' + (canRun ? "" : " disabled") +
        ' title="The same settings twice, once with the temporal stage removed, so the two can be paired seed by seed">Train it and its w/o LSTM twin</button>' : "") +
      "</div>" + (S.torch === false ? '<p class="pg-faint">PyTorch is not installed, so nothing can be trained here.</p>' : "") + "</div>";

    box.innerHTML = compare + actions + network + data;
    var run = $("#pg-run");
    if (run) { run.addEventListener("click", function () { submit([S.settings]); }); }
    var both = $("#pg-run-twin");
    if (both) {
      both.addEventListener("click", function () {
        var ablated = JSON.parse(JSON.stringify(S.settings));
        ablated.temporal = "last_cycle";
        submit([S.settings, ablated]);
      });
    }
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

  // ------------------------------------------------------------- results

  function loadRuns() {
    if (S.runs) { return Promise.resolve(S.runs); }
    return T.get("/api/playground/runs").then(function (data) { S.runs = data.runs; return S.runs; });
  }

  function drawResults() {
    var panel = $("#pg-panel");
    panel.innerHTML = '<div class="loading">Reading reports/…</div>';
    loadRuns().then(function (runs) {
      var f = S.filter;
      var shown = runs.filter(function (r) {
        return (f.subset === "all" || r.subset === f.subset) && (!f.results || !r.exploratory) && (!f.weights || r.has_weights);
      });
      panel.innerHTML =
        '<div class="pg-filters"><label>Sub-dataset <select id="pg-f-subset"><option value="all">all</option>' +
        ["FD001", "FD002", "FD003", "FD004"].map(function (s) {
          return "<option" + (f.subset === s ? " selected" : "") + ">" + s + "</option>";
        }).join("") + "</select></label>" +
        '<label><input type="checkbox" id="pg-f-results"' + (f.results ? " checked" : "") + "> only results (five seeds or more)</label>" +
        '<label><input type="checkbox" id="pg-f-weights"' + (f.weights ? " checked" : "") + "> only runs that kept their weights</label>" +
        '<span class="pg-faint">' + shown.length + " of " + runs.length + " runs · tick two to compare them</span></div>" +
        (shown.length ? runsTable(shown) : '<div class="pg-empty">No stored run matches. Design one in the first tab.</div>') +
        '<div id="pg-compare"></div><div id="pg-detail"></div>';
      $("#pg-f-subset").addEventListener("change", function (e) { f.subset = e.target.value; drawResults(); });
      $("#pg-f-results").addEventListener("change", function (e) { f.results = e.target.checked; drawResults(); });
      $("#pg-f-weights").addEventListener("change", function (e) { f.weights = e.target.checked; drawResults(); });
      $all("tr[data-run]").forEach(function (tr) {
        tr.addEventListener("click", function (event) {
          if (event.target.tagName === "INPUT") { return; }
          S.selected = tr.getAttribute("data-run"); S.detail = null; markSelected(); drawDetail();
        });
      });
      $all("input[data-compare]").forEach(function (box) {
        box.addEventListener("change", function () {
          var name = box.getAttribute("data-compare");
          S.compare = S.compare.filter(function (n) { return n !== name; });
          if (box.checked) { S.compare.push(name); }
          if (S.compare.length > 2) { S.compare = S.compare.slice(-2); }
          $all("input[data-compare]").forEach(function (b) { b.checked = S.compare.indexOf(b.getAttribute("data-compare")) >= 0; });
          drawComparison();
        });
      });
      markSelected();
      drawComparison();
      if (S.selected) { drawDetail(); }
    }).catch(function (error) {
      panel.innerHTML = '<div class="error-box"><strong>The runs could not be read</strong>' + esc(error.message) + "</div>";
    });
  }

  function markSelected() {
    $all("tr[data-run]").forEach(function (tr) {
      tr.classList.toggle("is-selected", tr.getAttribute("data-run") === S.selected);
    });
  }

  function runsTable(runs) {
    return '<div class="table-wrap pg-runs"><table class="data-table"><thead><tr><th></th><th>Finished (UTC)</th><th>Data</th>' +
      "<th>Temporal</th><th>Graph</th><th>Weights reported</th><th>Seeds</th><th class=\"num\">RMSE</th>" +
      "<th class=\"num\">Paper</th><th class=\"num\">Score</th><th class=\"num\">R²</th><th>Commit</th></tr></thead><tbody>" +
      runs.map(function (r) {
        var m = r.mean || {}, sd = r.std || {};
        return '<tr data-run="' + esc(r.name) + '"><td><input type="checkbox" data-compare="' + esc(r.name) + '"' +
          (S.compare.indexOf(r.name) >= 0 ? " checked" : "") + ' aria-label="compare"></td>' +
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
    var box = $("#pg-detail");
    if (!box || !S.selected) { return; }
    box.innerHTML = '<div class="loading">Reading the run…</div>';
    var load = S.detail && S.detail.name === S.selected ? Promise.resolve(S.detail)
      : T.get("/api/playground/run?name=" + encodeURIComponent(S.selected));
    load.then(function (d) {
      S.detail = d;
      var seeds = Object.keys(d.per_seed);
      var row = d.paper_row;
      var cards = MEASURES.map(function (m) {
        var mean = (d.mean || {})[m[0]], sd = (d.std || {})[m[0]];
        var paper = row ? row[m[0]] : null;
        return '<div class="pg-metric"><div class="pg-metric-name">' + m[1] + '</div><div class="pg-metric-value">' + fixed(mean, m[2]) +
          (sd !== null && sd !== undefined ? '<span> ± ' + fixed(sd, m[2]) + "</span>" : "") + "</div>" +
          '<div class="pg-faint">' + (paper !== null && paper !== undefined ? "paper " + fixed(paper, m[2]) + " · gap " +
            (mean - paper > 0 ? "+" : "") + fixed(mean - paper, m[2]) : "no published figure") + "</div></div>";
      }).join("");
      var perSeedPlot = plot({
        width: 420, height: 220, xlabel: "seed", ylabel: "test RMSE (cycles)", label: "test RMSE per seed",
        series: [{ name: "one seed", points: d.results.map(function (r) { return [r.seed, r.rmse]; }), colour: "var(--fig-accent)", dots: true, line: false, radius: 5 }],
        hlines: row ? [{ y: row.rmse, label: row.label + " " + row.rmse, colour: "var(--fig-lpc)" }] : []
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
      var predictions = seeds.length ? d.per_seed[seedChoice].predictions.slice().sort(function (a, b) { return a.rul - b.rul; }) : [];
      var predPlot = predictions.length ? plot({
        width: 640, height: 260, xlabel: "test engines, ordered by their true remaining life", ylabel: "remaining life (cycles)",
        label: "predictions against truth",
        series: [
          { name: "true remaining life", colour: "var(--fig-ink)", points: predictions.map(function (p, i) { return [i + 1, p.rul]; }), width: 2 },
          { name: "predicted", colour: "var(--fig-warn)", dots: true, line: false, points: predictions.map(function (p, i) { return [i + 1, p.predicted]; }) }
        ]
      }) : "";

      box.innerHTML = '<section class="pg-detail"><div class="pg-detail-head"><div><div class="eyebrow">Stored run</div><code class="pg-name">' +
        esc(d.name) + '</code><div class="chips">' + T.chip(d.dirty ? "warn" : "", "commit " + d.commit) +
        T.chip(d.exploratory ? "warn" : "good", d.seeds + " seed" + (d.seeds === 1 ? "" : "s") + (d.exploratory ? " · exploratory" : "")) +
        T.chip("", d.subset) + T.chip("", "temporal: " + temporalName(d.temporal)) + T.chip("", (d.graph || "") + " graph") +
        T.chip("", d.reported_weights === "final_epoch" ? "final pass reported (the paper's procedure)" : "best pass restored") +
        '</div></div><div class="pg-actions"><button class="button" id="pg-again">Set these settings up again</button>' +
        (d.has_weights ? '<button class="button pg-primary" id="pg-explain-this">Explain this run</button>' : "") + "</div></div>" +
        '<div class="pg-metrics">' + cards + "</div>" +
        (row ? '<p class="pg-faint">Compared with ' + esc(row.label) + " (" + esc(row.source) + ") on " + esc(d.subset) + ".</p>" :
          '<p class="pg-faint">' + esc(d.no_row_reason) + "</p>") + caveatList(d.caveats) +
        '<div class="pg-charts"><figure class="pg-figure"><figcaption>Test RMSE of every seed' + (row ? ", with the paper's figure dashed" : "") + "</figcaption>" + perSeedPlot + "</figure>" +
        (curves ? '<figure class="pg-figure pg-figure--wide"><figcaption>' + (hasValidation
          ? "Error on the held-back engines after each pass. A ring marks the pass whose weights were kept."
          : "Training loss after each pass. Nothing was held back, so the weights of the last pass are the ones reported.") +
          "</figcaption>" + curvePlot + "</figure>" : '<div class="pg-empty">This run was made before per-seed files were kept (2026-09-14), so it has no curves, predictions or weights.</div>') +
        (predPlot ? '<figure class="pg-figure pg-figure--wide"><figcaption>Every test engine, as in the paper\'s Fig. 8. Seed <select id="pg-pred-seed">' +
          seeds.map(function (s) { return "<option" + (s === seedChoice ? " selected" : "") + ">" + s + "</option>"; }).join("") +
          "</select>. A prediction can never exceed the label ceiling, because the output layer is a sigmoid scaled to it.</figcaption>" + predPlot + "</figure>" : "") +
        "</div></section>";

      var again = $("#pg-again");
      if (again) { again.addEventListener("click", function () { S.settings = settingsFromRun(d); saveSettings(); showTab("design"); }); }
      var explainThis = $("#pg-explain-this");
      if (explainThis) {
        explainThis.addEventListener("click", function () {
          S.explain.run = d.name; S.explain.detail = null; S.explain.shown = null; showTab("explain");
        });
      }
      var seedSelect = $("#pg-pred-seed");
      if (seedSelect) { seedSelect.addEventListener("change", function () { S.predSeed = seedSelect.value; drawDetail(); }); }
    }).catch(function (error) {
      box.innerHTML = '<div class="error-box">' + esc(error.message) + "</div>";
    });
  }

  function settingsFromRun(d) {
    var s = JSON.parse(JSON.stringify(S.options.defaults));
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

  // ------------------------------------------------------------- explain

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
        panel.innerHTML = '<div class="pg-explain">' +
          '<div class="pg-group pg-explain-form"><div class="pg-field"><span class="pg-label">Run</span><select id="pg-e-run">' +
          usable.map(function (r) {
            return '<option value="' + esc(r.name) + '"' + (r.name === e.run ? " selected" : "") + ">" + esc(when(r.finished) + " · " + r.subset + " · " +
              temporalName(r.temporal) + " · " + (r.graph || "") + " · RMSE " + fixed((r.mean || {}).rmse, 2) + " · " + r.seeds + " seeds") + "</option>";
          }).join("") + "</select></div>" +
          '<div class="pg-field"><span class="pg-label">Seed whose weights are explained</span><select id="pg-e-seed">' +
          seeds.map(function (s) { return "<option" + (s === String(e.seed) ? " selected" : "") + ">" + s + "</option>"; }).join("") + "</select></div>" +
          '<div class="pg-field"><span class="pg-label">Method</span><div class="pg-choices">' +
          radio("pg-e-method", "shap", e.method, "Exact Shapley values", sourceTag({ kind: "paper", ref: "Eq. 14–15" }) +
            " every one of the 16,384 subsets of sensors, for every window. An absent sensor is replaced by its average training window (" + refLink("A-20") + ").") +
          radio("pg-e-method", "occlusion", e.method, "Occlusion, one sensor at a time", sourceTag({ kind: "extension" }) +
            " 15 evaluations per window. The Shapley term in which the sensor arrives last; the two agree only when sensors do not interact.") +
          "</div></div>" +
          '<div class="pg-field"><span class="pg-label">What is explained</span><div class="pg-choices">' +
          radio("pg-e-scope", "engine", e.scope, "One engine, at every cycle", sourceTag({ kind: "paper", ref: "Section 4.4" }) +
            " the dynamic explanation of Table 12 and Fig. 10. The test set keeps only each engine's last window, so the engine comes from the held-back engines, or the training engines when none were held back.") +
          radio("pg-e-scope", "test", e.scope, "The last window of every test engine", sourceTag({ kind: "extension" }) +
            " one explanation per test engine, for a ranking across engines.") + "</div></div>" +
          (e.scope === "engine" ? '<div class="pg-inline"><div class="pg-field"><span class="pg-label">Engine number</span>' +
            '<input type="number" id="pg-e-engine" min="1" placeholder="longest" value="' + esc(e.engine) + '"></div>' +
            '<div class="pg-field"><span class="pg-label">Explain every n-th window</span><input type="number" id="pg-e-every" min="1" max="100" value="' + e.every + '"></div></div>' : "") +
          '<div class="pg-actions"><button class="button pg-primary" id="pg-e-go">Explain</button></div></div>' +
          '<div class="pg-group"><div class="pg-preset-title">Explanations stored for this run</div>' +
          (d.explanations.length ? '<table class="pg-mini pg-clickable"><thead><tr><th>Finished</th><th>Method</th><th>What</th><th>Seed</th><th class="num">Windows</th><th class="num">Largest sum gap</th><th>Commit</th></tr></thead><tbody>' +
            d.explanations.map(function (x) {
              return '<tr data-explanation="' + esc(x.name) + '"' + (x.name === e.shown ? ' class="is-selected"' : "") + "><td>" + when(x.finished_utc) + "</td><td>" + esc(x.method) +
                "</td><td>" + (x.scope === "engine" ? "engine " + x.unit + " (" + esc(x.source) + ")" + (x.every > 1 ? ", every " + x.every : "") : "test engines") +
                "</td><td>" + x.seed + '</td><td class="num">' + x.windows + '</td><td class="num">' + Number(x.sum_gap_max).toExponential(1) +
                "</td><td><code>" + esc(x.commit) + "</code></td></tr>";
            }).join("") + "</tbody></table>" : '<div class="pg-empty">None yet.</div>') + "</div>" +
          '<div id="pg-e-view"></div></div>';

        $("#pg-e-run").addEventListener("change", function (ev) { e.run = ev.target.value; e.detail = null; e.shown = null; drawExplain(); });
        $("#pg-e-seed").addEventListener("change", function (ev) { e.seed = ev.target.value; });
        $all("input[name=pg-e-method]").forEach(function (r) { r.addEventListener("change", function () { e.method = r.value; drawExplain(); }); });
        $all("input[name=pg-e-scope]").forEach(function (r) { r.addEventListener("change", function () { e.scope = r.value; drawExplain(); }); });
        var engine = $("#pg-e-engine"), every = $("#pg-e-every");
        if (engine) { engine.addEventListener("change", function () { e.engine = engine.value; }); }
        if (every) { every.addEventListener("change", function () { e.every = Number(every.value) || 1; }); }
        $("#pg-e-go").addEventListener("click", function () {
          var request = { run: e.run, seed: Number(e.seed), method: e.method, scope: e.scope,
            engine: e.scope === "engine" && e.engine !== "" ? Number(e.engine) : null,
            every: e.scope === "engine" ? Number(e.every) || 1 : 1,
            label: (e.method === "shap" ? "SHAP" : "Occlusion") + " · " + (e.scope === "engine" ? "engine " + (e.engine || "longest") : "test engines") +
              " · seed " + e.seed + " · " + d.subset + " " + temporalName(d.temporal) };
          post("/api/playground/explain", request).then(function () {
            T.toast("Explanation queued."); clearTimeout(window.__thesisPlaygroundTimer); pollJobs();
          }).catch(function (error) { T.toast(error.message, 8000); });
        });
        $all("tr[data-explanation]").forEach(function (tr) {
          tr.addEventListener("click", function () { e.shown = tr.getAttribute("data-explanation"); e.values = null; e.index = 0; drawExplain(); });
        });
        if (e.shown) { drawExplanation(); }
      });
    }).catch(function (error) {
      panel.innerHTML = '<div class="error-box"><strong>Could not load</strong>' + esc(error.message) + "</div>";
    });
  }

  function radio(name, value, current, label, note) {
    return '<label class="pg-choice' + (value === current ? " is-active" : "") + '"><input type="radio" name="' + name + '" value="' + value + '"' +
      (value === current ? " checked" : "") + '><span class="pg-choice-label">' + esc(label) + '</span><span class="pg-choice-source">' + note + "</span></label>";
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
      var symbols = Object.keys(sum.mean_abs);
      var sensorOrder = Object.keys(rows[0]).filter(function (k) { return symbols.indexOf(k) >= 0; });
      if (e.index >= rows.length) { e.index = 0; }
      var exact = sum.method === "shap";
      var head = '<section class="pg-detail"><div class="eyebrow">' + (exact ? "Exact Shapley values" : "Occlusion") + " · seed " + sum.seed + " · commit " + esc(sum.commit) + "</div>" +
        "<p>" + sum.windows + " window" + (sum.windows === 1 ? "" : "s") + (sum.scope === "engine" ? " of engine " + sum.unit + ", one of the " + esc(sum.source) + " engines" : ", the last of each test engine") +
        ", computed in " + duration(sum.seconds) + ". At the baseline window the model predicts " + fixed(sum.baseline_prediction, 2) + " cycles; each value is how many cycles a sensor moved the prediction away from that.</p>" +
        '<p class="pg-caveat pg-caveat--' + (exact ? "info" : "warn") + '">' + (exact
          ? "Equation 14 says a window's values add up to its prediction minus the baseline prediction. Checked here: the largest gap is " + Number(sum.sum_gap_max).toExponential(1) + " cycles."
          : "Occlusion values need not add up to anything, and here they miss the prediction's distance from the baseline by up to " + fixed(sum.sum_gap_max, 2) + " cycles. That gap is the part the sensors do together.") + "</p>" +
        caveatList([
          { level: "info", ref: "F-08", text: "These values are signed and in cycles. The paper's Table 12 cannot hold signed Shapley values satisfying its own Equation 14, so do not set the two side by side." },
          { level: "info", ref: "", text: "A large value says the model leaned on that sensor for this prediction. It is not evidence that the sensor causes the engine's wear." }
        ]);

      var ranking = symbols.map(function (s) { return [s, sum.mean_abs[s]]; });
      var maxRank = ranking[0][1] || 1;
      var bars = '<div class="pg-bars">' + ranking.map(function (r) {
        return '<div class="pg-bar-row"><span>' + esc(r[0]) + '</span><div><i style="width:' + (r[1] / maxRank * 100).toFixed(1) + '%"></i></div><b>' + fixed(r[1], 2) + "</b></div>";
      }).join("") + "</div>";

      var body;
      if (sum.scope === "engine") {
        var current = rows[e.index];
        var lifePlot = plot({
          width: 640, height: 220, xlabel: "cycle at the end of the window", ylabel: "remaining life (cycles)",
          series: [
            { name: "true remaining life (capped)", colour: "var(--fig-ink)", points: rows.map(function (r) { return [r.end_cycle, r.rul]; }) },
            { name: "predicted", colour: "var(--fig-warn)", points: rows.map(function (r) { return [r.end_cycle, r.predicted]; }), dots: rows.length < 80 },
            { name: "window shown below", colour: "var(--fig-accent)", line: false, dots: true, radius: 6, points: [[current.end_cycle, current.predicted]] }
          ]
        });
        body = head + '<div class="pg-charts"><figure class="pg-figure pg-figure--wide"><figcaption>The prediction along the engine\'s life</figcaption>' + lifePlot + "</figure>" +
          '<figure class="pg-figure pg-figure--wide"><figcaption>Every sensor at every window. Blue raised the predicted remaining life, orange lowered it; the stronger the colour, the more cycles.</figcaption>' +
          heatmap(rows, sensorOrder) + "</figure>" +
          '<figure class="pg-figure pg-figure--wide"><figcaption>One window, drawn the way the paper\'s Fig. 9 draws it. <label class="pg-slider">Window ' +
          '<input type="range" id="pg-e-index" min="0" max="' + (rows.length - 1) + '" value="' + e.index + '"> <span>cycle ' + current.end_cycle + ", true life " + fixed(current.rul, 0) +
          ", predicted " + fixed(current.predicted, 1) + "</span></label></figcaption>" +
          '<div class="pg-split">' + sensorGraph(current, sensorOrder) + signedBars(current, sensorOrder) + "</div>" +
          '<p class="pg-faint">The paper colours its four most influential sensors red, with intensity by contribution, and the rest blue from deep to light. It does not say whether influence is the size of the value or the value itself; here it is the size, ignoring the sign.</p></figure>' +
          '<figure class="pg-figure"><figcaption>Average size over the windows shown</figcaption>' + bars + "</figure></div></section>";
      } else {
        var mean = {};
        sensorOrder.forEach(function (s) { mean[s] = sum.mean_abs[s]; });
        body = head + '<div class="pg-charts"><figure class="pg-figure"><figcaption>Average size of each sensor\'s value over the test engines</figcaption>' + bars + "</figure>" +
          '<figure class="pg-figure"><figcaption>The same ranking on the sensor graph, coloured as in Fig. 9</figcaption>' + sensorGraph(mean, sensorOrder) + "</figure>" +
          '<figure class="pg-figure pg-figure--wide"><figcaption>Every test engine (columns, by engine number) and sensor. Blue raised the prediction, orange lowered it.</figcaption>' + heatmap(rows, sensorOrder) + "</figure></div></section>";
      }
      box.innerHTML = body;
      var slider = $("#pg-e-index");
      if (slider) {
        slider.addEventListener("input", function () {
          e.index = Number(slider.value);
          var y = window.scrollY;
          drawExplanation();
          window.scrollTo(0, y);
        });
      }
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

  // Fig. 9's scheme: the four sensors with the largest values in red, deeper
  // for larger; the other ten in blue, from deep to light as they matter less.
  function sensorGraph(values, sensors) {
    var g = S.options.graph;
    var ranked = sensors.slice().sort(function (a, b) { return Math.abs(values[b]) - Math.abs(values[a]); });
    var top = Math.abs(values[ranked[0]]) || 1;
    var scale = 0.62;
    var out = '<svg class="pg-graph" viewBox="0 0 ' + g.width * scale + " " + (g.height - 30) * scale + '">';
    g.edges.forEach(function (edge) {
      var a = g.positions[edge[0]], b = g.positions[edge[1]];
      out += '<line x1="' + a[0] * scale + '" y1="' + a[1] * scale + '" x2="' + b[0] * scale + '" y2="' + b[1] * scale + '" stroke="var(--fig-edge)" stroke-width="1.2"/>';
    });
    ranked.forEach(function (s, rank) {
      var p = g.positions[s];
      var colour = rank < 4 ? "var(--fig-burn)" : "var(--fig-fan)";
      var opacity = rank < 4 ? 0.45 + 0.55 * Math.abs(values[s]) / top : 1 - (rank - 4) / 10 * 0.8;
      out += '<g><title>' + esc(s + ": rank " + (rank + 1) + ", " + (values[s] >= 0 ? "+" : "") + Number(values[s]).toFixed(3) + " cycles") + "</title>" +
        '<circle cx="' + p[0] * scale + '" cy="' + p[1] * scale + '" r="17" fill="' + colour + '" fill-opacity="' + opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/>' +
        '<text x="' + p[0] * scale + '" y="' + (p[1] * scale + 30) + '" text-anchor="middle" class="pg-node">' + esc(s) + "</text></g>";
    });
    return out + "</svg>";
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
