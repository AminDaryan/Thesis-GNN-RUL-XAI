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
  var STEP = "thesis-playground-step";
  var PALETTE = ["var(--fig-accent)", "var(--fig-warn)", "var(--fig-lpt)", "var(--fig-lpc)",
    "var(--fig-core)", "var(--fig-cool)", "var(--fig-burn)", "var(--fig-fan)"];
  // The one vocabulary for temporal stages, job states and methods: the same
  // words in every label, chip, filter, toast and table cell.
  var TEMPORAL = { lstm: "LSTM", gru: "GRU", last_cycle: "w/o LSTM", mean_window: "window average",
    window_summary: "window summary", raw_window: "whole window" };
  var STATUS = { queued: "Queued", running: "Running", cancelling: "Stopping", done: "Done",
    failed: "Failed", cancelled: "Stopped", interrupted: "Interrupted" };
  var METHODS = { shap: "Exact Shapley values", occlusion: "Occlusion" };
  // What the Explain tab's options do, in plain words, above the line of each one's i.
  var EXPLAIN_ABOUT = {
    shap: "Compares the prediction from the recorded readings with the prediction when every sensor is swapped " +
      "for its training average, and divides that change among the 14 sensors, in cycles. A sensor's share " +
      "is how far swapping it back to its recorded readings moves the prediction, averaged over every mix " +
      "of the others recorded or swapped, each size of mix counting equally.",
    occlusion: "A quick first look. Each sensor in turn has its readings swapped for their training average while " +
      "the other 13 stay as recorded; its value is the prediction with every sensor recorded minus the " +
      "prediction with it swapped. Far quicker than trying every mix of sensors, but the values need not " +
      "add up the way the exact ones do.",
    engine: "Follows one engine through its recorded life, which ends in failure, explaining its windows one " +
      "after another. It shows how much each sensor raises or lowers the predicted remaining life, and how " +
      "that changes as the engine wears out. With no engines held back, the network learned from that " +
      "engine.",
    test: "Explains the predictions the run is scored on: one per test engine, at the last cycle it recorded, " +
      "for engines the network never learned from. An engine too short to fill a window has no prediction " +
      "and is skipped. With one window per engine, it cannot show how a sensor's effect changes as an " +
      "engine wears out."
  };
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
  // Choices short enough to sit side by side in one control, and a choice of
  // five that shares one source and sits beside the two rates it connects,
  // in a menu. Every other choice is a grid of options, each with the source
  // of its value.
  var SEGMENTED = { subset: true, device: true };
  var MENU = { lr_schedule: true };
  var SUBSETS = ["FD001", "FD002", "FD003", "FD004"];
  // The value that takes a part out of the pipeline. Each is the setting of a
  // variant of Table 11: "w/o LSTM" and "w/o Physics prior".
  var REMOVED = { temporal: "last_cycle", graph: "complete" };
  // What the pipeline calls the sensor graph the graph layers run on.
  var PRIOR = { physics: "Physics prior", correlation: "Correlation graph" };
  var GCN_FIXED = "The graph layers cannot be taken out: a network without them is not built here";
  var CROSS = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3 L9 9 M9 3 L3 9" ' +
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
  var BACK = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M7.5 2.5 L4 6 L7.5 9.5" fill="none" ' +
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var NEXT = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4.5 2.5 L8 6 L4.5 9.5" fill="none" ' +
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var PLUS = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2.5 V9.5 M2.5 6 H9.5" ' +
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
  var INFO = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" ' +
    'stroke-width="1.3"/><path d="M8 7.2 V11.2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>' +
    '<circle cx="8" cy="4.9" r=".95" fill="currentColor"/></svg>';
  // One picture for each box of the pipeline and for what goes in and comes
  // out of it: the files (a store of records), preprocessing (sliders: the
  // readings are adjusted), the temporal stage (a clock: it reads the cycles
  // in time), the graph layers (sensors joined in a graph) and the remaining
  // life (an hourglass). The same picture stands on a box, on its
  // flowchart's title and beside the words that lead into or out of it.
  function icon(paths) {
    return '<svg class="pg-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" ' +
      'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' + paths + "</svg>";
  }
  var ICONS = {
    data: icon('<ellipse cx="12" cy="6" rx="7" ry="2.5"/><path d="M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6"/>' +
      '<path d="M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5"/>'),
    preprocessing: icon('<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/>' +
      '<circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/>'),
    temporal: icon('<circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 2"/>'),
    graph: icon('<circle cx="6" cy="7" r="2.2"/><circle cx="18" cy="7" r="2.2"/><circle cx="12" cy="17.5" r="2.2"/>' +
      '<path d="M8.2 7h7.6M7.1 9l3.8 6.6M16.9 9l-3.8 6.6"/>'),
    output: icon('<path d="M7 3.5h10M7 20.5h10M8 3.5v2.2c0 2.3 4 3.8 4 6.3s-4 4-4 6.3v2.2M16 3.5v2.2c0 2.3-4 3.8-4 6.3s4 ' +
      '4 4 6.3v2.2"/>')
  };

  // A small drawing of what one step of a box's flowchart does, so the step
  // is recognised and remembered by its shape as well as its name. The server
  // names the drawing for each step (its "picture"); the steps of the form
  // that decide a step carry the same drawing.
  function picture(paths) {
    return '<svg class="pg-picture" viewBox="0 0 48 32" aria-hidden="true" fill="none" stroke="currentColor" ' +
      'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' + paths + "</svg>";
  }
  var DOT = function (x, y, r) { return '<circle cx="' + x + '" cy="' + y + '" r="' + (r || 2.5) + '"/>'; };
  var FILLED = function (x, y, r) { return '<circle cx="' + x + '" cy="' + y + '" r="' + (r || 2.5) + '" fill="currentColor"/>'; };
  var PICTURES = {
    // a text file of rows
    file: picture('<path d="M15 3h13l6 6v20H15z M28 3v6h6 M19 14h11 M19 19h11 M19 24h7"/>'),
    // remaining life falling to zero at failure
    life: picture('<path d="M6 4v24h36"/><path d="M9 7 L39 26"/>' + FILLED(39, 26, 2)),
    // 14 of the 21 sensors kept
    select: picture('<path d="M7 9l3 3 5-6M7 21l3 3 5-6M21 9h20M21 21h20"/>'),
    // life flat at the ceiling, then falling
    cap: picture('<path d="M6 4v24h36"/><path d="M9 9h14l16 17"/><path d="M9 5 L23 9" stroke-dasharray="2 3"/>'),
    // jagged readings made smooth
    smooth: picture('<path d="M5 22l4-9 4 11 4-10 4 8 4-9 4 10 4-7 4 8 4-6" opacity=".45"/>' +
      '<path d="M5 18c8-4 14 3 21-1s12-3 17-4"/>'),
    // one engine of five set aside
    hold: picture('<rect x="4" y="10" width="6" height="14" rx="1.5"/><rect x="12" y="10" width="6" height="14" rx="1.5"/>' +
      '<rect x="20" y="10" width="6" height="14" rx="1.5"/><rect x="28" y="10" width="6" height="14" rx="1.5"/>' +
      '<rect x="38" y="10" width="6" height="14" rx="1.5" stroke-dasharray="2 2"/>'),
    // rows grouped by operating condition
    conditions: picture(DOT(10, 10) + DOT(14, 14) + DOT(9, 17) + DOT(30, 8) + DOT(35, 11) + DOT(32, 14) +
      DOT(22, 24) + DOT(27, 27) + DOT(19, 28)),
    // readings of different sizes brought to one range
    rescale: picture('<path d="M7 28V5M13 28V18M19 28V24"/><path d="M24 16h5m-2-2 2 2-2 2"/><path d="M34 28V14M39 28V14M44 28V14"/>'),
    // a window of cycles cut from an engine's history
    window: picture('<path d="M4 22h40"/><path d="M8 20v4M12 20v4M16 20v4M20 20v4M24 20v4M28 20v4M32 20v4M36 20v4M40 20v4"/>' +
      '<rect x="14" y="8" width="16" height="20" rx="2"/>'),
    // training, held-back and test windows
    sets: picture('<rect x="4" y="8" width="20" height="16" rx="2"/><rect x="27" y="8" width="8" height="16" rx="2"/>' +
      '<rect x="38" y="8" width="6" height="16" rx="2"/>'),
    // one window split into one sequence per sensor
    split: picture('<path d="M5 16h10M15 16 27 6h16M15 16h28M15 16 27 26h16"/>'),
    // a recurrent network: a cell whose output comes back into it
    recurrent: picture('<rect x="15" y="10" width="18" height="14" rx="3"/><path d="M5 17h10M33 17h10"/>' +
      '<path d="M38 17v-12H24v5"/><path d="M22 8l2 2 2-2"/>'),
    // only the last of the sequence kept
    last: picture('<rect x="5" y="12" width="6" height="8" rx="1"/><rect x="13" y="12" width="6" height="8" rx="1"/>' +
      '<rect x="21" y="12" width="6" height="8" rx="1"/><rect x="29" y="12" width="6" height="8" rx="1"/>' +
      '<rect x="37" y="12" width="6" height="8" rx="1" fill="currentColor"/>'),
    // many numbers made into few, each a weighted sum
    project: picture('<path d="M6 6 30 16M6 12 30 16M6 20 30 16M6 26 30 16M30 16h12"/>' + FILLED(30, 16, 2.5)),
    // readings replaced by their average
    average: picture('<path d="M5 22l5-8 5 6 5-10 5 9 5-5 5 7 5-9" opacity=".45"/><path d="M5 16h38" stroke-dasharray="3 2"/>'),
    // the change over the window, a fitted line's slope
    slope: picture(DOT(8, 24, 1.8) + DOT(15, 20, 1.8) + DOT(22, 21, 1.8) + DOT(29, 14, 1.8) + DOT(36, 12, 1.8) +
      '<path d="M5 26 41 9"/>'),
    // three numbers stacked per sensor
    stack: picture('<rect x="16" y="4" width="16" height="7" rx="1.5"/><rect x="16" y="13" width="16" height="7" rx="1.5"/>' +
      '<rect x="16" y="22" width="16" height="7" rx="1.5"/>'),
    // sensors joined in a graph
    graph: picture(DOT(9, 9) + DOT(24, 6) + DOT(39, 11) + DOT(14, 25) + DOT(33, 26) +
      '<path d="M11 10l11-3M26 7l11 3M10 11l3 12M16 25h15M38 13l-4 11M23 8l-8 15"/>'),
    // the graph as a table of rescaled numbers
    table: picture('<rect x="10" y="3" width="28" height="26" rx="2"/><path d="M10 12h28M10 20h28M19 3v26M28 3v26"/>'),
    // a graph layer: each sensor gathers from its neighbours
    gcn: picture(FILLED(24, 16, 3.5) + DOT(7, 7) + DOT(7, 25) + DOT(41, 7) + DOT(41, 25) +
      '<path d="M9.5 8.5 20 14M9.5 23.5 20 18M38.5 8.5 28 14M38.5 23.5 28 18"/>'),
    // each sensor's numbers laid end to end
    flatten: picture('<rect x="4" y="4" width="10" height="10" rx="1"/><rect x="4" y="18" width="10" height="10" rx="1"/>' +
      '<path d="M17 16h5m-2-2 2 2-2 2"/><rect x="25" y="12" width="20" height="8" rx="1"/>'),
    // every number joined to every output
    dense: picture(DOT(8, 7) + DOT(8, 16) + DOT(8, 25) + DOT(40, 10) + DOT(40, 22) +
      '<path d="M10.5 7.5 37.5 10M10.5 16 37.5 10.5M10.5 24.5 37.5 11M10.5 7.5 37.5 21.5M10.5 16 37.5 22M10.5 24.5 37.5 22.5"/>'),
    // the sigmoid: any number squeezed between 0 and 1
    sigmoid: picture('<path d="M5 26h38M5 6h38" stroke-dasharray="2 3" opacity=".6"/><path d="M5 25c14 0 14-18 19-18s5 0 19 0"/>'),
    // passes over the training data, one after another
    passes: picture('<path d="M34 10a11 11 0 1 0 2 11"/><path d="M36 3v7h-7"/>'),
    // the learning rate, falling in steps
    rate: picture('<path d="M6 4v24h36"/><path d="M9 8h10v8h10v6h11"/>'),
    // a die: the seed decides the random draws
    seeds: picture('<rect x="12" y="4" width="24" height="24" rx="4"/>' + FILLED(18, 10, 2) + FILLED(24, 16, 2) +
      FILLED(30, 22, 2)),
    // a chip: which one does the arithmetic
    device: picture('<rect x="15" y="7" width="18" height="18" rx="2"/><rect x="20" y="12" width="8" height="8" rx="1"/>' +
      '<path d="M19 3v4M24 3v4M29 3v4M19 25v4M24 25v4M29 25v4M11 11h4M11 16h4M11 21h4M33 11h4M33 16h4M33 21h4"/>'),
    // the fraction times the ceiling: a scale in cycles
    scale: picture('<rect x="4" y="12" width="40" height="10" rx="1.5"/><path d="M10 12v4M16 12v6M22 12v4M28 12v6M34 12v4M40 12v6"/>')
  };

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

  // The same moment without its year or zone, "09-14 23:39", for a column
  // whose heading says the zone; the full form is kept for the pointer.
  function whenShort(stamp) {
    var m = /^(\d{4})-?(\d\d)-?(\d\d)T(\d\d):?(\d\d)/.exec(stamp || "");
    return m ? m[2] + "-" + m[3] + " " + m[4] + ":" + m[5] : (stamp || "—");
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

  // The words behind an i: first what the thing is for, one line of plain
  // text per paragraph, then, under a thin line, where its value comes from
  // (HTML from sourceNote), and last a "Read more" that opens the part of the
  // documentation that explains it in a new tab. Any part may be missing. A
  // line after the first that opens "Option words: ..." explains one option,
  // and its option's words are set in the heavier weight, so the eye finds them.
  function infoCard(about, source, more) {
    var lines = about ? String(about).split("\n").filter(Boolean).map(function (line, i) {
      var m = i > 0 && /^([^:]{1,48}): (.+)$/.exec(line);
      return '<span class="pg-info-line">' + (m ? "<strong>" + esc(m[1]) + ":</strong> " + esc(m[2]) : esc(line)) +
        "</span>";
    }).join("") : "";
    return lines + (source ? '<span class="pg-info-source">' + source + "</span>" : "") +
      (more ? '<span class="pg-info-more"><a href="' + esc(more) + '" target="_blank" rel="noopener">Read more' +
        NEXT + "</a></span>" : "");
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

  // How wide a chart is drawn: the width it is shown at, so one unit of the
  // drawing is one pixel on the screen and its words keep their size at every
  // width, rather than shrinking on a phone and swelling on a wide screen.
  // The charts stand in one column no wider than --measure-wide (1080), less
  // the figure's padding and border (26).
  function chartWidth() {
    var panel = $("#pg-panel");
    var shown = panel ? Math.min(panel.clientWidth, 1080) : 666;
    return Math.max(240, Math.round(shown - 26));
  }

  // A chart's height follows its width within limits: never so short that
  // its lines crowd on a phone, never taller than a laptop screen can show.
  function chartHeight(width) {
    return Math.round(Math.max(220, Math.min(360, width * 0.4)));
  }

  // The charts are drawn again when the width they would be drawn at
  // changes, since each was drawn for the width it had. The watcher is kept
  // here rather than in S, which every visit to the playground replaces, and
  // it stops itself once its panel has left the page, so it never redraws,
  // or scrolls, a page the reader has moved on to.
  var chartWatch = null, chartTimer = null;
  function watchCharts() {
    var panel = $("#pg-panel");
    if (chartWatch) { chartWatch.disconnect(); chartWatch = null; }
    clearTimeout(chartTimer);
    if (!panel || !window.ResizeObserver) { return; }
    var watch = chartWatch = new ResizeObserver(function () {
      clearTimeout(chartTimer);
      if (!panel.isConnected) { watch.disconnect(); return; }
      chartTimer = setTimeout(function () {
        if (!panel.isConnected || !onPage() || chartWidth() === S.chartsDrawnAt) { return; }
        var y = window.scrollY;
        if (S.tab === "results" && S.selected && S.detail) { drawDetail(); }
        else if (S.tab === "explain" && S.explain.shown && S.explain.values) { drawExplanation(); }
        else { return; }
        window.scrollTo({ top: y, behavior: "instant" });
      }, 200);
    });
    watch.observe(panel);
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
  function beeswarm(rows, ranked, width) {
    var hasReadings = rows.length > 0 && rows[0]["reading_" + ranked[0]] !== undefined;
    var rowH = 20, left = 64, right = 18, top = 10, bottom = 34, W = width || 640, dot = 2.4, step = 5.2;
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
  function rankedStrip(meanAbs, ranked, width) {
    var st = fig10Style(meanAbs, ranked);
    // Each row is tall enough for a circle and the average written under it,
    // so the frame round the four largest encloses both. Drawn at the width
    // it is shown at, it takes as many rows as that width needs, with the
    // four largest always together in the first; the circles draw closer on
    // a narrow screen before a row gives up a place. A circle is 50 pixels
    // across, room for the longest name, htBleed, at 11 pixels (45).
    var W = width || 640, r = 25, rowH = 2 * r + 32, room = W - 2 * r - 24;
    var perRow = Math.max(4, Math.min(Math.ceil(ranked.length / 2), Math.floor(room / 88) + 1));
    var gap = Math.min(88, room / (perRow - 1));
    var x0 = (W - (perRow - 1) * gap) / 2;
    var H = Math.ceil(ranked.length / perRow) * rowH + 4;
    var out = '<svg class="pg-plot pg-strip" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="The sensors ranked">';
    var box = function (from, to, colour) {
      var row = Math.floor(from / perRow), col = from % perRow, cols = to - from;
      return '<rect x="' + (x0 + col * gap - r - 12) + '" y="' + (row * rowH + 4) + '" width="' + ((cols - 1) * gap + 2 * r + 24) +
        '" height="' + (2 * r + 26) + '" rx="8" fill="none" stroke="' + colour + '" stroke-width="1.4" stroke-dasharray="4 3"/>';
    };
    out += box(0, Math.min(4, perRow), "var(--fig-burn)");
    ranked.forEach(function (s, rank) {
      var row = Math.floor(rank / perRow), col = rank % perRow;
      var cx = x0 + col * gap, cy = row * rowH + r + 11, k = st.style[s];
      out += '<g><title>' + esc(s + ": rank " + (rank + 1) + " of " + ranked.length) + "</title>" +
        '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/>' +
        '<text x="' + cx + '" y="' + (cy + 4) + '" text-anchor="middle" class="pg-strip-label">' + esc(s) + "</text>" +
        '<text x="' + cx + '" y="' + (cy + r + 13) + '" text-anchor="middle" class="pg-tick">' + fixed(meanAbs[s], 2) + "</text></g>";
    });
    return out + "</svg>";
  }

  // -------------------------------------------------------------- controls
  // Each returns the HTML of one control. `data` is the attribute string that
  // tells the page's one change listener what the control sets.

  // A unit drawn inside the field's box, "cycles", is read out with it too.
  function numberField(id, spec, value, data, unit) {
    var input = '<input class="field" type="number" id="' + id + '" min="' + spec.min + '" max="' + spec.max +
      '" step="' + (spec.step || 1) + '" value="' + esc(value) + '" ' + data +
      (unit ? ' aria-describedby="' + id + '-unit"' : "") + ">";
    return unit ? '<span class="field-unit">' + input + '<span id="' + id + '-unit">' + esc(unit) + "</span></span>" : input;
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

  // Where a setting comes from, or what an option does, behind an i beside
  // its name, so the form itself shows only names. It opens while the pointer
  // rests on the i, while the keyboard is on it, and once the i is chosen,
  // until something else is. html is what it says; about names the setting.
  // A link among the words is a stop for the Tab key only while the words
  // are opened, by the i being chosen: words shown while the pointer rests
  // on the i close as soon as the focus leaves it, and a stop inside them
  // would lead nowhere.
  function info(html, about, id, untabbed) {
    if (!html) { return ""; }
    id = id || "pg-info-" + (++S.infos);
    return '<span class="pg-info"><button type="button" class="pg-info-button" data-action="info"' +
      (untabbed ? ' tabindex="-1"' : "") + ' aria-label="About ' + esc(about) + '" aria-describedby="' + id + '"' +
      ' aria-expanded="false">' + INFO + '</button><span class="pg-info-body" id="' + id + '"><span class="pg-info-card">' +
      html.replace(/<a /g, '<a tabindex="-1" ') +
      "</span></span></span>";
  }

  // An i's words opened or closed by choosing the i: its links join the Tab
  // order while they are open, and leave it when they close.
  function setOpen(i, open) {
    i.classList.toggle("is-open", open);
    var button = i.querySelector(".pg-info-button");
    if (button) { button.setAttribute("aria-expanded", String(open)); }
    Array.prototype.forEach.call(i.querySelectorAll(".pg-info-card a"), function (a) { a.tabIndex = open ? 0 : -1; });
  }

  // A name followed by its i. The browser will break a line where a name's
  // element ends and the i begins, whatever character stands between them,
  // and leave the i alone on a line; so the name is drawn in two parts, and
  // its last word is kept on one line with the i. part(words, n) draws part
  // n, 0 or 1; both parts together name the control.
  function withInfo(text, about, part) {
    var m = /^(.*\s)?(\S+)$/.exec(text) || [text, "", text];
    return part(m[1] || "", 0) + '<span class="pg-tail">' + part(m[2], 1) + about + "</span>";
  }

  // The ids of a name drawn in two parts, for aria-labelledby.
  function nameIds(key) { return key + "-label " + key + "-label-end"; }

  // An option's own i is not a stop for the Tab key: the keyboard reaches
  // its words through the radio button, which opens them and names them as
  // its description.
  function optionList(name, choices, current, data, noteOf, labelledBy) {
    return '<div class="opts" role="radiogroup"' + (labelledBy ? ' aria-labelledby="' + labelledBy + '"' : "") + ">" +
      choices.map(function (c, i) {
        var note = noteOf(c);
        var id = name + "-" + i;
        return '<label class="opt' + (c.value === current ? " is-active" : "") + '"><input type="radio" name="' + name +
          '" value="' + esc(c.value) + '"' + (c.value === current ? " checked" : "") + " " + data +
          ' aria-labelledby="' + nameIds(id) + '"' + (note ? ' aria-describedby="' + id + '-note"' : "") + ">" +
          '<span class="opt-label">' + withInfo(c.label, info(note, c.label, id + "-note", true), function (words, n) {
            return '<span id="' + id + (n ? "-label-end" : "-label") + '">' + esc(words) + "</span>";
          }) + "</span></label>";
      }).join("") + "</div>";
  }

  function selectField(id, choices, current, data) {
    return '<select class="field" id="' + id + '" ' + data + ">" + choices.map(function (c) {
      return '<option value="' + esc(c.value) + '"' + (String(c.value) === String(current) ? " selected" : "") + ">" +
        esc(c.label) + "</option>";
    }).join("") + "</select>";
  }

  // One setting: its name, with the i that says what it is for, directly
  // above its control, so the eye goes straight down from one to the other;
  // a switch (spec.inline) has its name beside it instead, on the controls'
  // line, as a switch has everywhere else.
  // Settings stand side by side in a grid of cells (.pg-fields); spec.size
  // "wide" takes the grid's whole width, for a list of options.
  // Reset sits where it can appear without moving what the pointer is on:
  // beside a narrow cell's control, on a wide cell's first line beside its
  // name, or under a list its step's tab names, which has no name of its own.
  function cell(spec) {
    var about = info(spec.hint, spec.label || spec.infoName || "this setting");
    var name = !spec.label ? "" : withInfo(spec.label, about, spec.id
      ? function (words, n) {
        return '<label for="' + spec.id + '"' + (n ? "" : ' id="' + spec.id + '-label"') + ">" + esc(words) + "</label>";
      }
      : function (words, n) { return '<span id="' + spec.key + (n ? "-label-end" : "-label") + '">' + esc(words) + "</span>"; });
    var named = name || about ? '<span class="pg-field-name">' + (name || about) + "</span>" : "";
    var inline = spec.inline && !!name;
    var top = spec.size === "wide" && !!name;
    var size = spec.size === "wide" && !name && !about ? "wide pg-field--bare" : spec.size;
    return '<div class="pg-field' + (size ? " pg-field--" + size : "") + '"' +
      (spec.key ? ' data-row="' + spec.key + '"' : "") + (spec.hidden ? " hidden" : "") + ">" +
      '<div class="pg-field-label">' + (inline ? "" : named + (top ? spec.reset || "" : "")) + "</div>" +
      '<div class="pg-field-control">' + spec.control + (inline ? named : "") + (top ? "" : spec.reset || "") +
      "</div></div>";
  }

  function chip(kind, label) { return T.chip(kind, label); }

  // ------------------------------------------------------------------ page

  function start(target, tools) {
    T = tools;
    root = target;
    if (window.__thesisPlaygroundTimer) { clearTimeout(window.__thesisPlaygroundTimer); }
    // seen: the last state each job was polled in. mine: the jobs this page
    // started, whose end is announced even when the first poll finds them ended.
    // restore: what each part taken out of the pipeline was, so + puts that back.
    S = { tab: S.tab || "design", jobs: [], seen: {}, mine: {}, runs: null, selected: null, detail: null,
      restore: S.restore || {}, infos: 0, step: S.step || recall(STEP) || "data", zoom: S.zoom || null, zoomed: null,
      inside: null,
      compare: [], resultsTab: S.resultsTab || "runs", page: { runs: 0, explanations: 0 }, explanations: null,
      filter: { subset: "all", temporal: "all", graph: "all", pass: "all", state: "any", results: false, files: false },
      xfilter: { method: "all", scope: "all", subset: "all", state: "any" },
      explain: { run: null, detail: null, seed: null, method: "shap", scope: "engine", engine: "",
        every: 1, shown: null, values: null, index: 0 } };

    // The browser tab is named by app.js, from the menu, as every page is.
    return tools.get("/api/playground/options").then(function (options) {
      S.options = options;
      S.fields = {};
      options.fields.forEach(function (f) { S.fields[f.key] = f; });
      S.settings = loadSettings(options.defaults);
      // A step remembered from an older page that no longer has it opens the first.
      if (!groupLabel(S.step)) { S.step = options.groups[0].key; }
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
      '<div class="pg-steps" role="tablist" aria-label="The three steps">' +
      tabButton("design", 1, "Design a run") + tabButton("results", 2, "Results", '<span class="pg-jobs-badge" ' +
      'id="pg-jobs-badge"></span>') + tabButton("explain", 3, "Explain") + "</div>" +
      '<div id="pg-panel" class="pg-panel" role="tabpanel"></div></div>';
    // The same functions every time, so coming back to the page, which
    // draws into the same element, never adds a listener twice.
    root.addEventListener("click", onClick);
    root.addEventListener("change", onChange);
    root.addEventListener("input", onInput);
    root.addEventListener("focusin", onFocusin);
    root.addEventListener("mouseover", onMouseover);
    watchCharts();
    if (!window.__thesisPlaygroundKeys) {
      window.__thesisPlaygroundKeys = true;
      document.addEventListener("keydown", function (event) { if (onPage()) { onKeydown(event); } });
    }
  }

  // An i's words that were opened close once the focus moves somewhere
  // outside them, so they never cover the field the keyboard moves to.
  function onFocusin(event) {
    $all(".pg-info.is-open").forEach(function (open) { if (!open.contains(event.target)) { setOpen(open, false); } });
    unquiet();
    placeInfo(event.target);
  }

  function onMouseover(event) {
    if (event.target.closest(".pg-info")) { placeInfo(event.target); } else { unquiet(); }
  }

  function tabButton(key, step, label, after) {
    return '<button type="button" role="tab" class="pg-step-tab" id="pg-step-tab-' + key + '" data-tab="' + key +
      '" aria-controls="pg-panel"><span class="pg-step-n">' + step + "</span>" + label + (after || "") + "</button>";
  }

  // One tab of the row is a stop for the Tab key, the chosen one; the arrow
  // keys move along the row, and the panel is named by its tab.
  function showTab(tab) {
    if (tab === "jobs") { tab = "results"; }  // the old Jobs tab's jobs are rows of Results
    S.tab = tab;
    $all(".pg-step-tab").forEach(function (button) {
      var on = button.getAttribute("data-tab") === tab;
      button.setAttribute("aria-selected", String(on));
      button.tabIndex = on ? 0 : -1;
    });
    $("#pg-panel").setAttribute("aria-labelledby", "pg-step-tab-" + tab);
    // The banner is hidden while the Jobs tab lists the same job, so it is
    // checked again whenever the tab changes, not only when jobs are polled.
    paintBanner();
    if (tab === "design") { drawDesign(); }
    else if (tab === "results") { drawResults(); }
    else { drawExplain(); }
  }

  // ------------------------------------------------------------ dispatch
  // The page's three listeners. An element names what it wants with
  // data-action, data-tab, data-field and the like; the handler is found here.

  function onClick(event) {
    $all(".pg-info.is-open").forEach(function (open) { if (!open.contains(event.target)) { setOpen(open, false); } });
    // The words behind an option's i sit inside the option's label, and a
    // click on them would otherwise choose the option. A link among them still works.
    if (event.target.closest(".pg-info-body") && !event.target.closest("a")) { event.preventDefault(); return; }
    // A procedure the settings only half follow is shown chosen, and choosing
    // it again, which fires no change, sets both of its settings.
    if (event.target.matches("input[data-procedure]")) { chooseProcedure(event.target.value); return; }
    // A unit is drawn inside its field's box, and a click on it types in the field.
    var unit = event.target.closest(".field-unit");
    if (unit && !event.target.matches("input")) { unit.querySelector("input").focus(); return; }
    var target = event.target.closest("[data-tab], [data-action], [data-goto], [data-zoom], [data-inside], " +
      "[data-step-tab], [data-reset], [data-page], [data-results-tab], tr[data-run], tr[data-explanation]");
    if (!target || !root.contains(target)) { return; }
    if (target.hasAttribute("data-tab")) { showTab(target.getAttribute("data-tab")); return; }
    if (target.hasAttribute("data-goto")) {
      goTo(target.getAttribute("data-goto"), target.getAttribute("data-goto-row"), target.getAttribute("data-goto-index"));
      return;
    }
    if (target.hasAttribute("data-zoom")) { zoomTo(target.getAttribute("data-zoom")); return; }
    if (target.hasAttribute("data-inside")) { openInside(target.getAttribute("data-inside")); return; }
    if (target.hasAttribute("data-page")) { turnPage(target.getAttribute("data-page")); return; }
    if (target.hasAttribute("data-results-tab")) { setResultsTab(target.getAttribute("data-results-tab")); return; }
    if (target.hasAttribute("data-step-tab")) { setStep(target.getAttribute("data-step-tab")); return; }
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
      if (!event.target.closest("td.pg-tick-cell")) { openRun(target.getAttribute("data-run")); }
      return;
    }
    if (target.matches("tr[data-explanation]")) {
      var e = S.explain;
      e.shown = target.getAttribute("data-explanation"); e.values = null; e.index = 0;
      drawExplain();
    }
  }

  var ACTIONS = {
    // Every setting back to its default, each control set to match, and
    // nothing else drawn again: the page stays where it is, and so does the
    // focus, on the button just pressed.
    "reset-all": function () {
      S.settings = copy(S.options.defaults);
      S.restore = {};
      var keys = Object.keys(S.fields);
      keys.forEach(syncControl);
      settingsMoved(keys);
    },
    "unzoom": function () { unzoom(); },
    "info": function (button) { placeInfo(button); setOpen(button.parentNode, !button.parentNode.classList.contains("is-open")); },
    // The pipeline's × and +: a part taken out, or put back as it was.
    "remove-part": function (button) { removePart(button.getAttribute("data-part")); },
    "add-part": function (button) { addPart(button.getAttribute("data-part")); },
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
    "again": function () { S.settings = settingsFromRun(S.detail); S.restore = {}; saveSettings(); showTab("design"); },
    "explain-this": function () {
      S.explain.run = S.detail.name; S.explain.detail = null; S.explain.shown = null; showTab("explain");
    },
    // The Explain tab has no run picker: another run is chosen in Results,
    // among the runs that kept their weights.
    "choose-run": function () {
      S.selected = null; S.resultsTab = "runs"; S.filter.files = true; S.page.runs = 0; showTab("results");
    },
    "explain": function () { submitExplanation(); }
  };

  function onChange(event) {
    var input = event.target;
    if (input.hasAttribute("data-procedure")) { chooseProcedure(input.value); return; }
    if (input.hasAttribute("data-field")) { fieldChanged(input); return; }
    // The slider redraws when it is let go, not while it is being dragged.
    if (input.hasAttribute("data-window-index")) {
      S.explain.index = Number(input.value);
      var y = window.scrollY;
      drawExplanation();
      window.scrollTo({ top: y, behavior: "instant" });
      return;
    }
    // A filter changed: the list goes back to its first page, and the
    // focus stays on the filter.
    if (input.hasAttribute("data-filter") || input.hasAttribute("data-xfilter")) {
      var runs = input.hasAttribute("data-filter");
      var f = runs ? S.filter : S.xfilter, which = input.getAttribute(runs ? "data-filter" : "data-xfilter");
      f[which] = input.type === "checkbox" ? input.checked : input.value;
      S.page[S.resultsTab] = 0;
      drawResultsBody();
      var again = $("#" + input.id);
      if (again) { again.focus({ preventScroll: true }); }
      return;
    }
    if (input.hasAttribute("data-compare")) { compareTicked(input); return; }
    // Fig. 8 is drawn again with its controls, so the focus is put back on
    // the control that was changed.
    if (input.hasAttribute("data-pred-seed") || input.hasAttribute("data-pred-order")) {
      if (input.hasAttribute("data-pred-seed")) { S.predSeed = input.value; } else { S.predOrder = input.value; }
      var again = input.hasAttribute("data-pred-seed") ? "#pg-pred-seed" : 'input[name="pg-pred-order"]:checked';
      drawDetail().then(function () {
        var control = $(again);
        if (control) { control.focus({ preventScroll: true }); }
      });
      return;
    }
    if (input.hasAttribute("data-explain")) { explainChanged(input); }
  }

  // Escape closes whatever i is open, until the focus or the pointer moves on.
  // When no i's words are showing, it leads from a step's flowchart back to
  // the pipeline instead.
  // In a row of tabs the arrow keys, Home and End move to another tab and
  // choose it, as tabs do everywhere else.
  function onKeydown(event) {
    var tab = event.target && event.target.closest ? event.target.closest('[role="tab"]') : null;
    // With Alt, Ctrl or the command key held the keys are the browser's own:
    // Alt with an arrow goes back or forward.
    var plain = !event.altKey && !event.ctrlKey && !event.metaKey && !event.shiftKey;
    if (tab && plain && /^(ArrowLeft|ArrowRight|Home|End)$/.test(event.key)) {
      var all = Array.prototype.slice.call(tab.parentNode.querySelectorAll('[role="tab"]'));
      var at = all.indexOf(tab), n = all.length;
      var next = event.key === "Home" ? all[0] : event.key === "End" ? all[n - 1]
        : all[(at + (event.key === "ArrowRight" ? 1 : n - 1)) % n];
      event.preventDefault();
      next.focus();
      next.click();
      return;
    }
    var row = event.target && event.target.matches && event.target.matches("tr[data-action]") ? event.target : null;
    if (row && plain && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); row.click(); return; }
    if (event.key !== "Escape") { return; }
    // Words placed against the window have no offsetParent even while shown,
    // so what is drawn is asked instead.
    var showing = $all(".pg-info-body").some(function (body) { return body.getClientRects().length > 0; });
    $all(".pg-info").forEach(function (i) { setOpen(i, false); i.classList.add("is-quiet"); });
    var target = event.target && event.target.nodeType === 1 ? event.target : document.body;
    // A link inside the words that just closed would leave the focus
    // nowhere, so it goes back to the words' own i.
    var card = target.closest(".pg-info-card");
    if (card) { card.closest(".pg-info").querySelector(".pg-info-button").focus(); }
    var here = target.closest("#pg-pipeline") || target === document.body || target.matches("main, .content");
    if (!showing && here && S.zoom && S.tab === "design") { unzoom(); }
  }

  function unquiet() {
    $all(".pg-info.is-quiet").forEach(function (i) { i.classList.remove("is-quiet"); });
  }

  // An i's words open under it, moved left as far as they must to stay
  // inside the window. Placed before they show, for the i the pointer or the
  // focus has reached, or for the option the focus is on.
  // Inside the pipeline, when it scrolls on its own beside the form, the
  // words are placed against the window instead of against the i, since the
  // pipeline would otherwise cut off whatever passes its edges; they open
  // above the i when there is no room below it.
  function placeInfo(element) {
    var holder = element && element.closest && element.closest(".pg-info, .opt");
    var i = holder && (holder.classList.contains("pg-info") ? holder : holder.querySelector(".pg-info"));
    if (!i) { return; }
    var at = i.getBoundingClientRect();
    var width = Math.min(360, window.innerWidth - 32);
    var left = Math.max(16, Math.min(at.left - 8, window.innerWidth - 16 - width));
    var body = i.querySelector(".pg-info-body");
    var side = i.closest(".pg-pipeline-body");
    var loose = side && getComputedStyle(side).overflowY !== "visible";
    i.classList.toggle("is-loose", !!loose);
    if (!loose) {
      body.style.left = Math.round(left - at.left) + "px";
      body.style.top = body.style.bottom = "";
      i.classList.remove("is-above");
      return;
    }
    var above = at.bottom + (body.offsetHeight || 160) > window.innerHeight - 16;
    i.classList.toggle("is-above", above);
    body.style.left = Math.round(left) + "px";
    body.style.top = above ? "auto" : Math.round(at.bottom) + "px";
    body.style.bottom = above ? Math.round(window.innerHeight - at.top) + "px" : "auto";
  }

  // The pipeline beside the form is never taller than the form, so the form
  // alone decides how tall their row is: a flowchart taller than the form
  // would otherwise lengthen the row, and the pipeline would scroll away with
  // the page instead of staying in view. The stylesheet reads the height.
  function watchForm() {
    var form = $(".pg-form"), design = $(".pg-design");
    if (S.formWatch) { S.formWatch.disconnect(); S.formWatch = null; }
    if (!form || !design) { return; }
    var set = function () { design.style.setProperty("--pg-form-h", form.offsetHeight + "px"); };
    set();
    if (window.ResizeObserver) {
      S.formWatch = new ResizeObserver(set);
      S.formWatch.observe(form);
    }
  }

  // The pipeline's i's that are showing follow their i when the pipeline's
  // body or the page scrolls, and close once their i has scrolled out of the
  // body's view, rather than point at nothing. The body is drawn anew with
  // every change of view, so its scrolling is caught on the way down, from
  // the pipeline, which stays.
  function watchPipeline() {
    var box = $("#pg-pipeline");
    if (!box || box.__scrolls) { return; }
    box.__scrolls = true;
    var follow = function () {
      var scroller = box.querySelector(".pg-pipeline-body");
      if (!scroller) { return; }
      var view = scroller.getBoundingClientRect();
      Array.prototype.forEach.call(box.querySelectorAll(".pg-info.is-loose"), function (i) {
        if (!i.querySelector(".pg-info-body").getClientRects().length) { return; }
        var at = i.getBoundingClientRect();
        if (at.bottom <= view.top || at.top >= view.bottom) {
          setOpen(i, false);
          i.classList.add("is-quiet");
        } else {
          placeInfo(i);
        }
      });
    };
    box.addEventListener("scroll", follow, { passive: true, capture: true });
    if (!window.__thesisPlaygroundScroll) {
      window.__thesisPlaygroundScroll = true;
      window.addEventListener("scroll", function () {
        var now = document.getElementById("pg-pipeline");
        if (now && onPage()) { now.dispatchEvent(new Event("scroll")); }
      }, { passive: true });
    }
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
    if (!$('tr[data-job="' + job.id + '"]')) { T.toast(statusWord(job.status) + ": " + job.label); }
    if (job.status !== "done") { return; }
    S.runs = null;
    S.explanations = null;
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

  // The banner above the tabs shows the one job running now, and the
  // Results tab's badge counts the jobs running or waiting, whose rows it
  // lists; its rows follow the jobs while it is open.
  function drawJobs() {
    paintResultJobs();
    paintBanner();
    var badge = $("#pg-jobs-badge");
    if (badge) {
      var live = S.jobs.filter(isLive).length;
      badge.textContent = live ? count(live) : "";
      badge.title = live ? plural(live, "job") + " running or waiting" : "";
    }
  }

  // Hidden while the job's own row is on screen in Results.
  function paintBanner() {
    var box = $("#pg-current");
    if (!box) { return; }
    var current = orderedJobs().filter(isLive)[0];
    var listed = current && !!$('tr[data-job="' + current.id + '"]');
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

  // ---------------------------------------------------------------- design

  function fieldsOf(group) {
    return S.options.fields.filter(function (f) { return f.group === group; });
  }


  function groupLabel(key) {
    var found = S.options.groups.filter(function (g) { return g.key === key; })[0];
    return found ? found.label : "";
  }

  // The data are what goes into the pipeline, not one of its steps: they are
  // drawn the way the remaining life that comes out is, and their tab carries
  // their picture in place of a number. The steps are numbered from 1 after them.
  var INPUT = "data";

  function stepOf(group) {
    return S.options.groups.map(function (g) { return g.key; }).filter(function (k) { return k !== INPUT; })
      .indexOf(group) + 1;
  }

  // A step's number, filled in while that step is the one being worked on.
  // The same step's number in the pipeline follows it.
  function stepNumber(group) {
    var input = group === INPUT;
    return '<span class="pg-step-n pg-step-n--form' + (input ? " pg-step-n--input" : "") +
      (S.step === group ? " is-current" : "") + '" data-step="' + group + '">' + (input ? ICONS.data : stepOf(group)) +
      "</span>";
  }

  // The step being worked on: its tab is chosen and its settings shown, its
  // number is filled in on its tab and on its box in the pipeline, and it is
  // remembered, so the page opens on it next time.
  function setStep(group) {
    if (!groupLabel(group)) { return; }
    S.step = group;
    remember(STEP, group);
    $all("[data-step]").forEach(function (n) { n.classList.toggle("is-current", n.getAttribute("data-step") === group); });
    $all("[data-step-tab]").forEach(function (tab) {
      var on = tab.getAttribute("data-step-tab") === group;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
    });
    $all(".pg-tabpanel").forEach(function (panel) {
      var on = panel.getAttribute("data-group") === group;
      panel.classList.toggle("is-active", on);
      if (on) { panel.removeAttribute("inert"); } else { panel.setAttribute("inert", ""); }
    });
    // The pipeline's numbers were just changed in place, so the markup it is
    // compared against when settings move is brought up to date with them.
    var pipeline = $("#pg-pipeline");
    if (pipeline && pipeline.__zoom === S.zoom) { pipeline.__html = pipelineInner(); }
    fitTabs();
  }

  function capitalise(text) { return text.charAt(0).toUpperCase() + text.slice(1); }

  function visible(key) {
    var s = S.settings;
    if (key === "temporal_hidden" || key === "temporal_out_features" || key === "shared_across_sensors") {
      return s.temporal === "lstm" || s.temporal === "gru";
    }
    if (key === "smoothing_alpha") { return !!s.smoothing; }
    if (key === "validation_fraction") { return chosenProcedure().settings.validation_fraction > 0; }
    if (key === "early_stopping_patience") { return !!s.early_stopping; }
    return true;
  }

  function isChanged(f) { return !same(S.settings[f.key], f.default); }

  // How many settings of a group differ from their defaults. Training counts
  // the procedure as one setting, and the inputs only it shows as theirs.
  function changedIn(group) {
    var changed = fieldsOf(group).filter(function (f) { return visible(f.key) && isChanged(f); }).length;
    if (group === "training") {
      changed += fieldsOf("procedure").filter(function (f) {
        return f.kind !== "bool" && visible(f.key) && isChanged(f);
      }).length + (chosenProcedure() === defaultProcedure() && followsProcedure() ? 0 : 1);
    }
    return changed;
  }

  // The procedure some settings follow. A procedure decides two things,
  // whether any engines are held back and whether training stops early, so a
  // held-back fraction other than its own still counts as that procedure.
  // Settings that fit neither (read back from an older run, say) count as the
  // procedure that holds engines back, whose inputs then show them.
  function procedureFor(settings) {
    var list = S.options.procedures;
    var found = list.filter(function (p) {
      return Object.keys(p.settings).every(function (k) {
        var want = p.settings[k], have = settings[k];
        return typeof want === "number" ? (Number(have) > 0) === (want > 0) : have === want;
      });
    })[0];
    return found || list.filter(function (p) { return p.settings.validation_fraction > 0; })[0] || list[0];
  }

  function chosenProcedure() { return procedureFor(S.settings); }

  // Whether the settings follow the procedure shown exactly, both things it
  // decides. Settings read back from an older run can follow neither.
  function followsProcedure() {
    var p = chosenProcedure();
    return Object.keys(p.settings).every(function (k) {
      var want = p.settings[k], have = S.settings[k];
      return typeof want === "number" ? (Number(have) > 0) === (want > 0) : have === want;
    });
  }

  function defaultProcedure() { return procedureFor(S.options.defaults); }

  function choiceLabel(key, value) {
    var found = (S.fields[key].choices || []).filter(function (c) { return c.value === value; })[0];
    return found ? found.label : String(value);
  }

  // A label such as "Window length (cycles)" is a name and a unit.
  function splitUnit(label) {
    var m = /^(.*?)\s*\(([^)]+)\)$/.exec(label);
    return m ? { label: m[1], unit: m[2] } : { label: label, unit: "" };
  }

  // The pipeline, then the form in the order the work is done: the settings,
  // one step at a time under six tabs with Reset all at the foot of their
  // card, then the card that runs them. On a wide page the pipeline stands
  // to the right of the form and stays in view while the page scrolls; on a
  // narrow one it stands above it. The two cards beside each other start
  // level, since neither has a heading above it.
  function drawDesign() {
    $("#pg-panel").innerHTML =
      '<div class="pg-design">' + pipelineHtml() + '<div class="pg-form">' +
      '<section class="pg-block pg-settings" aria-label="Settings">' + tabsHtml() +
      '<div class="pg-tabpanels">' + S.options.groups.map(panelHtml).join("") + "</div>" +
      '<div class="pg-settings-foot"><button type="button" class="pg-reset" data-action="reset-all">Reset all</button></div>' +
      "</section>" +
      '<section class="pg-block pg-run" id="pg-run" aria-label="Run"></section></div></div>';
    // What the pipeline was just drawn with, so the first change of settings
    // finds it unchanged instead of drawing it again under the pointer.
    var pipeline = $("#pg-pipeline");
    pipeline.__html = pipelineInner();
    pipeline.__zoom = S.zoom;
    drawRunCard($("#pg-run"), null, null);
    watchPipeline();
    watchForm();
    watchTabs();
    fitFlow();
    fitTabs();
    refreshPreview();
    requestZoom(true);
  }

  // ----------------------------------------------------------------- steps
  // The six steps of the form as tabs, in the order the work is done. Every
  // step's settings are drawn, one on top of another, and only the chosen
  // one is shown, so the box is as tall as the tallest step and does not
  // change height when another tab is chosen. A tab carries a dot while any
  // setting of its step differs from its default and the step is not the
  // one on show, where each such setting has its own Reset.
  function tabsHtml() {
    return '<div class="pg-tabs" role="tablist" aria-label="Steps of the run" id="pg-tabs">' +
      S.options.groups.map(function (g) {
        var on = g.key === S.step;
        return '<button type="button" role="tab" class="pg-tab" id="pg-tab-' + g.key + '" data-step-tab="' + g.key +
          '" aria-selected="' + on + '" aria-controls="pg-tabpanel-' + g.key + '" tabindex="' + (on ? 0 : -1) + '">' +
          '<span class="pg-tab-num">' + stepNumber(g.key) + changedDot(g.key) + "</span>" +
          '<span class="pg-tab-name" id="pg-tab-name-' + g.key + '">' + esc(g.label) + "</span>" +
          changedWords(g.key) + "</button>";
      }).join("") + "</div>";
  }

  // The dot is drawn on the number; what it says is read after the step's
  // name, "2 Preprocessing, 1 setting changed".
  function changedDot(group) {
    var n = changedIn(group);
    return '<span class="pg-tab-changed" aria-hidden="true" data-n="' + n + '"' + (n ? "" : " hidden") + "></span>";
  }

  function changedWords(group) {
    var n = changedIn(group);
    return '<span class="pg-tab-words visually-hidden" data-n="' + n + '">' +
      (n ? ", " + n + (n === 1 ? " setting" : " settings") + " changed" : "") + "</span>";
  }

  function panelHtml(g) {
    var on = g.key === S.step;
    var body = '<div class="pg-sections">' + SECTIONS[g.key].map(function (section, i) {
      return sectionHtml(section, i, g.key);
    }).join("") + "</div>";
    return '<div class="pg-tabpanel' + (on ? " is-active" : "") + '" role="tabpanel" id="pg-tabpanel-' + g.key +
      '" data-group="' + g.key + '" aria-labelledby="pg-tab-' + g.key + '"' + (on ? "" : " inert") + ">" + body + "</div>";
  }

  // The settings of a box of the pipeline stand in groups named after the
  // steps of the box's own flowchart that they decide, in the order those
  // steps run, each with the step's picture: "Cap the labels" with the label
  // ceiling and whether the test labels are capped, then "Smooth each
  // sensor" with its weight, and so on. A group's name replaces the name of
  // a list of options it holds alone, whose i then stands beside the group's
  // name, and a switch that turns the whole step on or off stands before it.
  // So no i ever stands without a name beside it. Training and Repeats have
  // no box, and are grouped by subject in the same way.
  var SECTIONS = {
    data: [
      { name: "Sub-dataset", picture: "file", fields: ["subset"] }
    ],
    preprocessing: [
      { name: "Cap the labels", picture: "cap", fields: ["rul_max", "clip_test_labels"] },
      { name: "Smooth each sensor", picture: "smooth", toggle: "smoothing", fields: ["smoothing_alpha"] },
      { name: "Rescale each sensor", picture: "rescale", fields: ["normalization"] },
      { name: "Cut into windows", picture: "window", fields: ["window_size", "window_stride"] }
    ],
    temporal: [
      { name: "Temporal stage", picture: "window", fields: ["temporal"] },
      { name: function () { return "One " + networkWord() + (S.settings.shared_across_sensors ? " for all sensors" : " per sensor"); },
        picture: "recurrent", fields: ["temporal_hidden", "shared_across_sensors"] },
      { name: function () { return "Turn into " + plural(Number(S.settings.temporal_out_features) || 1, "number") + " per sensor"; },
        picture: "project", fields: ["temporal_out_features"] }
    ],
    graph: [
      { name: "Sensor graph", picture: "graph", fields: ["graph"] },
      { name: "Rescale the graph", picture: "table", fields: ["adjacency_normalization"] },
      { name: "Graph layers", picture: "gcn", fields: ["gcn_hidden", "gcn_bias"] },
      { name: "Fully connected", picture: "dense", fields: ["fc_hidden"] }
    ],
    training: [
      { name: "Procedure", picture: "hold", procedure: true },
      { name: "Passes", picture: "passes", fields: ["epochs", "batch_size"] },
      { name: "Learning rate", picture: "rate", fields: ["learning_rate", "final_learning_rate", "lr_schedule"] }
    ],
    repeats: [
      { name: "Seeds", picture: "seeds", fields: ["seed", "seeds", "fixed_split"] },
      { name: "Device", picture: "device", fields: ["device"] }
    ]
  };

  function sectionName(section) { return typeof section.name === "function" ? section.name() : section.name || ""; }

  // One group: its picture, its switch if it has one, its name and i, then its settings.
  function sectionHtml(section, index, group) {
    var fields = (section.fields || []).map(function (key) { return S.fields[key]; });
    var cells = function (named) {
      return '<div class="pg-fields">' + (section.procedure ? procedureHtml(named) : "") +
        fields.map(function (f) { return fieldRow(f, named); }).join("") + "</div>";
    };
    var name = sectionName(section);
    if (!name) { return cells(null); }
    var lone = section.procedure || (fields.length === 1 && fields[0].kind === "choice" && !MENU[fields[0].key]);
    var toggle = section.toggle ? S.fields[section.toggle] : null;
    var id = "pg-section-" + group + "-" + index;
    var hidden = !toggle && fields.every(function (f) { return !visible(f.key); });
    return '<section class="pg-section" data-section="' + group + "-" + index + '" aria-labelledby="' + id + '"' +
      (hidden ? " hidden" : "") + '><div class="pg-section-head"' + (toggle ? ' data-row="' + toggle.key + '"' : "") + ">" +
      '<span class="pg-section-picture">' + PICTURES[section.picture] + "</span>" +
      (toggle ? switchControl("pg-" + toggle.key, !!S.settings[toggle.key],
        'data-field="' + toggle.key + '" aria-labelledby="' + id + '"') : "") +
      '<h3 class="pg-section-name pg-field-name"><span class="pg-section-label" id="' + id + '">' + esc(name) + "</span>" +
      info(section.procedure ? infoCard(S.options.procedure_about, "", S.options.procedure_more)
        : lone ? hintFor(fields[0]) : toggle ? hintFor(toggle) : "", name) + "</h3>" +
      (toggle ? resetHtml(toggle) : "") + "</div>" + cells(lone ? id : null) + "</section>";
  }

  // A group is hidden while every setting in it is, and a group's name that
  // follows the settings, "One GRU per sensor", follows them.
  function paintSections() {
    $all(".pg-section").forEach(function (box) {
      var at = box.getAttribute("data-section").split("-"), section = SECTIONS[at[0]][Number(at[1])];
      var cells = Array.prototype.slice.call(box.querySelectorAll(".pg-fields > .pg-field"));
      box.hidden = !section.toggle && cells.length > 0 && cells.every(function (c) { return c.hidden; });
      var label = box.querySelector(".pg-section-label"), name = sectionName(section);
      if (label && label.textContent !== name) { label.textContent = name; }
    });
  }

  // Every changed-setting dot, after settings moved.
  function paintTabs() {
    S.options.groups.forEach(function (g) {
      var dot = $("#pg-tab-" + g.key + " .pg-tab-changed");
      var words = $("#pg-tab-" + g.key + " .pg-tab-words");
      if (dot && dot.getAttribute("data-n") !== String(changedIn(g.key))) {
        dot.outerHTML = changedDot(g.key);
        words.outerHTML = changedWords(g.key);
      }
    });
    fitTabs();
  }

  // The tabs name every step when all six names fit in one row. When they do
  // not, only the chosen tab keeps its name on screen and the others show
  // their numbers (is-compact); on the narrowest phones, where even that row
  // does not fit, every tab shows only its number, the chosen one filled in
  // and underlined (is-numbers). Each tab keeps its name for a screen
  // reader, and shows it when the pointer rests on its number. Measured, not
  // guessed from the width, as the pipeline's row is.
  function fitTabs() {
    var tabs = $("#pg-tabs");
    if (!tabs) { return; }
    var over = function () { return tabs.scrollWidth > tabs.clientWidth + 1; };
    tabs.classList.remove("is-compact", "is-numbers");
    var tight = over();
    tabs.classList.toggle("is-compact", tight);
    tabs.classList.toggle("is-numbers", tight && over());
    Array.prototype.forEach.call(tabs.querySelectorAll(".pg-tab"), function (tab) {
      var num = tab.querySelector(".pg-tab-num");
      if (tight) { num.title = tab.querySelector(".pg-tab-name").textContent; } else { num.removeAttribute("title"); }
    });
  }

  function watchTabs() {
    var tabs = $("#pg-tabs");
    if (!tabs || tabs.__watched || !window.ResizeObserver) { return; }
    tabs.__watched = true;
    var width = tabs.clientWidth;
    new ResizeObserver(function () {
      if (tabs.clientWidth !== width) { width = tabs.clientWidth; fitTabs(); }
    }).observe(tabs);
  }

  // ------------------------------------------------------------- pipeline
  // The run as a flowchart, in the order the data flows through it. A box
  // opens that step's own flowchart in its place, and a button leads back;
  // × takes a part out and + puts it back, the two moves Table 11 makes one
  // at a time: the LSTM from the overview, the sensor graph from inside the
  // graph layers' own flowchart. S.zoom is the step whose flowchart is open, or null.

  function pipelineHtml() {
    return '<section class="pg-block pg-pipeline" id="pg-pipeline" aria-labelledby="pg-pipeline-title">' +
      pipelineInner() + "</section>";
  }

  // A heading, then a body that scrolls on its own beside the form, so the
  // heading, and the way back from a step's flowchart, stay in view. Inside
  // the temporal stage's flowchart one of its networks can be opened in turn
  // (S.inside, a sensor's name, or "shared"); the way back then leads to the
  // stage's flowchart.
  function pipelineInner() {
    if (!S.zoom) {
      return '<div class="pg-pipeline-head"><h2 class="pg-block-title" id="pg-pipeline-title">Pipeline</h2></div>' +
        '<div class="pg-pipeline-body"><div class="pg-flow" id="pg-flow" role="list">' + flowHtml() + "</div></div>";
    }
    var inside = insideOpen();
    return '<div class="pg-pipeline-head pg-zoom-head">' +
      '<button type="button" class="button button--quiet button--small pg-zoom-back" data-action="unzoom" ' +
      'aria-label="Back to the ' + (inside ? "temporal stage" : "pipeline") + '">' + BACK +
      esc(inside ? boxName("temporal") : "Pipeline") + "</button>" +
      '<h2 class="pg-zoom-title" id="pg-pipeline-title" data-name="' + esc(titleKey()) + '">' + zoomTitle() + "</h2>" +
      '<span class="pg-zoom-updating" id="pg-zoom-updating" role="status"></span></div>' +
      '<div class="pg-pipeline-body" id="pg-zoom-body">' + zoomBody() + "</div>";
  }

  // Whether one of the temporal stage's networks is open.
  function insideOpen() { return S.zoom === "temporal" && !!S.inside; }

  // What the title shows, to tell when it must be written again.
  function titleKey() { return boxName(S.zoom) + (insideOpen() ? " " + S.inside : ""); }

  // The open box's picture, number and name, as its box shows them; the
  // data, the pipeline's input, have no number. An open network is named by
  // its sensor.
  function zoomTitle() {
    var name = boxName(S.zoom);
    if (insideOpen()) { name = networkWord() + (S.inside === "shared" ? ", shared by all sensors" : " of " + S.inside); }
    return '<span class="pg-zoom-icon">' + ICONS[S.zoom] + "</span>" + (S.zoom === INPUT ? "" : stepNumber(S.zoom)) +
      esc(name);
  }

  function networkWord() { return S.settings.temporal === "gru" ? "GRU" : "LSTM"; }

  // What a box says the step is now: the sub-dataset, the stage, the layers.
  function boxName(group) {
    var s = S.settings;
    if (group === "data") { return s.subset; }
    if (group === "temporal") {
      var name = temporalName(s.temporal);
      return name.indexOf("w/o") === 0 ? name : capitalise(name);
    }
    if (group === "graph") { return "GCN"; }
    return groupLabel(group);
  }

  // The data that go in, drawn as words with their picture as the remaining
  // life that comes out is, then preprocessing, the temporal stage when there
  // is one, the graph layers, and what comes out. The graph layers cannot be
  // taken out, and their cross says why instead of doing nothing silently;
  // the sensor graph they run on is a step of their own flowchart, taken out there.
  function flowHtml() {
    var s = S.settings;
    var temporal = s.temporal === REMOVED.temporal
      ? flowLink(addButton("temporal"))
      : flowLink("") + stageBox("temporal", removeButton("temporal", "pg-stage-x")) + flowLink("");
    var fixedCross = '<button type="button" class="pg-stage-x" aria-disabled="true" title="' + GCN_FIXED + '" aria-label="' +
      GCN_FIXED + '">' + CROSS + "</button>";
    return '<div class="pg-flow-end pg-flow-in" role="listitem"><button type="button" class="pg-flow-input" data-zoom="' +
      INPUT + '" aria-label="' + esc("Data: " + s.subset + ", its flowchart") + '"><span class="pg-flow-input-icon' +
      (S.step === INPUT ? " is-current" : "") + '" data-step="' + INPUT + '">' + ICONS.data + "</span>" + esc(s.subset) +
      "</button></div>" + flowLink("") + stageBox("preprocessing", "") + temporal + stageBox("graph", fixedCross) +
      '<div class="pg-flow-link pg-flow-out" aria-hidden="true"></div>' +
      '<div class="pg-flow-end pg-flow-out" role="listitem">' + ICONS.output + "Remaining life</div>";
  }

  // One box: its step's number in the top left corner, its picture, and
  // under it what the step is now, "LSTM"; its step's name is on the step's
  // tab, and the box is read out with both. The whole box opens the step's
  // flowchart; x is its cross, in the top right corner.
  function stageBox(group, x) {
    var name = boxName(group);
    return '<div class="pg-stage" role="listitem">' +
      '<button type="button" class="pg-stage-main" data-zoom="' + group + '" aria-label="' +
      esc(stepOf(group) + ". " + (groupLabel(group) === name ? name : groupLabel(group) + ": " + name) + ", its flowchart") +
      '">' + stepNumber(group) + '<span class="pg-stage-icon">' + ICONS[group] + "</span>" +
      '<span class="pg-stage-name">' + esc(name) + "</span></button>" + x + "</div>";
  }

  // One step's own flowchart, as the server describes it from the code and
  // the data: top to bottom, each step with its picture, its values and how
  // many numbers it learns, and on the arrows the shape of what passes. Or,
  // with one of the temporal stage's networks open, that network from
  // inside. The last answer stays on screen while a newer one is asked for.
  function zoomBody() {
    var d = S.zoomed;
    if (!d || d.stage !== S.zoom) { return '<div class="loading">Reading the step…</div>'; }
    // An answer for settings that have since changed stays on screen, greyed,
    // until the new one arrives: FD004's preprocessing takes several seconds.
    var stale = d.asked !== zoomKey();
    var note = d.note ? '<p class="pg-zoom-note">' + esc(d.note) + "</p>" : "";
    var networks = d.steps && networksOf(d.steps);
    var body = d.error ? errorBox(d.error) : insideOpen() && networks ? insideHtml(networks) : vflowHtml(d.steps);
    return '<div class="pg-zoom-steps' + (stale ? " is-stale" : "") + '">' + note + body + "</div>";
  }

  // The request an answer belongs to: the open step and every setting.
  function zoomKey() { return S.zoom + JSON.stringify(S.settings); }

  // The temporal stage's networks, as the step that runs them describes them.
  function networksOf(steps) {
    return steps.filter(function (step) { return step.networks; }).map(function (step) { return step.networks; })[0] || null;
  }

  // The shape of what an item of a flowchart hands to the next one; a row
  // of steps side by side names what the row hands on, and a branch ends in
  // its own arrow, so names none.
  function handedOn(item) { return item && item.kind !== "branch" ? item.shape || "" : ""; }

  // Every item of a box's flowchart, each followed by the arrow to the next
  // with the shape of what passes along it written on it: so each step reads
  // from top to bottom as what comes in, what is done to it and what goes
  // out, and every shape is written once. A branch, the sensor graph, is
  // drawn beside the arrow before it, as a second input of the step after it.
  function vflowHtml(steps) {
    var items = [];
    steps.forEach(function (item, i) {
      if (item.kind === "branch") { return; }
      var next = steps[i + 1], last = i === steps.length - 1;
      var after = last ? "" : next && next.kind === "branch" ? joinHtml(handedOn(item), next) : arrowHtml(handedOn(item));
      items.push('<li class="pg-vitem' + (item.kind === "edge" ? " pg-vitem--edge" : "") + '">' +
        vbodyHtml(item, steps[i - 1], i === 0, last) + after + "</li>");
    });
    return '<ol class="pg-vflow">' + items.join("") + "</ol>";
  }

  // An item itself. The words that open or close a flowchart stand with the
  // picture of the box they come from or go to, as the overview draws its
  // input and output; the opening words' shape goes on the arrow under them,
  // and the closing words show their own only when it differs from the one
  // on the arrow above them.
  function vbodyHtml(item, before, first, last) {
    if (item.kind === "edge") {
      var own = item.shape && !(first && !last) && !(last && item.shape === handedOn(before));
      return '<span class="pg-vedge">' + (ICONS[item.icon] || "") + "<span>" + esc(item.name) + "</span></span>" +
        (own ? '<code class="pg-vshape">' + esc(item.shape) + "</code>" : "");
    }
    if (item.kind === "group") {
      return '<div class="pg-vgroup">' + item.steps.map(function (s) { return vstepHtml(s, true); }).join("") + "</div>";
    }
    return vstepHtml(item, false);
  }

  // The arrow under an item, carrying the shape it hands on, if any.
  function arrowHtml(shape, long) {
    return '<div class="pg-varrow' + (long ? " pg-varrow--long" : "") + '">' + (shape ? '<span class="visually-hidden">' +
      "hands on </span>" + '<code class="pg-vshape">' + esc(shape) + "</code>" : "") + "</div>";
  }

  // The main line runs straight down to the step after the branch, and the
  // branch's steps stand beside it and end in an arrow into the same step.
  function joinHtml(shape, branch) {
    return '<div class="pg-vjoin"><div class="pg-vjoin-main">' + arrowHtml(shape, true) + "</div>" +
      '<ol class="pg-vjoin-side">' + branch.steps.map(function (s) {
        return '<li class="pg-vitem">' + vstepHtml(s, true) + arrowHtml(handedOn(s)) + "</li>";
      }).join("") + "</ol></div>";
  }

  // One step: its picture, its name, the values that decide it (each a link
  // to its setting) and how many numbers it learns, centred; what it does and
  // where it comes from are behind its i. Steps side by side on one row
  // (small) carry what they hand on inside, since no single arrow can. The
  // step that runs the temporal stage's networks shows each of them.
  function vstepHtml(step, small) {
    var values = (step.parts || []).filter(function (part) { return part.text; }).map(valueHtml).join(
      '<span class="pg-vstep-dot" aria-hidden="true"> · </span>');
    var words = infoCard(step.does, step.source ? sourceTag(step.source) : "", step.more);
    var part = partControl(step);
    return '<div class="pg-vstep' + (small ? " pg-vstep--small" : "") + (part ? " has-part" : "") + '">' +
      (PICTURES[step.picture] ? '<span class="pg-vstep-picture">' + PICTURES[step.picture] + "</span>" : "") +
      '<span class="pg-vstep-name">' + withInfo(step.name, info(words, step.name), function (w) { return esc(w); }) +
      "</span>" + (values ? '<span class="pg-vstep-values">' + values + "</span>" : "") +
      (step.networks ? networksHtml(step.networks) : "") +
      (step.learned ? '<span class="pg-vstep-learned">' + count(step.learned) + " learned numbers</span>" : "") +
      (small && step.shape ? '<code class="pg-vshape">' + esc(step.shape) + "</code>" : "") + part + "</div>";
  }

  // The recurrent networks, one per sensor, each a button that opens it; or
  // the one network every sensor shares.
  function networksHtml(n) {
    var word = n.kind === "gru" ? "GRU" : "LSTM";
    var names = n.shared ? ["shared"] : n.sensors;
    return '<span class="pg-nets' + (n.shared ? " is-shared" : "") + '">' + names.map(function (sensor) {
      var shown = sensor === "shared" ? "all sensors" : sensor;
      return '<button type="button" class="pg-net" data-inside="' + esc(sensor) + '" aria-label="' +
        esc(word + (sensor === "shared" ? " shared by all sensors" : " of " + sensor) + ", from inside") + '">' +
        '<span class="pg-net-kind">' + word + "</span>" + esc(shown) + "</button>";
    }).join("") + "</span>";
  }

  // One recurrent network from inside: what it takes in at every cycle, the
  // gates that decide what it keeps, and what it gives out, drawn and then
  // said in words, with the learned numbers of each gate from the built network.
  function insideHtml(n) {
    var lstm = n.kind !== "gru";
    var who = S.inside === "shared" ? "each sensor in turn" : S.inside;
    var carried = lstm ? "the two lists of " + count(n.hidden) + " numbers it carried from the cycle before, the " +
      "hidden state and the memory," : "the list of " + count(n.hidden) + " numbers it carried from the cycle before, " +
      "the hidden state,";
    var gates = lstm ? [
      ["Forget gate", "how much of the memory carried from the cycle before is kept"],
      ["Input gate", "how much of the candidate is added to the memory"],
      ["Candidate", "the new values that could be added, made from this cycle's reading and the hidden state"],
      ["Output gate", "how much of the updated memory, squeezed between −1 and 1, is handed on as the new hidden state"]
    ] : [
      ["Reset gate", "how much of the hidden state is used in making the candidate"],
      ["Update gate", "how much of the old hidden state is kept rather than replaced by the candidate"],
      ["Candidate", "the new hidden state that could replace the old one, made from this cycle's reading and the hidden state"]
    ];
    return '<div class="pg-inside">' +
      '<p class="pg-inside-lead">The same network runs ' + plural(n.window, "time") + ", once for each cycle of the " +
      "window, oldest first. Each time it takes " + esc(who) + "'s reading at that cycle and " + carried +
      " and gives out new ones.</p>" +
      '<div class="pg-inside-drawing">' + (lstm ? LSTM_DRAWING : GRU_DRAWING) + "</div>" +
      '<ul class="pg-inside-gates">' + gates.map(function (g) {
        return "<li><strong>" + esc(g[0]) + ":</strong> " + esc(g[1]) + '. <span class="pg-faint">' +
          count(n.per_gate) + " learned numbers</span></li>";
      }).join("") + "</ul>" +
      '<p class="pg-inside-foot">After the last cycle only the hidden state, ' + plural(n.hidden, "number") +
      ", leaves the network. " + (n.shared ? "Every sensor's window runs through this one network, with the same " +
        "learned numbers, and nothing in it says which sensor a window came from."
        : "Each of the other " + count(n.sensors.length - 1) + " sensors has a network with the same parts and " +
          "learned numbers of its own.") + " " + count(n.learned_each) + " learned numbers in all.</p></div>";
  }

  // The inside of an LSTM, the way it is usually drawn: the memory runs
  // along the top from one cycle to the next; this cycle's reading and the
  // hidden state feed four gates from below; the forget gate scales the
  // memory, the input gate and the candidate add to it, and the output gate
  // decides how much of it, squeezed, becomes the new hidden state.
  var LSTM_DRAWING = '<svg viewBox="0 0 620 272" role="img" aria-label="The inside of an LSTM: the memory along the top, ' +
    'four gates fed by the reading and the hidden state, and the new hidden state on the right">' +
    '<g class="pg-draw-line"><path d="M16 48H604"/><path d="M16 214H500"/><path d="M60 262V214"/>' +
    '<path d="M230 214V170M320 214V170M410 214V170M500 214V170"/><path d="M230 140V60"/>' +
    '<path d="M320 140 L356 108M410 140 L374 108"/><path d="M365 88V60"/>' +
    '<path d="M560 48V88M560 112V178"/><path d="M540 158 L552 181"/><path d="M560 202V216"/></g>' +
    '<g class="pg-draw-box"><rect x="190" y="140" width="80" height="30" rx="6"/><rect x="280" y="140" width="80" ' +
    'height="30" rx="6"/><rect x="370" y="140" width="80" height="30" rx="6"/><rect x="460" y="140" width="80" height="30" ' +
    'rx="6"/><rect x="524" y="88" width="72" height="24" rx="6"/></g>' +
    '<g class="pg-draw-op"><circle cx="230" cy="48" r="12"/><circle cx="365" cy="48" r="12"/><circle cx="365" cy="100" ' +
    'r="12"/><circle cx="560" cy="190" r="12"/></g>' +
    '<g class="pg-draw-sign"><text x="230" y="53">×</text><text x="365" y="53">+</text><text x="365" y="105">×</text>' +
    '<text x="560" y="195">×</text></g>' +
    '<g class="pg-draw-name"><text x="230" y="160">forget</text><text x="320" y="160">input</text>' +
    '<text x="410" y="160">candidate</text><text x="500" y="160">output</text><text x="560" y="105">squeeze</text></g>' +
    '<g class="pg-draw-note"><text x="16" y="34">memory, from the cycle before</text>' +
    '<text x="604" y="34" text-anchor="end">memory, to the next cycle</text>' +
    '<text x="100" y="234">hidden state, from the cycle before</text>' +
    '<text x="68" y="266">the reading at this cycle</text>' +
    '<text x="560" y="234" text-anchor="middle">new hidden state,</text>' +
    '<text x="560" y="252" text-anchor="middle">to the next cycle</text></g></svg>';

  // The inside of a GRU: one list, the hidden state, runs along the top; the
  // reset gate decides how much of it the candidate is made from, and the
  // update gate mixes the old hidden state with the candidate.
  var GRU_DRAWING = '<svg viewBox="0 0 620 272" role="img" aria-label="The inside of a GRU: the hidden state along the top, ' +
    'three gates fed by the reading and the hidden state, and the mix that makes the new hidden state">' +
    '<g class="pg-draw-line"><path d="M16 48H604"/><path d="M16 214H450"/><path d="M60 262V214"/>' +
    '<path d="M230 214V170M340 214V170M450 214V170"/><path d="M230 140V120H430V140"/><path d="M340 140 L465 58"/>' +
    '<path d="M450 140V92L468 62"/></g>' +
    '<g class="pg-draw-box"><rect x="190" y="140" width="80" height="30" rx="6"/><rect x="300" y="140" width="80" ' +
    'height="30" rx="6"/><rect x="410" y="140" width="80" height="30" rx="6"/></g>' +
    '<g class="pg-draw-op"><circle cx="480" cy="48" r="18"/></g>' +
    '<g class="pg-draw-sign"><text x="480" y="53">mix</text></g>' +
    '<g class="pg-draw-name"><text x="230" y="160">reset</text><text x="340" y="160">update</text>' +
    '<text x="450" y="160">candidate</text></g>' +
    '<g class="pg-draw-note"><text x="16" y="34">hidden state, from the cycle before</text>' +
    '<text x="604" y="34" text-anchor="end">hidden state, to the next cycle</text>' +
    '<text x="100" y="234">hidden state, from the cycle before</text>' +
    '<text x="68" y="266">the reading at this cycle</text></g></svg>';

  // A value a setting decides leads to that setting, as the boxes of the
  // overview lead to their steps. A setting the form hides for these
  // settings leads to the one that decides it.
  function valueHtml(part) {
    var field = part.setting && S.fields[part.setting];
    if (!field) { return '<span class="pg-vstep-value">' + esc(part.text) + "</span>"; }
    var group = field.group === "procedure" ? "training" : field.group;
    var row = part.setting === "validation_fraction" && !visible(part.setting) ? "procedure" : part.setting;
    var index = part.index === null || part.index === undefined ? "" : ' data-goto-index="' + part.index + '"';
    // Read out on its own, "at 125 cycles" says nothing; its name also says
    // which setting it leads to, and keeps the words shown, as a name must.
    var dest = row === "procedure" ? "Procedure" : splitUnit(S.fields[row].label).label;
    return '<button type="button" class="pg-vstep-value" data-goto="' + group + '" data-goto-row="' + row + '"' +
      index + ' aria-label="' + esc(part.text + ", " + dest + " setting") + '">' + esc(part.text) + "</button>";
  }

  // A box's flowchart opens in the pipeline, and its step's tab in the form,
  // so the step's settings are beside the drawing of what they decide.
  function zoomTo(group) {
    S.zoom = group;
    S.inside = null;
    setStep(group);
    paintPipeline();
    requestZoom(true);
    var back = $(".pg-zoom-back");
    if (back) { back.focus(); }
  }

  // One of the temporal stage's networks opened from inside, in the
  // pipeline's place; the way back leads to the stage's flowchart.
  function openInside(sensor) {
    S.inside = sensor;
    $("#pg-pipeline").__zoom = null;
    paintPipeline();
    var back = $(".pg-zoom-back");
    if (back) { back.focus(); }
  }

  // Back one level: from a network to the stage's flowchart, with the focus
  // on that network's button; from a flowchart to the overview, with the
  // focus on the box that was open, or, when that box has since been taken
  // out, on the + that would put it back.
  function unzoom() {
    if (insideOpen()) {
      var sensor = S.inside;
      S.inside = null;
      $("#pg-pipeline").__zoom = null;
      paintPipeline();
      var button = $('[data-inside="' + sensor + '"]');
      if (button) { button.focus(); }
      return;
    }
    S.inside = null;
    var was = S.zoom;
    clearTimeout(S.zoomTimer);
    S.zoom = null;
    paintPipeline();
    var box = $('button[data-zoom="' + was + '"]') || $('[data-action="add-part"][data-part="' + was + '"]');
    if (box) { box.focus(); }
  }

  // Ask the server for the open step's flowchart, for the settings on screen.
  function requestZoom(now) {
    if (!S.zoom) { return; }
    clearTimeout(S.zoomTimer);
    S.zoomTimer = setTimeout(function () {
      if (!S.zoom) { return; }  // the flowchart was left while this waited
      var asked = zoomKey(), stage = S.zoom;
      post("/api/playground/stage", { stage: stage, settings: S.settings }).then(function (answer) {
        if (asked !== zoomKey()) { return; }  // a newer request is on its way
        answer.asked = asked;
        S.zoomed = answer;
        paintZoom();
      }).catch(function (error) {
        if (asked !== zoomKey()) { return; }
        S.zoomed = { stage: stage, asked: asked, error: error.message };
        paintZoom();
      });
    }, now ? 0 : 220);
  }

  // A new answer replaces the steps. A button the keyboard was on is
  // replaced with them, so the focus moves to its counterpart: the same
  // action on the same part, the + that replaced a ×, or else the button in
  // the same place.
  function paintZoom() {
    var body = $("#pg-zoom-body");
    if (body) {
      var active = document.activeElement;
      var was = active && body.contains(active) ? active : null;
      var place = was ? Array.prototype.indexOf.call(body.querySelectorAll("button"), was) : -1;
      body.innerHTML = zoomBody();
      $("#pg-pipeline").__html = null;
      if (was) {
        var part = was.getAttribute("data-part");
        var again = (part && body.querySelector('[data-part="' + part + '"]')) || body.querySelectorAll("button")[place];
        if (again) { again.focus({ preventScroll: true }); }
      }
    }
    paintStatus();
  }

  // "Updating…" beside the title while the steps shown are for older settings.
  function paintStatus() {
    var status = $("#pg-zoom-updating");
    var d = S.zoomed;
    if (status) { status.textContent = d && d.stage === S.zoom && d.asked !== zoomKey() ? "Updating…" : ""; }
  }

  // The arrow between two boxes: a drawing, hidden from screen readers, unless
  // it carries the + of a part taken out from between them.
  function flowLink(inner) {
    return inner ? '<div class="pg-flow-link pg-flow-link--gap" role="listitem">' + inner + "</div>"
      : '<div class="pg-flow-link" aria-hidden="true"></div>';
  }

  // The sensor graph is taken out from its own step in the graph layers'
  // flowchart: × joins every sensor to every other, the setting of "w/o
  // Physics prior", and + puts back the graph that was there. The button
  // follows the settings on screen, not the answer being shown, so it is
  // right even while an older answer waits to be replaced.
  function partControl(step) {
    var part = (step.parts || []).filter(function (p) { return p.setting === "graph"; })[0];
    if (S.zoom !== "graph" || !part) { return ""; }
    return '<span class="pg-vstep-part" data-removed="' + (S.settings.graph === REMOVED.graph) + '">' + partButton() + "</span>";
  }

  function partButton() {
    return S.settings.graph === REMOVED.graph ? addButton("graph") : removeButton("graph", "pg-stage-x");
  }

  // What the pipeline calls a part with a given value: "LSTM", "Physics prior".
  function partName(part, value) {
    return part === "temporal" ? capitalise(temporalName(value)) : (PRIOR[value] || choiceLabel(part, value));
  }

  function removeButton(part, className) {
    var label = "Remove " + partName(part, S.settings[part]);
    return '<button type="button" class="' + className + '" data-action="remove-part" data-part="' + part +
      '" aria-label="' + esc(label) + '" title="' + esc(label) + '">' + CROSS + "</button>";
  }

  function addButton(part) {
    var name = partName(part, restoreValue(part));
    return '<button type="button" class="pg-add" data-action="add-part" data-part="' + part +
      '" aria-label="Add ' + esc(name) + '">' + PLUS + esc(name) + "</button>";
  }

  // What + puts back: the part as it was taken out, or its default.
  function restoreValue(part) {
    return S.restore[part] || S.fields[part].default;
  }

  // A part's setting moved from before to its value now. When that took the
  // part out, by the cross or in the form, what it was is kept for +.
  function noteRemoval(key, before) {
    if (REMOVED[key] !== undefined && S.settings[key] === REMOVED[key] && before !== REMOVED[key]) {
      S.restore[key] = before;
    }
  }

  // The pipeline after settings moved. Its markup is written again only
  // when it has changed: a change to a number field is reported as the
  // field loses the focus, which a press of the mouse causes, so rewriting
  // here would replace the very button being clicked and lose the click.
  // With a step's flowchart open, only what the settings decide on screen
  // is updated in place: the title, the grey of an answer for older
  // settings, and the sensor graph's cross or +. The steps themselves are
  // replaced when the new answer arrives.
  function paintPipeline() {
    var box = $("#pg-pipeline");
    if (!box) { return; }
    var body = $("#pg-zoom-body");
    if (S.zoom && body && box.__zoom === S.zoom) {
      var title = $("#pg-pipeline-title");
      if (title && title.getAttribute("data-name") !== titleKey()) {
        title.innerHTML = zoomTitle();
        title.setAttribute("data-name", titleKey());
      }
      var steps = body.querySelector(".pg-zoom-steps");
      if (steps) { steps.classList.toggle("is-stale", !!S.zoomed && S.zoomed.asked !== zoomKey()); }
      var removed = String(S.settings.graph === REMOVED.graph);
      $all(".pg-vstep-part").forEach(function (slot) {
        if (slot.getAttribute("data-removed") !== removed) {
          slot.innerHTML = partButton();
          slot.setAttribute("data-removed", removed);
        }
      });
    } else {
      var html = pipelineInner();
      if (box.__html !== html) { box.innerHTML = html; box.__html = html; }
      box.__zoom = S.zoom;
    }
    fitFlow();
    paintStatus();
  }

  // Beside the form the overview always stands in a column, top to bottom.
  // Above it, the overview stands in a row when its boxes fit, however long
  // their names; else it first drops the output's words, then stands in a
  // column. Measured, not guessed from the width, and again whenever the
  // width moves. Only the column beside the form is pinned (sticky).
  function fitFlow() {
    var box = $("#pg-pipeline"), flow = $("#pg-flow");
    if (!box) { return; }
    box.classList.remove("is-tight", "is-stacked");
    if (!flow) { return; }
    if (getComputedStyle(box).position === "sticky") { box.classList.add("is-stacked"); }
    if (flow.scrollWidth > flow.clientWidth + 1) { box.classList.add("is-tight"); }
    if (flow.scrollWidth > flow.clientWidth + 1) { box.classList.add("is-stacked"); }
    if (!box.__watched && window.ResizeObserver) {
      box.__watched = true;
      var width = box.clientWidth;
      new ResizeObserver(function () {
        if (box.clientWidth !== width) { width = box.clientWidth; fitFlow(); }
      }).observe(box);
    }
  }

  // A value in a flowchart was chosen: its step's tab opens and the setting
  // comes into view, marked for a moment so the eye finds it. The focus
  // moves to its control, so the keyboard carries on from there.
  function goTo(group, rowKey, index) {
    setStep(group);
    var target = rowKey && $('.pg-tabpanel[data-group="' + group + '"] [data-row="' + rowKey + '"]:not([hidden])');
    if (!target) { return; }
    target.scrollIntoView({ block: "nearest" });
    var focus = (index !== null && index !== undefined && target.querySelector('input[data-index="' + index + '"]')) ||
      target.querySelector("input:checked") || target.querySelector("input, select");
    if (focus) { focus.focus({ preventScroll: true }); }
    // Taking the class off and reading the layout restarts the mark when the
    // same box is chosen again while it still shows.
    target.classList.remove("is-target");
    void target.offsetWidth;
    target.classList.add("is-target");
  }

  function removePart(part) {
    var before = S.settings[part];
    S.settings[part] = REMOVED[part];
    noteRemoval(part, before);
    setPart(part, REMOVED[part], '[data-action="add-part"][data-part="' + part + '"]');
  }

  function addPart(part) {
    setPart(part, restoreValue(part), '[data-action="remove-part"][data-part="' + part + '"]');
  }

  // A part taken out or put back: its setting changes, its control in the
  // form follows, and the focus lands on what replaced the button pressed.
  function setPart(part, value, next) {
    S.settings[part] = value;
    syncControl(part);
    settingsMoved([part]);
    var button = $(next);
    if (button) { button.focus(); }
  }

  // ---------------------------------------------------------------- groups

  // The first choice in Training: the procedure, drawn as one setting though
  // it sets two, with the inputs only it controls set in under its options.
  // It is the only control for holding engines back, and moves what the LSTM
  // is worth (F-16).
  // named: the id of the group's name that names the procedure in its place.
  function procedureHtml(named) {
    var choices = S.options.procedures.map(function (p) {
      return { value: p.key, label: p.label, about: p.about, source: p.source, more: p.more };
    });
    return cell({
      key: "procedure", label: named ? "" : "Procedure", size: "wide",
      hint: named ? "" : infoCard(S.options.procedure_about, "", S.options.procedure_more),
      control: optionList("pg-procedure", choices, chosenProcedure().key, "data-procedure", optionNote,
        named || nameIds("procedure")) +
        '<div class="pg-fields pg-fields--sub">' +
        fieldsOf("procedure").filter(function (f) { return f.kind !== "bool"; })
          .map(function (f) { return fieldRow(f); }).join("") + "</div>",
      reset: procedureReset()
    });
  }

  function procedureReset() {
    var back = defaultProcedure();
    return '<span class="pg-reset-slot">' + (chosenProcedure() !== back || !followsProcedure()
      ? '<button type="button" class="pg-reset" data-reset="procedure" title="Back to ' + esc(back.label) +
        '" aria-label="Reset Procedure">Reset</button>'
      : "") + "</span>";
  }

  // The words behind a setting's i: what it is for and, for the sub-dataset,
  // what Table 5 says the chosen one holds; then where its default comes from.
  function hintFor(f) {
    var about = [f.about];
    if (f.key === "subset") {
      var t5 = (S.options.dataset || {})[S.settings.subset];
      if (t5) {
        about.push(S.settings.subset + ": " + count(t5.train_engines) + " training and " + count(t5.test_engines) +
          " test engines, " + plural(t5.conditions, "operating condition") + ", " + plural(t5.fault_modes, "fault mode") + ".");
      }
    }
    return infoCard(about.filter(Boolean).join("\n"), sourceNote(f.source), f.more);
  }

  // The words behind one option's i, when the option has any of its own.
  function optionNote(c) { return c.about || c.source ? infoCard(c.about, sourceNote(c.source), c.more) : ""; }

  // named: the id of the group's name that names this setting in its place.
  function fieldRow(f, named) {
    var value = S.settings[f.key];
    var id = "pg-" + f.key;
    var data = 'data-field="' + f.key + '"';
    var parts = splitUnit(f.label);
    // A list of options whose name is its step's name is labelled by the
    // step's tab, and one alone in a group by the group's name.
    var byTab = f.kind === "choice" && !SEGMENTED[f.key] && f.label === groupLabel(f.group);
    var bare = byTab || !!named;
    var size = "", control;
    if (f.kind === "bool") {
      control = switchControl(id, !!value, data);
    } else if (f.kind === "choice" && SEGMENTED[f.key]) {
      control = segmented(id, f.choices, value, data, named || nameIds(f.key));
      id = null;
    } else if (f.kind === "choice" && MENU[f.key]) {
      control = selectField(id, f.choices, value, data);
    } else if (f.kind === "choice") {
      size = "wide";
      control = optionList(id, f.choices, value, data, optionNote,
        named || (byTab ? "pg-tab-name-" + f.group : nameIds(f.key)));
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
    // A list its step's tab names has no name of its own, so its i, saying
    // what the step is for, stands alone above its options; a list a group
    // names has its i beside the group's name.
    return cell({
      key: f.key, id: id, label: bare ? "" : parts.label, infoName: f.label, hidden: !visible(f.key),
      size: f.group === "data" ? "wide" : size, inline: f.kind === "bool",
      hint: named ? "" : hintFor(f),
      control: control,
      reset: resetHtml(f)
    });
  }

  function resetHtml(f) {
    return '<span class="pg-reset-slot">' + (isChanged(f)
      ? '<button type="button" class="pg-reset" data-reset="' + f.key + '" title="Back to ' + esc(defaultText(f)) +
        '" aria-label="' + esc("Reset " + splitUnit(f.label).label) + '">Reset</button>'
      : "") + "</span>";
  }

  function defaultText(f) {
    if (f.kind === "choice") { return choiceLabel(f.key, f.default); }
    if (f.kind === "bool") { return f.default ? "on" : "off"; }
    if (f.kind === "widths") { return f.default.join(", "); }
    return String(f.default);
  }

  // Choosing a procedure sets both things it decides, and the share held back
  // comes back as the procedure's own, so that input is set to match.
  function chooseProcedure(key) {
    if (chosenProcedure().key === key && followsProcedure()) { return; }
    var preset = S.options.procedures.filter(function (p) { return p.key === key; })[0];
    var keys = Object.keys(preset.settings);
    keys.forEach(function (k) { S.settings[k] = preset.settings[k]; syncControl(k); });
    settingsMoved(keys);
  }

  // Resetting the procedure puts back both settings it decides. The
  // controls are set to match and the rest follows as for any other change,
  // so the page does not move; the focus goes to the setting's control,
  // since the Reset that was pressed is gone.
  function resetField(key) {
    var keys = key === "procedure" ? ["validation_fraction", "early_stopping"] : [key];
    keys.forEach(function (k) {
      S.settings[k] = copy(S.fields[k].default);
      syncControl(k);
    });
    settingsMoved(keys);
    var box = $('[data-row="' + key + '"]');
    var control = box && (box.querySelector("input:checked") || box.querySelector("input, select"));
    if (control) { control.focus({ preventScroll: true }); }
  }

  // A control changed: the setting is read from it, then what depends on it is redrawn.
  function fieldChanged(input) {
    var key = input.getAttribute("data-field");
    var f = S.fields[key];
    var before = S.settings[key];
    if (f.kind === "bool") { S.settings[key] = input.checked; }
    else if (f.kind === "choice") { S.settings[key] = input.value; }
    else if (f.kind === "widths") {
      var widths = S.settings[key].slice();
      widths[+input.getAttribute("data-index")] = input.value === "" ? input.value : Number(input.value);
      S.settings[key] = widths;
    } else { S.settings[key] = input.value === "" ? input.value : Number(input.value); }
    noteRemoval(key, before);
    settingsMoved([key]);
  }

  // A setting changed from somewhere other than its own control, the
  // pipeline or the procedure: the control is set to match it.
  function syncControl(key) {
    var value = S.settings[key];
    $all('[data-field="' + key + '"]').forEach(function (input) {
      if (input.type === "radio") { input.checked = input.value === String(value); }
      else if (input.type === "checkbox") { input.checked = !!value; }
      else if (input.hasAttribute("data-index")) { input.value = value[+input.getAttribute("data-index")]; }
      else { input.value = value; }
    });
  }

  // Settings moved, from a control or from the pipeline. Only what depends
  // on them is redrawn: their rows, the rows they show or hide, the
  // procedure, the pipeline and the step's flowchart, the tabs' dots and the
  // run card. The controls themselves are left alone, so a number being
  // typed keeps its focus.
  function settingsMoved(keys) {
    saveSettings();
    keys.forEach(paintRow);
    S.options.fields.forEach(function (other) {
      var box = $('[data-row="' + other.key + '"]');
      if (box) { box.hidden = !visible(other.key); }
    });
    paintProcedure();
    paintSections();
    paintPipeline();
    requestZoom(false);
    paintTabs();
    refreshPreview();
  }

  function paintRow(key) {
    var f = S.fields[key];
    var box = $('[data-row="' + key + '"]');
    if (!box) { return; }
    paintSlot(box, isChanged(f), resetHtml(f));
    Array.prototype.forEach.call(box.querySelectorAll(".opt"), function (option) {
      option.classList.toggle("is-active", option.querySelector("input").checked);
    });
    var words = box.querySelector(".pg-field-name .pg-info-card");
    var hint = hintFor(f);
    if (words && words.getAttribute("data-hint") !== hint) {
      words.innerHTML = hint.replace(/<a /g, '<a tabindex="-1" ');
      words.setAttribute("data-hint", hint);
    }
  }

  // A setting's Reset slot, written again only when Reset should appear or
  // go: rewriting it at every change would replace a Reset being clicked.
  function paintSlot(box, changed, html) {
    var slot = box.querySelector(".pg-reset-slot");
    if (slot && !!slot.firstChild !== changed) { slot.outerHTML = html; }
  }

  function paintProcedure() {
    var box = $('[data-row="procedure"]');
    if (!box) { return; }
    var chosen = chosenProcedure().key;
    Array.prototype.forEach.call(box.querySelectorAll(".opt"), function (option) {
      var input = option.querySelector("input");
      input.checked = input.value === chosen;
      option.classList.toggle("is-active", input.checked);
    });
    paintSlot(box, chosenProcedure() !== defaultProcedure() || !followsProcedure(), procedureReset());
  }

  function refreshPreview() {
    clearTimeout(S.previewTimer);
    S.previewTimer = setTimeout(function () {
      var box = $("#pg-run");
      if (!box) { return; }
      var asked = JSON.stringify(S.settings);
      post("/api/playground/preview", S.settings).then(function (preview) {
        if (asked !== JSON.stringify(S.settings)) { return; }  // a newer request is on its way
        drawRunCard(box, preview, null);
      }).catch(function (error) {
        if (asked !== JSON.stringify(S.settings)) { return; }
        drawRunCard(box, null, error.message);
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

  // The card that ends the form: the network these settings build, then only
  // what stops the run or qualifies it, then the one action, so each caution
  // sits directly above the button it is about.
  function drawRunCard(box, preview, error) {
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
        '<details class="pg-details" data-layers' + (S.layersOpen ? " open" : "") + "><summary>Layer by layer</summary>" +
        '<table class="pg-mini"><thead><tr><th>Layer</th><th>Out</th><th class="num">Learned numbers</th></tr></thead><tbody>' +
        table4Rows(preview.layers).map(function (l) {
          return "<tr><td>" + esc(l.layer) + (l.from ? ' <span class="pg-faint">from ' + esc(l.from) + "</span>" : "") +
            "</td><td><code>" + esc(l.out) + '</code></td><td class="num">' + count(Number(l.parameters)) + "</td></tr>";
        }).join("") + "</tbody></table>" +
        '<p class="pg-faint">B is the number of windows in a batch. Table 4 prints no counts; each is read from the built model.</p>' +
        "</details></div>";
    }

    // The card is drawn again after every change of settings. What the
    // keyboard was on in it keeps the focus, and "Layer by layer" stays as
    // the reader left it, rather than closing at every change.
    var active = box.contains(document.activeElement) ? document.activeElement : null;
    var had = active && (active.matches("summary") ? "summary" : active.matches(".pg-train") ? ".pg-train" : null);
    box.innerHTML = network + caveatList(notes) + '<div class="pg-run-action">' +
      '<button type="button" class="button button--primary pg-train" data-action="run"' + (canRun ? "" : " disabled") +
      ">Train " + plural(S.settings.seeds, "seed") + "</button></div>";
    var layers = box.querySelector("details[data-layers]");
    if (layers) { layers.addEventListener("toggle", function () { S.layersOpen = layers.open; }); }
    var again = had && box.querySelector(had);
    if (again && !again.disabled) { again.focus({ preventScroll: true }); }
    else if (active) { box.setAttribute("tabindex", "-1"); box.focus({ preventScroll: true }); }
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
  // One tab for everything that has been run, with a tab inside it for the
  // training runs and one for the explanations. A job that has not finished,
  // or that failed or was stopped, is a row of its own table with its state,
  // at the top; a finished one is its stored run or explanation. Ten rows a
  // page, as wide as the page, filtered by what the runs are.

  var DIRTY = "-dirty: uncommitted changes, so no committed state reproduces this run";
  var PAGE_ROWS = 10;
  var RESULTS_TABS = [["runs", "Training runs", "train"], ["explanations", "Explanations", "explain"]];

  function loadRuns() {
    if (S.runs) { return Promise.resolve(S.runs); }
    return T.get("/api/playground/runs").then(function (data) { S.runs = data.runs; return S.runs; });
  }

  function loadExplanations() {
    if (S.explanations) { return Promise.resolve(S.explanations); }
    return T.get("/api/playground/explanations").then(function (data) {
      S.explanations = data.explanations;
      return S.explanations;
    });
  }

  function setResultsTab(which) {
    S.resultsTab = which;
    S.selected = null;
    drawResults();
    var tab = $('[data-results-tab="' + which + '"]');
    if (tab) { tab.focus(); }
  }

  // The kind of job a tab of Results lists.
  function tabKind(which) { return RESULTS_TABS.filter(function (t) { return t[0] === which; })[0][2]; }

  // The list, or, once a run is chosen, that run on its own with a way back.
  function drawResults() {
    var panel = $("#pg-panel");
    if (S.selected) {
      panel.innerHTML = '<div class="loading">Reading reports/…</div>';
      loadRuns().then(function () { if (S.tab === "results") { drawDetail(); } })
        .catch(function (error) { panel.innerHTML = errorBox(error.message); });
      return;
    }
    panel.innerHTML = '<div class="pg-results">' + resultsTabsHtml() +
      '<div class="pg-tabpanel-results" role="tabpanel" aria-labelledby="pg-results-tab-' + S.resultsTab + '" ' +
      'id="pg-results-body"><div class="loading">Reading reports/…</div></div></div>';
    (S.resultsTab === "explanations" ? loadExplanations() : loadRuns()).then(function () {
      if (S.tab === "results" && !S.selected) { drawResultsBody(); }
    }).catch(function (error) {
      var body = $("#pg-results-body");
      if (body) { body.innerHTML = errorBox(error.message); }
    });
  }

  // The two tabs, each with the number of its jobs running or waiting.
  function resultsTabsHtml() {
    return '<div class="pg-subtabs" role="tablist" aria-label="What was run">' + RESULTS_TABS.map(function (t) {
      var on = S.resultsTab === t[0];
      var live = S.jobs.filter(function (job) { return job.kind === t[2] && isLive(job); }).length;
      return '<button type="button" role="tab" class="pg-subtab" id="pg-results-tab-' + t[0] + '" data-results-tab="' +
        t[0] + '" aria-selected="' + on + '" tabindex="' + (on ? 0 : -1) + '">' + t[1] +
        (live ? ' <span class="pg-jobs-badge" title="' + esc(plural(live, "job") + " running or waiting") + '">' +
          count(live) + "</span>" : "") + "</button>";
    }).join("") + "</div>";
  }

  // The rows of the open tab after its filters: its jobs that are not done,
  // what is running first, then its stored runs or explanations.
  function resultRows() {
    var which = S.resultsTab, kind = tabKind(which);
    var f = which === "runs" ? S.filter : S.xfilter;
    var jobs = orderedJobs().filter(function (job) {
      if (job.kind !== kind || job.status === "done") { return false; }
      var state = isLive(job) ? "live" : "ended";
      return f.state === "any" || f.state === state;
    }).map(function (job) { return { job: job }; });
    var stored = f.state === "any" || f.state === "done" ? (which === "runs" ? S.runs : S.explanations) || [] : [];
    stored = stored.filter(which === "runs" ? runShown : explanationShown).map(function (item) { return { item: item }; });
    return jobs.concat(stored);
  }

  function runShown(r) {
    var f = S.filter;
    return (f.subset === "all" || r.subset === f.subset) && (f.temporal === "all" || r.temporal === f.temporal) &&
      (f.graph === "all" || r.graph === f.graph) &&
      (f.pass === "all" || (f.pass === "final") === (r.reported_weights === "final_epoch")) &&
      (!f.results || !r.exploratory) && (!f.files || r.has_weights);
  }

  function explanationShown(x) {
    var f = S.xfilter;
    return (f.method === "all" || x.method === f.method) && (f.scope === "all" || x.scope === f.scope) &&
      (f.subset === "all" || x.run_subset === f.subset);
  }

  // The filters, the comparison of two ticked runs, one page of rows and the
  // way to the next.
  function drawResultsBody() {
    var body = $("#pg-results-body");
    if (!body) { return; }
    var which = S.resultsTab;
    var rows = resultRows();
    var pages = Math.max(1, Math.ceil(rows.length / PAGE_ROWS));
    S.page[which] = Math.min(S.page[which] || 0, pages - 1);
    var first = S.page[which] * PAGE_ROWS;
    var shown = rows.slice(first, first + PAGE_ROWS);
    var none = which === "runs" ? "No run matches these filters." : "No explanation matches these filters.";
    body.innerHTML = '<div class="pg-filters">' + (which === "runs" ? runFilters() : explanationFilters()) + "</div>" +
      (which === "runs" ? '<div id="pg-compare"></div>' : "") +
      (shown.length ? (which === "runs" ? runsTable(shown) : explanationsList(shown)) +
        pagerHtml(first, shown.length, rows.length) : '<div class="pg-empty">' + none + "</div>");
    S.jobRows = jobSignature();
    if (which === "runs") { drawComparison(); }
    paintBanner();
  }

  // A filter: a name above a menu of choices, "All" first.
  function filterSelect(name, key, attr, choices, current) {
    var id = "pg-" + attr + "-" + key;
    return '<label class="pg-filter" for="' + id + '"><span>' + name + "</span>" +
      selectField(id, [{ value: "all", label: "All" }].concat(choices), current, attr + '="' + key + '"') + "</label>";
  }

  // The values a list holds for one key, in the words of the form.
  function present(list, key, words) {
    var seen = {};
    list.forEach(function (item) { if (item[key] !== null && item[key] !== undefined) { seen[item[key]] = true; } });
    return Object.keys(seen).sort().map(function (v) { return { value: v, label: words(v) }; });
  }

  function stateFilter(attr, current) {
    return filterSelect("State", "state", attr, [{ value: "done", label: "Finished" },
      { value: "live", label: "Running or waiting" }, { value: "ended", label: "Failed or stopped" }], current);
  }

  function runFilters() {
    var f = S.filter, runs = S.runs || [];
    var graphWords = function (v) { return (S.fields.graph.choices.filter(function (c) { return c.value === v; })[0] || { label: v }).label; };
    return filterSelect("Sub-dataset", "subset", "data-filter", SUBSETS.map(function (s) { return { value: s, label: s }; }), f.subset) +
      filterSelect("Temporal stage", "temporal", "data-filter", present(runs, "temporal", temporalName), f.temporal) +
      filterSelect("Sensor graph", "graph", "data-filter", present(runs, "graph", graphWords), f.graph) +
      filterSelect("Weights from", "pass", "data-filter", [{ value: "best", label: "The best pass" },
        { value: "final", label: "The final pass" }], f.pass) +
      stateFilter("data-filter", f.state) +
      '<label class="pg-filter pg-filter--switch">' + switchControl("pg-f-results", f.results, 'data-filter="results"') +
      "Five seeds or more</label>" +
      '<label class="pg-filter pg-filter--switch">' + switchControl("pg-f-files", f.files, 'data-filter="files"') +
      "Kept their weights</label>";
  }

  function explanationFilters() {
    var f = S.xfilter;
    return filterSelect("Method", "method", "data-xfilter", [{ value: "shap", label: METHODS.shap },
      { value: "occlusion", label: METHODS.occlusion }], f.method) +
      filterSelect("What", "scope", "data-xfilter", [{ value: "engine", label: "One engine, at every cycle" },
        { value: "test", label: "The last window of every test engine" }], f.scope) +
      filterSelect("Sub-dataset", "subset", "data-xfilter", SUBSETS.map(function (s) { return { value: s, label: s }; }), f.subset) +
      stateFilter("data-xfilter", f.state);
  }

  // "11–20 of 34", between the ways to the page before and the page after.
  function pagerHtml(first, shown, total) {
    if (total <= PAGE_ROWS) { return ""; }
    var which = S.resultsTab;
    return '<nav class="pg-pager" aria-label="Pages">' +
      '<button type="button" class="button button--small" data-page="prev"' + (first === 0 ? " disabled" : "") + ">" +
      BACK + "Previous</button><span>" + count(first + 1) + "–" + count(first + shown) + " of " + count(total) + "</span>" +
      '<button type="button" class="button button--small" data-page="next"' + (first + shown >= total ? " disabled" : "") +
      ">Next" + NEXT + "</button></nav>";
  }

  function turnPage(way) {
    S.page[S.resultsTab] = (S.page[S.resultsTab] || 0) + (way === "next" ? 1 : -1);
    drawResultsBody();
    var list = $("#pg-results-body");
    if (list && list.getBoundingClientRect().top < 0) { list.scrollIntoView({ block: "start" }); }
  }

  // A job's row: its state, its name, its bar and what is happening now, or
  // its last lines when it failed, and what can be done with it.
  function jobRow(job, columns) {
    var line = jobLine(job);
    var forget = !isLive(job)
      ? '<button type="button" class="button button--small button--quiet" data-action="forget" data-id="' + esc(job.id) + '">Remove</button>' : "";
    return '<tr class="pg-jobrow pg-job--' + job.status + '" data-job="' + esc(job.id) + '"><td>' + statusPill(job) + "</td>" +
      '<td colspan="' + (columns - 2) + '"><div class="pg-job-head"><strong>' + esc(job.label) + "</strong>" +
      (job.elapsed ? '<span class="pg-faint">' + duration(job.elapsed) + "</span>" : "") + "</div>" +
      '<div class="pg-job-live">' + jobBar(job) + (line ? '<div class="pg-job-line">' + esc(line) + "</div>" : "") + "</div>" +
      (job.log ? '<details class="pg-details"><summary>Last lines it printed</summary><pre class="pg-log">' + esc(job.log) +
        "</pre></details>" : "") +
      '</td><td class="pg-jobrow-actions">' + stopButton(job) + forget + "</td></tr>";
  }

  // What the job rows show, to tell a poll that changes only a bar from one
  // that adds, removes or ends a job.
  function jobSignature() {
    return S.jobs.map(function (job) { return job.id + ":" + job.status + ":" + (job.log ? 1 : 0); }).join(",");
  }

  // After a poll, with Results open: the rows are drawn again when a job
  // came, went or changed state; otherwise only their bars and lines move.
  function paintResultJobs() {
    if (S.tab !== "results" || S.selected || !$("#pg-results-body")) { return; }
    var tabs = $(".pg-subtabs");
    if (tabs) { tabs.outerHTML = resultsTabsHtml(); }
    if (S.jobRows !== jobSignature()) { drawResultsBody(); return; }
    S.jobs.forEach(function (job) {
      var live = $('tr[data-job="' + job.id + '"] .pg-job-live');
      if (live) {
        var line = jobLine(job);
        live.innerHTML = jobBar(job) + (line ? '<div class="pg-job-line">' + esc(line) + "</div>" : "");
      }
    });
  }

  function openRun(name) {
    S.selected = name; S.detail = null;
    drawDetail();
  }

  // One row per run. The sub-dataset has a column only while the list holds
  // more than one; ticked rows are marked, and a commit with uncommitted
  // changes is coloured, its meaning given once on the column's head.
  function runsTable(rows) {
    var subsets = {};
    rows.forEach(function (row) { if (row.item) { subsets[row.item.subset] = true; } });
    var showData = S.filter.subset === "all" && Object.keys(subsets).length > 1;
    var columns = 9 + (showData ? 1 : 0);
    // The result comes straight after when the run finished, so a phone
    // shows it without scrolling the table; the settings that produced it
    // follow, and the commit, which a narrow panel leaves out, comes last.
    // The whole first cell ticks the box, and a click there never opens the run.
    return '<div class="pg-runs"><table class="data-table"><thead><tr><th><span class="visually-hidden">Compare</span></th>' +
      '<th>Finished, UTC</th><th class="num">RMSE</th><th class="num">Paper</th>' + (showData ? "<th>Data</th>" : "") +
      '<th>Temporal stage</th><th>Graph</th><th>Seeds</th><th>Weights</th>' +
      '<th class="col-commit" title="' + esc(DIRTY) + '">Commit</th></tr></thead><tbody>' +
      rows.map(function (row) {
        if (row.job) { return jobRow(row.job, columns); }
        var r = row.item, m = r.mean || {}, sd = r.std || {};
        var ticked = S.compare.indexOf(r.name) >= 0;
        return '<tr data-run="' + esc(r.name) + '"' + (ticked ? ' class="is-selected"' : "") + '><td class="pg-tick-cell"><label>' +
          '<input type="checkbox" data-compare="' + esc(r.name) + '"' + (ticked ? " checked" : "") +
          ' aria-label="Compare this run"></label></td>' +
          '<td title="' + esc(when(r.finished)) + '">' + whenShort(r.finished) + "</td>" +
          '<td class="num">' + number(m.rmse, 3) + (sd.rmse !== null && sd.rmse !== undefined ? ' <span class="sd">± ' + number(sd.rmse, 3) + "</span>" : "") + "</td>" +
          '<td class="num" title="' + esc(r.paper_row ? r.paper_row.label + ", " + r.paper_row.source : r.no_row_reason) + '">' +
          (r.paper_row ? fixed(r.paper_row.rmse, 2) : "—") + "</td>" + (showData ? "<td>" + esc(r.subset) + "</td>" : "") +
          "<td>" + esc(stageName(r.temporal, r.temporal_hidden)) + "</td><td>" + esc(graphName(r.graph, r.adjacency_normalization)) + "</td>" +
          '<td><span class="chip' + (r.exploratory ? " chip--warn" : "") + '">' + count(r.seeds) + "</span></td>" +
          "<td>" + (r.reported_weights === "final_epoch" ? "final pass" : "best pass") +
          (r.has_weights ? ' <span class="chip chip--good">files</span>' : "") + "</td>" +
          '<td class="col-commit"><code class="pg-commit' + (r.dirty ? " pg-commit--dirty" : "") + '">' + esc(r.commit) + "</code></td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  // One row per explanation, newest first, with the run it explains; a row
  // opens the explanation in the Explain tab.
  function explanationsList(rows) {
    var columns = 7;
    return '<div class="pg-runs"><table class="data-table"><thead><tr><th>Finished, UTC</th><th>Method</th><th>What</th>' +
      '<th>Seed</th><th class="num">Windows</th><th>Data</th><th>Run</th></tr></thead><tbody>' +
      rows.map(function (row) {
        if (row.job) { return jobRow(row.job, columns); }
        var x = row.item;
        return '<tr data-action="open-explanation" data-run="' + esc(x.run) + '" data-name="' + esc(x.name) + '" tabindex="0">' +
          '<td title="' + esc(when(x.finished_utc)) + '">' + whenShort(x.finished_utc) + "</td><td>" +
          esc(METHODS[x.method] || x.method) + "</td><td>" + esc(explainedWhat(x)) + "</td><td>" + esc(x.seed) + "</td>" +
          '<td class="num">' + count(x.windows) + "</td><td>" + esc(x.run_subset || "") + "</td><td>" +
          esc([stageName(x.run_temporal, x.run_temporal_hidden), graphName(x.run_graph)].filter(Boolean).join(" · ")) +
          ' <span class="pg-faint">' + esc(whenShort(x.run_finished)) + "</span></td></tr>";
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

  // A date is never broken at its hyphens; the time may still move to the
  // next line.
  function keepDate(text) {
    return esc(text).replace(/^(\d{4}-\d\d-\d\d)/, '<span class="pg-nowrap">$1</span>');
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
          '<table class="pg-mini"><thead><tr><th>Setting that differs</th><th>First, ' + keepDate(runTime(c.first.name)) +
          "</th><th>Second, " + keepDate(runTime(c.second.name)) + "</th></tr></thead><tbody>" +
          (c.differences.length ? c.differences.map(function (d) {
            var whole = function (v) {
              var words = esc(settingWords(d.setting, v));
              return d.setting === "commit" ? '<span class="pg-nowrap">' + words + "</span>" : words;
            };
            return "<tr><td>" + esc(settingLabel(d.setting)) + "</td><td>" + whole(d.first) +
              "</td><td>" + whole(d.second) + "</td></tr>";
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

  // A run already read is drawn again over its old drawing, with no loading
  // line between: the page never shrinks to one line and back, which would
  // throw the reader to its top.
  function drawDetail() {
    var panel = $("#pg-panel");
    if (!panel || !S.selected) { return Promise.resolve(); }
    var held = S.detail && S.detail.name === S.selected;
    if (!held) { panel.innerHTML = '<div class="loading">Reading the run…</div>'; }
    var load = held ? Promise.resolve(S.detail)
      : T.get("/api/playground/run?name=" + encodeURIComponent(S.selected));
    return load.then(function (d) {
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

      var W = S.chartsDrawnAt = chartWidth();
      var perSeedPlot = plot({
        width: W, height: 200, xlabel: "seed", ylabel: "test RMSE (cycles)", label: "test RMSE per seed",
        series: [{ points: d.results.map(function (r) { return [r.seed, r.rmse]; }), colour: "var(--fig-accent)", dots: true, line: false, radius: 5 }],
        hlines: paperRow ? [{ y: paperRow.rmse, label: "paper", colour: "var(--fig-lpc)" }] : []
      });
      var curves = seeds.length && d.per_seed[seeds[0]].history.length;
      var hasValidation = curves && d.per_seed[seeds[0]].history[0].validation_rmse_cycles !== undefined &&
        d.per_seed[seeds[0]].history[0].validation_rmse_cycles !== null;
      var curvePlot = curves ? plot({
        width: W, height: chartHeight(W), xlabel: "pass over the training data", label: "training curves",
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
        width: W, height: chartHeight(W), ylabel: "remaining life (cycles)", label: "predictions against truth", xlabel: "test engine",
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
        '<div class="pg-charts"><figure class="pg-figure pg-figure--wide">' + perSeedPlot + "</figure>" +
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
    return '<section class="pg-block"><div class="pg-fields">' +
      cell({ key: "pg-e-run", label: "Run", size: "wide",
        control: '<span class="pg-run-name">' + esc(named) + "</span>" +
          '<button type="button" class="pg-link" data-action="choose-run">Choose another run</button>' }) +
      // Six seeds or fewer sit side by side, named by the cell's name; more
      // go in a list, which the name's label points at.
      (seeds.length <= 6
        ? cell({ key: "pg-e-seed", label: "Seed whose weights are explained", size: "wide",
          control: segmented("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"', nameIds("pg-e-seed")) })
        : cell({ id: "pg-e-seed", label: "Seed whose weights are explained", size: "wide",
          control: selectField("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"') })) +
      cell({ key: "pg-e-method", label: "Method", size: "wide",
        control: optionList("pg-e-method", [
          { value: "shap", label: METHODS.shap, about: EXPLAIN_ABOUT.shap, note: sourceTag({ kind: "paper", ref: "Eq. 14–15" }) +
            " every one of the " + count(16384) + " subsets of sensors, for every window. An absent sensor is replaced by its average training window (" + refLink("A-20") + ")." },
          { value: "occlusion", label: "Occlusion, one sensor at a time", about: EXPLAIN_ABOUT.occlusion, note: sourceTag({ kind: "extension" }) +
            " 15 evaluations per window. The Shapley term in which the sensor arrives last; the two agree only when sensors do not interact." }
        ], e.method, 'data-explain="method"', explainNote, nameIds("pg-e-method")) }) +
      cell({ key: "pg-e-scope", label: "What is explained", size: "wide",
        control: optionList("pg-e-scope", [
          { value: "engine", label: "One engine, at every cycle", about: EXPLAIN_ABOUT.engine, note: sourceTag({ kind: "paper", ref: "Section 4.4" }) +
            " the dynamic explanation of Table 12 and Fig. 10. The test set keeps only each engine's last window, so the engine comes from the held-back engines, or the training engines when none were held back." },
          { value: "test", label: "The last window of every test engine", about: EXPLAIN_ABOUT.test, note: sourceTag({ kind: "extension" }) +
            " one explanation per test engine, for a ranking across engines." }
        ], e.scope, 'data-explain="scope"', explainNote, nameIds("pg-e-scope")) }) +
      cell({ id: "pg-e-engine", key: "pg-e-engine-row", label: "Engine number", hidden: !forEngine,
        hint: infoCard("Blank: the engine with the longest history.", "", S.options.explain_more.engine),
        control: '<input class="field" type="number" id="pg-e-engine" min="1" value="' + esc(e.engine) + '" data-explain="engine">' }) +
      cell({ id: "pg-e-every", key: "pg-e-every-row", label: "Explain every n-th window", hidden: !forEngine,
        hint: infoCard("Larger is quicker and coarser.", "", S.options.explain_more.every),
        control: '<input class="field" type="number" id="pg-e-every" min="1" max="100" value="' + e.every + '" data-explain="every">' }) +
      cell({ control: '<button type="button" class="button button--primary" data-action="explain">Explain</button>' }) +
      "</div></section>";
  }

  // An Explain option's i: what it does in plain words, then its source and
  // the technical detail under the line.
  function explainNote(c) { return infoCard(c.about, c.note, S.options.explain_more[c.value]); }

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
    var held = e.values && e.values.name === e.shown;
    var load = held ? Promise.resolve(e.values)
      : T.get("/api/playground/explanation?run=" + encodeURIComponent(e.run) + "&name=" + encodeURIComponent(e.shown))
        .then(function (data) { data.name = e.shown; return data; });
    if (!held) { box.innerHTML = '<div class="loading">Reading the explanation…</div>'; }
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
      var W = S.chartsDrawnAt = chartWidth();

      // The paper's figures, in the order its Section 4.4 introduces them, each
      // drawn from this explanation. Every caption opens by saying which of the
      // paper's figures it follows, or that the paper has none like it.
      var swarm = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, left.</strong> Every value of every sensor, one dot per ' +
        (forEngine ? "window" : "engine") + ", the sensors ranked by the average size of their values.</figcaption>" + beeswarm(rows, symbols, W) + "</figure>";
      var strip = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 9, the two boxes above its graph.</strong> ' +
        "Deeper for larger, lighter for smaller.</figcaption>" + rankedStrip(mean, symbols, Math.min(W, 640)) + "</figure>";

      var body;
      if (forEngine) {
        var current = rows[e.index];
        var subset = e.detail ? e.detail.subset : "";
        var lifePlot = plot({
          width: W, height: chartHeight(W), xlabel: "cycle at the end of the window", ylabel: "remaining life, and error (cycles)",
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
    // At 0.8 of the layout's own size no circle meets a neighbour's name;
    // the names are drawn after every circle, so none is ever covered, and
    // each carries its circle's words, shown when the pointer rests on it.
    // The graph scrolls sideways rather than shrink past reading on a phone.
    var scale = 0.8, names = "";
    var out = '<svg class="pg-graph" viewBox="0 0 ' + g.width * scale + " " + (g.height - 30) * scale + '">';
    g.edges.forEach(function (edge) {
      var a = g.positions[edge[0]], b = g.positions[edge[1]];
      out += '<line x1="' + a[0] * scale + '" y1="' + a[1] * scale + '" x2="' + b[0] * scale + '" y2="' + b[1] * scale + '" stroke="var(--fig-edge)" stroke-width="1.2"/>';
    });
    st.ranked.forEach(function (s, rank) {
      var p = g.positions[s], k = st.style[s];
      var title = "<title>" + esc(s + ": rank " + (rank + 1) + ", " + signedValue(Number(values[s])) + " cycles") + "</title>";
      out += "<g>" + title +
        '<circle cx="' + p[0] * scale + '" cy="' + p[1] * scale + '" r="17" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-bg)" stroke-width="2"/></g>';
      names += "<g>" + title + '<text x="' + p[0] * scale + '" y="' + (p[1] * scale + 30) + '" text-anchor="middle" class="pg-node">' + esc(s) + "</text></g>";
    });
    return '<div class="pg-graph-stage">' + out + names + "</svg></div>";
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
    var st = fig10Style(values, sensors), names = "";
    // The names after every circle, so a later circle never covers one.
    ring.forEach(function (s) {
      var p = at[s], k = st.style[s], v = Number(values[s]);
      out += "<g><title>" + esc(label + " · " + s + ": rank " + (k.rank + 1) + " of " + n + ", " + (unit ? signedValue(v) + " " + unit : fixed(v, 2))) + "</title>" +
        '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="8.4" fill="' + k.colour + '" fill-opacity="' + k.opacity.toFixed(2) + '" stroke="var(--fig-ink)" stroke-width="0.35"/></g>';
      if (k.rank < 4) { names += ringLabel(p, s); }
    });
    return out + names + "</svg><figcaption>" + esc(label) + "</figcaption></figure>";
  }

  // The ring once more with every sensor named and nothing coloured: the key to where each one sits.
  function keyRing() {
    var at = ringPositions();
    var out = '<figure class="pg-ring pg-ring--key"><svg viewBox="0 0 100 100" role="img" aria-label="Where each sensor sits">' + ringEdges(at);
    var names = "";
    S.options.table12.ring.forEach(function (s) {
      var p = at[s];
      out += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="8.4" fill="var(--fig-bg)" stroke="var(--fig-axis)" stroke-width="0.5"/>';
      names += ringLabel(p, s);
    });
    return out + names + "</svg><figcaption>key</figcaption></figure>";
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
