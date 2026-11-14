/* The playground: set up a run, follow it, read its results, explain it.
   docsite/playground/README.md explains what each part is for and why every run
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
  var METHODS = { shap: "Exact Shapley values", occlusion: "Occlusion", walks: "Walks through the sensor graph" };
  // What the Explain tab's options do, in plain words, above the line of each one's i.
  var EXPLAIN_ABOUT = {
    shap: "Explains how the recorded input changes a prediction relative to an average training window. Start with that window and restore " +
      "each sensor's recorded window in turn. Its Shapley value is the average change in predicted remaining " +
      "life over every possible restoration order, in cycles. Together, the values equal the original " +
      "prediction minus the starting prediction, apart from numerical rounding. They explain the model, " +
      "not the causes of engine failure.",
    occlusion: "Measures what changes when one sensor's window is replaced by its average training window, " +
      "with every other sensor unchanged. The value is the original prediction minus the replacement " +
      "prediction, in cycles: positive means this sensor's recorded readings raise the prediction. " +
      "The values need not add to the prediction's difference from the input where every sensor is replaced. They describe " +
      "the model, not the causes of engine failure.",
    walks: "Splits the model's score, before conversion to remaining-life cycles, into shares along graph routes. " +
      "A route takes one connection per graph layer, starting from a sensor's temporal output. Each layer's numbers " +
      "are divided among their sources by the rule of the method's authors, under the setting chosen below. Route shares and a " +
      "separate share from added constants sum to the score within rounding error. The authors' own check, adding and " +
      "removing sensors in the order the routes rank them, is run beside it. They do not establish physical causes.",
    engine: "Explains successive windows of one engine's history to show how sensor contributions change " +
      "over time. The engine comes from those held back from training; if none were held back, it comes " +
      "from the training engines. The window spacing and the setting below determine which windows are explained.",
    test: "Explains one prediction per test engine, using its last available window. These engines were " +
      "not used to fit the model. Engines too short to fill a window are excluded. This supports comparison " +
      "across engines, but one window per engine cannot show changes over that engine's life."
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
  var SOURCE_TITLES = { paper: "Supported by the cited part of the paper", assumption: "A project choice where the paper is incomplete or ambiguous",
    project: "A rule of this repository", extension: "An option offered for comparison" };
  // Choices short enough to sit side by side in one control, and a choice of
  // five that shares one source and sits beside the two rates it connects,
  // in a menu. Every other choice is a grid of options, each with the source
  // of its value.
  var SEGMENTED = { subset: true, device: true };
  var MENU = { lr_schedule: true };
  var SUBSETS = ["FD001", "FD002", "FD003", "FD004"];
  // The value that takes a part out of the pipeline. Each is the setting of a
  // variant of Table 11: "w/o LSTM", "w/o Physics prior" and "w/o GCN".
  var REMOVED = { temporal: "last_cycle", graph: "complete", spatial: "none" };
  // What the pipeline calls the sensor graph the graph layers run on.
  var PRIOR = { physics: "Physics prior", correlation: "Correlation graph" };
  var CROSS = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3 L9 9 M9 3 L3 9" ' +
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
  var BACK = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M7.5 2.5 L4 6 L7.5 9.5" fill="none" ' +
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var NEXT = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M4.5 2.5 L8 6 L4.5 9.5" fill="none" ' +
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  // Two links of a chain, for the button that copies a run's address.
  var LINK_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6.8 9.2 L9.2 6.8 M7.4 4.6 L8.6 3.4 a2.4 2.4 0 0 1 3.4 3.4 ' +
    'L10.8 8 M8.6 11.4 L7.4 12.6 a2.4 2.4 0 0 1 -3.4 -3.4 L5.2 8" fill="none" stroke="currentColor" stroke-width="1.5" ' +
    'stroke-linecap="round" stroke-linejoin="round"/></svg>';
  var PLUS = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M6 2.5 V9.5 M2.5 6 H9.5" ' +
    'fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
  var INFO = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" ' +
    'stroke-width="1.3"/><path d="M8 7.2 V11.2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>' +
    '<circle cx="8" cy="4.9" r=".95" fill="currentColor"/></svg>';
  // One picture for each box of the pipeline and for what goes in and comes
  // out of it: the files (a store of records), preprocessing (sliders: the
  // readings are adjusted), the temporal stage (a memory cell and return
  // arrow), the graph layers (sensors joined in a graph) and the remaining
  // life (an hourglass). The same picture stands on a box, on its
  // flowchart's title and beside the words that lead into or out of it.
  function icon(paths) {
    return '<svg class="pg-icon" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" ' +
      'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' + paths + "</svg>";
  }
  var ICONS = {
    explain: icon('<circle cx="10.5" cy="10.5" r="6.5"/><path d="m15.2 15.2 5 5"/>'),
    history: icon('<path d="M4 5v5h5M4.5 10a8 8 0 1 1 .6 6M12 7v5l3 2"/>'),
    data: icon('<ellipse cx="12" cy="6" rx="7" ry="2.5"/><path d="M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6"/>' +
      '<path d="M5 12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5"/>'),
    preprocessing: icon('<path d="M4 7h9M17 7h3M4 12h3M11 12h9M4 17h11M19 17h1"/><circle cx="15" cy="7" r="2"/>' +
      '<circle cx="9" cy="12" r="2"/><circle cx="17" cy="17" r="2"/>'),
    temporal: icon('<rect x="7" y="8" width="10" height="10" rx="2"/><path d="M2 13h5m10 0h5M20 13V3H4v5m-2-2 2 2 2-2"/>' +
      '<path d="M10 11h4m-4 4h4"/>'),
    graph: icon('<path d="M5.5 5.5 10 10M14 11l4-3M13 14l3 4M9.5 13l-4 4M4 7v9M7 19h8"/>' +
      '<circle cx="4" cy="4" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="19" r="2"/>' +
      '<circle cx="18" cy="20" r="2"/><circle cx="12" cy="12" r="3" fill="currentColor"/>'),
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
    // A memory cell, with its state carried back to the next cycle.
    recurrent: picture('<rect x="15" y="11" width="18" height="16" rx="3"/><path d="M4 19h11m18 0h11M39 19V4H9v10m-3-3 3 3 3-3"/>' +
      '<path d="M21 16h6m-6 6h6"/>'),
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
    gcn: picture('<path d="M8 9 11 23M14 25l20 1M28 5l11 5" opacity=".45"/>' +
      '<path d="M10 8l9 6m-4-1 4 1-1-4M14 23l6-4m-4 0h4v4M26 7l-2 5m-2-3 2 3 3-1' +
      'M38 12l-11 3m3-3-3 3 4 1M34 24l-8-5m1 4-1-4 4 0"/>' +
      DOT(7, 6) + DOT(12, 26) + DOT(27, 4) + DOT(41, 11) + DOT(36, 26) + FILLED(23, 16, 3)),
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

  // ------------------------------------------------------------- addresses
  // Each run has its own address, #/explore/playground/runs/<folder>, so it
  // can be bookmarked, reloaded and reached with Back and Forward. The list
  // is #/explore/playground/runs, explaining is .../explain, and the bare
  // address is the design tab. Opening or leaving a run adds a step to the
  // browser's history; moving between tabs only corrects the address. The
  // part of a run on show follows the run, "…/runs/<folder>#charts"; the
  // first part, the overview, adds nothing.

  var RUN_TABS = [["overview", "Overview"], ["charts", "Charts"], ["explanations", "Explanations"], ["settings", "Settings"]];
  // The sections a run's page had before it had parts: an address that names
  // one still opens the part that holds it.
  var RUN_SECTIONS = { "pg-run-results": "overview", "pg-run-interpretation": "overview", "pg-run-charts": "charts",
    "pg-run-settings": "settings" };

  function runAddress(name) { return ROUTE + "/runs/" + encodeURIComponent(name).replace(/%2F/g, "/"); }

  function address() {
    if (S.tab === "results") { return S.selected ? runAddress(S.selected) : ROUTE + "/runs"; }
    return S.tab === "explain" ? ROUTE + "/explain" : ROUTE;
  }

  // The address on show, without what a link may have added after it: a
  // section of a page, or the part of a run.
  function shownAddress() { return window.location.hash.replace(/^(#\/[^#]*).*$/, "$1"); }

  // The part of the run on show, and the part the address names.
  function runTabShown() {
    return RUN_TABS.some(function (t) { return t[0] === S.runTab; }) ? S.runTab : "overview";
  }
  function runTabInAddress() {
    var part = window.location.hash.split("#")[2] || "";
    part = RUN_SECTIONS[part] || part;
    return RUN_TABS.some(function (t) { return t[0] === part; }) ? part : "overview";
  }
  function runPartAddress() {
    var part = S.tab === "results" && S.selected ? runTabShown() : "overview";
    return part === "overview" ? "" : "#" + part;
  }

  function setAddress() {
    if (!onPage()) { return; }
    var next = address(), now = shownAddress(), whole = next + runPartAddress();
    if (next !== now) {
      var runPage = /\/runs\/./;
      history[runPage.test(next) || runPage.test(now) ? "pushState" : "replaceState"](null, "", whole);
    } else if (S.tab === "results" && S.selected && window.location.hash !== whole) {
      // The same run at another of its parts.
      history.replaceState(null, "", whole);
    }
    pageTitle();
  }

  // The browser tab names the run on show, so a bookmark or a list of tabs does too.
  function pageTitle() {
    if (!S.baseTitle) { return; }
    document.title = S.tab === "results" && S.selected ? "Run " + runTime(S.selected) + " · " + S.baseTitle : S.baseTitle;
  }

  // What the part of an address after the playground's own asks for.
  function readAddress(sub) {
    var run = /^runs(?:\/(.+))?$/.exec(sub || "");
    var name = null;
    if (run && run[1]) {
      try { name = decodeURIComponent(run[1]); } catch (e) { name = run[1]; }
    }
    return run ? { tab: "results", selected: name } : { tab: sub === "explain" ? "explain" : "design", selected: null };
  }

  // The page's scroll position, which is the content column's on a wide
  // screen and the window's on a narrow one.
  function scroller() { return root.closest(".content") || document.scrollingElement; }
  function scrolledTo() { var box = scroller(); return Math.max(box.scrollTop || 0, window.scrollY || 0); }
  function scrollToTop(y) {
    var box = scroller();
    box.scrollTo({ top: y || 0, behavior: "instant" });
    if (box !== document.scrollingElement) { window.scrollTo({ top: 0, behavior: "instant" }); }
  }
  // The page moved on by a number of pixels, whichever of the two scrolls.
  function scrollOn(by) {
    var box = scroller(), before = box.scrollTop;
    box.scrollBy({ top: by, behavior: "instant" });
    if (box.scrollTop === before && box !== document.scrollingElement) { window.scrollBy({ top: by, behavior: "instant" }); }
  }

  // Back, Forward, a typed address or a link: app.js hands the new address
  // here instead of building the whole page again.
  function follow(sub) {
    if (!S.options || !$("#pg-panel")) { return false; }
    var want = readAddress(sub), before = S.selected, beforeTab = S.tab;
    var part = want.selected ? runTabInAddress() : "overview";
    if (want.tab === beforeTab && want.selected === before) {
      // The run on show at another of its parts, from an address typed or pasted.
      if (want.selected && part !== runTabShown()) { S.runTab = part; drawDetail(); }
      pageTitle();
      return true;
    }
    if (want.selected && !before) { S.listScroll = beforeTab === "results" ? scrolledTo() : 0; }
    S.fromList = !!want.selected && !before && beforeTab === "results";
    if (want.selected !== before) { S.selected = want.selected; S.detail = null; }
    S.runTab = part;
    // Back on the list, its place is found again once its rows are drawn.
    if (!want.selected && before) { S.pendingScroll = S.listScroll || 0; }
    showTab(want.tab);
    if (want.selected && want.selected !== before) { scrollToTop(0); }
    return true;
  }
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
      return '<a href="#/doc/docs/audits/AUDIT_2026-09-08_scientific_validity.md">' + ref + "</a>";
    }
    return esc(ref);
  }

  // Where a default comes from: the paper, a numbered choice, a project rule,
  // or an option the paper does not have. docsite/playground/README.md lists the four.
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
  // text per paragraph, then a footer with where its value comes from and a
  // "Read more" link to the part of the documentation that explains it.
  // Any part may be missing. A
  // line after the first that opens "Option words: ..." explains one option,
  // and its option's words are set in the heavier weight, so the eye finds them.
  function infoCard(about, source, more) {
    var lines = about ? String(about).split("\n").filter(Boolean).map(function (line, i) {
      var m = i > 0 && /^([^:]{1,48}): (.+)$/.exec(line);
      return '<span class="pg-info-line">' + (m ? "<strong>" + esc(m[1]) + ":</strong> " + esc(m[2]) : esc(line)) +
        "</span>";
    }).join("") : "";
    var footer = (source ? '<span class="pg-info-source">' + source + "</span>" : "") +
      (more ? '<span class="pg-info-more"><a href="' + esc(more) + '" target="_blank" rel="noopener">Read more' +
        NEXT + "</a></span>" : "");
    return lines + (footer ? '<span class="pg-info-footer">' + footer + "</span>" : "");
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

  // A stored run's Graph column: its graph, or "w/o GCN" for a network with
  // no graph layers, which never used the graph its data were prepared with.
  function runGraphName(run) {
    return run.spatial === REMOVED.spatial ? "w/o GCN" : graphName(run.graph, run.adjacency_normalization);
  }

  function statusWord(status) { return STATUS[status] || status; }

  // A setting two runs differ in, named as the form names it, with its value in
  // the form's own words. A record the form has no field for keeps its name.
  var RECORD_LABELS = { seeds: "Seeds", split: "Held-back engines", commit: "Commit" };
  var RECORD_WORDS = { varies_with_seed: "a different draw for every seed", fixed: "the same for every seed",
    none: "none held back" };

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
  // shades the error between the prediction and its "True RUL" curve, which is
  // the label, the remaining life capped at 125 (assumption A-01).
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
    if (spec.zero && ylo < 0 && yhi > 0) {
      out += '<line x1="' + left + '" x2="' + (W - right) + '" y1="' + Y(0).toFixed(1) + '" y2="' + Y(0).toFixed(1) +
        '" stroke="var(--fig-axis)"/>';
    }
    (spec.vlines || []).forEach(function (v) {
      var x = X(v.x), late = x > W - right - 96;
      out += '<line x1="' + x.toFixed(1) + '" x2="' + x.toFixed(1) + '" y1="' + top + '" y2="' + (H - bottom) +
        '" stroke="var(--fig-ink)" stroke-opacity="0.6" stroke-dasharray="3 3"/>' +
        (v.label ? '<text x="' + (x + (late ? -6 : 6)).toFixed(1) + '" y="' + (top + 10) + '" text-anchor="' + (late ? "end" : "start") +
          '" class="pg-tick">' + esc(v.label) + "</text>" : "");
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
  function chartWidth() {
    var panel = $("#pg-panel");
    var shown = panel ? panel.clientWidth : 666;
    return Math.max(240, Math.round(shown - 26));
  }

  // Explanation charts occupy their whole tab panel, as a run's charts do.
  function explanationChartWidth() {
    var panel = $("#pg-panel");
    var shown = panel ? panel.clientWidth : 666;
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
        var width = S.tab === "explain" ? explanationChartWidth() : chartWidth();
        if (!panel.isConnected || !onPage() || width === S.chartsDrawnAt) { return; }
        var y = window.scrollY;
        if (S.tab === "results" && S.selected && S.detail && runTabShown() === "charts") { drawDetail(); }
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

  // What a sensor's name stands for is shown beside the pointer wherever the
  // name is drawn: engine-view.js holds the words and draws the tooltip. This
  // marks an element as carrying a sensor's name; note is a second line.
  function sensorTip(s, note) {
    var known = window.ThesisEngineView && window.ThesisEngineView.locations[s];
    return known ? ' data-sensor-tip="' + esc(s) + '"' + (note ? ' data-tip-note="' + esc(note) + '"' : "") : "";
  }

  // Every value of every sensor as one dot, the way the paper's Fig. 9 (left)
  // draws it: the sensors ranked by the average size of their values, a red
  // frame round the four largest and a blue one round the rest, and each dot
  // coloured by the sensor's own reading at that window. An explanation
  // stored before the readings were kept colours by the window's place in
  // the sequence instead, and the legend says which. unit is left out for
  // values in cycles and "" for shares of the score before the sigmoid.
  function beeswarm(rows, ranked, width, unit) {
    var inCycles = unit === undefined;
    var hasReadings = rows.length > 0 && rows[0]["reading_" + ranked[0]] !== undefined;
    // A row is tall enough for the dots that share a value to stand in a
    // column beside one another, so a crowd of windows reads as a crowd.
    var W = width || 640, wide = W >= 700;
    var rowH = wide ? 38 : 28, left = 64, right = 18, top = 10, bottom = 36, dot = wide ? 2.8 : 2.4, step = wide ? 6 : 5.2;
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
      out += '<text x="' + X(t).toFixed(1) + '" y="' + (H - bottom + 15) + '" text-anchor="middle" class="pg-tick">' + tickLabel(t) + "</text>";
    });
    out += '<text x="' + ((left + W - right) / 2) + '" y="' + (H - 4) + '" text-anchor="middle" class="pg-axis">' +
      (inCycles ? "value, in cycles of remaining life" : "share of the score before the sigmoid") + "</text>";
    ranked.forEach(function (s, j) {
      var cy = top + j * rowH + rowH / 2;
      out += '<text x="' + (left - 10) + '" y="' + (cy + 4) + '" text-anchor="end" class="pg-tick pg-sensor-name"' + sensorTip(s) + ">" + esc(s) + "</text>";
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
          esc(s + " · " + p.label + ": " + (p.x >= 0 ? "+" : "") + p.x.toFixed(2) + (inCycles ? " cycles" : "") +
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
      out += "<g" + (sensorTip(s, "rank " + (rank + 1) + " of " + ranked.length) || ' aria-label="' + esc(s + ": rank " + (rank + 1) + " of " + ranked.length) + '"') + ">" +
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
        var note = noteOf ? noteOf(c) : "";
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
    var size = !name && !about && !inline ? (spec.size ? spec.size + " pg-field--bare" : "bare") : spec.size;
    return '<div class="pg-field' + (size ? " pg-field--" + size : "") + '"' +
      (spec.key ? ' data-row="' + spec.key + '"' : "") + (spec.hidden ? " hidden" : "") + ">" +
      '<div class="pg-field-label">' + (inline ? "" : named + (top ? spec.reset || "" : "")) + "</div>" +
      '<div class="pg-field-control">' + spec.control + (inline ? named : "") + (top ? "" : spec.reset || "") +
      "</div></div>";
  }

  function chip(kind, label) { return T.chip(kind, label); }

  // ------------------------------------------------------------------ page

  function start(target, tools, sub) {
    T = tools;
    root = target;
    if (window.__thesisPlaygroundTimer) { clearTimeout(window.__thesisPlaygroundTimer); }
    // seen: the last state each job was polled in. mine: the jobs this page
    // started, whose end is announced even when the first poll finds them ended.
    // restore: what each part taken out of the pipeline was, so + puts that back.
    S = { tab: S.tab || "design", baseTitle: document.title, jobs: [], seen: {}, mine: {}, runs: null, selected: null, detail: null,
      restore: S.restore || {}, infos: 0, step: S.step || recall(STEP) || "data", zoom: S.zoom || null, zoomed: null,
      inside: null,
      compare: [], resultsTab: S.resultsTab || "runs", page: { runs: 0, explanations: 0 }, explanations: null,
      filter: noFilters(false),
      xfilter: { method: "all", scope: "all", subset: "all", state: "all" },
      explain: { run: null, detail: null, seed: null, method: "shap", scope: "engine", engine: "",
        every: 1, gamma: null, shown: null, values: null, index: 0, chart: "engine" } };

    // The browser tab is named by app.js, from the menu, as every page is.
    return tools.get("/api/playground/options").then(function (options) {
      S.options = options;
      S.fields = {};
      options.fields.forEach(function (f) { S.fields[f.key] = f; });
      S.settings = loadSettings(options.defaults);
      if (S.step === "repeats") { S.step = "training"; remember(STEP, S.step); }
      // A step remembered from an older page that no longer has it opens the first.
      if (!groupLabel(S.step)) { S.step = options.groups[0].key; }
      frame();
      if (sub) {
        var want = readAddress(sub);
        S.tab = want.tab; S.selected = want.selected; S.runTab = runTabInAddress();
      }
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
      '<div class="pg-steps" role="tablist" aria-label="Playground sections">' +
      tabButton("design", ICONS.preprocessing, "Design a run") + tabButton("explain", ICONS.explain, "Explain predictions") +
      tabButton("results", ICONS.history, "Run history", '<span class="pg-jobs-badge" id="pg-jobs-badge"></span>') + "</div>" +
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

  function tabButton(key, symbol, label, after) {
    return '<button type="button" role="tab" class="pg-step-tab" id="pg-step-tab-' + key + '" data-tab="' + key +
      '" aria-controls="pg-panel"><span class="pg-step-icon">' + symbol + '</span><span class="pg-step-label">' +
      esc(label) + "</span>" + (after || "") + "</button>";
  }

  // One tab of the row is a stop for the Tab key, the chosen one; the arrow
  // keys move along the row, and the panel is named by its tab.
  function showTab(tab) {
    if (tab === "jobs") { tab = "results"; }  // the old Jobs tab's jobs are rows of Results
    S.tab = tab;
    setAddress();
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
    // A link that also names an action, such as the way back to the list,
    // still opens in a new tab with a modifier key held.
    if (target.tagName === "A" && target.hasAttribute("data-action")) {
      if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) { return; }
      event.preventDefault();
    }
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
      e.engineView = {}; e.chart = "engine";
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
    "locate-operation": function (button) {
      S.locate = button.getAttribute("data-operation");
      zoomTo(button.getAttribute("data-group"));
      locateOperation();
    },
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
      S.fromList = false;
      S.selected = button.getAttribute("data-run"); S.detail = null; S.runTab = "overview"; S.runs = null;
      showTab("results"); scrollToTop(0);
    },
    "open-explanation": function (button) {
      S.explain.run = button.getAttribute("data-run"); S.explain.detail = null;
      S.explain.shown = button.getAttribute("data-name"); S.explain.values = null; S.explain.chart = "engine";
      showTab("explain");
    },
    // Back to the list: the browser's own Back when the run was opened from
    // it, so the list returns where it was left; otherwise a new step.
    "back": function () {
      if (S.fromList) { history.back(); return; }
      S.selected = null; S.detail = null; setAddress(); drawResults(); scrollToTop(0);
    },
    "copy-link": function () {
      var link = window.location.href.split("#")[0] + address() + runPartAddress();
      var done = function () { T.toast("Link to this run copied."); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(link).then(done, function () { T.toast(link, 9000); });
      } else { T.toast(link, 9000); }
    },
    "run-part": function (control) {
      showRunPart(control.getAttribute("data-part"), control.getAttribute("data-reach"), control.getAttribute("role") === "tab");
    },
    // The filters for every parameter, shown or put away; the ones in use
    // stay in use either way.
    "more-filters": function () { S.filter.open = !S.filter.open; filtersChanged("pg-more-toggle", true); },
    "drop-filter": function (button) { delete S.filter.more[button.getAttribute("data-param")]; filtersChanged("pg-more-toggle"); },
    "clear-filters": function () { S.filter = noFilters(S.filter.open); filtersChanged("pg-more-toggle"); },
    "again": function () { S.settings = settingsFromRun(S.detail); S.restore = {}; saveSettings(); showTab("design"); },
    "explain-this": function () {
      S.explain.run = S.detail.name; S.explain.detail = null; S.explain.shown = null; showTab("explain");
    },
    "explain": function () { submitExplanation(); },
    "chart-tab": function (button) { selectExplanationChart(button.getAttribute("data-chart")); },
    // A walk chosen in the list of the largest is drawn on the engine, where
    // the same list is the picker's.
    "walk-show": function (button) {
      var e = S.explain, box = $("#pg-e-view");
      if (!box || !e.values) { return; }
      if (!e.engineView) { e.engineView = {}; }
      e.engineView.walk = Number(button.getAttribute("data-walk"));
      e.engineView.sensor = null;
      mountEngine(box, e.values);
      selectExplanationChart("engine");
      var tab = $("#pg-chart-tab-engine");
      if (tab) { tab.focus(); }
    },
    "explain-previous": function () { selectExplanationIndex(S.explain.index - 1); },
    "explain-next": function () { selectExplanationIndex(S.explain.index + 1); }
  };

  function onChange(event) {
    var input = event.target;
    if (input.hasAttribute("data-procedure")) { chooseProcedure(input.value); return; }
    if (input.hasAttribute("data-field")) { fieldChanged(input); return; }
    if (input.hasAttribute("data-explain-index")) {
      selectExplanationIndex(Number(input.value));
      return;
    }
    // A filter changed: the list goes back to its first page, and the
    // focus stays on the filter.
    if (input.hasAttribute("data-filter") || input.hasAttribute("data-xfilter")) {
      var runs = input.hasAttribute("data-filter");
      var f = runs ? S.filter : S.xfilter, which = input.getAttribute(runs ? "data-filter" : "data-xfilter");
      f[which] = input.type === "checkbox" ? input.checked : input.value;
      filtersChanged(input.id);
      return;
    }
    // One of the filters under "More filters": "All" takes it away again.
    if (input.hasAttribute("data-param")) {
      var param = input.getAttribute("data-param");
      if (input.value === "all") { delete S.filter.more[param]; } else { S.filter.more[param] = input.value; }
      filtersChanged(input.id);
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
  // Inside the diagram and settings scroll areas, the
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
    var side = i.closest(".pg-pipeline-body, .pg-tabpanel");
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

  // Open help cards follow their i when either scroll area or the page
  // scrolls, and close once their i has scrolled out of the
  // body's view, rather than point at nothing. The body is drawn anew with
  // every change of view, so its scrolling is caught on the way down, from
  // the pipeline, which stays.
  function watchPipeline() {
    var box = $(".pg-design");
    if (!box || box.__scrolls) { return; }
    box.__scrolls = true;
    var follow = function () {
      Array.prototype.forEach.call(box.querySelectorAll(".pg-info.is-loose"), function (i) {
        if (!i.querySelector(".pg-info-body").getClientRects().length) { return; }
        var scroller = i.closest(".pg-pipeline-body, .pg-tabpanel");
        if (!scroller) { return; }
        var view = scroller.getBoundingClientRect();
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
        var now = document.querySelector(".pg-design");
        if (now && onPage()) { now.dispatchEvent(new Event("scroll")); }
      }, { passive: true });
    }
  }

  // Update the selected data while keeping the dragged control attached.
  function onInput(event) {
    var input = event.target;
    if (!input.hasAttribute("data-explain-index")) { return; }
    selectExplanationIndex(Number(input.value));
  }

  // What a stored "rul" column holds, in words. The windows of one engine carry
  // the label the network trains toward, the remaining life capped at the
  // ceiling (assumption A-01); test engines carry it capped only when the run
  // capped its test labels (assumption A-02), true when it did not, and a run
  // that recorded neither is not guessed at.
  function lifeLabel(capped) {
    return capped === true ? "capped remaining life" : capped === false ? "true remaining life" : "test label";
  }

  function lifeSeriesName(capped, ceiling) {
    return capped === true && ceiling ? "remaining life capped at " + fixed(ceiling, 0) : lifeLabel(capped);
  }

  // The Fig. 8 chart's name for a screen reader, after the series it is drawn against.
  function predictionsChartName(capped, ceiling) {
    return "predictions against the " + lifeSeriesName(capped, ceiling);
  }

  function explanationCapped(data) {
    return data.summary.scope === "engine" ? true : (S.explain.detail || {}).clip_test_labels;
  }

  function explanationCaption(row) {
    var data = S.explain.values;
    var life = lifeLabel(data ? explanationCapped(data) : undefined) + " " + fixed(row.rul, 0);
    if (data && data.summary.scope === "test") {
      return "engine " + row.unit + ", " + life + ", predicted " + fixed(row.predicted, 1);
    }
    return "cycle " + row.end_cycle + ", " + life + ", predicted " + fixed(row.predicted, 1);
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
    return '<span class="pg-step-n pg-step-n--form pg-step-n--input' +
      (S.step === group ? " is-current" : "") + '" data-step="' + group + '">' + (ICONS[group] || PICTURES.passes) +
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
    if (key === "validation_fraction") { return chosenProcedure().settings.validation_fraction > 0; }
    if (key === "early_stopping_patience") { return !!s.early_stopping; }
    return true;
  }

  function isChanged(f) { return !same(S.settings[f.key], f.default); }

  function settingsChanged() { return !same(S.settings, S.options.defaults); }

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

  // The pipeline, then the form with the run summary inside Training.
  // On a wide page the pipeline stands
  // to the right of the form and stays in view while the page scrolls; on a
  // narrow one it stands above it. The two cards beside each other start
  // level, since neither has a heading above it.
  function drawDesign() {
    $("#pg-panel").innerHTML =
      '<div class="pg-design">' + pipelineHtml() + '<div class="pg-form">' +
      '<section class="pg-block pg-settings" aria-label="Settings">' + tabsHtml() +
      '<div class="pg-tabpanels">' + S.options.groups.map(panelHtml).join("") + "</div>" +
      '<div class="pg-settings-foot"><button type="button" class="button button--small pg-reset-all" data-action="reset-all"' +
      (settingsChanged() ? "" : " disabled") + ">Reset all</button></div>" +
      "</section></div></div>";
    // What the pipeline was just drawn with, so the first change of settings
    // finds it unchanged instead of drawing it again under the pointer.
    var pipeline = $("#pg-pipeline");
    pipeline.__html = pipelineInner();
    pipeline.__zoom = S.zoom;
    drawRunCard($("#pg-run"), null, null);
    watchPipeline();
    watchTabs();
    fitFlow();
    fitTabs();
    refreshPreview();
    requestZoom(true);
  }

  // ----------------------------------------------------------------- steps
  // The five tabs of the form share the available height and scroll within
  // it, so their pipeline stays steady. A tab carries a dot while any
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
    if (g.key === "data") {
      body += dataDetailsHtml(null);
    }
    if (g.key === "training") {
      body += '<section class="pg-run" id="pg-run" aria-label="Run summary"></section>';
    }
    return '<div class="pg-tabpanel' + (on ? " is-active" : "") + '" role="tabpanel" id="pg-tabpanel-' + g.key +
      '" data-group="' + g.key + '" aria-labelledby="pg-tab-' + g.key + '"' + (on ? "" : " inert") + ">" + body + "</div>";
  }

  function dataHistoryWords(history) {
    if (!history) { return ""; }
    var median = Number(history.median);
    var middle = Number.isInteger(median) ? count(median) : number(median, 1);
    return count(history.shortest) + "–" + count(history.longest) + " cycles; median " + middle;
  }

  function dataFileHtml(group, label, filename, role, rows, columns, engines, history) {
    return '<tr>' + group + '<th scope="row">' + esc(label) +
      '<span class="pg-data-column-key">' + esc(filename) + '</span></th><td>' + esc(role) +
      '</td><td>' + count(rows) + '</td><td>' + count(columns) + '</td><td>' + count(engines) +
      '</td><td>' + (history ? esc(dataHistoryWords(history)) : 'One answer per engine') + '</td></tr>';
  }

  function dataSampleValue(value, key) {
    if (value === null || value === undefined) { return "—"; }
    return key === "unit" || key === "cycle" ? count(value) : String(value);
  }

  function dataDetailsHtml(dataset) {
    var subset = S.settings.subset;
    var table5 = (dataset && dataset.table5) || (S.options.dataset || {})[subset] || {};
    var files = dataset && dataset.files;
    var schema = S.options.dataset_schema || {};
    var sensors = schema.sensors || [];
    var sample = files && files.sample;
    var ready = !!dataset;
    var status = !ready ? "Reading file details…" : dataset.error ? "Refresh failed" : files ? "" : "Files not found";
    var fileRows = dataFileHtml('<th scope="rowgroup">Training</th>', "Training histories", "train_" + subset + ".txt",
      "Sensor readings from each engine's first recorded cycle through its last cycle before failure.",
      files && files.train_rows, schema.raw_column_count, files && files.train_engines, files && files.train_history) +
      '</tbody><tbody class="pg-data-testing">' +
      dataFileHtml('<th scope="rowgroup" rowspan="2">Testing<span class="pg-data-pair">Used together</span></th>', "Test histories", "test_" + subset + ".txt",
        "Readings stop before failure. The model uses these readings to predict life left.", files && files.test_rows, schema.raw_column_count,
        files && files.test_engines, files && files.test_history) +
      dataFileHtml('', "Test answers", "RUL_" + subset + ".txt",
        "Known life left at each test engine's last recorded cycle, used to score its prediction.", files && files.labels, 1, files && files.test_engines, null);
    var groups = (schema.column_groups || []).map(function (group) {
      return '<div><dt>' + esc(group.name) + '</dt><dd><strong>' + count(group.count) + '</strong><span class="pg-data-code">' +
        esc(group.columns) + "</span></dd></div>";
    }).join("");
    var sensorRows = sensors.map(function (sensor) {
      return '<tr class="' + (sensor.used ? "is-used" : "is-unused") + '"><td><span class="pg-data-use">' +
        (sensor.used ? "Used" : "Not used") + "</span>" +
        '</td><th scope="row">' + esc(sensor.symbol) + '</th><td class="pg-data-code">' + esc(sensor.column) +
        "</td><td>" + esc(sensor.description) + "</td><td>" + esc(sensor.unit === "-" ? "—" : sensor.unit) +
        "</td><td>" + esc(sensor.station) + "</td></tr>";
    }).join("");
    var sampleHead = sample ? sample.columns.map(function (column) {
      return '<th scope="col"><span>' + esc(column.label) + '</span><span class="pg-data-column-key">' +
        esc(column.detail) + "</span></th>";
    }).join("") : "";
    var sampleRows = sample ? sample.rows.map(function (row) {
      return "<tr>" + sample.columns.map(function (column, index) {
        var tag = index === 0 ? 'th scope="row"' : "td";
        return "<" + tag + ">" + esc(dataSampleValue(row[column.key], column.key)) + "</" + tag.split(" ")[0] + ">";
      }).join("") + "</tr>";
    }).join("") : "";
    var sampleTable = sample ? '<div class="pg-data-table-wrap" tabindex="0" role="region" aria-label="All dataset columns; scroll horizontally"><table class="pg-data-table pg-data-sample"><caption class="visually-hidden">Ten consecutive rows from ' +
      esc(sample.source) + '</caption><thead><tr>' + sampleHead + "</tr></thead><tbody>" + sampleRows +
      "</tbody></table></div>" : "";
    var unavailable = ready && !dataset.error && !files ? '<p class="pg-data-note pg-data-note--warn">The three files are not in data/raw/, so row counts and history lengths cannot be measured here.</p>' : "";
    var failed = dataset && dataset.error ? '<p class="pg-data-note pg-data-note--warn">File details could not be refreshed: ' +
      esc(dataset.error) + "</p>" : "";
    return '<section class="pg-data-detail" id="pg-data-detail" aria-labelledby="pg-data-detail-title" aria-live="polite" aria-busy="' +
      (!ready) + '"><header class="pg-data-detail-head"><div><h3 id="pg-data-detail-title">' + esc(subset) +
      '</h3></div>' +
      (status ? '<span class="pg-data-status">' + esc(status) + '</span>' : '') + '</header><dl class="pg-data-metrics"><div><dt>Training engines</dt><dd>' +
      count(table5.train_engines) + '</dd></div><div><dt>Test engines</dt><dd>' + count(table5.test_engines) +
      '</dd></div><div><dt>Operating conditions</dt><dd>' + count(table5.conditions) +
      '</dd></div><div><dt>Fault modes</dt><dd>' + count(table5.fault_modes) +
      '</dd></div></dl><div class="pg-data-detail-body"><section aria-labelledby="pg-data-files-title"><h4 id="pg-data-files-title">Files</h4>' +
      '<div class="pg-data-table-wrap" tabindex="0" role="region" aria-label="Dataset files"><table class="pg-data-table pg-data-files"><caption class="visually-hidden">Training file and complementary testing files</caption><thead><tr><th>Use</th><th>File</th><th>Purpose</th><th>Rows</th><th>Columns</th><th>Engines</th><th>Recorded cycles per engine</th></tr></thead><tbody>' + fileRows + "</tbody></table></div>" + unavailable + failed + "</section>" +
      '<section class="pg-data-sensors" aria-labelledby="pg-data-sensors-title"><div class="pg-data-sensors-head"><div><h4 id="pg-data-sensors-title">Sensor columns</h4>' +
      '<p>The paper uses 14 of the 21 sensor readings as model inputs. The other seven stay in the raw files.</p></div></div>' +
      '<div class="pg-data-table-wrap"><table class="pg-data-table"><caption class="visually-hidden">Every sensor column in ' +
      esc(subset) + ', its meaning, unit, location and whether the paper uses it</caption><thead><tr><th>Paper input</th><th>Sensor</th>' +
      '<th>File column</th><th>Measurement</th><th>Unit</th><th>Location</th></tr></thead><tbody>' + sensorRows +
      '</tbody></table></div></section><section class="pg-data-rows" aria-labelledby="pg-data-columns-title"><h4 id="pg-data-columns-title">Each history row</h4>' +
      '<p class="pg-data-intro">Each row records one engine at one cycle. These ten consecutive rows from ' +
      esc(sample ? sample.source : "the training file") + ' show all ' + count(schema.raw_column_count) +
      ' columns in file order. Scroll horizontally to see every sensor.</p><dl class="pg-data-columns">' + groups + "</dl>" + sampleTable +
      "</section></div></section>";
  }

  function paintDatasetDetails(dataset) {
    var box = $("#pg-data-detail");
    if (box) { box.outerHTML = dataDetailsHtml(dataset); }
  }

  // The settings of a box of the pipeline stand in groups named after the
  // steps of the box's own flowchart that they decide, in the order those
  // steps run, each with the step's picture: "Cap the labels" with the label
  // ceiling and whether the test labels are capped, then "Smooth each
  // sensor" with its weight, and so on. A group's name replaces the name of
  // a list of options it holds alone, whose i then stands beside the group's
  // name, and a switch that turns the whole step on or off stands before it.
  // So no i ever stands without a name beside it. Training has
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
      { name: "Turn into numbers per sensor",
        picture: "project", fields: ["temporal_out_features"] }
    ],
    // With the graph layers out of the network, the sensor graph, its
    // rescaling and the layers' sizes decide nothing: their groups stay in
    // view, faded and out of use.
    graph: [
      { name: "Graph layers in the network", picture: "gcn", fields: ["spatial"] },
      { name: "Sensor graph", picture: "graph", fields: ["graph"], unused: graphLayersOut },
      { name: "Rescale the graph", picture: "table", fields: ["adjacency_normalization"], unused: graphLayersOut },
      { name: "Graph layers", picture: "gcn", fields: ["gcn_hidden", "gcn_bias"], unused: graphLayersOut,
        about: "Each layer combines a sensor's numbers with those of its connected neighbours, using the graph's fixed weights. " +
          "Learned multipliers then turn that combination into new numbers for the sensor. Optional learned offsets are added, " +
          "and negative results become zero. The multipliers are shared across sensors. Each layer's width sets how many " +
          "numbers it produces per sensor. Repeating the operation lets information travel through successive connections." },
      { name: "Fully connected", picture: "dense", fields: ["fc_hidden"] }
    ],
    training: [
      { name: "Procedure", picture: "hold", procedure: true },
      { name: "Passes", picture: "passes", fields: ["epochs", "batch_size"] },
      { name: "Learning rate", picture: "rate", fields: ["learning_rate", "final_learning_rate", "lr_schedule"] },
      { name: "Repeats", picture: "seeds", fields: ["seed", "seeds", "fixed_split"] },
      { name: "Device", picture: "device", fields: ["device"] }
    ]
  };

  function sectionName(section) { return typeof section.name === "function" ? section.name() : section.name || ""; }

  // Whether the graph layers are out of the network, the setting of "w/o GCN".
  function graphLayersOut() { return S.settings.spatial === REMOVED.spatial; }

  // Whether a group's settings are out of use: its step switched off, or a
  // step it depends on taken out of the network.
  function sectionOff(section) {
    return (!!section.toggle && !S.settings[section.toggle]) || (!!section.unused && section.unused());
  }

  function sectionNote(section) {
    var text = section.note ? section.note() : "";
    return '<p class="pg-faint pg-section-note"' + (text ? "" : " hidden") + ">" + esc(text) + "</p>";
  }

  function sectionHelp(section, destination) {
    if (section.procedure) { return infoCard(S.options.procedure_about, "", S.options.procedure_more); }
    var keys = section.toggle ? [section.toggle] : section.fields || [];
    var more = destination || (S.options.operation_more || {})[sectionName(section)];
    return keys.length === 1 ? hintFor(S.fields[keys[0]], more) : infoCard(
      section.about || keys.map(function (key) { return S.fields[key].about; }).filter(Boolean).join("\n"),
      keys.map(function (key) { return sourceNote(S.fields[key].source); }).filter(Boolean).join(" "),
      more || (keys.length ? S.fields[keys[0]].more : ""));
  }

  function sectionForStep(step) {
    var sections = SECTIONS[S.zoom] || [];
    return sections.filter(function (section) { return sectionName(section) === step.name; })[0] ||
      sections.filter(function (section) {
        return (step.parts || []).some(function (part) {
          return part.setting && (section.fields || []).indexOf(part.setting) >= 0;
        });
      })[0];
  }

  function stepHelp(step, section) {
    if (section && section.fields.length === 1 && section.fields[0] === "temporal") {
      var choice = S.fields.temporal.choices.filter(function (option) { return option.value === S.settings.temporal; })[0];
      return infoCard(choice.about, sourceNote(choice.source), step.more);
    }
    return section ? sectionHelp(section, step.more) :
      infoCard(step.does, step.source ? sourceTag(step.source) : "", step.more);
  }

  function locateOperation() {
    if (!S.locate || !S.zoomed || S.zoomed.stage !== S.zoom) { return; }
    var target = $all(".pg-vstep[data-operation]").filter(function (box) {
      return box.getAttribute("data-operation") === S.locate;
    })[0];
    if (target) {
      target.scrollIntoView({ block: "nearest" });
      target.classList.add("is-target");
      var button = target.querySelector(".pg-operation-link");
      if (button) { button.focus({ preventScroll: true }); }
      S.locate = null;
    }
  }

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
    var hidden = !toggle && fields.length > 0 && fields.every(function (f) { return !visible(f.key); });
    var off = sectionOff(section);
    var body = cells(lone ? id : null) + (section.note ? sectionNote(section) : "");
    if (off) { body = body.replace(/<(input|select)\b/g, "<$1 disabled"); }
    return '<section class="pg-section' + (off ? " is-off" : "") + '" data-section="' + group + "-" + index +
      '" aria-labelledby="' + id + '"' + (hidden ? " hidden" : "") + '><div class="pg-section-head"' +
      (toggle ? ' data-row="' + toggle.key + '"' : "") + ">" +
      '<span class="pg-section-picture">' + PICTURES[section.picture] + "</span>" +
      '<h3 class="pg-section-name pg-field-name">' + (group !== "training" && group !== "data"
        ? '<button type="button" class="pg-section-label pg-operation-link" id="' + id + '" data-action="locate-operation" data-group="' + group + '" data-operation="' + esc(name) + '">' + esc(name) + '</button>'
        : '<span class="pg-section-label" id="' + id + '">' + esc(name) + '</span>') +
      info(sectionHelp(section), name) + "</h3>" +
      (toggle ? '<span class="pg-section-switch">' + resetHtml(toggle) + switchControl("pg-" + toggle.key,
        !!S.settings[toggle.key], 'data-field="' + toggle.key + '" aria-labelledby="' + id + '"') + "</span>" : "") +
      "</div>" + body + "</section>";
  }

  // A group is hidden while every setting in it is, and a group's name that
  // follows the settings, "One GRU per sensor", follows them. A group whose
  // step is switched off stays in view, faded, with its settings out of use,
  // so the reader sees what switching it back on would bring.
  function paintSections() {
    $all(".pg-section").forEach(function (box) {
      var at = box.getAttribute("data-section").split("-"), section = SECTIONS[at[0]][Number(at[1])];
      var cells = Array.prototype.slice.call(box.querySelectorAll(".pg-fields > .pg-field"));
      box.hidden = !section.toggle && cells.length > 0 && cells.every(function (c) { return c.hidden; });
      var off = sectionOff(section);
      box.classList.toggle("is-off", off);
      var note = box.querySelector(".pg-section-note");
      if (note) {
        var text = section.note ? section.note() : "";
        if (note.textContent !== text) { note.textContent = text; }
        note.hidden = !text;
      }
      Array.prototype.forEach.call(box.querySelectorAll(".pg-fields input, .pg-fields select"), function (control) {
        control.disabled = off;
      });
      var label = box.querySelector(".pg-section-label"), name = sectionName(section);
      if (label && label.textContent !== name) { label.textContent = name; }
      if (label && label.hasAttribute("data-operation")) { label.setAttribute("data-operation", name); }
      var help = box.querySelector(".pg-section-head .pg-info-card");
      if (help) { help.innerHTML = sectionHelp(section).replace(/<a /g, '<a tabindex="-1" '); }
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
    return '<span class="pg-zoom-icon">' + ICONS[S.zoom] + "</span>" +
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
  // is one, the graph layers when there are any, and what comes out. A part
  // taken out leaves the + that puts it back on the arrow between the boxes
  // either side of it, so the flow reads preprocessing → temporal stage →
  // remaining life without the graph layers. The sensor graph they run on is
  // a step of their own flowchart, taken out there.
  function flowHtml() {
    var s = S.settings;
    var html = '<div class="pg-flow-end pg-flow-in" role="listitem"><button type="button" class="pg-flow-input" data-zoom="' +
      INPUT + '" aria-label="' + esc("Data: " + s.subset + ", its flowchart") + '"><span class="pg-flow-input-icon' +
      (S.step === INPUT ? " is-current" : "") + '" data-step="' + INPUT + '">' + ICONS.data + "</span>" + esc(s.subset) +
      "</button></div>" + flowLink("") + stageBox("preprocessing", "");
    var gap = s.temporal === REMOVED.temporal;
    html += gap ? flowLink(addButton("temporal"))
      : flowLink("") + stageBox("temporal", removeButton("temporal", "pg-stage-x"));
    if (s.spatial === REMOVED.spatial) {
      html += flowLink(addButton("spatial"));
    } else {
      html += (gap ? "" : flowLink("")) + stageBox("graph", removeButton("spatial", "pg-stage-x")) +
        '<div class="pg-flow-link pg-flow-out" aria-hidden="true"></div>';
    }
    return html + '<div class="pg-flow-end pg-flow-out" role="listitem">' + ICONS.output + "Remaining life</div>";
  }

  // One box: its step's number in the top left corner, its picture, and
  // under it what the step is now, "LSTM"; its step's name is on the step's
  // tab, and the box is read out with both. The whole box opens the step's
  // flowchart; x is its cross, in the top right corner.
  function stageBox(group, x) {
    var name = boxName(group);
    return '<div class="pg-stage" role="listitem">' +
      '<button type="button" class="pg-stage-main" data-zoom="' + group + '" aria-label="' +
      esc((groupLabel(group) === name ? name : groupLabel(group) + ": " + name) + ", its flowchart") +
      '"><span class="pg-stage-icon">' + ICONS[group] + "</span>" +
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

  // Every displayed shape uses the same help card, including branches and endpoints.
  function outputInfo(item) {
    var outputs = item.handoff_outputs || item.outputs || [];
    if (!outputs.length) { return ""; }
    var note = item.output_note || (item.handoff_outputs ?
      "The arrow summarises the training file. The groups below stay separate." : "");
    var sameAxes = outputs.every(function (output) { return output.axes === outputs[0].axes; });
    return info((note ? '<span class="pg-info-line">' + esc(note) + '</span>' : '') +
      outputs.map(function (output) {
        return '<span class="pg-transfer-output"><span>' + esc(output.label) + '</span><code>' + esc(output.shape) + '</code></span>' +
          (!sameAxes ? '<span class="pg-transfer-axes">' + esc(output.axes) + '</span>' : '');
      }).join('') + (sameAxes ? '<span class="pg-transfer-axes">' + esc(outputs[0].axes) + '</span>' : ''),
      "outputs after " + item.name);
  }

  function shapeHtml(shape, details) {
    return '<span class="pg-arrow-label"><code class="pg-vshape">' + esc(shape) + '</code>' + (details || '') + '</span>';
  }

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
      var details = outputInfo(item);
      var after = last || next.detached ? "" : item.handoff ?
        '<div class="pg-varrow pg-varrow--named"><span class="pg-transfer">' +
        esc(item.handoff) + details + "</span></div>" : next && next.kind === "branch" ?
        joinHtml(handedOn(item), next, details) : arrowHtml(handedOn(item), false, details);
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
    if (item.kind === "evaluation") {
      return '<section class="pg-evaluation-path"><h3>' + esc(item.name) + '</h3><p>' + esc(item.summary) + '</p>' + vflowHtml(item.steps) + '</section>';
    }
    if (item.kind === "edge") {
      var own = item.shape && !(first && !last) && !(last && item.shape === handedOn(before));
      return '<span class="pg-vedge">' + (ICONS[item.icon] || "") + "<span>" + esc(item.name) + "</span></span>" +
        (item.handoff ? (last ? '<span class="pg-transfer">' + esc(item.handoff) + "</span>" : "") :
          own ? shapeHtml(item.shape, outputInfo(item)) : "") + storyHtml(item);
    }
    if (item.kind === "group") {
      if (item.lanes) {
        return '<div class="pg-vlanes">' + item.lanes.map(function (lane) {
          return '<section class="pg-vlane"><h4>' + esc(lane.name) + '</h4>' +
            vflowHtml(lane.steps) + (lane.shape ? arrowHtml(lane.shape, false, outputInfo(lane)) : "") + '</section>';
        }).join("") + '</div>';
      }
      return '<div class="pg-vgroup">' + item.steps.map(function (s) { return vstepHtml(s, true); }).join("") + "</div>";
    }
    return vstepHtml(item, false);
  }

  // The arrow under an item, carrying the shape it hands on, if any.
  function arrowHtml(shape, long, details) {
    return '<div class="pg-varrow' + (long ? " pg-varrow--long" : "") + '">' + (shape ? '<span class="visually-hidden">' +
      "hands on </span>" + shapeHtml(shape, details) : "") + "</div>";
  }

  // The main line runs straight down to the step after the branch, and the
  // branch's steps stand beside it and end in an arrow into the same step.
  function joinHtml(shape, branch, details) {
    return '<div class="pg-vjoin"><div class="pg-vjoin-main">' + arrowHtml(shape, true, details) + "</div>" +
      '<ol class="pg-vjoin-side">' + branch.steps.map(function (s) {
        return '<li class="pg-vitem">' + vstepHtml(s, false) + arrowHtml(handedOn(s), false, outputInfo(s)) + "</li>";
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
    var section = sectionForStep(step), operation = section && sectionName(section);
    var words = stepHelp(step, section);
    var row = section && (section.toggle || section.fields[0]);
    var destination = row ? ' data-goto="' + S.fields[row].group + '" data-goto-row="' + row + '"' : '';
    var part = partControl(step);
    return '<div class="pg-vstep' + (small ? " pg-vstep--small" : "") + (part ? " has-part" : "") + '"' +
      (operation ? ' data-operation="' + esc(operation) + '"' + destination : '') + '>' +
      (PICTURES[step.picture] ? '<span class="pg-vstep-picture">' + PICTURES[step.picture] + "</span>" : "") +
      '<span class="pg-vstep-name">' + (destination ? '<button type="button" class="pg-operation-link"' + destination + '>' + esc(step.name) + '</button>' + info(words, step.name)
        : withInfo(step.name, info(words, step.name), function (w) { return esc(w); })) +
      "</span>" + (values ? '<span class="pg-vstep-values">' + values + "</span>" : "") +
      storyHtml(step) +
      (step.networks ? networksHtml(step.networks) : "") +
      (step.learned ? '<span class="pg-vstep-learned">' + count(step.learned) + " learned numbers</span>" : "") +
      (small && step.shape ? shapeHtml(step.shape, outputInfo(step)) : "") + part + "</div>";
  }

  // Keep the flow readable; full explanations belong in the step's help card.
  function storyHtml(step) {
    return step.summary ? '<p class="pg-story-summary">' + esc(step.summary) + "</p>" : "";
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
    // The graph layers' box is taken out by its own setting, not by its group's name.
    var part = was === "graph" ? "spatial" : was;
    var box = $('button[data-zoom="' + was + '"]') || $('[data-action="add-part"][data-part="' + part + '"]');
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
        paintSections();
        paintZoom();
        locateOperation();
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

  // What the pipeline calls a part with a given value: "LSTM", "Physics prior",
  // "GCN" for the graph layers, as their box names them.
  function partName(part, value) {
    if (part === "spatial") { return "GCN"; }
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
      control: '<p class="pg-procedure-note">This choice sets the engine groups during Preprocessing, before rescaling. ' +
        'Validation checks progress later, during training.</p>' +
        optionList("pg-procedure", choices, chosenProcedure().key, "data-procedure", null,
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
  function hintFor(f, more) {
    var about = [f.about];
    if (f.key === "subset") {
      var t5 = (S.options.dataset || {})[S.settings.subset];
      if (t5) {
        about.push(S.settings.subset + ": " + count(t5.train_engines) + " training and " + count(t5.test_engines) +
          " test engines, " + plural(t5.conditions, "operating condition") + ", " + plural(t5.fault_modes, "fault mode") + ".");
      }
    }
    return infoCard(about.filter(Boolean).join("\n"), sourceNote(f.source), more || f.more);
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
      control = optionList(id, f.choices, value, data, null,
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
    // Design settings keep their help on the section title, not on each control.
    return cell({
      key: f.key, id: id, label: bare ? "" : parts.label, infoName: f.label, hidden: !visible(f.key),
      size: f.group === "data" ? "wide" : size, inline: f.kind === "bool",
      control: control,
      reset: resetHtml(f)
    });
  }

  // The sub-dataset and the graph layers have no Reset of their own: each is
  // a short list whose first option is the default, one click away, and
  // Reset all still returns both.
  function resetHtml(f) {
    if (f.key === "subset" || f.key === "spatial") { return ""; }
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
    if (keys.indexOf("subset") >= 0 || keys.indexOf("window_size") >= 0) {
      paintDatasetDetails(null);
    }
    S.options.fields.forEach(function (other) {
      var box = $('[data-row="' + other.key + '"]');
      if (box) { box.hidden = !visible(other.key); }
    });
    paintProcedure();
    paintSections();
    paintPipeline();
    requestZoom(false);
    paintTabs();
    var resetAll = $('[data-action="reset-all"]');
    if (resetAll) { resetAll.disabled = !settingsChanged(); }
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
        paintDatasetDetails(preview.dataset);
      }).catch(function (error) {
        if (asked !== JSON.stringify(S.settings)) { return; }
        drawRunCard(box, null, error.message);
        paintDatasetDetails({ table5: (S.options.dataset || {})[S.settings.subset], error: error.message });
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
      network = '<div class="pg-network">' +
        '<div class="pg-big">' + count(preview.parameters) + " <span>learned numbers</span></div>" +
        '<section class="pg-layers" aria-labelledby="pg-layers-title"><h4 id="pg-layers-title">Layer by layer</h4>' +
        '<table class="pg-mini"><thead><tr><th>Layer</th><th>Out</th><th class="num">Learned numbers</th></tr></thead><tbody>' +
        // layer_summary() already folds the temporal stage into Table 4's Input row.
        preview.layers.map(function (l) {
          return "<tr><td>" + esc(l.layer) + (l.from ? ' <span class="pg-faint">from ' + esc(l.from) + "</span>" : "") +
            "</td><td><code>" + esc(l.out) + '</code></td><td class="num">' + count(Number(l.parameters)) + "</td></tr>";
        }).join("") + "</tbody></table>" +
        '<p class="pg-faint">B is the number of windows in a batch. Table 4 prints no counts; each is read from the built model.</p>' +
        "</section></div>";
    }

    // The card is drawn again after every change of settings. What the
    // keyboard was on in it keeps the focus.
    var active = box.contains(document.activeElement) ? document.activeElement : null;
    var had = active && (active.matches(".pg-train") ? ".pg-train" : null);
    var summary = [S.settings.subset, stageName(S.settings.temporal, S.settings.temporal_hidden),
      graphLayersOut() ? "w/o GCN: no graph layers" : graphName(S.settings.graph, S.settings.adjacency_normalization) + " graph",
      S.settings.window_size + " cycles per window"].join(" · ");
    box.innerHTML = '<div class="pg-section-head"><span class="pg-section-picture">' + PICTURES.dense +
      '</span><h3 class="pg-section-name">Run summary</h3></div><p class="pg-run-summary">' + esc(summary) +
      '</p>' + network + caveatList(notes) + '<div class="pg-run-action">' +
      '<button type="button" class="button button--primary pg-train" data-action="run"' + (canRun ? "" : " disabled") +
      ">Train " + plural(S.settings.seeds, "seed") + "</button></div>";
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
      if (S.tab === "results" && !S.selected) {
        drawResultsBody();
        if (S.pendingScroll !== undefined && S.pendingScroll !== null) { scrollToTop(S.pendingScroll); S.pendingScroll = null; }
      }
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
  // "all" is every state. Until 2026-11-14 the lists began at a value of
  // their own, "any", which the State menu could not return to: choosing a
  // state and then "All" again left both lists empty.
  function resultRows() {
    var which = S.resultsTab, kind = tabKind(which);
    var f = which === "runs" ? S.filter : S.xfilter;
    var jobs = orderedJobs().filter(function (job) {
      if (job.kind !== kind || job.status === "done") { return false; }
      var state = isLive(job) ? "live" : "ended";
      return f.state === "all" || f.state === state;
    }).map(function (job) { return { job: job }; });
    var stored = f.state === "all" || f.state === "done" ? (which === "runs" ? S.runs : S.explanations) || [] : [];
    stored = stored.filter(which === "runs" ? runShown : explanationShown).map(function (item) { return { item: item }; });
    return jobs.concat(stored);
  }

  function runShown(r) {
    var f = S.filter;
    return (f.subset === "all" || r.subset === f.subset) && (f.temporal === "all" || r.temporal === f.temporal) &&
      (f.graph === "all" || r.graph === f.graph) &&
      (f.pass === "all" || (f.pass === "final") === (r.reported_weights === "final_epoch")) &&
      (!f.results || !r.exploratory) && (!f.files || r.has_weights) &&
      Object.keys(f.more).every(function (key) { return paramToken(r, key) === f.more[key]; });
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
    var params = which === "runs" ? runParameters(S.runs || []) : [];
    if (which === "runs") { tidyFilters(params); }
    var rows = resultRows();
    var stored = rows.filter(function (row) { return row.item; }).length;
    var pages = Math.max(1, Math.ceil(rows.length / PAGE_ROWS));
    S.page[which] = Math.min(S.page[which] || 0, pages - 1);
    var first = S.page[which] * PAGE_ROWS;
    var shown = rows.slice(first, first + PAGE_ROWS);
    var none = which === "runs" ? "No run matches these filters." : "No explanation matches these filters.";
    body.innerHTML = '<div class="pg-filters">' + (which === "runs" ? runFilters() : explanationFilters()) + "</div>" +
      (which === "runs" ? filterBar(params, stored) + moreFilters(params) + '<div id="pg-compare"></div>' : "") +
      (shown.length ? (which === "runs" ? runsTable(shown) : explanationsList(shown)) +
        pagerHtml(first, shown.length, rows.length) : '<div class="pg-empty">' + none + "</div>");
    S.jobRows = jobSignature();
    if (which === "runs") { drawComparison(); }
    paintBanner();
  }

  // A filter: a name above a menu of choices, "All" first.
  function filterSelect(name, key, attr, choices, current) {
    var id = "pg-" + attr + "-" + key;
    return '<label class="pg-filter' + (current !== "all" ? " is-used" : "") + '" for="' + id + '"><span>' + name + "</span>" +
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

  // ---- a filter for every parameter
  //
  // A stored run arrives with every setting it recorded (its "parameters"),
  // named as the comparison of two runs names them: "preprocessing.rul_max",
  // "model.gcn_bias", "training.epochs", "seeds", "split", "commit".

  // No filter in use. open says whether the filters under "More filters"
  // are on show, which clearing the filters leaves as it is.
  function noFilters(open) {
    return { subset: "all", temporal: "all", graph: "all", pass: "all", state: "all", results: false, files: false,
      more: {}, open: !!open };
  }

  // The parameters the first row of filters already covers.
  var FIRST_ROW = { "preprocessing.subset": true, "model.temporal": true, "preprocessing.graph": true, "reported_weights": true };
  // The value of a menu's choice for the runs that recorded nothing. A
  // recorded value is written the way JSON writes it, so none can equal it.
  var UNRECORDED = "unrecorded";
  // The groups of the design form, in its order, then what only a run's record holds.
  var PARAM_GROUPS = ["data", "preprocessing", "temporal", "graph", "training"];
  // How the engines were split is recorded as "split"; the form's switch for it is "fixed_split".
  var PARAM_FIELD = { split: "fixed_split" };
  var FILTER_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 4.5h11M4.5 8h7M6.5 11.5h3" fill="none" ' +
    'stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
  var DOWN = '<svg class="pg-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 4.5 L6 8 L9.5 4.5" fill="none" ' +
    'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  // What a run recorded for one parameter, as text two runs can be compared
  // by, or UNRECORDED when it recorded nothing.
  function paramToken(run, key) {
    var p = run.parameters || {};
    return Object.prototype.hasOwnProperty.call(p, key) ? JSON.stringify(p[key]) : UNRECORDED;
  }

  // A parameter's name: the design form's where the form has a control for
  // it, otherwise the name a run's Settings part gives the record.
  function paramLabel(key) {
    var last = key.split(".").pop();
    var field = S.fields && S.fields[last];
    return field ? field.label : RECORD_LABELS[key] || TRAINING_LABELS[last] || sentenceCase(last.replace(/_/g, " "));
  }

  // The group of the design form a parameter stands in. The form has no
  // control for how the engines were split, for what only the training
  // files hold or for the commit; they join the group they belong to.
  function paramGroup(key) {
    var parts = key.split("."), field = S.fields && S.fields[parts[parts.length - 1]];
    var group = field ? (field.group === "procedure" ? "training" : field.group)
      : key === "split" ? "training" : parts.length < 2 ? "record" : parts[0] === "model" ? "graph" : parts[0];
    return PARAM_GROUPS.indexOf(group) < 0 ? "record" : group;
  }

  // A value in the design form's words. A setting left empty says what that
  // means, and a switch says On or Off.
  function paramWords(key, value) {
    if (value === null) { return UNSET_WORDS[key.split(".").pop()] || "None"; }
    if (value === true || value === false) { return value ? "On" : "Off"; }
    if (key === "split") { return sentenceCase(RECORD_WORDS[value] || String(value)); }
    return settingWords(key, value);
  }

  // Every parameter the stored runs recorded that the first row has no
  // filter for, in the design form's order, each with the values the runs
  // hold for it. It varies when they hold at least two.
  function runParameters(runs) {
    var seen = {}, keys = [], place = {};
    runs.forEach(function (run) {
      var p = run.parameters || {};
      Object.keys(p).forEach(function (key) {
        if (FIRST_ROW[key]) { return; }
        if (!seen[key]) { seen[key] = { values: {}, recorded: 0 }; keys.push(key); }
        seen[key].values[JSON.stringify(p[key])] = p[key];
        seen[key].recorded += 1;
      });
    });
    ((S.options && S.options.fields) || []).forEach(function (f, i) { place[f.key] = i; });
    return keys.map(function (key) {
      var values = seen[key].values, last = key.split(".").pop();
      var choices = Object.keys(values).map(function (token) {
        return { value: token, raw: values[token], label: paramWords(key, values[token]) };
      }).sort(function (a, b) {
        return typeof a.raw === "number" && typeof b.raw === "number" ? a.raw - b.raw
          : a.label.localeCompare(b.label, "en", { numeric: true });
      });
      if (seen[key].recorded < runs.length) { choices.push({ value: UNRECORDED, label: "Not recorded" }); }
      var at = place[PARAM_FIELD[key] || last];
      return { key: key, label: paramLabel(key), group: paramGroup(key), place: at === undefined ? 1000 : at,
        choices: choices, varies: Object.keys(values).length > 1 };
    }).filter(function (p) {
      // What only a run's record holds, such as its commit, is given no
      // filter (the author's decision of 2026-11-14): it is not a setting.
      return p.group !== "record";
    }).sort(function (a, b) {
      return PARAM_GROUPS.indexOf(a.group) - PARAM_GROUPS.indexOf(b.group) || a.place - b.place ||
        a.label.localeCompare(b.label, "en");
    });
  }

  // A filter on a value no stored run holds any more is taken away, so no
  // menu shows "All" while it still narrows the list.
  function tidyFilters(params) {
    var more = S.filter.more;
    Object.keys(more).forEach(function (key) {
      var p = params.filter(function (x) { return x.key === key; })[0];
      if (!p || !p.varies || !p.choices.some(function (c) { return c.value === more[key]; })) { delete more[key]; }
    });
  }

  // How many filters are in use: in the first row, and under "More filters".
  function filtersInUse() {
    var f = S.filter;
    return {
      first: ["subset", "temporal", "graph", "pass"].filter(function (key) { return f[key] !== "all"; }).length +
        (f.state !== "all" ? 1 : 0) + (f.results ? 1 : 0) + (f.files ? 1 : 0),
      more: Object.keys(f.more).length
    };
  }

  // After a filter moved: the list goes back to its first page, and the
  // focus stays on the control that was used.
  function filtersChanged(id, samePage) {
    if (!samePage) { S.page[S.resultsTab] = 0; }
    drawResultsBody();
    var again = id ? $("#" + id) : null;
    if (again) { again.focus({ preventScroll: true }); }
  }

  // Under the first row: the way to the filters for every parameter, each
  // one in use while they are put away, how many runs are left, and the way
  // to take every filter off at once.
  function filterBar(params, shown) {
    var f = S.filter, used = filtersInUse(), total = (S.runs || []).length;
    var chips = f.open ? "" : Object.keys(f.more).map(function (key) {
      var p = params.filter(function (x) { return x.key === key; })[0];
      var choice = p.choices.filter(function (c) { return c.value === f.more[key]; })[0];
      return '<li class="pg-filter-chip"><span>' + esc(p.label) + ": <strong>" + esc(choice.label) + "</strong></span>" +
        '<button type="button" data-action="drop-filter" data-param="' + esc(key) + '" aria-label="Remove the filter ' +
        esc(p.label) + '">' + CROSS + "</button></li>";
    }).join("");
    return '<div class="pg-filterbar"><button type="button" class="button button--small pg-more-toggle" id="pg-more-toggle" ' +
      'data-action="more-filters" aria-expanded="' + f.open + '" aria-controls="pg-more">' + FILTER_ICON + "More filters" +
      (used.more ? '<span class="pg-badge" title="' + esc(plural(used.more, "filter") + " in use here") + '">' +
        count(used.more) + "</span>" : "") + DOWN + "</button>" +
      (chips ? '<ul class="pg-filter-chips" aria-label="Filters in use">' + chips + "</ul>" : "") +
      '<p class="pg-filterbar-count" aria-live="polite">' +
      (shown === total ? plural(total, "run") : count(shown) + " of " + plural(total, "run")) + "</p>" +
      (used.first + used.more ? '<button type="button" class="button button--small button--quiet" data-action="clear-filters">' +
        "Clear filters</button>" : "") + "</div>";
  }

  // The filters for every parameter, in the groups of the design form. A
  // parameter on which every run agrees is named with its value instead of
  // being given a menu, since there is nothing to choose between. A group of
  // up to three menus takes half the panel's width where the panel is wide,
  // so two short groups share a line; beside is how many halves are open on
  // the line being filled, and decides whether the last block fits there.
  function moreFilters(params) {
    var f = S.filter, beside = 0;
    var groups = PARAM_GROUPS.map(function (group) {
      var own = params.filter(function (p) { return p.group === group && p.varies; });
      if (!own.length) { return ""; }
      var half = own.length <= 3;
      beside = half ? (beside + 1) % 2 : 0;
      return '<div class="pg-more-group' + (half ? " pg-more-half" : "") + '" role="group" aria-labelledby="pg-more-' + group +
        '"><h4 id="pg-more-' + group + '">' +
        esc(groupLabel(group)) + '</h4><div class="pg-filters">' + own.map(function (p) {
          var id = "pg-param-" + p.key.replace(/[^a-z0-9]+/gi, "-");
          return '<label class="pg-filter' + (f.more[p.key] ? " is-used" : "") + '" for="' + id + '"><span>' + esc(p.label) +
            "</span>" + selectField(id, [{ value: "all", label: "All" }].concat(p.choices), f.more[p.key] || "all",
              'data-param="' + esc(p.key) + '"') + "</label>";
        }).join("") + "</div></div>";
    }).join("");
    var same = params.filter(function (p) { return !p.varies; });
    return '<section class="pg-more" id="pg-more" aria-label="More filters"' + (f.open ? "" : " hidden") + ">" +
      (groups || '<p class="pg-hint">The stored runs differ in nothing but what the filters above choose.</p>') +
      (same.length ? '<div class="pg-more-same' + (beside ? " pg-more-half" : "") +
        '"><p>The same in every stored run that recorded it, so there is nothing to filter by:</p><dl>' +
        same.map(function (p) { return "<div><dt>" + esc(p.label) + "</dt><dd>" + esc(p.choices[0].label) + "</dd></div>"; }).join("") +
        "</dl></div>" : "") + "</section>";
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
    return filterSelect("Method", "method", "data-xfilter", Object.keys(METHODS).map(function (m) {
      return { value: m, label: METHODS[m] };
    }), f.method) +
      filterSelect("What", "scope", "data-xfilter", [{ value: "engine", label: "One engine over time" },
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
    S.listScroll = scrolledTo(); S.fromList = true;
    S.selected = name; S.detail = null; S.runTab = "overview";
    setAddress();
    scrollToTop(0);
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
          "<td>" + esc(stageName(r.temporal, r.temporal_hidden)) + "</td><td>" + esc(runGraphName(r)) + "</td>" +
          '<td><span class="chip' + (r.exploratory ? " chip--warn" : "") + '">' + count(r.seeds) + "</span></td>" +
          "<td>" + (r.reported_weights === "final_epoch" ? "final pass" :
            r.reported_weights === "best_validation_epoch" ? "best pass" : "not recorded") +
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
          esc(methodLabel(x)) + "</td><td>" + esc(explainedWhat(x)) + "</td><td>" + esc(x.seed) + "</td>" +
          '<td class="num">' + count(x.windows) + "</td><td>" + esc(x.run_subset || "") + "</td><td>" +
          esc([stageName(x.run_temporal, x.run_temporal_hidden),
            runGraphName({ spatial: x.run_spatial, graph: x.run_graph })].filter(Boolean).join(" · ")) +
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
  // What a setting left empty means, and the names of the training settings:
  // said the same way in a run's Settings part and in the filters of the list.
  var UNSET_WORDS = { smoothing_alpha: "Off", early_stopping_patience: "Off", gradient_clip: "Off",
    lr_step_epoch: "Automatic: halfway when using the step schedule", temporal_hidden: "Not used" };
  var TRAINING_LABELS = {
    epochs: "Maximum epochs (passes over the training data)", batch_size: "Windows per update",
    learning_rate: "Initial learning rate", final_learning_rate: "Final learning-rate setting",
    optimizer: "Parameter update rule", loss: "Training error measure", lr_schedule: "Learning-rate schedule",
    lr_step_epoch: "Pass at the learning-rate step", early_stopping_patience: "Passes without improvement before stopping",
    min_improvement: "Minimum improvement to reset the wait", gradient_clip: "Gradient limit (loss derivatives)",
    seed: "Training seed", device: "Requested computing device", notes: "Saved notes"
  };

  function savedSettingValue(key, value) {
    if (value === undefined) { return '<span class="pg-setting-missing">Not recorded</span>'; }
    if (value === null) { return esc(UNSET_WORDS[key] || "None"); }
    if (typeof value === "boolean") { return value ? "Yes" : "No"; }
    if (Array.isArray(value)) { return esc(JSON.stringify(value)); }
    if (typeof value === "object") { return esc(JSON.stringify(value)); }
    var names = {
      zscore: "Subtract the mean and divide by the spread", minmax_global: "One range per sensor across training readings",
      minmax_per_condition: "Separate ranges per sensor and operating condition", physics: "Physics graph",
      complete: "Every sensor joined to every other", correlation: "Most correlated sensor pairs",
      random: "41 sensor pairs picked at random", rewired: "The physics graph rewired, connection counts kept",
      row: "Divide each receiving row by its total", symmetric: "Rescale using both sensors' connection counts",
      lstm: "LSTM", gru: "GRU", last_cycle: "Keep the last reading", mean_window: "Average the window",
      raw_window: "Pass every cycle", window_summary: "Three numbers per sensor",
      adam: "Adam", mse: "Mean squared error", step: "One step down", exponential: "Exponential reduction",
      cosine: "Cosine reduction", plateau: "Reduce when validation stops improving", constant: "Constant rate",
      fixed: "Same engine split for every seed", varies_with_seed: "Engine split varies with the seed",
      none: "No engines held back, so no split",
      best_validation_epoch: "Best pass on held-back engines", final_epoch: "Final training pass",
      auto: "Automatic device selection", cpu: "CPU", cuda: "CUDA graphics processor"
    };
    // "none" already names a split; for the graph layers it means there are none.
    if (key === "spatial") { names = { gcn: "Three graph layers", none: "No graph layers (w/o GCN)" }; }
    var namedChoice = ["normalization", "graph", "adjacency_normalization", "temporal", "optimizer",
      "loss", "lr_schedule", "split", "reported_weights", "device", "spatial"].indexOf(key) >= 0;
    return namedChoice && Object.prototype.hasOwnProperty.call(names, value)
      ? esc(names[value]) + ' <span class="pg-setting-raw">(' + esc(value) + ")</span>" : esc(String(value));
  }

  function savedSettingsGroup(title, source, labels, values, renderValue) {
    var keys = Object.keys(labels);
    Object.keys(values).forEach(function (key) { if (keys.indexOf(key) < 0) { keys.push(key); } });
    return '<section class="pg-settings-group"><div class="pg-settings-group-head"><h4>' + esc(title) + '</h4><p class="pg-setting-source">' +
      esc(source) + '</p></div><dl>' + keys.map(function (key) {
        return '<div class="pg-setting-row" data-setting="' + esc(key) + '"><dt>' +
          esc(labels[key] || key.replace(/_/g, " ")) + '</dt><dd>' +
          (renderValue ? renderValue(key) : savedSettingValue(key, values[key])) + '</dd></div>';
      }).join("") + '</dl></section>';
  }

  function runConfiguration(d) {
    var summary = d.recorded_summary || {}, preprocessing = copy(d.preprocessing || {});
    var model = copy(d.model || {}), training = copy(summary.training || d.training || {});
    // A run from before the training settings were stored wrote its
    // learning-rate schedule in its summary alone; it is shown from there,
    // as the list's filter for it reads it.
    if (training.lr_schedule === undefined && summary.lr_schedule !== undefined && summary.lr_schedule !== null) {
      training.lr_schedule = summary.lr_schedule;
    }
    var preparationFallback = {}, modelFallback = {};
    ["subset", "graph", "adjacency_normalization", "clip_test_labels", "validation_fraction"].forEach(function (key) {
      if (preprocessing[key] === undefined && summary[key] !== undefined) {
        preprocessing[key] = summary[key]; preparationFallback[key] = true;
      }
    });
    Object.keys({temporal: 1, temporal_hidden: 1}).forEach(function (key) {
      var saved = summary[key === "temporal" ? "temporal_stage" : key];
      if (model[key] === undefined && saved !== undefined) { model[key] = saved; modelFallback[key] = true; }
    });
    // A run stored before the graph layers could be left out recorded no
    // choice and had them; the server says so, and the page marks it.
    var spatialAssumed = model.spatial === undefined && d.spatial !== undefined;
    if (spatialAssumed) { model.spatial = d.spatial; }
    var seeds = Object.keys(d.per_seed || {});
    (d.results || []).forEach(function (row) {
      if (row.seed !== null && row.seed !== undefined && seeds.indexOf(String(row.seed)) < 0) { seeds.push(String(row.seed)); }
    });
    seeds.sort(function (a, b) { return Number(a) - Number(b); });
    var trainingKeys = copy(training), recordedSeeds = {};
    seeds.forEach(function (seed) {
      var saved = ((d.per_seed || {})[seed] || {}).training || {};
      recordedSeeds[seed] = saved;
      Object.keys(saved).forEach(function (key) { trainingKeys[key] = saved[key]; });
    });
    function trainingValue(key) {
      if (!seeds.length) { return savedSettingValue(key, training[key]); }
      var rows = seeds.map(function (seed) {
        var saved = recordedSeeds[seed], fromSummary = !Object.keys(saved).length;
        return { seed: seed, value: savedSettingValue(key, (fromSummary ? training : saved)[key]), fromSummary: fromSummary };
      });
      var same = rows.every(function (row) { return row.value === rows[0].value && row.fromSummary === rows[0].fromSummary; });
      function rendered(row) {
        return row.value + (row.fromSummary && training[key] !== undefined ? ' <span class="pg-setting-raw">(run summary)</span>' : "");
      }
      return same ? rendered(rows[0]) : '<ul class="pg-setting-seeds">' + rows.map(function (row) {
        return '<li><span>Seed ' + esc(row.seed) + ':</span> ' + rendered(row) + '</li>';
      }).join("") + '</ul>';
    }
    var raw = { "summary.json": summary, "preprocessing_config.yaml": d.preprocessing || {},
      "model_config.yaml": d.recorded_model || {} };
    seeds.forEach(function (seed) { raw['seed_' + seed + '/training_config.yaml'] = recordedSeeds[seed]; });
    return '<section class="pg-run-section pg-run-settings" id="pg-run-settings" aria-labelledby="pg-run-settings-title">' +
      '<div class="pg-run-section-head"><h3 id="pg-run-settings-title">Settings used for this run</h3>' +
      '<p class="pg-settings-intro">Values saved with this run. Missing values are marked “Not recorded”. ' +
      'Training settings are shown for every seed when they differ.</p>' +
      (d.name ? '<p class="pg-run-folder"><span>Folder</span><code>reports/' + esc(d.name) + "</code></p>" : "") +
      '</div><div class="pg-settings-grid">' +
      savedSettingsGroup("Data and preprocessing", "Saved preprocessing configuration; summary values identified when used", {
        subset: "Data subset", window_size: "Cycles per window", window_stride: "Cycles between windows",
        rul_max: "Remaining-life ceiling (cycles)", clip_test_labels: "Cap test answers at that ceiling",
        normalization: "Rescale sensor readings", smoothing_alpha: "Smoothing weight on the new reading",
        graph: "Sensor graph", adjacency_normalization: "Rescale the graph",
        validation_fraction: "Fraction of engines held back", seed: "First engine-split seed", notes: "Saved notes"
      }, preprocessing, function (key) {
        return savedSettingValue(key, preprocessing[key]) + (preparationFallback[key] ? ' <span class="pg-setting-raw">(run summary)</span>' : '');
      }) +
      savedSettingsGroup("Temporal algorithm and GNN", (summary.model ? "Saved model settings in the run summary." : "Saved model configuration.") +
        (model.temporal && model.temporal !== "lstm" && model.temporal !== "gru" ?
          " Memory size, conversion width and recurrent sharing are unused by this temporal route." : ""), {
        temporal: "Temporal algorithm", temporal_hidden: "Memory size", temporal_out_features: "Graph-input values per sensor",
        shared_across_sensors: "Share the recurrent network across sensors", spatial: "Graph layers in the network",
        gcn_hidden: "Values after each graph layer", fc_hidden: "Fully connected width", gcn_bias: "Add offsets in graph layers"
      }, model, function (key) {
        if (key === "spatial" && spatialAssumed) {
          return savedSettingValue(key, model[key]) + ' <span class="pg-setting-raw">(not recorded: stored before the choice existed)</span>';
        }
        return savedSettingValue(key, model[key]) + (modelFallback[key] ? ' <span class="pg-setting-raw">(run summary)</span>' : '');
      }) +
      savedSettingsGroup("Training", "Per-seed training files, with run-summary values identified when used",
        TRAINING_LABELS, trainingKeys, trainingValue) + '</div>' +
      '<details class="pg-recorded-files"><summary>Inspect the complete saved configuration and source record</summary>' +
      '<p>These are the recorded fields, including additional settings and the software environment when available. ' +
      'An empty object means no readable record was available.</p><pre><code>' +
      esc(JSON.stringify(raw, null, 2)) + '</code></pre></details></section>';
  }

  function runInterpretation(cautions, compared) {
    var titles = { "A-26": "Temporal choices", "F-16": "Choosing the reported weights", "A-02": "Test-label ceiling",
      "D-6": "Last-reading temporal route", "Section 2.2": "Shared sensor network", "Table 4": "Different layer widths",
      "A-17": "Learning-rate schedule", "D-2": "Complete sensor graph", "A-27": "Graph varies with the training engines",
      "A-23": "Test engines left out", "A-42": "No graph layers" };
    return '<section class="pg-run-section pg-run-interpretation" id="pg-run-interpretation" aria-labelledby="pg-run-interpretation-title">' +
      '<div class="pg-run-section-head"><h3 id="pg-run-interpretation-title">How to interpret this run</h3>' + compared + "</div>" +
      '<div class="pg-run-notes">' + cautions.map(function (c) {
        var warn = c.level === "warn";
        return '<article class="pg-run-note' + (warn ? ' pg-run-note--warn' : '') + '"><p class="pg-run-note-kind">' +
          '<span class="pg-run-note-mark" aria-hidden="true">' + (warn ? "!" : "i") + "</span>" + (warn ? "Caution" : "Note") +
          "</p><h4>" + esc(titles[c.ref] || "Comparison limits") + '</h4><p>' + esc(c.text) + '</p>' +
          (c.ref ? '<span class="pg-run-note-ref">Background: ' + refLink(c.ref) + '</span>' : '') + '</article>';
      }).join("") + '</div></section>';
  }

  // One stored run. Its page shows one part at a time, chosen in a row of
  // tabs under its name: the results with the notes that qualify them, the
  // charts, the explanations stored for it, and the settings it ran with.
  function drawDetail() {
    var panel = $("#pg-panel");
    if (!panel || !S.selected) { return Promise.resolve(); }
    var held = S.detail && S.detail.name === S.selected;
    if (!held) { panel.innerHTML = '<div class="pg-run-page">' + runBack() + '<div class="loading">Reading the run…</div></div>'; }
    var load = held ? Promise.resolve(S.detail)
      : T.get("/api/playground/run?name=" + encodeURIComponent(S.selected));
    return load.then(function (d) {
      if (S.tab !== "results") { return; }
      S.detail = d;
      var which = runTabShown();
      var cautions = (d.caveats || []).filter(function (c) { return cautionFor(c, "run"); });
      S.chartsDrawnAt = chartWidth();
      var part = which === "charts" ? runCharts(d, cautions) : which === "explanations" ? runExplanations(d)
        : which === "settings" ? runConfiguration(d) : runOverview(d, cautions);
      panel.innerHTML = '<div class="pg-run-page">' + runBack() + '<section class="pg-detail">' + runHeader(d) + runTabs(d, which) +
        '<div class="pg-run-part" role="tabpanel" id="pg-run-part" aria-labelledby="pg-run-tab-' + which + '">' + part +
        "</div></section></div>";
      pageTitle();
    }).catch(function (error) {
      panel.innerHTML = '<div class="pg-run-page">' + runBack() + errorBox(error.message) + "</div>";
    });
  }

  function sentenceCase(text) { return text ? text.charAt(0).toUpperCase() + text.slice(1) : text; }

  // The way back to the list of runs.
  function runBack() {
    return '<nav class="pg-crumbs" aria-label="Breadcrumb"><a href="' + ROUTE + '/runs" data-action="back" class="pg-crumb-back">' +
      BACK + "All runs</a></nav>";
  }

  // Which run this is, what it was, and what can be done with it.
  function runHeader(d) {
    return '<header class="pg-run-head"><div class="pg-run-id">' +
      '<h2 class="pg-run-title" id="pg-run-title"><span class="pg-run-kind">Training run</span> ' + keepDate(runTime(d.name)) + "</h2>" +
      '<div class="chips">' +
      chip(d.dirty ? "warn" : "", "commit " + d.commit) + chip("", plural(d.seeds, "seed")) + chip("", d.subset) +
      (d.temporal ? chip("", stageName(d.temporal, d.temporal_hidden)) : "") +
      (d.graph ? chip("", graphName(d.graph, d.adjacency_normalization) + " graph") : "") +
      chip("", d.reported_weights === "final_epoch" ? "final pass reported" :
        d.reported_weights === "best_validation_epoch" ? "best pass restored" : "reported weights not recorded") + "</div></div>" +
      '<div class="pg-actions pg-run-actions">' +
      '<button type="button" class="button button--quiet pg-run-copy" data-action="copy-link" title="Copy a link to this run">' +
      LINK_ICON + "Copy link</button>" +
      '<button type="button" class="button" data-action="again">Set these settings up again</button>' +
      (d.has_weights ? '<button type="button" class="button button--primary" data-action="explain-this">Explain this run</button>' : "") +
      "</div></header>";
  }

  // The parts of the run, with how many explanations are stored for it.
  function runTabs(d, which) {
    var stored = (d.explanations || []).length;
    return '<div class="pg-subtabs pg-run-tabs" role="tablist" aria-label="Parts of this run">' + RUN_TABS.map(function (t) {
      var on = t[0] === which;
      return '<button type="button" role="tab" class="pg-subtab" id="pg-run-tab-' + t[0] + '" data-action="run-part" data-part="' + t[0] +
        '" aria-controls="pg-run-part" aria-selected="' + on + '" tabindex="' + (on ? 0 : -1) + '">' + t[1] +
        (t[0] === "explanations" ? ' <span class="pg-run-count" title="' + esc(plural(stored, "explanation") + " stored for this run") +
          '">' + count(stored) + "</span>" : "") + "</button>";
    }).join("") + "</div>";
  }

  // Another part of the run on show. The address names it without adding a
  // step to the browser's history. A page scrolled past its row of tabs
  // moves back to where the new part begins, so its start is never off screen.
  function showRunPart(which, reach, fromTab) {
    S.runTab = which;
    setAddress();
    return drawDetail().then(function () {
      var tabs = $(".pg-run-tabs"), head = $(".pg-run-head");
      if (tabs && head) {
        var past = tabs.getBoundingClientRect().top - head.getBoundingClientRect().bottom;
        if (past > 1) { scrollOn(-past); }
      }
      var place = reach ? document.getElementById(reach) : null;
      if (place) { place.scrollIntoView({ behavior: "instant", block: "start" }); }
      var tab = fromTab ? $("#pg-run-tab-" + which) : null;
      if (tab) { tab.focus({ preventScroll: true }); }
    });
  }

  // The first part: each measure with its spread over the seeds, the paper's
  // figure at the paper's own precision with the gap to it, and under them
  // the notes that say how far the comparison goes.
  function runOverview(d, cautions) {
    var seeds = Object.keys(d.per_seed);
    var paperRow = d.paper_row;
    var rmseOnly = !!paperRow && (paperRow.score === null || paperRow.score === undefined);
    var cards = MEASURES.map(function (m) {
      var mean = (d.mean || {})[m[0]], sd = (d.std || {})[m[0]];
      var paper = paperRow ? paperRow[m[0]] : null;
      var published = paper !== null && paper !== undefined;
      var shown = published ? paperNumber(m[0], paper) : null;
      var gap = published && mean !== null && mean !== undefined ? mean - paper : null;
      // Where the line under the tiles already says the paper prints RMSE
      // only, the other three tiles do not each repeat it.
      var saidBelow = rmseOnly && m[0] !== "rmse";
      return '<div class="pg-metric"><div class="pg-metric-name">' + m[1] + '</div><div class="pg-metric-value">' + number(mean, m[2]) +
        (sd !== null && sd !== undefined ? '<span class="pg-metric-sd"> ± ' + number(sd, m[2]) + "</span>" : "") + "</div>" +
        (published ? '<dl class="pg-metric-paper"><div><dt>paper</dt><dd>' + shown.text + "</dd></div>" +
          (gap !== null ? "<div><dt>gap</dt><dd>" + (gap > 0 ? "+" : "") + number(gap, shown.places) + "</dd></div>" : "") + "</dl>"
          : paperRow && !saidBelow ? '<div class="pg-metric-paper pg-metric-paper--none">The paper prints none</div>' : "") +
        // ASUE averages over the engines predicted early; runs since 2026-10-03
        // record how many, and with none it is not a number (assumption A-39).
        (m[0] === "asue" && (d.mean || {}).n_early !== undefined && (d.mean || {}).n_early !== null ?
          '<div class="pg-faint">over ' + number(d.mean.n_early, 1) + " engines predicted early</div>" : "") + "</div>";
    }).join("");
    var lower = MEASURES.filter(function (m) { return m[0] !== "r2"; }).map(function (m) { return m[1]; });
    var higher = MEASURES.filter(function (m) { return m[0] === "r2"; })[0][1];
    var better = "Lower is better for " + lower.slice(0, -1).join(", ") + " and " + lower[lower.length - 1] +
      "; higher is better for " + higher + ".";
    var compared = paperRow
      ? "Compared with " + esc(paperRow.label) + ", " + esc(paperRow.source) + (rmseOnly ? ", which prints RMSE only." : ".")
      : esc(d.no_row_reason);
    return '<section class="pg-run-section" id="pg-run-results" aria-labelledby="pg-run-results-title">' +
      '<div class="pg-run-section-head"><h3 id="pg-run-results-title">Results on the test engines</h3>' +
      "<p>Mean ± standard deviation over " + plural(seeds.length || d.seeds, "seed") + ".</p></div>" +
      '<div class="pg-metrics">' + cards + "</div>" +
      '<p class="pg-run-foot">' + compared + " " + better + "</p></section>" +
      (cautions.length ? runInterpretation(cautions, "") : "");
  }

  // Where results are shown without their notes beside them, one line says
  // how many notes qualify them and leads to the notes.
  function runNotesPointer(d, cautions) {
    if (!cautions.length) { return ""; }
    var warn = cautions.filter(function (c) { return c.level === "warn"; }).length, info = cautions.length - warn;
    var said = [warn ? plural(warn, "caution") : "", info ? plural(info, "note") : ""].filter(Boolean).join(" and ");
    return '<p class="pg-run-pointer' + (warn ? " pg-run-pointer--warn" : "") + '">' +
      '<span class="pg-run-note-mark" aria-hidden="true">' + (warn ? "!" : "i") + "</span><span>" + said +
      (cautions.length === 1 ? " applies" : " apply") + ' to these results. <a href="' + runAddress(d.name) +
      '" data-action="run-part" data-part="overview" data-reach="pg-run-interpretation">Read ' +
      (cautions.length === 1 ? "it" : "them") + " in Overview</a></span></p>";
  }

  // The widths a run's charts are drawn at. On a wide panel the per-seed
  // chart stands beside the training curves, a third of the row and two
  // thirds; the predictions keep the whole row. The numbers follow the style
  // sheet: 12 pixels between two figures, and 13 of border and padding on
  // each side of one.
  function runChartWidths() {
    var full = chartWidth(), panel = full + 26;
    if (panel < 1000) { return { full: full, seeds: full, curves: full, beside: false }; }
    var third = (panel - 12) / 3;
    return { full: full, seeds: Math.round(third - 26), curves: Math.round(2 * third - 26), beside: true };
  }

  function runCharts(d, cautions) {
    var seeds = Object.keys(d.per_seed);
    var paperRow = d.paper_row;
    var widths = runChartWidths(), W = widths.full;
    var rowHeight = chartHeight(widths.curves);
    var perSeedPlot = plot({
      width: widths.seeds, height: widths.beside ? rowHeight : 200, xlabel: "seed", ylabel: "test RMSE (cycles)", label: "test RMSE per seed",
      series: [{ points: d.results.map(function (r) { return [r.seed, r.rmse]; }), colour: "var(--fig-accent)", dots: true, line: false, radius: 5 }],
      hlines: paperRow ? [{ y: paperRow.rmse, label: "paper", colour: "var(--fig-lpc)" }] : []
    });
    var curves = seeds.length && d.per_seed[seeds[0]].history.length;
    var hasValidation = curves && d.per_seed[seeds[0]].history[0].validation_rmse_cycles !== undefined &&
      d.per_seed[seeds[0]].history[0].validation_rmse_cycles !== null;
    var curvePlot = curves ? plot({
      width: widths.curves, height: rowHeight, xlabel: "epoch (one pass over the training data)", label: "training curves",
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
    // points of each series; ordering them by their label instead
    // shows how the error changes along the life, so both are offered.
    var byUnit = S.predOrder !== "rul";
    var testLife = lifeSeriesName(d.clip_test_labels, d.rul_max);
    var predictions = seeds.length ? d.per_seed[seedChoice].predictions.slice().sort(byUnit
      ? function (a, b) { return a.unit - b.unit; } : function (a, b) { return a.rul - b.rul; }) : [];
    var predPlot = predictions.length ? plot({
      width: W, height: chartHeight(W), ylabel: "remaining life (cycles)", label: predictionsChartName(d.clip_test_labels, d.rul_max), xlabel: "test engine",
      series: byUnit ? [
        { name: testLife, colour: "var(--fig-burn)", dots: true, radius: 2.8, width: 1.3, points: predictions.map(function (p) { return [p.unit, p.rul]; }) },
        { name: "predicted", colour: "var(--fig-accent)", dots: true, radius: 2.8, width: 1.3, points: predictions.map(function (p) { return [p.unit, p.predicted]; }) }
      ] : [
        { name: testLife, colour: "var(--fig-ink)", points: predictions.map(function (p, i) { return [i + 1, p.rul]; }), width: 2 },
        { name: "predicted", colour: "var(--fig-warn)", dots: true, line: false, points: predictions.map(function (p, i) { return [i + 1, p.predicted]; }) }
      ]
    }) : "";
    return '<section class="pg-run-section" id="pg-run-charts" aria-label="Charts">' + runNotesPointer(d, cautions) +
      '<div class="pg-charts pg-run-charts"><figure class="pg-figure"><h3 class="pg-figure-title">Test RMSE per seed</h3>' + perSeedPlot + "</figure>" +
      (curves ? '<figure class="pg-figure"><h3 class="pg-figure-title">Training curves</h3>' + curvePlot + "</figure>"
        : '<div class="pg-empty">This run kept no per-seed files, so it has no curves, predictions or weights.</div>') +
      (predPlot ? '<figure class="pg-figure pg-figure--wide"><div class="pg-figure-head"><div><h3 class="pg-figure-title">' +
        esc(sentenceCase(predictionsChartName(d.clip_test_labels, d.rul_max))) + "</h3>" +
        '<p class="pg-figure-as"><strong>As the paper\'s Fig. 8.</strong></p></div>' +
        '<div class="pg-toolbar pg-figure-toolbar"><label for="pg-pred-seed">Seed</label>' +
        selectField("pg-pred-seed", seeds.map(function (s) { return { value: s, label: s }; }), seedChoice, "data-pred-seed") +
        segmented("pg-pred-order", [{ value: "unit", label: "by unit number" }, { value: "rul", label: "by " + lifeLabel(d.clip_test_labels) }],
          byUnit ? "unit" : "rul", 'data-pred-order') + "</div></div>" + predPlot + "</figure>" : "") +
      "</div></section>";
  }

  // The explanations stored for this run, newest first. A row opens one in
  // the Explain tab, where its charts stand with the words that say how to
  // read them; none of its values is repeated here without those words.
  function runExplanations(d) {
    var stored = d.explanations || [];
    var open = '<section class="pg-run-section" id="pg-run-explanations" aria-labelledby="pg-run-explanations-title">' +
      '<div class="pg-run-section-head"><h3 id="pg-run-explanations-title">Explanations stored for this run</h3>';
    if (!stored.length) {
      return open + "</div>" + '<div class="pg-empty pg-run-none">' + (d.has_weights
        ? "None yet. Explain this run, above, opens Explain predictions with this run chosen."
        : "None, and none can be made: this run kept no weights.") + "</div></section>";
    }
    return open + "<p>Select a row to open it in Explain predictions.</p></div>" +
      '<div class="pg-runs"><table class="data-table pg-run-stored"><thead><tr><th>Method</th><th>What it explains</th><th>Seed</th>' +
      '<th class="num">Windows</th><th>Finished</th><th class="col-commit">Commit</th>' +
      '<th class="pg-run-open"><span class="visually-hidden">Open</span></th></tr></thead><tbody>' +
      stored.map(function (x) {
        return '<tr data-action="open-explanation" data-run="' + esc(d.name) + '" data-name="' + esc(x.name) + '" tabindex="0"><td>' +
          esc(methodLabel(x)) + "</td><td>" + esc(sentenceCase(explainedWhat(x))) + '</td><td data-label="Seed">' + esc(x.seed) +
          '</td><td class="num" data-label="Windows">' + count(x.windows) + '</td><td data-label="Finished">' + when(x.finished_utc) +
          '</td><td class="col-commit"><code class="pg-commit">' + esc(x.commit || "") + "</code></td>" +
          '<td class="pg-run-open">' + NEXT + "</td></tr>";
      }).join("") + "</tbody></table></div></section>";
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
    // A run trained from the command line can hold a value the form does not
    // offer, such as a graph drawn at random (assumption A-41); the form keeps
    // its default for it and says so.
    Object.keys(s).forEach(function (k) {
      var field = S.fields && S.fields[k];
      if (field && field.choices && !field.choices.some(function (c) { return c.value === s[k]; })) {
        T.toast(field.label + ": this run used “" + String(s[k]) + "”, which the form does not offer, " +
          "so the form's default is set instead.", 7000);
        s[k] = S.options.defaults[k];
      }
    });
    return s;
  }

  // --------------------------------------------------------------- explain

  // Explain the selected trained run, or the newest usable run on first entry.
  function drawExplain() {
    var panel = $("#pg-panel");
    var e = S.explain;
    var request = S.explainRequest = (S.explainRequest || 0) + 1;
    panel.innerHTML = '<div class="loading">Reading reports/…</div>';
    return loadRuns().then(function (runs) {
      if (S.tab !== "explain" || request !== S.explainRequest) { return; }
      var usable = runs.filter(function (r) { return r.has_weights; });
      if (!usable.length) {
        panel.innerHTML = '<div class="pg-empty">No trained run is available to explain. Train a run first, then return here.</div>' +
          '<button type="button" class="button" data-tab="design">Design a run</button>';
        return;
      }
      if (!e.run || !usable.some(function (r) { return r.name === e.run; })) {
        e.run = usable[0].name; e.detail = null; e.shown = null;
      }
      var run = usable.filter(function (r) { return r.name === e.run; })[0];
      var load = e.detail && e.detail.name === e.run ? Promise.resolve(e.detail)
        : T.get("/api/playground/run?name=" + encodeURIComponent(e.run));
      return load.then(function (d) {
        if (S.tab !== "explain" || request !== S.explainRequest) { return; }
        e.detail = d;
        var seeds = Object.keys(d.per_seed).filter(function (s) { return d.per_seed[s].has_weights; });
        if (seeds.indexOf(String(e.seed)) < 0) { e.seed = seeds[0]; }
        panel.innerHTML = '<div class="pg-explain">' + explainForm(run, seeds, usable) +
          '<section class="pg-block"><div class="pg-block-head"><h2 class="pg-block-title">Explanations stored for this run</h2>' +
          (d.explanations.length ? '<span class="pg-faint">click a row to open it</span>' : "") + "</div>" +
          (d.explanations.length ? explanationsTable(d.explanations) : '<div class="pg-empty">None yet.</div>') +
          '</section><div id="pg-e-view"></div></div>';
        if (e.shown) { drawExplanation(); }
      });
    }).catch(function (error) {
      if (S.tab === "explain" && request === S.explainRequest) { panel.innerHTML = errorBox(error.message); }
    });
  }

  function explainRunLabel(run) {
    var graph = graphName(run.graph, run.adjacency_normalization);
    return [when(run.finished), run.subset, stageName(run.temporal, run.temporal_hidden),
      run.spatial === REMOVED.spatial ? "w/o GCN" : graph ? graph + " graph" : "", plural(run.seeds, "seed")]
      .filter(Boolean).join(" · ");
  }

  // Whether a stored run's network has graph layers; a run stored before the
  // choice existed recorded none, and has them.
  function hasGraphLayers(run) { return !run || run.spatial !== REMOVED.spatial; }

  function explainForm(run, seeds, runs) {
    var e = S.explain;
    var labels = runs.map(explainRunLabel);
    var choices = runs.map(function (r, i) {
      var duplicate = labels.indexOf(labels[i]) !== labels.lastIndexOf(labels[i]);
      return { value: r.name, label: labels[i] + (duplicate ? " · " + r.name : "") };
    });
    var seedChoices = seeds.map(function (s) { return { value: s, label: s }; });
    var forEngine = e.scope === "engine";
    // Walks follow the sensor graph through the graph layers, so a network
    // without them offers only the two methods that vary the input.
    var layered = hasGraphLayers(run);
    if (!layered && e.method === "walks") { e.method = "shap"; }
    return '<section class="pg-block"><div class="pg-fields">' +
      cell({ id: "pg-e-run", label: "Trained run", size: "wide",
        control: selectField("pg-e-run", choices, e.run, 'data-explain="run"') +
          '<button type="button" class="pg-link" data-action="open-run" data-run="' + esc(run.name) + '">View run details</button>' }) +
      // Six seeds or fewer sit side by side, named by the cell's name; more
      // go in a list, which the name's label points at.
      (seeds.length <= 6
        ? cell({ key: "pg-e-seed", label: "Seed whose weights are explained", size: "wide",
          control: segmented("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"', nameIds("pg-e-seed")) })
        : cell({ id: "pg-e-seed", label: "Seed whose weights are explained", size: "wide",
          control: selectField("pg-e-seed", seedChoices, e.seed, 'data-explain="seed"') })) +
      cell({ key: "pg-e-method", label: "Method", size: "wide",
        control: optionList("pg-e-method", [
          { value: "shap", label: METHODS.shap, about: EXPLAIN_ABOUT.shap, note: sourceTag({ kind: "paper", ref: "Eq. 15" }) +
            " All " + count(16384) + " sensor combinations are evaluated. Replacement readings are averaged across training windows at each position (" + refLink("A-20") + "); their prediction is our starting point (" + refLink("A-24") + ")." },
          { value: "occlusion", label: "Occlusion, one sensor at a time", about: EXPLAIN_ABOUT.occlusion, note: sourceTag({ kind: "extension" }) +
            " Evaluates the original input and one replacement per sensor. Replacement readings are averaged across training windows separately at each window position (" + refLink("A-20") + ")." },
          { value: "walks", label: METHODS.walks, about: EXPLAIN_ABOUT.walks, note: sourceTag({ kind: "extension" }) +
            " GNN-LRP of Schnake et al. (2022) (" + refLink("A-29") + "). " +
            "Their node-flipping check is stored with every explanation (" + refLink("A-33") +
            "). Identical graph outputs are grouped where routes end (" + refLink("F-18") + ")." }
        ].filter(function (option) { return layered || option.value !== "walks"; }),
        e.method, 'data-explain="method"', explainNote, nameIds("pg-e-method")) +
          (layered ? "" : '<p class="pg-faint" data-no-walks>Walks through the sensor graph are not offered: this ' +
            "run's network has no graph layers, so no sensor's numbers travel along a connection.</p>") }) +
      // The walk rule's setting, for walks only; the server takes its name,
      // never numbers, and every name gives the first graph layer 0.
      cell({ key: "pg-e-gamma", label: "How walk shares are divided (γ)", size: "wide", hidden: e.method !== "walks",
        hint: infoCard(S.options.gamma_about, "", S.options.explain_more.gamma),
        control: optionList("pg-e-gamma", S.options.gamma_settings, explainGamma(), 'data-explain="gamma"', optionNote,
          nameIds("pg-e-gamma")) }) +
      cell({ key: "pg-e-scope", label: "What is explained", size: "wide",
        control: optionList("pg-e-scope", [
          { value: "engine", label: "One engine over time", about: EXPLAIN_ABOUT.engine, note: sourceTag({ kind: "paper", ref: "Section 4.4" }) +
            " Follows the paper's idea of explaining successive windows; the engine selection described above is this project's procedure." },
          { value: "test", label: "The last window of every test engine", about: EXPLAIN_ABOUT.test, note: sourceTag({ kind: "extension" }) +
            " Ranks sensors by their average contribution size across the included engines; the sign is ignored for this ranking." }
        ], e.scope, 'data-explain="scope"', explainNote, nameIds("pg-e-scope")) }) +
      cell({ id: "pg-e-engine", key: "pg-e-engine-row", label: "Engine number", hidden: !forEngine,
        hint: infoCard("Choose an engine from this seed's held-back set, or its training set if none were held back. " +
          "Leave blank to choose the engine with the most available windows. Engine numbers refer to the NASA data files.", "", S.options.explain_more.engine),
        control: '<input class="field" type="number" id="pg-e-engine" min="1" value="' + esc(e.engine) + '" data-explain="engine">' }) +
      cell({ id: "pg-e-every", key: "pg-e-every-row", label: "Explain every n-th window", hidden: !forEngine,
        hint: infoCard("Enter 1 to explain every available window. Larger values keep the first window, then " +
          "every n-th window after it, reducing computation but hiding changes between those windows. " +
          "This counts windows, not cycles; their spacing was set when the run was prepared.", "", S.options.explain_more.every),
        control: '<input class="field" type="number" id="pg-e-every" min="1" max="100" value="' + e.every + '" data-explain="every">' }) +
      // The button ends the form on a row of its own, at the leading edge
      // where every setting's name starts, instead of in a cell beside the
      // last settings.
      '</div><div class="pg-form-actions"><button type="button" class="button button--primary" data-action="explain">Explain</button></div></section>';
  }

  // An Explain option's i: what it does in plain words, then its source and
  // the technical detail under the line.
  function explainNote(c) { return infoCard(c.about, c.note, S.options.explain_more[c.value]); }

  // The walk rule's setting chosen in the form: the first the server lists, until another is chosen.
  function explainGamma() {
    var e = S.explain;
    if (!e.gamma) { e.gamma = S.options.gamma_settings[0].value; }
    return e.gamma;
  }

  // A setting of the walk rule by its name in the form, or "" for one the form does not offer.
  function gammaLabel(value) {
    var setting = (S.options.gamma_settings || []).filter(function (s) { return s.value === value; })[0];
    return setting ? setting.label : "";
  }

  function lowerFirst(text) { return text.charAt(0).toLowerCase() + text.slice(1); }

  // A stored explanation's method in the page's words. A walk explanation made
  // with another setting of the rule than the first the form offers names it.
  function methodLabel(x) {
    var name = METHODS[x.method] || x.method;
    var setting = x.method === "walks" ? gammaSettingOf(x) : null;
    var label = setting && setting !== S.options.gamma_settings[0].value ? gammaLabel(setting) : "";
    return label ? name + " · " + lowerFirst(label) : name;
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
          esc(methodLabel(x)) + "</td><td>" + esc(explainedWhat(x)) + "</td><td>" + esc(x.seed) +
          '</td><td class="num">' + count(x.windows) + "</td><td>" + when(x.finished_utc) + "</td></tr>";
      }).join("") + "</tbody></table></div>";
  }

  function explainChanged(input) {
    var e = S.explain, which = input.getAttribute("data-explain");
    if (which === "run") {
      e.run = input.value; e.detail = null; e.seed = null; e.engine = "";
      e.shown = null; e.values = null; e.index = 0; e.chart = "engine";
      drawExplain().then(function () {
        var picker = $("#pg-e-run");
        if (S.tab === "explain" && picker) { picker.focus({ preventScroll: true }); }
      });
      return;
    }
    if (which === "seed") { e.seed = input.value; return; }
    if (which === "engine") { e.engine = input.value; return; }
    if (which === "every") { e.every = Number(input.value) || 1; return; }
    if (which === "method") { e.method = input.value; }
    if (which === "scope") { e.scope = input.value; }
    if (which === "gamma") { e.gamma = input.value; }
    // The engine rows apply to one scope only, the rule's setting to walks
    // only; the option lists show their pick.
    var forEngine = e.scope === "engine";
    ["pg-e-engine-row", "pg-e-every-row"].forEach(function (key) {
      var box = $('[data-row="' + key + '"]');
      if (box) { box.hidden = !forEngine; }
    });
    var rule = $('[data-row="pg-e-gamma"]');
    if (rule) { rule.hidden = e.method !== "walks"; }
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
    if (e.method === "walks") { request.gamma = explainGamma(); }
    post("/api/playground/explain", request).then(function (job) {
      if (job && job.id) { S.mine[job.id] = true; }
      return repoll();
    }).then(function () {
      if (!bannerInView()) { T.toast("Explanation queued."); }
    }).catch(function (error) { T.toast(error.message, 8000); });
  }

  // Every view of an explanation lives in one full-width deck. The tab row
  // keeps the views beside one another without making the reader scroll past
  // every chart to reach the next one. Only the chosen panel is shown, while
  // the ordinary tab-key and arrow-key handling above remains shared with the
  // other tab rows on the page.
  function explanationCharts(charts) {
    charts = charts.filter(function (chart) { return !!chart.html; });
    var keys = charts.map(function (chart) { return chart.key; });
    if (keys.indexOf(S.explain.chart) < 0) { S.explain.chart = keys[0]; }
    return '<div class="pg-charts pg-explanation-charts">' + explanationSelection(S.explain.values) +
      '<div class="pg-chart-tabs" role="tablist" aria-label="Explanation charts">' +
      charts.map(function (chart) {
        var on = chart.key === S.explain.chart;
        return '<button type="button" role="tab" class="pg-chart-tab" id="pg-chart-tab-' + chart.key +
          '" data-action="chart-tab" data-chart="' + chart.key + '" aria-controls="pg-chart-panel-' + chart.key +
          '" aria-selected="' + on + '" tabindex="' + (on ? 0 : -1) + '">' + esc(chart.label) + "</button>";
      }).join("") + '</div><div class="pg-chart-panels">' + charts.map(function (chart) {
        var on = chart.key === S.explain.chart;
        return '<div class="pg-chart-panel" id="pg-chart-panel-' + chart.key + '" role="tabpanel" aria-labelledby="pg-chart-tab-' +
          chart.key + '"' + (on ? "" : " hidden") + ">" + chart.html + "</div>";
      }).join("") + "</div></div>";
  }

  function selectExplanationChart(key) {
    var button = $('[data-chart="' + key + '"]');
    if (!button) { return; }
    S.explain.chart = key;
    var deck = button.closest(".pg-charts");
    Array.prototype.forEach.call(deck.querySelectorAll(".pg-chart-tab"), function (tab) {
      var on = tab.getAttribute("data-chart") === key;
      tab.setAttribute("aria-selected", String(on));
      tab.tabIndex = on ? 0 : -1;
    });
    Array.prototype.forEach.call(deck.querySelectorAll(".pg-chart-panel"), function (panel) {
      panel.hidden = panel.id !== "pg-chart-panel-" + key;
    });
    if (S.explain.updateCharts) { S.explain.updateCharts(); }
  }

  // One shared, persistent control selects a stored cycle or test engine.
  function explanationSelection(data) {
    var rows = data.values, forEngine = data.summary.scope === "engine";
    var detail = S.explain.detail || {};
    var ceiling = (forEngine || detail.clip_test_labels) ? (data.summary.rul_max || detail.rul_max) : null;
    var life = lifeLabel(explanationCapped(data));
    var label = forEngine ? "Engine cycle" : "Test engine";
    var item = forEngine ? "cycle" : "test engine";
    var name = function (row) { return (forEngine ? "Cycle " : "Engine ") + (forEngine ? row.end_cycle : row.unit); };
    return '<section class="pg-selection" aria-label="Shared chart selection">' +
      '<div class="pg-selection-main"><label for="pg-explain-pick">' + label + '</label>' +
      '<div class="pg-selection-stepper"><button type="button" class="button" data-action="explain-previous" aria-label="Previous saved ' + item + '">' + BACK + '</button>' +
      '<select id="pg-explain-pick" data-explain-index>' + rows.map(function (row, index) {
        return '<option value="' + index + '"' + (index === S.explain.index ? ' selected' : '') + '>' + esc(name(row)) + '</option>';
      }).join("") + '</select><button type="button" class="button" data-action="explain-next" aria-label="Next saved ' + item + '">' + NEXT + '</button></div></div>' +
      '<div class="pg-selection-metric"><span>' + esc(life.charAt(0).toUpperCase() + life.slice(1)) + '</span><strong><span data-selection-actual></span> <small>cycles</small></strong></div>' +
      '<div class="pg-selection-metric"><span>Predicted remaining life</span><strong><span data-selection-predicted></span> <small>cycles</small></strong></div>' +
      '<div class="pg-selection-track"><input type="range" data-explain-index min="0" max="' + (rows.length - 1) +
      '" step="1" value="' + S.explain.index + '" aria-label="' + label + '" aria-describedby="pg-selection-help">' +
      '<div class="pg-selection-ends"><span>' + esc(name(rows[0])) + '</span><span data-selection-position></span><span>' + esc(name(rows[rows.length - 1])) + '</span></div></div>' +
      '<p id="pg-selection-help">' + (forEngine
        ? 'Engine ' + esc(rows[0].unit) + ': each position uses the readings ending at that cycle.'
        : 'Each position is a different test engine, using its last available readings—not a step in time.') +
      ' Drag to update the selected prediction live. Overview charts still show all saved ' + (forEngine ? 'cycles' : 'engines') + '.' +
      (ceiling ? ' The remaining life shown is capped at ' + esc(fixed(ceiling, 0)) + ' cycles, the ceiling of the labels the network learned from: where it reads ' +
        esc(fixed(ceiling, 0)) + ', more may truly remain.' : '') + '</p></section>';
  }

  // Update labels without replacing the range while it is being dragged.
  function updateExplanationSelection() {
    var e = S.explain, rows = e.values.values, row = rows[e.index], box = $(".pg-selection");
    if (!box) { return; }
    Array.prototype.forEach.call(box.querySelectorAll("[data-explain-index]"), function (input) {
      input.value = String(e.index);
      input.disabled = rows.length < 2;
      if (input.type === "range") {
        input.setAttribute("aria-valuetext", explanationCaption(row) + " cycles");
        input.style.setProperty("--selection-progress", (rows.length > 1 ? 100 * e.index / (rows.length - 1) : 0) + "%");
      }
    });
    box.querySelector('[data-action="explain-previous"]').disabled = e.index === 0;
    box.querySelector('[data-action="explain-next"]').disabled = e.index === rows.length - 1;
    box.querySelector("[data-selection-actual]").textContent = fixed(row.rul, 0);
    box.querySelector("[data-selection-predicted]").textContent = fixed(row.predicted, 1);
    box.querySelector("[data-selection-position]").textContent = (e.index + 1) + " of " + rows.length + " saved";
  }

  // Coalesce rapid input events into one visual update per animation frame.
  function selectExplanationIndex(index) {
    var e = S.explain, data = e.values;
    if (!data || !data.values.length || !Number.isFinite(index)) { return; }
    e.index = Math.max(0, Math.min(Math.round(index), data.values.length - 1));
    updateExplanationSelection();
    if (e.selectionFrame) { window.cancelAnimationFrame(e.selectionFrame); }
    e.selectionFrame = window.requestAnimationFrame(function () {
      e.selectionFrame = null;
      if (S.tab === "explain" && e.values === data && e.shown === data.name && e.updateCharts) { e.updateCharts(); }
    });
  }

  // Refresh only selection-dependent content, preserving controls and open details.
  function wireExplanationSelection(box, data, sensorOrder, W) {
    var e = S.explain;
    e.updateCharts = function () {
      var row = data.values[e.index];
      if (e.chart === "engine" && e.engineController) { e.engineController.update(e.index); }
      var panel = box.querySelector("#pg-chart-panel-" + e.chart);
      if (e.chart === "life" && panel) { panel.innerHTML = lifeFigure(data.values, row, W); }
      var total = box.querySelector("[data-walk-sum]"), along = box.querySelector("[data-walk-parts]");
      if (e.chart === "parts" && total) { total.innerHTML = scoreSum(data, e.index); }
      if (e.chart === "parts" && along) { along.innerHTML = walkParts(data, e.index, W); }
      var check = box.querySelector("[data-walk-check]");
      if (e.chart === "check" && check) { check.innerHTML = flippingFigure(data, e.index, W); }
      var example = box.querySelector("[data-walk-example]");
      if (example) { example.innerHTML = walkExample(data, e.index); }
      if (historyCursor) { historyCursor(e.index); }
      if (e.chart === "history" && ribbonsRedraw) { ribbonsRedraw(); }
    };
    updateExplanationSelection();
  }

  function drawExplanation() {
    var e = S.explain, box = $("#pg-e-view");
    if (!box) { return; }
    var runName = e.run, shown = e.shown;
    var held = e.values && e.values.name === e.shown;
    var load = held ? Promise.resolve(e.values)
      : T.get("/api/playground/explanation?run=" + encodeURIComponent(e.run) + "&name=" + encodeURIComponent(e.shown))
        .then(function (data) { data.name = shown; return data; });
    if (!held) { box.innerHTML = '<div class="loading">Reading the explanation…</div>'; }
    return load.then(function (data) {
      if (S.tab !== "explain" || e.run !== runName || e.shown !== shown) { return; }
      e.values = data;
      var sum = data.summary, rows = data.values;
      if (!rows || !rows.length) {
        box.innerHTML = '<div class="pg-empty">This explanation stored no windows.</div>';
        return;
      }
      // The method decides what the stored numbers are, so it is read first:
      // a walk explanation stores shares of the score, not values in cycles.
      if (sum.method === "walks") { drawWalks(box, data); return; }
      if (sum.method !== "shap" && sum.method !== "occlusion") {
        box.innerHTML = errorBox("This page cannot draw an explanation made with the method \"" + sum.method + "\".");
        return;
      }
      var symbols = Object.keys(sum.mean_abs);
      var sensorOrder = Object.keys(rows[0]).filter(function (k) { return symbols.indexOf(k) >= 0; });
      if (e.index >= rows.length) { e.index = 0; }
      var exact = sum.method === "shap";
      var forEngine = sum.scope === "engine";

      // The charts carry the reading: each axis names its unit and each
      // legend its colours. Only a method the paper does not use says so
      // above them, once, as the walk view does.
      var head = '<section class="pg-detail">' + (exact ? ""
        : '<p class="pg-lead">Occlusion is a comparison added by this project; the paper does not use it.</p>');

      var mean = {};
      sensorOrder.forEach(function (s) { mean[s] = sum.mean_abs[s]; });
      var W = S.chartsDrawnAt = explanationChartWidth();

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
        // The two angles the 3D chart is seen from belong to the
        // explanation on screen: another explanation opens at the first view.
        if (e.ribbonViewName !== e.shown) { e.ribbonView = { az: RIBBON_VIEW.az, el: RIBBON_VIEW.el }; e.ribbonViewName = e.shown; }
        var valueName = exact ? "SHAP value" : "Occlusion value";
        var history = '<figure class="pg-figure pg-figure--wide"><figcaption><strong>One chart per sensor.</strong> ' +
          "How each sensor's value changed over the engine's life: the cycle along the bottom, the value up the side, " +
          "in cycles of remaining life. Point at any chart to read every sensor at that cycle.</figcaption>" +
          sensorHistory(rows, symbols, e.index, W) + "</figure>" +
          '<figure class="pg-figure pg-figure--wide"><figcaption><strong>Every sensor in one space.</strong> ' +
          "The same values as the charts above: each sensor is a ribbon along the engine's cycles, and its height is its " +
          esc(valueName) + ", in cycles of remaining life. Blue stands above zero and orange below; the dots mark the engine cycle chosen above. " +
          "Drag the chart to turn it.</figcaption>" +
          '<div class="pg-3d-stage" data-ribbons tabindex="0" role="group" aria-label="Three-dimensional chart. The arrow keys turn it; Home returns it to its first view.">' +
          ribbons(rows, symbols, W, e.ribbonView, e.index, valueName) + "</div>" +
          '<div class="pg-faint">Features are the fourteen sensors, in the order of the average size of their values. ' +
          "The ribbons are not joined into a surface, because no value exists between two sensors. " +
          "The arrow keys turn the chart too; a double click returns it to this view.</div></figure>";
        body = head + explanationCharts([
          { key: "engine", label: "Engine", html: engineFigure() },
          { key: "life", label: "Remaining life", html: lifeFigure(rows, current, W) },
          { key: "values", label: "All values", html: swarm },
          { key: "ranking", label: "Sensor ranking", html: strip },
          { key: "history", label: "Over time", html: history },
          { key: "paper", label: "Paper comparison", html: exact ? paperComparison(rows, sensorOrder, sum) : "" }
        ]) + "</section>";
      } else {
        // The heat map takes the place of the history chart, which needs
        // one engine's life.
        var heat = '<figure class="pg-figure pg-figure--wide"><figcaption>Every sensor at every ' +
          "test engine (columns, by engine number). Blue raised the predicted remaining life, orange lowered it; the stronger the colour, the more cycles.</figcaption>" +
          heatmap(rows, sensorOrder) + "</figure>";
        body = head + explanationCharts([
          { key: "engine", label: "Engine", html: engineFigure() },
          { key: "values", label: "All values", html: swarm },
          { key: "ranking", label: "Sensor ranking", html: strip },
          { key: "engines", label: "All engines", html: heat }
        ]) + "</section>";
      }
      box.innerHTML = body;
      mountEngine(box, data);
      wireHistory(box, forEngine ? rows : [], symbols, W);
      wireRibbons(box, forEngine ? rows : [], symbols, W, exact ? "SHAP value" : "Occlusion value");
      wireExplanationSelection(box, data, sensorOrder, W);
    }).catch(function (error) {
      box.innerHTML = errorBox(error.message);
    });
  }

  function engineFigure() {
    return '<figure class="pg-figure pg-figure--wide ev-figure" data-engine-view></figure>';
  }

  function mountEngine(box, data) {
    var e = S.explain;
    if (e.engineViewName !== e.shown) { e.engineView = {}; e.engineViewName = e.shown; }
    // A network with no graph layers is sent no connections: its sensors
    // are drawn where they sit on the engine, with no line between them.
    e.engineController = window.ThesisEngineView.mount(box.querySelector("[data-engine-view]"), data, {
      index: e.index, graph: S.options.graph,
      edges: data.graph_edges === undefined ? null : data.graph_edges
    }, e.engineView || (e.engineView = {}));
  }

  // The paper's Fig. 7 for one engine: the capped and the predicted remaining
  // life at every window, the error between them shaded, the chosen window marked.
  function lifeFigure(rows, current, W) {
    var subset = S.explain.detail ? S.explain.detail.subset : "";
    var ceiling = (S.explain.values && S.explain.values.summary.rul_max) || (S.explain.detail || {}).rul_max;
    var lifePlot = plot({
      width: W, height: chartHeight(W), xlabel: "cycle at the end of the window", ylabel: "remaining life, and error (cycles)",
      label: "the prediction along the engine's life",
      areas: [{ name: "predicted minus capped remaining life: the paper's \"new error area\"", colour: "var(--fig-lpc)", opacity: 0.35,
        points: rows.map(function (r) { return [r.end_cycle, r.predicted - r.rul]; }) }],
      series: [
        { name: lifeSeriesName(true, ceiling), colour: "var(--fig-accent)", dash: "7 4", width: 2, points: rows.map(function (r) { return [r.end_cycle, r.rul]; }) },
        { name: "predicted", colour: "var(--fig-lpt)", width: 1.8, points: rows.map(function (r) { return [r.end_cycle, r.predicted]; }) },
        { name: "selected engine cycle", colour: "var(--fig-ink)", line: false, dots: true, radius: 5.5, points: [[current.end_cycle, current.predicted]] }
      ]
    });
    return '<figure class="pg-figure pg-figure--wide"><figcaption><strong>As the paper\'s Fig. 7.</strong>' +
      (FIG7_UNIT[subset] ? " The paper's " + esc(subset) + " engine is unit " + FIG7_UNIT[subset] + "." : "") +
      "</figcaption>" + lifePlot + "</figure>";
  }

  // "a", "a and b", "a, b and c".
  function andList(list) {
    return list.length < 2 ? list.join("") : list.slice(0, -1).join(", ") + " and " + list[list.length - 1];
  }

  // A share of the score before the sigmoid, with its sign: "+0.570", "-0.006".
  function scoreShare(v) {
    return (v > 0 ? "+" : "") + fixed(v, 3);
  }

  // The sensors a walk explanation covers; explanations saved before 2026-10-01
  // named them only through an average-size ranking.
  function walkSensors(sum) {
    return sum.sensors || Object.keys(sum.mean_abs_start || {});
  }

  // Which setting of the rule a saved walk explanation used, by the value the
  // Explain form gives it. One saved before the setting was recorded, on
  // 2026-10-02, is recognised by its numbers: the authors' default schedule,
  // or gamma 0 everywhere; anything else gives null.
  function gammaSettingOf(sum) {
    if (sum.gamma_setting) { return sum.gamma_setting; }
    var gamma = sum.gamma || [];
    if (gamma.length && gamma.every(function (g) { return g === 0; })) { return "gnn-gi"; }
    var authors = gamma.length > 1 && gamma[0] === 0 && gamma.every(function (g, i) {
      return Math.abs(g - 3 * (1 - i / (gamma.length - 1))) < 1e-9 || i === 0;
    });
    return authors ? "authors" : null;
  }

  // Which setting of the rule a saved walk explanation used, in words.
  function gammaNote(sum) {
    var setting = gammaSettingOf(sum);
    var listed = (sum.gamma || []).map(function (g) { return fixed(g, 2); }).join(", ");
    if (setting === "authors") {
      return "The rule's setting \u03b3 is " + listed + " at the graph layers in order: the authors' default schedule, falling from 3 to 0, " +
        "with 0 at the first layer because its inputs can be negative (" + refLink("A-29") + ").";
    }
    if (setting === "gnn-gi" && sum.gamma_setting) {
      return "This explanation was made with \u03b3 0 at every graph layer, the simpler baseline the authors call GNN-GI, " +
        "chosen when it was made in place of their own setting (" + refLink("A-29") + ").";
    }
    if (setting === "gnn-gi") {
      return "This saved explanation used \u03b3 0 at every graph layer, the simpler baseline the authors call GNN-GI, not their own setting. " +
        "Explain the run again for the authors' setting (" + refLink("A-29") + ").";
    }
    // The option's name is in this explanation's row of the table above, so
    // only its numbers are given here.
    if (gammaLabel(setting)) {
      return "The rule's setting \u03b3 is " + listed + " at the graph layers in order, chosen when this explanation was made in place of the authors' own, " +
        "with 0 at the first layer because its inputs can be negative (" + refLink("A-29") + ").";
    }
    return "The rule's setting \u03b3 is " + listed + " at the graph layers in order, which is not the authors' default (" + refLink("A-29") + ").";
  }

  // Show the saved arithmetic for the selected engine and cycle.
  function walkExample(data, index) {
    var selected = data.values[index];
    var sensorTotal = walkSensors(data.summary).reduce(function (total, sensor) {
      return total + Number(selected["start_" + sensor]);
    }, 0);
    return '<strong>Example from the selected ' + (data.summary.scope === "engine" ? 'engine cycle' : 'test engine') + '.</strong> Engine ' + esc(selected.unit) +
      ', cycle ' + esc(selected.end_cycle) + ': the sensor contributions total <strong>' + scoreShare(sensorTotal) +
      '</strong>, and what the network adds contributes <strong>' + scoreShare(Number(selected.constant)) +
      '</strong>. Together they give the internal score <strong>' + scoreShare(Number(selected.score)) +
      '</strong>, which the model converts to <strong>' + fixed(selected.predicted, 1) +
      ' cycles</strong> of remaining life. These displayed numbers are rounded.';
  }

  // What every tab of a walk explanation opens with: the question the tab
  // answers, then what is drawn, in a sentence or two.
  function walkTabHead(question, text) {
    return '<header class="pg-tabhead"><h3>' + question + "</h3><p>" + text + "</p></header>";
  }

  // What a walk is, in one clause, with the number of graph layers the
  // explanation itself recorded.
  function walkMeaning(sum) {
    return "the route one sensor's reading takes through the network's " + (sum.steps ? sum.steps + " " : "") + "graph layers";
  }

  // The three parts a walk explanation splits a score into, named for what
  // they are. Where the graph cannot tell some sensors apart they count as
  // one group, and a walk stays when it never leaves its group; with no such
  // group it stays when it never leaves its sensor.
  function walkPartList(sum) {
    var home = (sum.groups || []).length ? "group" : "sensor";
    return [
      { key: "constant", name: "Added by the network", colour: "var(--fig-core)",
        about: "What the numbers the network learned to add inside its calculations, its biases, contribute. It traces back to no sensor." },
      { key: "stays", name: "Walks that stay in their " + home, colour: "var(--fig-lpt)",
        about: "The share of every walk that never leaves the " + home + " it starts in." },
      { key: "moves", name: "Walks that cross to another " + home, colour: "var(--fig-lpc)",
        about: "The share of every walk that reaches another " + home + " at least once, even if it comes back." }
    ];
  }

  // A sensor's name as a chip that says what the sensor measures when pointed at.
  function sensorChip(s) {
    return '<span class="pg-chip"' + sensorTip(s) + ">" + esc(s) + "</span>";
  }

  // One position of a walk: a sensor, or a group of sensors the graph layers
  // cannot tell apart, written with slashes as the stored walk names it.
  function walkStep(step) {
    var names = String(step).split("/");
    if (names.length < 2) { return sensorChip(step); }
    return '<span class="pg-chip pg-chip--group">' + names.map(function (s) {
      return "<span" + sensorTip(s) + ">" + esc(s) + "</span>";
    }).join("<i>/</i>") + "</span>";
  }

  // What a reader of the score's parts is most likely to misread, said once
  // under the sum: what a group is, where the explanation has any, and why
  // the part no sensor owns is not the same at every cycle or engine.
  function scoreNotes(sum) {
    var groups = sum.groups || [];
    return '<div class="pg-notes">' + (groups.length ? "<p><strong>What a group is.</strong> Sensors joined to exactly the same sensors get identical numbers " +
      "from every graph layer, so the network cannot tell them apart and they count as one group: " +
      groups.map(function (g) { return walkStep(g.join("/")); }).join(" ") +
      ". Every other sensor is a group of its own (" + refLink("F-18") + "). A group belongs to the network's arithmetic; it is not a part of the engine.</p>" : "") +
      "<p><strong>Why what the network adds is not one fixed number.</strong> Its learned numbers are fixed after training, but what they contribute differs from one " +
      (sum.scope === "engine" ? "cycle" : "engine") + " to the next, because the readings decide which calculations pass them on. " +
      "It is not the score of a reference engine.</p></div>";
  }

  // Words wrapped onto lines of at most so many letters, for a label drawn in a chart.
  function wrapWords(text, width) {
    var lines = [""];
    String(text).split(" ").forEach(function (word) {
      var last = lines[lines.length - 1];
      if (last && (last + " " + word).length > width) { lines.push(word); } else { lines[lines.length - 1] = last ? last + " " + word : word; }
    });
    return lines;
  }

  // The range every score sum of an explanation is drawn on: wide enough for
  // each running total of each cycle or engine, so that bars compare as the
  // slider moves.
  function scoreSumScale(rows) {
    var lo = 0, hi = 0;
    rows.forEach(function (r) {
      var c = Number(r.constant), s = c + Number(r.stays), m = s + Number(r.moves);
      [c, s, m, Number(r.score)].forEach(function (v) {
        if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      });
    });
    var room = (hi - lo) * 0.04 || 1;
    return { lo: lo - room, hi: hi + room };
  }

  // One score written the way a sum is written by hand: the parts in a
  // column, each bar starting where the one above ended, a rule, and the
  // total under it. The total's bar ends where the last part's does, which
  // is the adding-up drawn rather than asserted.
  function scoreSum(data, index) {
    var sum = data.summary, rows = data.values, forEngine = sum.scope === "engine";
    var row = rows[Math.max(0, Math.min(index || 0, rows.length - 1))];
    var k = scoreSumScale(rows), span = k.hi - k.lo;
    var at = function (v) { return ((v - k.lo) / span * 100).toFixed(2); };
    var ticks = niceTicks(k.lo, k.hi, 6);
    var grid = ticks.map(function (t) {
      return '<span class="pg-sum-grid' + (t === 0 ? " pg-sum-grid--zero" : "") + '" style="left:' + at(t) + '%"></span>';
    }).join("");
    var bar = function (from, to, colour) {
      var a = Math.min(from, to), b = Math.max(from, to);
      return '<span class="pg-sum-bar' + (colour ? "" : " pg-sum-bar--total") + '" style="left:' + at(a) + "%;width:" +
        Math.max(0.2, (b - a) / span * 100).toFixed(2) + "%" + (colour ? ";background:" + colour : "") + '"></span>';
    };
    var join = function (v, way) { return '<span class="pg-sum-join pg-sum-join--' + way + '" style="left:' + at(v) + '%"></span>'; };
    var from = 0;
    var body = walkPartList(sum).map(function (part, i) {
      var value = Number(row[part.key]), to = from + value;
      var html = '<div class="pg-sum-row"><div class="pg-sum-name"><i style="background:' + part.colour + '"></i><b>' + esc(part.name) +
        "</b><span>" + esc(part.about) + '</span></div><div class="pg-sum-value">' + scoreShare(value) +
        '</div><div class="pg-sum-track" aria-hidden="true">' + grid + (i ? join(from, "in") : "") + bar(from, to, part.colour) + join(to, "out") +
        "</div></div>";
      from = to;
      return html;
    }).join("");
    var score = Number(row.score), ceiling = Number(sum.rul_max);
    var total = '<div class="pg-sum-row pg-sum-row--total"><div class="pg-sum-name"><i aria-hidden="true">=</i><b>Score</b>' +
      '<span>The three parts added together.</span></div><div class="pg-sum-value">' + scoreShare(score) +
      '</div><div class="pg-sum-track" aria-hidden="true">' + grid + join(from, "in") + bar(0, score, "") + "</div></div>";
    var axis = '<div class="pg-sum-row pg-sum-row--axis" aria-hidden="true"><div></div><div></div><div class="pg-sum-axis">' +
      ticks.map(function (t) { return '<span style="left:' + at(t) + '%">' + tickLabel(t) + "</span>"; }).join("") +
      "<em>score, a number with no unit</em></div></div>";
    var who = forEngine ? "cycle " + esc(row.end_cycle) + " of engine " + esc(row.unit) : "test engine " + esc(row.unit);
    return '<h4 class="pg-figure-title">The score of ' + who + ", added up</h4>" +
      '<div class="pg-sum" role="group" aria-label="The parts of the score and their total">' + body + total + axis + "</div>" +
      '<p class="pg-sum-result">The network turns this score into <strong>' + fixed(row.predicted, 1) +
      " cycles</strong> of predicted remaining life. A higher score always gives a longer prediction" +
      (isFinite(ceiling) && ceiling > 0 ? ", and none can pass the ceiling of " + fixed(ceiling, 0) + " cycles" : "") + ".</p>" +
      scoreNotes(sum);
  }

  // A walk explanation. Its numbers are shares of the score before the
  // sigmoid, never cycles: the server adds each window's walks up by the
  // sensor they start at (start_), by the connection they pass along
  // (link_) and by whether they leave their sensor's group (stays, moves).
  // Each sensor's share is copied under its own name so the charts drawn for
  // the other methods can read it. The paper has no walks, so nothing here
  // is set beside its figures, and the beeswarm already ranks the sensors.
  function drawWalks(box, data) {
    var e = S.explain, sum = data.summary, rows = data.values;
    var symbols = walkSensors(sum);
    if (!symbols.length) {
      box.innerHTML = errorBox("This walk explanation stored no share for any sensor.");
      return;
    }
    var sensorOrder = Object.keys(rows[0]).filter(function (k) {
      return k.indexOf("start_") === 0 && symbols.indexOf(k.slice(6)) >= 0;
    }).map(function (k) { return k.slice(6); });
    if (e.index >= rows.length) { e.index = 0; }
    var forEngine = sum.scope === "engine";
    var groups = sum.groups || [];
    var detailsOpen = box.querySelector("[data-walk-guide][open]");

    // Explain the visible numbers first, with the calculation available on request.
    var methodNotes = '<p><strong>How a sensor gets its number.</strong> The model combines information from connected sensors ' +
      'in several calculation steps. A sequence of sensors followed through those steps is called a <strong>walk</strong>. ' +
      'This explanation adds the contributions from all walks starting at a sensor to get the number shown beneath its label.</p>' +
      '<p><strong>How the total becomes a prediction.</strong> Inside its calculations, the model also adds numbers learned during training, called biases. ' +
      'Their contribution is the part named "Added by the network" in the Score breakdown tab. Add that contribution to all the sensor contributions ' +
      'to get the internal score. The model then converts that total into remaining life; a larger total gives a longer prediction.</p>' +
      '<p data-walk-example>' + walkExample(data, e.index) + '</p>' +
      '<p><strong>Check on the addition.</strong> Across this saved explanation, the largest difference between the sum ' +
      'of those parts and the model\'s internal score is ' + Number(sum.conservation_error_max).toExponential(1) +
      ' score units. This checks the explanation\'s arithmetic; it does not check whether the prediction is correct (' + refLink("A-29") + ').</p>' +
      '<p><strong>The rule.</strong> At each graph layer, a sensor\'s number is divided among the sensors it came from in proportion to what each sent, ' +
      'and a setting called \u03b3 decides how much more heavily positive contributions count (Schnake et al., 2022). ' + gammaNote(sum) + '</p>';
    if (groups.length === 1 && groups[0].length === symbols.length) {
      methodNotes += '<p><strong>Why sensor names are grouped.</strong> In this model every sensor is connected to every other sensor. ' +
        'The steps that combine connected sensors therefore produce the same result at every sensor position. ' +
        'The walk display combines these positions into one group; walks here are distinguished by their starting sensor (' + refLink("F-18") + ').</p>';
    } else if (groups.length) {
      methodNotes += '<p><strong>Why some sensor names are grouped.</strong> These groups have matching connections in this model: ' +
        esc(groups.map(andList).join("; ")) + '. When the model combines information from connected sensors, it produces the same result ' +
        'at every position within a group. The walk display combines those positions and separates their names with slashes. ' +
        'This describes the model\'s calculation; the sensors can still measure different things (' + refLink("F-18") + ').</p>';
    }
    if (sum.walks_start_after_learned_stage) {
      methodNotes += '<p><strong>Which moments are explained?</strong> This model first processes each sensor\'s readings over time. ' +
        'The walk calculation starts with the result of that processing, so it cannot say which individual cycle in the selected ' +
        'window contributed most (' + refLink("A-29") + ').</p>';
    }
    methodNotes += '<p><strong>What the connecting lines mean.</strong> Their widths show the size of the score contribution assigned ' +
      'to each connection. They do not tell you how much the prediction would change if you removed that connection. ' +
      'Answering that question requires changing the connections and running the model again (' + refLink("A-30") + ').</p>' +
      '<p>Time taken to compute this saved explanation: ' + (sum.seconds < 1 ? 'under a second' : duration(sum.seconds)) + '.</p>';
    var head = '<section class="pg-detail"><div class="pg-explanation-guide" aria-label="How to read the walk explanation">' +
      '<p class="pg-method-origin">Walk explanations are added by this project; the paper does not use this method (' + refLink("A-29") + ').</p>' +
      '<p><strong>What do the numbers beneath the sensors mean?</strong> In the Engine tab, the number beneath each sensor label ' +
      'is that sensor\'s contribution to the model\'s calculation for the selected ' + (forEngine ? 'engine cycle' : 'test engine') +
      '. It is not the sensor reading or a number of remaining cycles.</p>' +
      '<p>Blue, positive numbers add to an internal score; orange, negative numbers subtract from it. ' +
      'The model converts the total score into its prediction of remaining life. A number further above zero supports a longer ' +
      'prediction; a number further below zero supports a shorter one.</p>' +
      '<p>Use the ' + (forEngine ? 'Engine cycle' : 'Test engine') + ' control above the chart tabs to choose ' + (forEngine ? 'a moment in the engine\'s life' : 'a test engine') +
      ', then click a sensor to read what it measures and its contribution. These charts explain the model\'s calculation; ' +
      'they do not identify damaged parts or prove what caused the engine to wear.</p>' +
      '<details class="pg-details" data-walk-guide' + (detailsOpen ? ' open' : '') +
      '><summary>How this explanation is calculated and checked</summary>' + methodNotes + '</details></div>';

    var W = S.chartsDrawnAt = explanationChartWidth();
    var item = forEngine ? "cycle" : "test engine";
    // Each tab opens with the question it answers. Score breakdown: the
    // chosen score as a sum, then the same parts for every cycle or engine.
    var parts = walkTabHead("What is the score made of?",
      "Before it gives a number of cycles, the network computes one number, the score. A walk explanation splits that score into three parts " +
      "that add up to it exactly: two kinds of walk, a walk being " + walkMeaning(sum) + ", and what the network adds by itself.") +
      '<figure class="pg-figure pg-figure--wide" data-walk-sum>' + scoreSum(data, e.index) + "</figure>" +
      '<figure class="pg-figure pg-figure--wide" data-walk-parts>' + walkParts(data, e.index, W) + "</figure>";
    var walks = topWalks(data);
    var check = walkTabHead("Does the explanation agree with the network?",
      "The check the method's own authors run on their explanations (Schnake et al., 2022, Section 4.1). An explanation says what any set of sensors " +
      "is worth to the score. The check adds sensors to an empty graph, or removes them from the full one, in the order that worth gives, " +
      "runs the network on the graph that is left after every step, and follows the score. The curves are for the chosen " + item + ".") +
      '<div data-walk-check>' + flippingFigure(data, e.index, W) + "</div>";
    var fleetFlow = forEngine ? "" : walkTabHead("How much passed along each connection, on average?",
      "Each line is one connection of the sensor graph, as wide as the share of the score that passed along it, both directions and all graph layers together, " +
      "averaged over the " + count(rows.length) + " test engines. The Test engine control above does not change this picture; the Engine tab draws the chosen engine. " +
      "A wide line is not what the prediction would lose if that connection were removed.") +
      '<figure class="pg-figure pg-figure--wide">' + flowGraph(rows, null, sensorOrder) + "</figure>";

    // Only views the method's own paper uses, or this project's split of the
    // score: no view borrowed from the Shapley explanation or the main paper.
    var body = head + explanationCharts([
      { key: "engine", label: "Engine", html: engineFigure() },
      { key: "parts", label: "Score breakdown", html: parts },
      { key: "connections", label: "Connections", html: fleetFlow },
      { key: "walks", label: "Largest walks", html: walks },
      { key: "check", label: "Explanation check", html: check }
    ]) + "</section>";
    box.innerHTML = body;
    mountEngine(box, data);
    wireExplanationSelection(box, data, sensorOrder, W);
  }

  // The authors' own check of the explanation (Schnake et al., Section 4.1):
  // sensors added to an empty graph, or removed from the full one, in the order
  // each ranking gives, with the score followed at every step.
  var FLIP_STYLES = {
    "GNN-LRP, walks": { colour: "var(--fig-ink)", width: 2.2 },
    "GNN-LRP, starting sensors": { colour: "var(--fig-ink)", dash: "6 4" },
    "GNN-GI, walks": { colour: "var(--fig-cool)", width: 2.2 },
    "GNN-GI, starting sensors": { colour: "var(--fig-cool)", dash: "6 4" },
    // The axis grey, not the paler one beside it: a drawn line needs 3 to 1 against the page.
    "random": { colour: "var(--fig-axis)", dash: "2 4" }
  };
  var FLIP_NAMES = {
    "GNN-LRP, walks": "Whole walks, the authors' setting",
    "GNN-LRP, starting sensors": "Sensor totals, the authors' setting",
    "GNN-GI, walks": "Whole walks, the simpler baseline (\u03b3\u00a00)",
    "GNN-GI, starting sensors": "Sensor totals, the simpler baseline (\u03b3\u00a00)",
    "random": "Random orders, averaged"
  };

  // A third setting of the rule, chosen when the explanation was made, is
  // checked under the name "GNN-LRP <setting>" (scripts/explain.py) and named
  // as the form names it. It is drawn in gold, a colour this view never uses
  // for a sign: orange already means a negative share here, and the accent
  // blue sits too close to the baseline's blue-green (CIE76 colour difference
  // 32 in light mode and 23 in dark; gold is 78 and 83 from it, and 69 and 63
  // from the authors' ink).
  function chosenRanking(ranking) {
    var m = /^GNN-LRP (\S+), (walks|starting sensors)$/.exec(ranking);
    return m && gammaLabel(m[1]) ? { label: gammaLabel(m[1]), walks: m[2] === "walks" } : null;
  }

  function flipName(ranking) {
    var chosen = chosenRanking(ranking);
    // Named the way the two stored settings are: the kind of ranking, then the setting.
    if (chosen) { return (chosen.walks ? "Whole walks, " : "Sensor totals, ") + lowerFirst(chosen.label); }
    return FLIP_NAMES[ranking] || ranking;
  }

  function flipStyle(ranking) {
    var chosen = chosenRanking(ranking);
    if (chosen) { return chosen.walks ? { colour: "var(--fig-core)", width: 2.2 } : { colour: "var(--fig-core)", dash: "6 4" }; }
    return FLIP_STYLES[ranking] || { colour: "var(--fig-axis)" };
  }

  // Which stored ranking is this explanation's own: its whole walks, under
  // the setting of the rule it was made with.
  function ownRanking(sum, rows) {
    var setting = gammaSettingOf(sum);
    var wanted = setting === "gnn-gi" ? "GNN-GI, walks" : setting && setting !== "authors" ? "GNN-LRP " + setting + ", walks" : "GNN-LRP, walks";
    return rows.some(function (r) { return r.ranking === wanted; }) ? wanted : "GNN-LRP, walks";
  }

  // A short sample of the line a ranking is drawn with, for a table or a sentence.
  function lineKey(style) {
    return '<svg class="pg-check-key" viewBox="0 0 28 10" aria-hidden="true"><line x1="1" x2="27" y1="5" y2="5" stroke="' + style.colour +
      '" stroke-width="' + (style.width || 1.7) + '"' + (style.dash ? ' stroke-dasharray="' + style.dash + '"' : "") + "/></svg>";
  }

  // The check's two tests for the chosen cycle or engine, side by side where
  // there is room: the curves, the order this explanation's own ranking
  // took the sensors in, and a table that is the curves' legend as well,
  // with each curve's mean here and over every saved cycle or engine.
  function flippingFigure(data, index, W) {
    var sum = data.summary, forEngine = sum.scope === "engine";
    var rows = (data.flipping || []).filter(function (r) { return Number(r.window) === index; });
    if (!rows.length) {
      return '<div class="pg-empty">This saved explanation was made before the check was stored with it. Explain the run again to add it.</div>';
    }
    var chosen = data.values[index], sensors = walkSensors(sum);
    var here = forEngine ? "cycle " + chosen.end_cycle : "engine " + chosen.unit;
    var averages = sum.flipping_aufc || [], own = ownRanking(sum, rows);
    var steps = Object.keys(rows[0]).filter(function (k) { return k.indexOf("step_") === 0; }).length;
    // Two cards share the panel's width less the 16 between them; each keeps 26 for its own frame.
    var two = W >= 1180, cardW = two ? Math.floor(W / 2 - 21) : W;
    var H = Math.round(Math.max(220, Math.min(320, cardW * 0.46)));
    var all = averages.length ? "all " + count(averages[0].windows) + (forEngine ? " cycles" : " engines") : "";

    function card(task) {
      var adding = task === "activation";
      var mine = rows.filter(function (r) { return r.task === task; });
      var curve = function (r) {
        var points = [];
        for (var k = 0; k < steps; k++) { points.push([k, Number(r["step_" + k])]); }
        return points;
      };
      var chart = plot({ width: cardW, height: H,
        series: mine.map(function (r) {
          var style = flipStyle(r.ranking);
          return { name: flipName(r.ranking), colour: style.colour, width: style.width, dash: style.dash, points: curve(r), nolegend: true };
        }),
        label: adding ? "The score as sensors are added" : "How far the score has moved as sensors are removed",
        xlabel: adding ? "sensors added so far" : "sensors removed so far",
        ylabel: adding ? "score" : "distance from the full graph's score" });
      var mineOwn = mine.filter(function (r) { return r.ranking === own; })[0];
      var order = mineOwn && mineOwn.order ? String(mineOwn.order).split(" ").map(function (i) { return sensors[Number(i)]; }) : [];
      var chips = order.length && order.every(Boolean) ? '<p class="pg-check-order"><span>At ' + esc(here) + ", this explanation " +
        (adding ? "adds" : "removes") + " them in this order:</span>" + order.map(sensorChip).join("") + "</p>" : "";
      var stats = averages.filter(function (a) { return a.task === task; });
      var lo = Infinity, hi = -Infinity;
      stats.forEach(function (a) {
        var spread = Number(a.std) || 0;
        lo = Math.min(lo, a.mean - spread); hi = Math.max(hi, a.mean + spread);
      });
      var room = (hi - lo) * 0.05 || 1;
      lo -= room; hi += room;
      var pos = function (v) { return ((v - lo) / (hi - lo) * 100).toFixed(2); };
      var body = mine.map(function (r) {
        var style = flipStyle(r.ranking), a = stats.filter(function (x) { return x.ranking === r.ranking; })[0];
        var spread = a && a.std !== null && a.std !== undefined ? Number(a.std) : null;
        // The mean of a curve's points is what the authors call its area.
        var area = isFinite(Number(r.aufc)) && r.aufc !== null ? Number(r.aufc)
          : curve(r).reduce(function (total, p) { return total + p[1]; }, 0) / steps;
        return "<tr" + (r.ranking === own ? ' class="is-own"' : "") + '><td><div class="pg-check-name">' + lineKey(style) + "<span>" + esc(flipName(r.ranking)) +
          (r.ranking === own ? ' <span class="pg-tag">this explanation</span>' : "") + '</span></div></td><td class="num">' + fixed(area, 3) +
          '</td><td class="num">' + (a ? fixed(a.mean, 3) + (spread !== null ? " ± " + fixed(spread, 3) : "") : "") + "</td>" +
          (stats.length ? '<td class="pg-check-range" aria-hidden="true">' + (a ? '<span class="pg-range">' +
            (spread !== null ? '<span class="pg-range-line" style="left:' + pos(a.mean - spread) + "%;width:" +
              (2 * spread / (hi - lo) * 100).toFixed(2) + '%"></span>' : "") +
            '<span class="pg-range-dot" style="left:' + pos(a.mean) + "%;background:" + style.colour + '"></span></span>' : "") + "</td>" : "") + "</tr>";
      }).join("");
      var table = '<table class="pg-check-table"><thead><tr><th>Order taken from</th><th class="num">At ' + esc(here) + "</th>" +
        (stats.length ? '<th class="num">Mean ± spread, ' + all + '</th><th aria-hidden="true"></th>' : '<th class="num"></th>') +
        "</tr></thead><tbody>" + body + "</tbody></table>";
      return '<section class="pg-figure pg-check-card"><div class="pg-check-head"><h4 class="pg-figure-title">' +
        (adding ? "Adding sensors" : "Removing sensors") + '</h4><span class="pg-tag pg-tag--plain">' +
        (adding ? "higher is better" : "lower is better") + '</span></div><p class="pg-figure-lead">' +
        (adding ? "Start from a graph with no sensor in it and add one at a time, each time the sensor that the ranking says makes the set worth most. " +
          "A ranking that describes the network well raises the score fastest."
          : "Start from the full graph and remove one at a time, each time the sensor whose loss the ranking says costs least. " +
          "A ranking that describes the network well moves the score least.") + "</p>" + chart + chips + table + "</section>";
    }

    var kinds = '<div class="pg-check-kinds"><p>' + lineKey({ colour: "var(--fig-ink)", width: 2.2 }) +
      "<span><b>Whole walks</b> value a set of sensors by every walk that lies wholly inside it.</span></p><p>" +
      lineKey({ colour: "var(--fig-ink)", dash: "6 4" }) +
      "<span><b>Sensor totals</b> value it by its sensors' own numbers added up, the ones the Engine tab shows: every walk that starts in the set, wherever it goes.</span></p><p>" +
      lineKey(FLIP_STYLES.random) + "<span><b>Random orders</b>, averaged, show what an order that knows nothing about the network reaches.</span></p></div>";
    return kinds + '<div class="pg-check' + (two ? " pg-check--two" : "") + '">' + card("activation") + card("pruning") + "</div>" +
      '<p class="pg-faint">The number given for a curve is the mean of its points, which the method\'s authors call the area under the flipping curve.' +
      (averages.length ? " The spread is the standard deviation between " + all.replace("all ", "the ") + "." : "") +
      " A sensor that is out of the graph keeps its readings but has every connection cut, its link to itself included, " +
      "so nothing of it reaches the score through the graph layers (" + refLink("A-33") + ").</p>";
  }

  // The same three parts and their total for every cycle of one engine, as
  // lines with the chosen cycle marked, or across the test engines, as one
  // row of dots per part with the chosen engine ringed.
  function walkParts(data, index, W) {
    var sum = data.summary, rows = data.values, forEngine = sum.scope === "engine";
    var at = Math.max(0, Math.min(index || 0, rows.length - 1)), row = rows[at];
    var parts = walkPartList(sum).concat([{ key: "score", name: "Score", colour: "var(--fig-ink)" }]);
    if (!forEngine) {
      return '<h4 class="pg-figure-title">The same parts across the ' + count(rows.length) + " test engines</h4>" +
        '<p class="pg-figure-lead">One dot is one engine, and the ringed dot is the chosen one. The engines are not set out in order, ' +
        "because an engine's number is a name and not a quantity.</p>" + walkPartDistribution(rows, W, at, parts);
    }
    var series = parts.map(function (part) {
      var line = { name: part.name, colour: part.colour,
        points: rows.map(function (r) { return [r.end_cycle, r[part.key]]; }),
        marks: [[row.end_cycle, row[part.key], part.name + " at cycle " + row.end_cycle + ": " + scoreShare(Number(row[part.key]))]] };
      if (part.key === "score") { line.dash = "5 4"; line.width = 1.5; }
      return line;
    });
    return '<h4 class="pg-figure-title">The same parts at every cycle of engine ' + esc(row.unit) + "</h4>" +
      '<p class="pg-figure-lead">The dashed line is the score, and at every cycle the three coloured lines add up to it. ' +
      "Where the dashed line moves, the coloured line that moves with it is the part that changed the prediction.</p>" +
      plot({ width: W, height: chartHeight(W), xlabel: "cycle at the end of the window", ylabel: "score (no unit)",
        label: "each part of the score along the engine's life", series: series, zero: true,
        vlines: [{ x: row.end_cycle, label: "cycle " + row.end_cycle }] });
  }

  // One row per part of the score, holding every test engine. Engine number
  // is deliberately absent: it is an identifier, not an ordered quantity from
  // which a trend can be read. The chosen engine is ringed in every row.
  function walkPartDistribution(rows, W, chosen, parts) {
    var narrow = W < 420, values = [];
    parts.forEach(function (part) {
      part.values = rows.map(function (row) { return Number(row[part.key]); });
      part.lines = wrapWords(part.name, narrow ? 16 : 22);
      values = values.concat(part.values);
    });
    var sortedAt = function (list, share) {
      var sorted = list.slice().sort(function (a, b) { return a - b; });
      var at = (sorted.length - 1) * share, below = Math.floor(at), above = Math.ceil(at);
      return sorted[below] + (sorted[above] - sorted[below]) * (at - below);
    };
    var lo = Math.min(0, Math.min.apply(null, values)), hi = Math.max(0, Math.max.apply(null, values));
    var pad = (hi - lo) * 0.06 || 1;
    lo -= pad; hi += pad;
    var left = narrow ? 112 : 156, right = 14, top = 16, rowH = 48, bottom = 40;
    var H = top + parts.length * rowH + bottom;
    var X = function (v) { return left + (v - lo) / (hi - lo) * (W - left - right); };
    var out = '<svg class="pg-plot" viewBox="0 0 ' + W + " " + H +
      '" role="img" aria-label="The distribution of each part of the score across the test engines">';
    niceTicks(lo, hi, 7).forEach(function (tick) {
      out += '<line x1="' + X(tick).toFixed(1) + '" x2="' + X(tick).toFixed(1) + '" y1="' + top +
        '" y2="' + (H - bottom) + '" stroke="var(--line-soft)"/><text x="' + X(tick).toFixed(1) + '" y="' +
        (H - bottom + 16) + '" text-anchor="middle" class="pg-tick">' + tickLabel(tick) + "</text>";
    });
    out += '<line x1="' + X(0).toFixed(1) + '" x2="' + X(0).toFixed(1) + '" y1="' + top + '" y2="' +
      (H - bottom) + '" stroke="var(--fig-axis)" stroke-width="1.2"/>';
    parts.forEach(function (part, partIndex) {
      var cy = top + partIndex * rowH + rowH / 2;
      var minimum = Math.min.apply(null, part.values), maximum = Math.max.apply(null, part.values);
      var q1 = sortedAt(part.values, 0.25), median = sortedAt(part.values, 0.5), q3 = sortedAt(part.values, 0.75);
      part.lines.forEach(function (line, i) {
        out += '<text x="' + (left - 10) + '" y="' + (cy + 4 + (i - (part.lines.length - 1) / 2) * 14) +
          '" text-anchor="end" class="pg-tick">' + esc(line) + "</text>";
      });
      out += '<line x1="' + X(minimum).toFixed(1) + '" x2="' + X(maximum).toFixed(1) + '" y1="' + cy +
        '" y2="' + cy + '" stroke="' + part.colour + '" stroke-opacity="0.55"/>';
      part.values.forEach(function (value, engineIndex) {
        var jitter = ((engineIndex * 5) % 9 - 4) * 1.4;
        out += '<circle cx="' + X(value).toFixed(1) + '" cy="' + (cy + jitter).toFixed(1) +
          '" r="2" fill="' + part.colour + '" fill-opacity="0.34"><title>' +
          esc(part.name + " · engine " + rows[engineIndex].unit + ": " + scoreShare(value)) + "</title></circle>";
      });
      out += '<rect x="' + X(q1).toFixed(1) + '" y="' + (cy - 7) + '" width="' +
        Math.max(1, X(q3) - X(q1)).toFixed(1) + '" height="14" rx="3" fill="' + part.colour +
        '" fill-opacity="0.20" stroke="' + part.colour + '"/><line x1="' + X(median).toFixed(1) + '" x2="' +
        X(median).toFixed(1) + '" y1="' + (cy - 10) + '" y2="' + (cy + 10) + '" stroke="' + part.colour +
        '" stroke-width="2"><title>' + esc(part.name + " median: " + scoreShare(median)) + "</title></line>";
      out += '<circle cx="' + X(part.values[chosen]).toFixed(1) + '" cy="' + cy + '" r="5" fill="var(--fig-bg)" stroke="var(--fig-ink)" stroke-width="2"><title>' +
        esc(part.name + " · engine " + rows[chosen].unit + ", the chosen one: " + scoreShare(part.values[chosen])) + "</title></circle>";
    });
    out += '<text x="' + ((left + W - right) / 2) + '" y="' + (H - 5) +
      '" text-anchor="middle" class="pg-axis">score (no unit)</text></svg>';
    var medians = parts.map(function (part) { return lowerFirst(part.name) + " " + scoreShare(sortedAt(part.values, 0.5)); });
    var positiveConstants = parts[0].values.filter(function (value) { return value > 0; }).length;
    return out + '<div class="pg-legend"><span><i style="background:var(--fig-axis);opacity:.45"></i>one test engine</span>' +
      '<span><i style="background:var(--fig-bg);box-shadow:inset 0 0 0 2px var(--fig-ink)"></i>engine ' + esc(rows[chosen].unit) + ", the chosen one</span>" +
      '<span><i style="background:var(--fig-accent);opacity:.20;border-radius:2px"></i>middle half of the engines</span>' +
      '<span><i style="background:var(--fig-ink);width:2px;border-radius:0"></i>median, the middle engine after ordering</span></div>' +
      '<p class="pg-faint">Medians: ' + esc(medians.join("; ")) + ". What the network adds is above zero for " + positiveConstants + " of " + rows.length +
      " engines. A part near zero can still hide large positive and negative walks that cancel.</p>";
  }

  // The sensor graph with what passed along each connection: one line per
  // connection this explanation's own graph has, read from its link_
  // columns, so a run on another graph shows that graph on the same places.
  // A line is as wide as what passed both ways together, blue where that
  // raised the prediction and orange where it lowered it; the widest line of
  // any window sets the scale, so widths compare as the slider moves. With
  // no window chosen (index null) each connection is averaged over the rows.
  // The sensors are left uncoloured, since the beeswarm already ranks them.
  function flowGraph(rows, index, sensors) {
    var g = S.options.graph, scale = 0.8, names = "";
    var pairs = [];
    sensors.forEach(function (a, i) {
      sensors.slice(i + 1).forEach(function (b) {
        var ab = "link_" + a + "_" + b, ba = "link_" + b + "_" + a;
        if ((rows[0][ab] !== undefined || rows[0][ba] !== undefined) && g.positions[a] && g.positions[b]) { pairs.push([a, b, ab, ba]); }
      });
    });
    var value = function (r, p) { return (Number(r[p[2]]) || 0) + (Number(r[p[3]]) || 0); };
    var shown = rows[index];
    if (index === null) {
      shown = {};
      Object.keys(rows[0]).forEach(function (k) {
        if (k.indexOf("link_") !== 0) { return; }
        var total = 0;
        rows.forEach(function (r) { total += Number(r[k]) || 0; });
        shown[k] = total / rows.length;
      });
    }
    var peak = 0;
    (index === null ? [shown] : rows).forEach(function (r) {
      pairs.forEach(function (p) { peak = Math.max(peak, Math.abs(value(r, p))); });
    });
    peak = peak || 1;
    // One direction of a connection in words: "T24 to T30: +0.570".
    var way = function (label, key) {
      return label + ": " + (shown[key] === undefined ? "no connection" : scoreShare(Number(shown[key])));
    };
    var out = '<svg class="pg-graph" viewBox="0 0 ' + g.width * scale + " " + (g.height - 30) * scale +
      '" role="img" aria-label="What passed along each connection">';
    // The narrow lines first, so a wide one is never hidden under them.
    pairs.map(function (p) { return { p: p, v: value(shown, p) }; })
      .sort(function (a, b) { return Math.abs(a.v) - Math.abs(b.v); })
      .forEach(function (l) {
        var a = g.positions[l.p[0]], b = g.positions[l.p[1]];
        var ends = 'x1="' + a[0] * scale + '" y1="' + a[1] * scale + '" x2="' + b[0] * scale + '" y2="' + b[1] * scale + '"';
        out += "<g><title>" + esc(way(l.p[0] + " to " + l.p[1], l.p[2]) + " · " + way(l.p[1] + " to " + l.p[0], l.p[3])) + "</title>" +
          "<line " + ends + ' stroke="' + (l.v >= 0 ? "var(--fig-accent)" : "var(--fig-warn)") + '" stroke-width="' +
          (1.2 + 6 * Math.abs(l.v) / peak).toFixed(2) + '" stroke-linecap="round"/>' +
          "<line " + ends + ' stroke="transparent" stroke-width="10"/></g>';
      });
    sensors.forEach(function (s) {
      var p = g.positions[s];
      if (!p) { return; }
      var self = shown["link_" + s + "_" + s];
      var note = self !== undefined ? "along its connection to itself: " + scoreShare(Number(self)) : "";
      var tip = sensorTip(s, note);
      var title = tip ? "" : "<title>" + esc(s + (note ? " · " + note : "")) + "</title>";
      out += "<g" + tip + ">" + title + '<circle cx="' + p[0] * scale + '" cy="' + p[1] * scale +
        '" r="17" fill="var(--fig-bg)" stroke="var(--fig-axis)" stroke-width="1.5"/></g>';
      names += "<g" + tip + ">" + title + '<text x="' + p[0] * scale + '" y="' + (p[1] * scale + 30) + '" text-anchor="middle" class="pg-node">' + esc(s) + "</text></g>";
    });
    return '<div class="pg-graph-stage">' + out + names + "</svg></div>" +
      '<div class="pg-legend"><span><i style="background:var(--fig-accent)"></i>raised the prediction</span>' +
      '<span><i style="background:var(--fig-warn)"></i>lowered it</span></div>' +
      '<p class="pg-faint">The widest line stands for ' + fixed(peak, 2) + " of the score before the sigmoid. Point at a line " +
      "to read what passed each way, and at a sensor for what passed along its connection to itself.</p>";
  }

  // The walks with the largest average size, each drawn as the sensors it
  // passes through, one per graph layer, beside how much it carries: its
  // average share as a solid bar, inside its average size with signs
  // ignored. A walk stays when every position after its start is its own
  // start's group, which is how the score's two walk parts are told apart.
  function topWalks(data) {
    var top = data.top_walks || [], sum = data.summary;
    if (!top.length) { return ""; }
    var forEngine = sum.scope === "engine", n = data.values.length;
    var over = "the " + count(n) + (forEngine ? " cycles of engine " + esc(data.values[0].unit) : " test engines");
    var home = (sum.groups || []).length ? "group" : "sensor";
    var groupOf = {};
    (sum.groups || []).forEach(function (g) { g.forEach(function (s) { groupOf[s] = g.join("/"); }); });
    var peak = Math.max.apply(null, top.map(function (w) { return Number(w.mean_abs) || 0; })) || 1;
    var layers = String(top[0].walk).split(" -> ").length - 1;
    var heads = "<span>Starts at</span>";
    for (var i = 1; i <= layers; i++) { heads += "<span>After layer " + i + "</span>"; }
    var half = function (v) { return (Math.abs(v) / peak * 50).toFixed(2); };
    var list = top.map(function (w, index) {
      var steps = String(w.walk).split(" -> ");
      var start = groupOf[steps[0]] || steps[0];
      var stays = steps.slice(1).every(function (step) { return step === start; });
      var mean = Number(w.mean), size = Number(w.mean_abs), side = mean >= 0 ? "left:50%" : "right:50%";
      var kind = stays ? "stays in its " + home : "crosses to another " + home;
      return '<li class="pg-walk" aria-label="' + esc("Walk " + (index + 1) + ": " + steps.join(", then ") + "; " + kind + "; average share " +
        scoreShare(mean) + "; average size " + fixed(size, 3)) + '"><span class="pg-walk-rank">' + (index + 1) + "</span>" +
        '<span class="pg-walk-steps" style="--steps:' + steps.length + '">' + steps.map(function (step) {
          return '<span class="pg-walk-step">' + walkStep(step) + "</span>";
        }).join("") + '</span><span class="pg-walk-kind"><i style="background:' + (stays ? "var(--fig-lpt)" : "var(--fig-lpc)") + '"></i>' + kind + "</span>" +
        '<span class="pg-walk-number" data-label="average share">' + scoreShare(mean) + "</span>" +
        '<span class="pg-walk-number" data-label="average size">' + fixed(size, 3) + "</span>" +
        '<span class="pg-walk-bar" aria-hidden="true"><span class="pg-walk-size" style="' + side + ";width:" + half(size) + '%"></span>' +
        '<span class="pg-walk-share pg-walk-share--' + (mean >= 0 ? "up" : "down") + '" style="' + side + ";width:" + half(mean) + '%"></span></span>' +
        '<button type="button" class="button button--quiet button--small" data-action="walk-show" data-walk="' + index + '">Show on engine</button></li>';
    }).join("");
    var cancel = Number(sum.cancellation_median);
    var callout = isFinite(cancel) ? '<div class="pg-callout"><strong>' + fixed(cancel, 1) + "×</strong><p><b>Walks mostly cancel one another.</b> " +
      "Added up with their signs ignored, the walks of one " + (forEngine ? "cycle" : "engine") + " come to " + fixed(cancel, 1) +
      " times their sum with signs kept, as a median over " + over + ": most of what one walk adds, another takes away. " +
      "So a large walk in this list is not, on its own, a large effect on the prediction.</p></div>" : "";
    return walkTabHead("Which walks carry the most?",
      "A walk is " + walkMeaning(sum) + ": the sensor it starts at, then the sensor" + (home === "group" ? " or group" : "") +
      " that holds it after each layer. These are the " + count(top.length) + " walks with the largest average size over " + over +
      ". They are averages, so the " + (forEngine ? "Engine cycle" : "Test engine") + " control above does not change this list.") + callout +
      '<figure class="pg-figure pg-figure--wide"><div class="pg-walks-head" aria-hidden="true"><span></span><span class="pg-walk-steps" style="--steps:' +
      (layers + 1) + '">' + heads + "</span><span>Kind</span><span class=\"num\">Average share</span><span class=\"num\">Average size</span>" +
      '<span class="pg-walk-ends"><span>lowers the score</span><span>raises it</span></span><span></span></div>' +
      '<ol class="pg-walks">' + list + "</ol>" +
      '<div class="pg-legend"><span><i class="pg-key-bar" style="background:var(--fig-accent)"></i>average share, positive</span>' +
      '<span><i class="pg-key-bar" style="background:var(--fig-warn)"></i>average share, negative</span>' +
      '<span><i class="pg-key-bar pg-key-bar--size"></i>average size, signs ignored</span></div>' +
      '<p class="pg-faint">A share is a part of the score, a number with no unit. Where the solid bar is much shorter than the pale one, the walk\'s share is positive at some ' +
      (forEngine ? "cycles" : "engines") + " and negative at others." +
      ((sum.groups || []).length ? " Names in one chip are a group of sensors the graph layers cannot tell apart, so the walks through them are added together." : "") +
      "</p></figure>";
  }

  // One chart per sensor, holding that sensor's value at every window of the
  // engine's life, every chart on ONE vertical scale so that a tall chart
  // means a large value and not a large axis. This is the counterpart of
  // plot_contributions_over_time in explain/plots.py, which draws the same
  // thing for the notebooks. Each chart is drawn at the size it is shown at,
  // so the cycles along its bottom and the values up its side are read at
  // the page's own text size. A cursor marks one window in every chart at
  // once and each heading reads that sensor's value there; wireHistory moves
  // it. As for the beeswarm, unit is left out for cycles and "" for score
  // shares.
  var HISTORY_GAP = 16, HISTORY_PAD = { left: 40, right: 12, top: 10, bottom: 24 };
  var historyCursor = null;   // moves the cursor to a window, while the chart is on screen

  // How many charts stand side by side at a width: four at most, and none
  // narrower than 340 pixels until the screen itself is.
  function historyColumns(width) {
    return Math.max(1, Math.min(4, Math.floor(width / 340)));
  }

  // Where a cycle and a value land in one chart: shared by every chart, by
  // the drawing and by the pointer that reads a cycle back from it.
  function historyScale(rows, symbols, width) {
    var cols = historyColumns(width), p = HISTORY_PAD;
    var W = Math.floor((width - (cols - 1) * HISTORY_GAP) / cols);
    var H = Math.round(Math.max(168, Math.min(220, W * 0.46)));
    var cycles = rows.map(function (r) { return r.end_cycle; });
    var x0 = Math.min.apply(null, cycles), x1 = Math.max.apply(null, cycles);
    var lo = 0, hi = 0;
    symbols.forEach(function (s) {
      rows.forEach(function (r) { if (isFinite(r[s])) { lo = Math.min(lo, r[s]); hi = Math.max(hi, r[s]); } });
    });
    var largest = Math.max(-lo, hi);
    var room = (hi - lo) * 0.06 || 1;
    lo -= room; hi += room;
    return {
      cols: cols, W: W, H: H, x0: x0, x1: x1, lo: lo, hi: hi, largest: largest,
      X: function (c) { return p.left + (c - x0) / ((x1 - x0) || 1) * (W - p.left - p.right); },
      Y: function (v) { return p.top + (hi - v) / (hi - lo) * (H - p.top - p.bottom); },
      cycleAt: function (x) { return x0 + (x - p.left) / (W - p.left - p.right) * (x1 - x0); }
    };
  }

  function signedValue(v) {
    return (v > 0 ? "+" : "") + fixed(v, 2);
  }

  function sensorHistory(rows, symbols, current, width, unit) {
    var k = historyScale(rows, symbols, width), X = k.X, Y = k.Y, p = HISTORY_PAD;
    var W = k.W, H = k.H, zero = Y(0), floor = H - p.bottom, edge = W - p.right;

    // The cycle at which the capped remaining life first falls below its
    // ceiling: before it the label is flat, after it the label is falling.
    var ceiling = Math.max.apply(null, rows.map(function (r) { return r.rul; }));
    var knee = null;
    for (var i = 0; i < rows.length; i++) {
      if (rows[i].rul < ceiling) { knee = rows[i].end_cycle; break; }
    }

    var marked = rows[current] || rows[0];
    var cursorX = X(marked.end_cycle).toFixed(1);
    // The same lines and numbers behind every chart: values up the side,
    // cycles along the bottom.
    var grid = niceTicks(k.lo, k.hi, 4).map(function (t) {
      var y = Y(t).toFixed(1);
      return '<line x1="' + p.left + '" x2="' + edge + '" y1="' + y + '" y2="' + y + '" stroke="var(--line-soft)"/>' +
        '<text x="' + (p.left - 6) + '" y="' + (Y(t) + 4).toFixed(1) + '" text-anchor="end" class="pg-tick">' + tickLabel(t) + "</text>";
    }).join("") + niceTicks(k.x0, k.x1, W < 400 ? 4 : 6).map(function (t) {
      var x = X(t).toFixed(1);
      return '<line x1="' + x + '" x2="' + x + '" y1="' + floor + '" y2="' + (floor + 4) + '" stroke="var(--fig-axis)"/>' +
        '<text x="' + x + '" y="' + (floor + 17) + '" text-anchor="middle" class="pg-tick">' + tickLabel(t) + "</text>";
    }).join("") + '<line x1="' + p.left + '" x2="' + edge + '" y1="' + floor + '" y2="' + floor + '" stroke="var(--fig-axis)"/>';

    var panels = symbols.map(function (s, index) {
      var pts = rows.map(function (r) { return X(r.end_cycle).toFixed(1) + "," + Y(r[s]).toFixed(1); }).join(" ");
      var area = X(rows[0].end_cycle).toFixed(1) + "," + zero.toFixed(1) + " " + pts + " " +
                 X(rows[rows.length - 1].end_cycle).toFixed(1) + "," + zero.toFixed(1);
      var up = "pg-up" + index, down = "pg-down" + index;
      return '<figure class="pg-sm" data-sensor="' + esc(s) + '">' +
        '<figcaption><b' + sensorTip(s) + ">" + esc(s) + '</b><span class="pg-sm-value">' + signedValue(marked[s]) + "</span></figcaption>" +
        '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="' + esc(s + " over the engine's life") + '">' +
        "<defs>" +
        '<clipPath id="' + up + '"><rect x="0" y="0" width="' + W + '" height="' + zero.toFixed(1) + '"/></clipPath>' +
        '<clipPath id="' + down + '"><rect x="0" y="' + zero.toFixed(1) + '" width="' + W + '" height="' + (H - zero).toFixed(1) + '"/></clipPath>' +
        "</defs>" + grid +
        (knee !== null ? '<line x1="' + X(knee).toFixed(1) + '" x2="' + X(knee).toFixed(1) +
          '" y1="' + p.top + '" y2="' + floor + '" stroke="var(--fig-axis)" stroke-dasharray="3 3"/>' : "") +
        '<polygon points="' + area + '" fill="var(--fig-accent)" fill-opacity="0.28" clip-path="url(#' + up + ')"/>' +
        '<polygon points="' + area + '" fill="var(--fig-warn)" fill-opacity="0.28" clip-path="url(#' + down + ')"/>' +
        '<line x1="' + p.left + '" x2="' + edge + '" y1="' + zero.toFixed(1) + '" y2="' + zero.toFixed(1) + '" stroke="var(--fig-axis)" stroke-width="0.8"/>' +
        '<polyline points="' + pts + '" fill="none" stroke="var(--fig-ink)" stroke-width="1.3" stroke-linejoin="round"/>' +
        '<line class="pg-sm-cursor" x1="' + cursorX + '" x2="' + cursorX + '" y1="' + p.top + '" y2="' + floor + '"/>' +
        '<circle class="pg-sm-dot" cx="' + cursorX + '" cy="' + Y(marked[s]).toFixed(1) + '" r="3"/>' +
        "</svg></figure>";
    }).join("");

    // The readout names the cycle only while the pointer is on a chart; the
    // rest of the time the slider's caption already says which window is marked.
    return '<div class="pg-sm-readout is-idle" aria-live="polite">Values at cycle <b>' + marked.end_cycle + "</b></div>" +
      '<div class="pg-smalls" style="grid-template-columns:repeat(' + k.cols + ',minmax(0,1fr))">' + panels + "</div>" +
      '<div class="pg-faint">Blue raised the predicted remaining life, orange lowered it. ' +
      "Every chart shares one scale; the largest value is " + fixed(k.largest, 2) + (unit === undefined ? " cycles" : "") + "." +
      (knee !== null ? " The dashed line is cycle " + knee + ", where the capped remaining life starts falling." : "") +
      "</div>";
  }

  // Pointing at any chart moves the cursor in all of them to the nearest
  // window; leaving puts it back on the window the slider has chosen.
  function wireHistory(scope, rows, symbols, width) {
    historyCursor = null;
    var grid = scope.querySelector(".pg-smalls");
    if (!grid || !rows.length) { return; }
    var scale = historyScale(rows, symbols, width);
    var readoutBox = scope.querySelector(".pg-sm-readout");
    var readout = readoutBox && readoutBox.querySelector("b");
    var panels = Array.prototype.map.call(grid.querySelectorAll(".pg-sm"), function (panel) {
      return {
        sensor: panel.getAttribute("data-sensor"),
        cursor: panel.querySelector(".pg-sm-cursor"),
        dot: panel.querySelector(".pg-sm-dot"),
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
        p.dot.setAttribute("cx", x);
        p.dot.setAttribute("cy", scale.Y(row[p.sensor]).toFixed(1));
        p.value.textContent = signedValue(row[p.sensor]);
      });
      if (readout) { readout.textContent = row.end_cycle; }
    }

    grid.addEventListener("pointermove", function (event) {
      var svg = event.target.closest && event.target.closest("svg");
      if (!svg) { return; }
      var box = svg.getBoundingClientRect();
      var cycle = scale.cycleAt((event.clientX - box.left) / (box.width || 1) * scale.W);
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

  // The same values in one space: a ribbon per sensor running along the
  // engine's cycles, its height the value. Sensors are separate things, so
  // the ribbons are not joined into a surface, which would draw values
  // between two sensors where none exist. Nothing shrinks with distance:
  // equal lengths along one axis stay equal on the screen, so two heights
  // can be compared from front to back. The ground the ribbons stand on is
  // the value zero. RIBBON_BOX holds the lengths of the feature, cycle and
  // value axes against one another, and RIBBON_VIEW the two angles the
  // chart is first seen from, in degrees: az turns it about the value axis
  // and el tilts its top towards the reader.
  var RIBBON_BOX = { a: 2, b: 2, c: 0.8 };
  var RIBBON_VIEW = { az: -45, el: 30 };
  var ribbonsRedraw = null;   // draws the 3D chart again, while it is on screen

  // How large the 3D chart is drawn inside a panel of a given width.
  function ribbonSize(width) {
    var W = Math.max(440, Math.min(width, 1100));
    return { W: W, H: Math.round(Math.max(360, Math.min(680, W * 0.62))) };
  }

  // Turns a place in the box (u along the features, v along the cycles, w
  // up the values, each measured from the box's centre) into a place on the
  // screen, for one pair of angles. The box is sized to the widest it can
  // look from any side, so turning it never changes its size.
  function ribbonCamera(view, W, H) {
    var t = view.az * Math.PI / 180, f = view.el * Math.PI / 180;
    var ct = Math.cos(t), st = Math.sin(t), cf = Math.cos(f), sf = Math.sin(f);
    var B = RIBBON_BOX, m = { left: 84, right: 84, top: 16, bottom: 56 };
    var across = Math.sqrt(B.a * B.a + B.b * B.b);
    var scale = Math.min((W - m.left - m.right) / across, (H - m.top - m.bottom) / (B.c * cf + across * sf));
    var cx = m.left + (W - m.left - m.right) / 2, cy = m.top + (H - m.top - m.bottom) / 2;
    return {
      // Which end of the feature axis and of the cycle axis faces the reader.
      near: { u: st > 0 ? -1 : 1, v: ct > 0 ? -1 : 1 },
      depth: function (u, v) { return u * st + v * ct; },
      at: function (u, v, w) {
        return [cx + (u * ct - v * st) * scale, cy - (w * cf + (u * st + v * ct) * sf) * scale];
      }
    };
  }

  // Every k-th label along an axis, k being the smallest step at which two
  // neighbours no longer overlap: from is one label's place on the screen,
  // to the next one's, and room the width a label needs.
  function labelStep(from, to, room) {
    var dx = Math.abs(to[0] - from[0]), dy = Math.abs(to[1] - from[1]);
    if (dx < 0.01 && dy < 0.01) { return 1; }
    return Math.max(1, Math.min(dx > 0.01 ? Math.ceil(room / dx) : Infinity, dy > 0.01 ? Math.ceil(13 / dy) : Infinity));
  }

  function ribbons(rows, symbols, width, view, current, valueName) {
    var size = ribbonSize(width), W = size.W, H = size.H, B = RIBBON_BOX;
    var cam = ribbonCamera(view, W, H), near = cam.near;
    // The box is sized for every side it can be seen from, so from any one
    // side some of its height is unused. The highest and lowest places drawn
    // are noted, and the drawing is set midway between the top and the
    // bottom of its frame.
    var highest = Infinity, lowest = -Infinity;
    var at = function (u, v, w) {
      var p = cam.at(u, v, w);
      highest = Math.min(highest, p[1]);
      lowest = Math.max(lowest, p[1]);
      return p;
    };
    var n = symbols.length;
    var cycles = rows.map(function (r) { return r.end_cycle; });
    var c0 = Math.min.apply(null, cycles), c1 = Math.max.apply(null, cycles);
    var lo = 0, hi = 0;
    symbols.forEach(function (s) {
      rows.forEach(function (r) { if (isFinite(r[s])) { lo = Math.min(lo, r[s]); hi = Math.max(hi, r[s]); } });
    });
    var room = (hi - lo) * 0.05 || 1;
    lo -= room; hi += room;
    var U = function (j) { return (n > 1 ? j / (n - 1) - 0.5 : 0) * B.a; };
    var V = function (c) { return ((c - c0) / ((c1 - c0) || 1) - 0.5) * B.b; };
    var Z = function (v) { return ((v - lo) / (hi - lo) - 0.5) * B.c; };
    var z0 = Z(0), half = { u: B.a / 2, v: B.b / 2, w: B.c / 2 };
    var pt = function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); };
    var line = function (a, b, stroke, more) {
      return '<line x1="' + a[0].toFixed(1) + '" y1="' + a[1].toFixed(1) + '" x2="' + b[0].toFixed(1) + '" y2="' + b[1].toFixed(1) +
        '" stroke="' + stroke + '"' + (more || "") + "/>";
    };
    var valueTicks = niceTicks(lo, hi, 5), cycleTicks = niceTicks(c0, c1, 6);
    var farU = -near.u * half.u, farV = -near.v * half.v, nearU = near.u * half.u, nearV = near.v * half.v;

    var out = '<svg class="pg-plot pg-3d" viewBox="0 0 ' + W + " " + H + '" style="max-width:' + W + 'px" role="img" aria-label="' +
      esc("Every sensor's " + valueName + " over the engine's cycles, in three dimensions") + '">';
    var body = "";

    // The two walls behind the ribbons, with a line at every value the
    // vertical axis names; the line at zero is the darker one.
    body += '<polygon points="' + [at(farU, -half.v, -half.w), at(farU, half.v, -half.w), at(farU, half.v, half.w), at(farU, -half.v, half.w)].map(pt).join(" ") +
      '" fill="var(--panel)" fill-opacity="0.6" stroke="var(--line)"/>' +
      '<polygon points="' + [at(-half.u, farV, -half.w), at(half.u, farV, -half.w), at(half.u, farV, half.w), at(-half.u, farV, half.w)].map(pt).join(" ") +
      '" fill="var(--panel)" fill-opacity="0.6" stroke="var(--line)"/>';
    valueTicks.forEach(function (t) {
      var stroke = t === 0 ? "var(--fig-axis)" : "var(--line)";
      body += line(at(farU, -half.v, Z(t)), at(farU, half.v, Z(t)), stroke) + line(at(-half.u, farV, Z(t)), at(half.u, farV, Z(t)), stroke);
    });
    // The ground at the value zero: its edge, and a line at every cycle the cycle axis names.
    body += '<polygon points="' + [at(-half.u, -half.v, z0), at(half.u, -half.v, z0), at(half.u, half.v, z0), at(-half.u, half.v, z0)].map(pt).join(" ") +
      '" fill="none" stroke="var(--fig-axis)" stroke-width="0.8"/>';
    cycleTicks.forEach(function (t) { body += line(at(-half.u, V(t), z0), at(half.u, V(t), z0), "var(--line)"); });

    // The ribbons, the farthest first, so a nearer one is drawn over it.
    var order = symbols.map(function (s, j) { return j; }).sort(function (a, b) { return cam.depth(U(b), 0) - cam.depth(U(a), 0); });
    order.forEach(function (j, place) {
      var s = symbols[j], u = U(j);
      var pts = rows.filter(function (r) { return isFinite(r[s]); }).map(function (r) { return [r.end_cycle, r[s]]; });
      if (!pts.length) { return; }
      // Each side of zero is filled in its own colour, so a point is added
      // wherever the line crosses zero between two windows.
      var path = [];
      pts.forEach(function (p, i) {
        if (i && (pts[i - 1][1] > 0) !== (p[1] > 0) && pts[i - 1][1] !== 0 && p[1] !== 0) {
          var before = pts[i - 1], share = before[1] / (before[1] - p[1]);
          path.push([before[0] + (p[0] - before[0]) * share, 0]);
        }
        path.push(p);
      });
      var side = function (keep) {
        return pt(at(u, V(path[0][0]), z0)) + " " + path.map(function (p) { return pt(at(u, V(p[0]), Z(keep(p[1])))); }).join(" ") + " " +
          pt(at(u, V(path[path.length - 1][0]), z0));
      };
      var nearness = n > 1 ? place / (n - 1) : 1;
      // Under its colour each side is first painted in the page's own, so a
      // ribbon hides most of what stands behind it instead of mixing with it.
      var above = side(function (v) { return Math.max(v, 0); }), below = side(function (v) { return Math.min(v, 0); });
      body += '<g class="pg-3d-ribbon"' + sensorTip(s) + ">" +
        '<polygon points="' + above + '" fill="var(--fig-bg)" fill-opacity="0.82"/>' +
        '<polygon points="' + below + '" fill="var(--fig-bg)" fill-opacity="0.82"/>' +
        '<polygon points="' + above + '" fill="var(--fig-accent)" fill-opacity="0.3"/>' +
        '<polygon points="' + below + '" fill="var(--fig-warn)" fill-opacity="0.3"/>' +
        line(at(u, -half.v, z0), at(u, half.v, z0), "var(--fig-axis)", ' stroke-width="0.8"') +
        '<polyline points="' + pts.map(function (p) { return pt(at(u, V(p[0]), Z(p[1]))); }).join(" ") +
        '" fill="none" stroke="var(--fig-ink)" stroke-width="1.1" stroke-linejoin="round" stroke-opacity="' + (0.6 + 0.4 * nearness).toFixed(2) + '"/></g>';
    });

    // The engine cycle chosen above: a line across the ground and a dot on every ribbon.
    var marked = rows[current] || rows[0], vm = V(marked.end_cycle);
    body += line(at(-half.u, vm, z0), at(half.u, vm, z0), "var(--fig-ink)", ' stroke-dasharray="4 3"');
    symbols.forEach(function (s, j) {
      if (!isFinite(marked[s])) { return; }
      var p = at(U(j), vm, Z(marked[s]));
      body += '<circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="2.6" fill="var(--fig-ink)" stroke="var(--fig-bg)" stroke-width="1"/>';
    });

    // The three axes. A name or a number is set outwards from its edge, on
    // the side the edge faces, and anchored on that side.
    var label = function (place, away, gap, text, cls, more) {
      var len = Math.sqrt(away[0] * away[0] + away[1] * away[1]) || 1, nx = away[0] / len, ny = away[1] / len;
      highest = Math.min(highest, place[1] + ny * gap - 10);
      lowest = Math.max(lowest, place[1] + ny * gap + 10);
      return '<text x="' + (place[0] + nx * gap).toFixed(1) + '" y="' + (place[1] + ny * gap + 4).toFixed(1) + '" text-anchor="' +
        (nx > 0.35 ? "start" : nx < -0.35 ? "end" : "middle") + '" class="' + cls + '"' + (more || "") + ">" + esc(text) + "</text>";
    };
    var centre = at(0, 0, z0);
    var awayV = [at(0, nearV, z0)[0] - centre[0], at(0, nearV, z0)[1] - centre[1]];
    var awayU = [at(nearU, 0, z0)[0] - centre[0], at(nearU, 0, z0)[1] - centre[1]];
    // Features: along the ground's edge that faces the reader, where each ribbon begins or ends.
    var stepF = n > 1 ? labelStep(at(U(0), nearV, z0), at(U(1), nearV, z0), 50) : 1;
    symbols.forEach(function (s, j) {
      if (j % stepF) { return; }
      body += label(at(U(j), nearV, z0), awayV, 10, s, "pg-3d-name", sensorTip(s));
    });
    body += label(at(0, nearV, z0), awayV, 76, "Features", "pg-axis pg-3d-title");
    // Cycles: along the other edge that faces the reader.
    var stepC = cycleTicks.length > 1 ? labelStep(at(nearU, V(cycleTicks[0]), z0), at(nearU, V(cycleTicks[1]), z0), 30) : 1;
    var end = at(nearU, farV, z0);
    cycleTicks.forEach(function (t, i) {
      var place = at(nearU, V(t), z0);
      // A cycle's number is left out where it would meet the numbers of the value axis, which starts where this edge ends.
      if (i % stepC || Math.abs(place[0] - end[0]) + Math.abs(place[1] - end[1]) < 24) { return; }
      body += label(place, awayU, 10, tickLabel(t), "pg-tick pg-3d-number");
    });
    body += label(at(nearU, 0, z0), awayU, 52, "Cycle", "pg-axis pg-3d-title");
    // Values: up the upright edge where the cycle axis ends, the one edge
    // of the outline that no feature's name stands beside.
    var post = [nearU, farV], foot = at(post[0], post[1], 0), side = foot[0] >= centre[0] ? 1 : -1;
    body += line(at(post[0], post[1], -half.w), at(post[0], post[1], half.w), "var(--fig-axis)");
    valueTicks.forEach(function (t) {
      var p = at(post[0], post[1], Z(t));
      body += line(p, [p[0] + side * 5, p[1]], "var(--fig-axis)") +
        '<text x="' + (p[0] + side * 9).toFixed(1) + '" y="' + (p[1] + 4).toFixed(1) + '" text-anchor="' + (side > 0 ? "start" : "end") +
        '" class="pg-tick pg-3d-number">' + tickLabel(t) + "</text>";
    });
    body += '<text transform="translate(' + (foot[0] + side * 52).toFixed(1) + "," + foot[1].toFixed(1) + ') rotate(-90)" text-anchor="middle" class="pg-axis pg-3d-title">' +
      esc(valueName) + "</text>";
    return out + '<g transform="translate(0,' + ((H - highest - lowest) / 2).toFixed(1) + ')">' + body + "</g></svg>";
  }

  // Dragging turns the chart, and so do the arrow keys while it holds the
  // keyboard; a double click, or Home, returns it to the view it opened in.
  // The angles are kept with the explanation, so drawing the page again at
  // another width leaves the chart as the reader turned it.
  function wireRibbons(scope, rows, symbols, width, valueName) {
    ribbonsRedraw = null;
    var stage = scope.querySelector("[data-ribbons]");
    if (!stage || !rows.length) { return; }
    var e = S.explain, view = e.ribbonView, frame = null, drag = null;
    function draw() {
      frame = null;
      if (stage.isConnected) { stage.innerHTML = ribbons(rows, symbols, width, view, e.index, valueName); }
    }
    function later() { if (!frame) { frame = window.requestAnimationFrame(draw); } }
    function turn(az, el) {
      view.az = ((az + 180) % 360 + 360) % 360 - 180;
      view.el = Math.max(8, Math.min(72, el));
      later();
    }
    stage.addEventListener("pointerdown", function (event) {
      if (event.button) { return; }
      drag = { x: event.clientX, y: event.clientY, az: view.az, el: view.el };
      stage.setPointerCapture(event.pointerId);
      stage.classList.add("is-turning");
    });
    stage.addEventListener("pointermove", function (event) {
      if (drag) { turn(drag.az + (event.clientX - drag.x) * 0.4, drag.el + (event.clientY - drag.y) * 0.25); }
    });
    ["pointerup", "pointercancel"].forEach(function (name) {
      stage.addEventListener(name, function () { drag = null; stage.classList.remove("is-turning"); });
    });
    stage.addEventListener("dblclick", function () { turn(RIBBON_VIEW.az, RIBBON_VIEW.el); });
    stage.addEventListener("keydown", function (event) {
      var step = { ArrowLeft: [-10, 0], ArrowRight: [10, 0], ArrowUp: [0, 6], ArrowDown: [0, -6] }[event.key];
      if (event.key === "Home") { event.preventDefault(); turn(RIBBON_VIEW.az, RIBBON_VIEW.el); }
      else if (step) { event.preventDefault(); turn(view.az + step[0], view.el + step[1]); }
    });
    ribbonsRedraw = later;
  }

  // Every sensor at every test engine, one coloured cell each; unit as for the beeswarm.
  function heatmap(rows, sensors, unit) {
    var after = unit === undefined ? " cycles" : "";
    var cell = Math.max(2, Math.min(18, Math.floor(640 / rows.length)));
    var W = 60 + cell * rows.length, H = 14 * 16 + 10;
    var peak = 0;
    rows.forEach(function (r) { sensors.forEach(function (s) { peak = Math.max(peak, Math.abs(r[s])); }); });
    peak = peak || 1;
    var out = '<div class="pg-heat-wrap"><svg class="pg-heat" viewBox="0 0 ' + W + " " + H + '" width="' + W + '" height="' + H + '">';
    sensors.forEach(function (s, j) {
      // This drawing is not scaled to fit, so its labels are a size on the screen and keep to the 11-pixel floor.
      out += '<text x="54" y="' + (j * 16 + 12) + '" text-anchor="end" class="pg-heat-label pg-sensor-name"' + sensorTip(s) + ">" + esc(s) + "</text>";
      rows.forEach(function (r, i) {
        var v = r[s];
        out += '<rect x="' + (60 + i * cell) + '" y="' + (j * 16) + '" width="' + cell + '" height="15" fill="' + (v >= 0 ? "var(--fig-accent)" : "var(--fig-warn)") +
          '" fill-opacity="' + (Math.abs(v) / peak).toFixed(3) + '"><title>' + esc(s + " · engine " + r.unit + ": " + signedValue(v) + after) + "</title></rect>";
      });
    });
    return out + '</svg></div><div class="pg-faint">Strongest colour: ' + peak.toFixed(2) + after + ".</div>";
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
      var note = label + " · rank " + (k.rank + 1) + " of " + n + ", " + (unit ? signedValue(v) + " " + unit : fixed(v, 2));
      var tip = sensorTip(s, note);
      out += "<g" + tip + ">" + (tip ? "" : "<title>" + esc(s + ": " + note) + "</title>") +
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
      out += "<g" + sensorTip(s) + '><circle cx="' + p[0].toFixed(1) + '" cy="' + p[1].toFixed(1) + '" r="8.4" fill="var(--fig-bg)" stroke="var(--fig-axis)" stroke-width="0.5"/></g>';
      names += ringLabel(p, s);
    });
    return out + names + "</svg><figcaption>key</figcaption></figure>";
  }

  // The paper's Fig. 10, redrawn from Table 12, above this engine at the same
  // graph numbers. ASSUMPTION A-35: graph g is the engine's g-th window, the
  // one ending at cycle g + 29; the paper never says which cycle a graph is.
  // So with every n-th window explained only graphs 1, 1 + n, 1 + 2n ... exist here.
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

  start.follow = follow;
  window.thesisPlayground = start;
})();
