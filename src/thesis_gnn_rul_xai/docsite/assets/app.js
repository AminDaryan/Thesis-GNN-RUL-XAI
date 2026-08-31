/* The documentation website's behaviour: routing, fetching and drawing.
   README.md in this folder explains what each view is for. */

(function () {
  "use strict";

  var state = {};        // what this machine can do, from /api/state
  var catalog = null;    // every document, from /api/catalog
  var order = [];        // the reading order, for the previous/next links
  var results = null;    // runs and assumption entries, fetched once

  var content = document.getElementById("content");
  var sidebar = document.getElementById("sidebar");
  var tocPanel = document.getElementById("toc");

  // ------------------------------------------------------------- utilities

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "html") { node.innerHTML = attrs[key]; }
      else if (key === "text") { node.textContent = attrs[key]; }
      else if (attrs[key] !== null && attrs[key] !== undefined) {
        node.setAttribute(key, attrs[key]);
      }
    });
    (children || []).forEach(function (child) {
      if (child) { node.appendChild(child); }
    });
    return node;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function get(url) {
    return fetch(url, { headers: { Accept: "application/json" } }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) { throw new Error(data.error || response.statusText); }
        return data;
      });
    });
  }

  function toast(message, milliseconds) {
    var box = document.getElementById("toast");
    box.textContent = message;
    box.hidden = false;
    clearTimeout(box._timer);
    box._timer = setTimeout(function () { box.hidden = true; }, milliseconds || 5000);
  }

  function fixed(value, places) {
    return value === null || value === undefined || isNaN(value)
      ? "—" : Number(value).toFixed(places === undefined ? 2 : places);
  }

  // ---------------------------------------------------------------- theme

  function applyTheme(choice) {
    var root = document.documentElement;
    if (choice === "auto") {
      var dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      root.setAttribute("data-theme", dark ? "dark" : "light");
    } else {
      root.setAttribute("data-theme", choice);
    }
    try { localStorage.setItem("docsite-theme", choice); } catch (e) { /* private mode */ }
  }

  function initTheme() {
    var stored = "auto";
    try { stored = localStorage.getItem("docsite-theme") || "auto"; } catch (e) { /* private mode */ }
    applyTheme(stored);
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function () {
      var current = "auto";
      try { current = localStorage.getItem("docsite-theme") || "auto"; } catch (e) { /* ignore */ }
      if (current === "auto") { applyTheme("auto"); }
    });
    document.getElementById("theme-button").addEventListener("click", function () {
      var now = document.documentElement.getAttribute("data-theme");
      applyTheme(now === "dark" ? "light" : "dark");
    });
  }

  // ------------------------------------------------------------ MathJax

  function loadMath() {
    window.MathJax = {
      tex: { inlineMath: [], displayMath: [] },
      options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] },
      startup: { typeset: false }
    };
    var script = document.createElement("script");
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
    script.async = true;
    script.onerror = function () {
      document.documentElement.classList.add("no-math");
      window._mathFailed = true;
    };
    document.head.appendChild(script);
  }

  function typeset(root) {
    if (window._mathFailed || !window.MathJax || !window.MathJax.tex2chtmlPromise) { return; }
    var nodes = root.querySelectorAll(".math-inline, .math-display");
    Array.prototype.forEach.call(nodes, function (node) {
      if (node.dataset.typeset) { return; }
      var display = node.classList.contains("math-display");
      window.MathJax.tex2chtmlPromise(node.textContent, { display: display })
        .then(function (rendered) {
          node.textContent = "";
          node.appendChild(rendered);
          node.dataset.typeset = "1";
          if (window.MathJax.startup && window.MathJax.startup.document) {
            window.MathJax.startup.document.clear();
            window.MathJax.startup.document.updateDocument();
          }
        })
        .catch(function () { document.documentElement.classList.add("no-math"); });
    });
  }

  // ------------------------------------------------------------- sidebar

  // Which sections the reader has folded away. Kept per browser so the menu
  // opens the way they left it; a browser that refuses storage just gets the
  // default, which is what openByDefault decides.
  function foldedSections() {
    try {
      return JSON.parse(localStorage.getItem("docsite-folded") || "{}") || {};
    } catch (e) {
      return {};
    }
  }

  function rememberFold(key, folded) {
    var state = foldedSections();
    state[key] = folded;
    try { localStorage.setItem("docsite-folded", JSON.stringify(state)); } catch (e) { /* private mode */ }
  }

  function buildSidebar() {
    var folded = foldedSections();
    sidebar.innerHTML = "";
    sidebar.appendChild(el("a", {
      class: "nav-link nav-home", href: "#/",
      html: "<strong>Home</strong><span class='nav-blurb-inline'>What this is, and where to start</span>"
    }));

    catalog.sections.forEach(function (section) {
      // Learn and Explore are where a new reader starts, so they open by
      // default. The long reference lists stay folded until asked for.
      var openByDefault = section.key === "learn" || section.key === "explore";
      var open = folded[section.key] === undefined ? openByDefault : !folded[section.key];

      var group = el("section", { class: "nav-section", "data-section": section.key });

      var toggle = el("button", {
        class: "nav-toggle",
        type: "button",
        "aria-expanded": String(open),
        "aria-controls": "nav-body-" + section.key,
        html:
          '<svg class="nav-chevron" viewBox="0 0 12 12" aria-hidden="true">' +
          '<path d="M4 2.5 L8 6 L4 9.5" fill="none" stroke="currentColor" ' +
          'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>' +
          "<span class='nav-toggle-title'>" + escapeHtml(section.title) + "</span>" +
          "<span class='nav-count'>" + section.documents.length + "</span>"
      });

      var body = el("div", {
        class: "nav-body",
        id: "nav-body-" + section.key,
        hidden: open ? null : "hidden"
      });
      body.appendChild(el("div", { class: "nav-blurb", text: section.blurb }));

      section.documents.forEach(function (doc) {
        var meta = doc.kind === "notebook" || doc.kind === "explore"
          ? "" : "<span class='nav-meta'>" + doc.minutes + " min</span>";
        body.appendChild(el("a", {
          class: "nav-link",
          href: routeFor(doc),
          "data-path": doc.path,
          html: meta + escapeHtml(doc.title) +
                (doc.blurb ? "<span class='nav-blurb-inline'>" + escapeHtml(doc.blurb) + "</span>" : "")
        }));
      });

      toggle.addEventListener("click", function () {
        var nowOpen = toggle.getAttribute("aria-expanded") !== "true";
        toggle.setAttribute("aria-expanded", String(nowOpen));
        body.hidden = !nowOpen;
        rememberFold(section.key, !nowOpen);
      });

      group.appendChild(toggle);
      group.appendChild(body);
      sidebar.appendChild(group);
    });
  }

  // A page the reader has navigated to must be visible in the menu, even if
  // they had folded the section it lives in.
  function revealSection(path) {
    var link = sidebar.querySelector('.nav-link[data-path="' + cssEscape(path) + '"]');
    if (!link) { return; }
    var group = link.closest(".nav-section");
    if (!group) { return; }
    var toggle = group.querySelector(".nav-toggle");
    var body = group.querySelector(".nav-body");
    if (toggle && body && body.hidden) {
      toggle.setAttribute("aria-expanded", "true");
      body.hidden = false;
      rememberFold(group.dataset.section, false);
    }
  }

  function cssEscape(value) {
    return window.CSS && CSS.escape ? CSS.escape(value) : value.replace(/["\\]/g, "\\$&");
  }

  function routeFor(doc) {
    if (doc.kind === "notebook") { return "#/notebook/" + doc.path; }
    if (doc.kind === "explore") { return "#/" + doc.path; }
    return "#/doc/" + doc.path;
  }

  function markActive(path) {
    Array.prototype.forEach.call(sidebar.querySelectorAll(".nav-link"), function (link) {
      if (link.dataset.path === path) { link.setAttribute("aria-current", "page"); }
      else { link.removeAttribute("aria-current"); }
    });
    Array.prototype.forEach.call(sidebar.querySelectorAll(".nav-section"), function (group) {
      var holdsCurrent = !!group.querySelector('.nav-link[aria-current="page"]');
      group.classList.toggle("nav-section--current", holdsCurrent);
    });
    if (path) { revealSection(path); }
  }

  // ----------------------------------------------------------------- TOC

  function buildToc(headings) {
    tocPanel.innerHTML = "";
    var useful = (headings || []).filter(function (h) { return h.level >= 2 && h.level <= 4; });
    if (useful.length < 2) { return; }
    tocPanel.appendChild(el("div", { class: "toc-title", text: "On this page" }));
    useful.forEach(function (heading) {
      tocPanel.appendChild(el("a", {
        href: "#" + heading.anchor,
        "data-level": heading.level,
        "data-anchor": heading.anchor,
        text: heading.text
      }));
    });
    spyOnScroll();
  }

  var spyObserver = null;
  function spyOnScroll() {
    if (spyObserver) { spyObserver.disconnect(); }
    var links = {};
    Array.prototype.forEach.call(tocPanel.querySelectorAll("a"), function (link) {
      links[link.dataset.anchor] = link;
    });
    spyObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var link = links[entry.target.id];
        if (!link) { return; }
        if (entry.isIntersecting) {
          Array.prototype.forEach.call(tocPanel.querySelectorAll("a.active"), function (other) {
            other.classList.remove("active");
          });
          link.classList.add("active");
        }
      });
    }, { rootMargin: "-70px 0px -75% 0px", threshold: 0 });
    Object.keys(links).forEach(function (anchor) {
      var target = document.getElementById(anchor);
      if (target) { spyObserver.observe(target); }
    });
  }

  // -------------------------------------------------------------- figures

  var FIGURE_PATTERN = /(?:<p>\s*)?\{\{\s*figure:([a-z0-9-]+)\s*\}\}(?:\s*<\/p>)?/gi;

  function expandFigures(html) {
    return html.replace(FIGURE_PATTERN, function (whole, name) {
      return '<figure class="figure" data-figure="' + name + '">' +
             '<div class="loading">Drawing ' + name + '…</div></figure>';
    });
  }

  function fillFigures(root) {
    Array.prototype.forEach.call(root.querySelectorAll("[data-figure]"), function (holder) {
      if (holder.dataset.filled) { return; }
      holder.dataset.filled = "1";
      fetch("/api/figure?name=" + encodeURIComponent(holder.dataset.figure))
        .then(function (r) { return r.text(); })
        .then(function (svg) {
          var caption = holder.getAttribute("data-caption");
          holder.innerHTML = svg + (caption ? "<figcaption>" + caption + "</figcaption>" : "");
        })
        .catch(function (error) {
          holder.innerHTML = '<div class="figure-notice figure-notice--error">' +
            "<strong>This figure could not be loaded</strong><p>" +
            escapeHtml(error.message) + "</p></div>";
        });
    });
  }

  function wrapTables(root) {
    Array.prototype.forEach.call(root.querySelectorAll(".prose > table"), function (table) {
      var wrapper = el("div", { class: "table-scroll" });
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  }

  // --------------------------------------------------------------- router

  function currentRoute() {
    var hash = window.location.hash.replace(/^#\/?/, "");
    if (!hash) { return { view: "home", argument: "" }; }
    var slash = hash.indexOf("/");
    if (slash < 0) { return { view: hash, argument: "" }; }
    var view = hash.slice(0, slash);
    var rest = hash.slice(slash + 1);
    return { view: view, argument: rest };
  }

  function navigate() {
    var route = currentRoute();
    var argument = route.argument;
    var fragment = "";
    var hashIndex = argument.indexOf("#");
    if (hashIndex >= 0) {
      fragment = argument.slice(hashIndex + 1);
      argument = argument.slice(0, hashIndex);
    }

    window.scrollTo(0, 0);
    tocPanel.innerHTML = "";
    sidebar.classList.remove("open");
    content.innerHTML = '<div class="loading">Loading…</div>';

    var done;
    if (route.view === "home" || !route.view) { done = viewHome(); }
    else if (route.view === "doc") { done = viewDocument(argument); }
    else if (route.view === "notebook") { done = viewNotebook(argument); }
    else if (route.view === "source") { done = viewSource(argument); }
    else if (route.view === "explore") { done = viewExplore(argument); }
    else if (route.view === "entry") { done = viewEntry(argument); }
    else if (route.view === "figures") { done = viewFigures(); }
    else { done = Promise.resolve(showError("No page at " + route.view, "")); }

    markActive(argument);
    Promise.resolve(done).then(function () {
      if (fragment) {
        var target = document.getElementById(fragment);
        if (target) { target.scrollIntoView(); }
      }
      content.focus({ preventScroll: true });
    });
  }

  function showError(title, detail) {
    content.innerHTML = "";
    content.appendChild(el("div", {
      class: "error-box",
      html: "<strong>" + escapeHtml(title) + "</strong>" + escapeHtml(detail || "")
    }));
  }

  function crumbs(parts) {
    return '<div class="crumbs">' + parts.map(function (part, index) {
      return (index ? " › " : "") + (part.href
        ? '<a href="' + part.href + '">' + escapeHtml(part.label) + "</a>"
        : escapeHtml(part.label));
    }).join("") + "</div>";
  }

  // ----------------------------------------------------------- view: home

  function viewHome() {
    var learn = catalog.sections.filter(function (s) { return s.key === "learn"; })[0];
    var chips = [];
    chips.push(chip(state.hasFD001 ? "good" : "warn",
      state.hasFD001 ? "FD001 data present" : "FD001 data missing"));
    chips.push(chip(state.torch ? "good" : "", state.torch ? "PyTorch available" : "PyTorch not installed"));
    chips.push(chip(state.jupyter ? "good" : "", state.jupyter ? "JupyterLab available" : "JupyterLab not installed"));
    chips.push(chip("", "commit " + state.commit));

    var html = '<div class="hero">' +
      "<h1>Predicting how much life is left in a jet engine — and explaining the answer</h1>" +
      "<p>This site is the documentation of a master's thesis that re-implements one " +
      "published method on the NASA C-MAPSS benchmark, and measures how far the paper " +
      "alone can carry it. Everything here is generated from the repository as it " +
      "stands right now.</p>" +
      '<div class="chips">' + chips.join("") + "</div></div>";

    html += '<div class="section-head"><h2>Start here</h2>' +
      "<p>Eight pages that build the whole method from nothing. Each one takes a few minutes.</p></div>";

    if (learn && learn.documents.length) {
      html += '<div class="grid grid--2">' + learn.documents.map(function (doc, index) {
        return '<a class="card" href="#/doc/' + doc.path + '">' +
          '<span class="card-step">Step ' + (index + 1) + " · " + doc.minutes + " min</span>" +
          "<h3>" + escapeHtml(doc.title) + "</h3>" +
          "<p>" + escapeHtml(doc.blurb) + "</p></a>";
      }).join("") + "</div>";
    } else {
      html += '<div class="error-box"><strong>The taught pages are missing</strong>' +
        "docs/learn/ holds no Markdown files. Everything else on this site still works.</div>";
    }

    html += '<div class="section-head"><h2>Or go straight to</h2>' +
      "<p>The notebooks run the whole chain on the real data. The explorers compute their figures now, from this repository.</p></div>";

    html += '<div class="grid grid--3">' +
      card("#/notebook/notebooks/01_preprocessing_fd001.ipynb", "Notebook 01",
           "The raw text files becoming model-ready arrays, with every number shown.") +
      card("#/notebook/notebooks/02_model_fd001.ipynb", "Notebook 02",
           "Building the network, training it, scoring it and explaining it.") +
      card("#/explore/graph", "The sensor graph",
           "All 14 sensors, their 41 edges, and what three layers can reach.") +
      card("#/explore/architecture", "The network",
           "Every layer and its parameter count, built and measured live.") +
      card("#/explore/results", "Stored runs",
           "Every result in reports/, with the commit that produced it.") +
      card("#/explore/assumptions", "Assumptions and findings",
           "The 22 choices and 11 findings, as cards you can filter.") +
      "</div>";

    html += '<div class="section-head"><h2>How to read anything on this site</h2>' +
      "<p>Three kinds of statement appear throughout, and they are never mixed.</p></div>" +
      '<div class="grid grid--3">' +
      '<div class="callout callout--paper"><div class="callout-title">The paper says X</div>' +
      "Verifiable in the PDF, and cited to a table or an equation.</div>" +
      '<div class="callout callout--choice"><div class="callout-title">We chose Y</div>' +
      "A judgement call the paper left open. Every one has a numbered entry.</div>" +
      '<div class="callout callout--measured"><div class="callout-title">We measured Z</div>' +
      "A result, with the command that produced it and the seeds it averages.</div>" +
      "</div>";

    content.innerHTML = html;
    return Promise.resolve();
  }

  function chip(kind, label) {
    return '<span class="chip' + (kind ? " chip--" + kind : "") + '">' + escapeHtml(label) + "</span>";
  }

  function card(href, title, body) {
    return '<a class="card" href="' + href + '"><h3>' + escapeHtml(title) + "</h3><p>" +
      escapeHtml(body) + "</p></a>";
  }

  // ------------------------------------------------------- view: document

  function viewDocument(path) {
    return get("/api/doc?path=" + encodeURIComponent(path)).then(function (doc) {
      var meta = [];
      if (doc.minutes) { meta.push("about " + doc.minutes + " minutes"); }
      if (doc.words) { meta.push(doc.words.toLocaleString("en") + " words"); }
      meta.push("<code>" + escapeHtml(doc.path) + "</code>");

      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: doc.title }]) +
        '<article class="prose">' + expandFigures(doc.html) + "</article>" +
        '<div class="page-meta">' + meta.join(" · ") + "</div>" +
        neighbourLinks(path);

      wrapTables(content);
      fillFigures(content);
      buildToc(doc.headings);
      typeset(content);
      document.title = doc.title + " — Thesis documentation";
    }).catch(function (error) {
      showError("That document could not be shown", error.message);
    });
  }

  function neighbourLinks(path) {
    var index = order.indexOf(path);
    if (index < 0) { return ""; }
    var previous = order[index - 1];
    var next = order[index + 1];
    if (!previous && !next) { return ""; }
    var html = '<nav class="footer-nav">';
    if (previous) {
      html += '<a href="' + routeForPath(previous) + '"><small>Previous</small><span>' +
        escapeHtml(titleOf(previous)) + "</span></a>";
    }
    if (next) {
      html += '<a class="next" href="' + routeForPath(next) + '"><small>Next</small><span>' +
        escapeHtml(titleOf(next)) + "</span></a>";
    }
    return html + "</nav>";
  }

  function documentAt(path) {
    var found = null;
    catalog.sections.forEach(function (section) {
      section.documents.forEach(function (doc) { if (doc.path === path) { found = doc; } });
    });
    return found;
  }

  function titleOf(path) {
    var doc = documentAt(path);
    return doc ? doc.title : path;
  }

  function routeForPath(path) {
    var doc = documentAt(path);
    return doc ? routeFor(doc) : "#/doc/" + path;
  }

  // ------------------------------------------------------- view: notebook

  function viewNotebook(path) {
    return get("/api/notebook?path=" + encodeURIComponent(path)).then(function (notebook) {
      var style = document.getElementById("notebook-style") || el("style", { id: "notebook-style" });
      style.textContent = scopeNotebookCss(notebook.css);
      document.head.appendChild(style);

      var runHere = state.jupyter
        ? '<button class="button" id="open-in-jupyter" type="button">Open this notebook in JupyterLab</button>'
        : '<span class="chip">JupyterLab is not installed — install it with <code>pip install -e .[notebook]</code> to run these cells</span>';

      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: "Notebooks" }, { label: notebook.title }]) +
        '<div class="page-head"><div class="eyebrow">Notebook</div>' +
        "<h1 style='font-family:var(--sans);font-size:31px;margin:0 0 10px;letter-spacing:-0.02em'>" +
        escapeHtml(titleOf(path) || notebook.title) + "</h1>" +
        "<p style='color:var(--ink-soft);margin:0 0 14px'>You are reading the outputs this notebook " +
        "was saved with, produced on the real FD001 files. Nothing is being executed here.</p>" +
        '<div class="chips">' + runHere + "</div></div>" +
        '<div class="notebook-frame">' + notebook.html + "</div>";

      var button = document.getElementById("open-in-jupyter");
      if (button) {
        button.addEventListener("click", function () { launchJupyter(path); });
      }
      buildToc(headingsIn(content));
      typeset(content);
      document.title = notebook.title + " — Thesis documentation";
    }).catch(function (error) {
      showError("That notebook could not be shown", error.message);
    });
  }

  function scopeNotebookCss(css) {
    if (!css) { return ""; }
    // nbconvert's stylesheet styles bare tags, which would leak into the site.
    // Every rule is confined to the frame the notebook is drawn inside.
    return css.replace(/(^|\})\s*([^@{}]+)\{/g, function (whole, brace, selectors) {
      var scoped = selectors.split(",").map(function (selector) {
        selector = selector.trim();
        if (!selector || selector.charAt(0) === "@") { return selector; }
        return ".notebook-frame " + selector;
      }).join(", ");
      return brace + " " + scoped + "{";
    });
  }

  function headingsIn(root) {
    var found = [];
    Array.prototype.forEach.call(root.querySelectorAll("h1, h2, h3"), function (node, index) {
      if (!node.id) { node.id = "nb-" + index; }
      found.push({
        level: Number(node.tagName.charAt(1)),
        text: node.textContent.replace(/¶/g, "").trim(),
        anchor: node.id
      });
    });
    return found;
  }

  // --------------------------------------------------------- view: source

  function viewSource(path) {
    return get("/api/source?path=" + encodeURIComponent(path)).then(function (file) {
      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: "Source" }, { label: file.title }]) +
        '<div class="page-head"><div class="eyebrow">Source file</div>' +
        "<h1 style='font-family:var(--sans);font-size:26px;margin:0'><code>" +
        escapeHtml(file.path) + "</code></h1>" +
        '<div class="page-meta">' + file.lines + " lines</div></div>" +
        '<article class="prose">' + file.html + "</article>";
      document.title = file.title + " — Thesis documentation";
    }).catch(function (error) {
      showError("That file could not be shown", error.message);
    });
  }

  // -------------------------------------------------------- view: explore

  function viewExplore(name) {
    if (name === "graph") { return exploreGraph(); }
    if (name === "architecture") { return exploreArchitecture(); }
    if (name === "results") { return exploreResults(); }
    if (name === "assumptions") { return exploreAssumptions(); }
    showError("No explorer called " + name, "");
    return Promise.resolve();
  }

  function figureBlock(name, caption) {
    return '<figure class="figure" data-figure="' + name + '" data-caption="' +
      escapeHtml(caption) + '"><div class="loading">Drawing…</div></figure>';
  }

  function exploreGraph() {
    return get("/api/graph").then(function (data) {
      var report = data.report;
      var rows = Object.keys(report.degrees).map(function (symbol) {
        return "<tr><td><code>" + symbol + "</code></td><td>" +
          escapeHtml(data.descriptions[symbol] || "") + "</td>" +
          '<td class="num">' + report.degrees[symbol] + "</td>" +
          "<td>" + escapeHtml(data.stations[symbol] || "") + "</td>" +
          '<td class="num">' + data.neighbours[symbol].join(", ") + "</td></tr>";
      }).join("");

      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: "Explore" }, { label: "The sensor graph" }]) +
        '<div class="page-head"><div class="eyebrow">Computed now, from graph/adjacency.py</div>' +
        "<h1 style='font-family:var(--sans);font-size:31px;margin:0 0 8px;letter-spacing:-0.02em'>The sensor graph</h1>" +
        "<p style='color:var(--ink-soft);max-width:74ch;margin:0'>The paper prints this graph twice — " +
        "as Table 3 and again as Fig. 3 — and the two agree. Everything below is measured from the " +
        "transcription in this repository, not copied from the paper's text.</p>" +
        '<div class="chips">' +
          chip("", report.n_nodes + " sensors") +
          chip("", report.n_edges + " edges") +
          chip("", "density " + fixed(report.density, 3)) +
          chip("", "diameter " + report.diameter) +
          chip(report.is_connected ? "good" : "warn", report.is_connected ? "connected" : "disconnected") +
          chip("warn", "articulation point: " + report.articulation_points.join(", ")) +
        "</div></div>" +
        figureBlock("sensor-graph", "<strong>The graph itself.</strong> Node size is the number of " +
          "neighbours; colour is the engine station the sensor sits at. T24 is circled because removing " +
          "it would split the graph in two — the bypass-side sensors reach the core through it and " +
          "nothing else.") +
        figureBlock("message-passing", "<strong>What one graph layer does.</strong> A value at P15 " +
          "spreads one edge per layer. After three layers it has reached everything except W32, which " +
          "is four edges away and can never be mixed with it.") +
        figureBlock("reach", "<strong>Why three layers is a real constraint.</strong> Of the 91 sensor " +
          "pairs, four are further apart than the network is deep.") +
        figureBlock("degrees", "<strong>How central the paper considers each sensor.</strong> " +
          "Read off the transcribed Table 3.") +
        figureBlock("adjacency", "<strong>The same graph as a table.</strong> The normalisation on the " +
          "right is assumption A-21: the paper describes the step but never prints the formula.") +
        '<div class="section-head"><h2>Every sensor, and who it is wired to</h2></div>' +
        '<div class="table-wrap"><table class="data-table"><thead><tr><th>Sensor</th>' +
        "<th>What it measures</th><th>Neighbours</th><th>Station</th><th>Wired to</th>" +
        "</tr></thead><tbody>" + rows + "</tbody></table></div>";

      fillFigures(content);
      buildToc([]);
      document.title = "The sensor graph — Thesis documentation";
    }).catch(function (error) {
      showError("The graph could not be measured", error.message);
    });
  }

  function exploreArchitecture() {
    return get("/api/model").then(function (data) {
      if (data.error) {
        content.innerHTML = '<div class="figure-notice figure-notice--torch"><strong>' +
          "PyTorch is not installed</strong><p>" + escapeHtml(data.error) + "</p></div>";
        return;
      }
      var rows = data.layers.map(function (layer) {
        return "<tr><td><strong>" + escapeHtml(layer.layer) + "</strong></td>" +
          "<td>" + escapeHtml(layer.type) + "</td>" +
          "<td><code>" + escapeHtml(layer["in"]) + "</code></td>" +
          "<td><code>" + escapeHtml(layer.out) + "</code></td>" +
          '<td class="num">' + Number(layer.parameters).toLocaleString("en") + "</td>" +
          "<td>" + escapeHtml(layer.activation) + "</td></tr>";
      }).join("");

      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: "Explore" }, { label: "The network" }]) +
        '<div class="page-head"><div class="eyebrow">Built now, from models/estgnn.py</div>' +
        "<h1 style='font-family:var(--sans);font-size:31px;margin:0 0 8px;letter-spacing:-0.02em'>The network</h1>" +
        "<p style='color:var(--ink-soft);max-width:74ch;margin:0'>This page builds the model in memory " +
        "and reads its layers back. The parameter counts below are what the code really has, not what " +
        "the documentation claims it has.</p>" +
        '<div class="chips">' +
          chip("good", Number(data.parameters).toLocaleString("en") + " trainable parameters") +
          chip("", data.nodes + " nodes") +
          chip("", data.in_features + " feature per node") +
          chip("", "ceiling " + data.rul_max + " cycles") +
        "</div></div>" +
        figureBlock("architecture", "<strong>Every layer of the paper's Table 4.</strong> The " +
          "per-sensor LSTM the paper places in front of the graph layers is deliberately absent " +
          "(assumption A-11), so results from this network compare against the paper's own " +
          "&ldquo;ESTGNN w/o LSTM&rdquo; row and never its headline figures.") +
        '<div class="section-head"><h2>Layer by layer</h2>' +
        "<p>The same table the paper prints, produced by <code>ESTGNN.layer_summary()</code>.</p></div>" +
        '<div class="table-wrap"><table class="data-table"><thead><tr><th>Layer</th><th>Type</th>' +
        "<th>In</th><th>Out</th><th>Parameters</th><th>Activation</th></tr></thead><tbody>" +
        rows + "</tbody></table></div>" +
        '<div class="callout callout--measured"><div class="callout-title">Why the last layer matters</div>' +
        "The network ends in a sigmoid, so its output is a fraction between 0 and 1 that is multiplied " +
        "by the ceiling. It is therefore <em>structurally incapable</em> of predicting more than " +
        data.rul_max + " cycles. That is why the test labels are capped too (assumption A-02), and " +
        "why that cap must be stated whenever these numbers are compared with published ones.</div>";

      fillFigures(content);
      buildToc([]);
      document.title = "The network — Thesis documentation";
    }).catch(function (error) {
      showError("The network could not be built", error.message);
    });
  }

  function exploreResults() {
    return loadResults().then(function (data) {
      var runs = data.runs;
      var rows = runs.map(function (run) {
        return "<tr>" +
          "<td>" + escapeHtml(run.finished.replace("T", " ").replace("Z", "")) + "</td>" +
          "<td><code>" + escapeHtml(run.commit) + "</code>" +
            (run.dirty ? ' <span class="status status--reversed">dirty</span>' : "") + "</td>" +
          "<td>" + escapeHtml(run.graph) + " / " + escapeHtml(run.adjacency_normalization) + "</td>" +
          '<td class="num">' + run.seeds + "</td>" +
          "<td>" + escapeHtml(run.split) + "</td>" +
          '<td class="num">' + fixed(run.mean.rmse) +
            (run.std.rmse === null ? "" : ' <span class="sd">± ' + fixed(run.std.rmse) + "</span>") + "</td>" +
          '<td class="num">' + fixed(run.mean.score, 0) + "</td>" +
          '<td class="num">' + fixed(run.mean.r2, 3) + "</td>" +
          '<td class="num">' + fixed(run.mean.asue) + "</td>" +
          "</tr>";
      }).join("");

      var dirty = runs.filter(function (r) { return r.dirty; }).length;
      var multi = runs.filter(function (r) { return r.seeds >= 5; }).length;

      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: "Explore" }, { label: "Stored runs" }]) +
        '<div class="page-head"><div class="eyebrow">Read from reports/</div>' +
        "<h1 style='font-family:var(--sans);font-size:31px;margin:0 0 8px;letter-spacing:-0.02em'>Stored runs</h1>" +
        "<p style='color:var(--ink-soft);max-width:74ch;margin:0'>Every run this repository has kept, " +
        "newest first, each with the commit that produced it. A run whose commit ends in " +
        "<code>-dirty</code> was made from a working tree with uncommitted changes, so it cannot be " +
        "reproduced exactly from any committed state.</p>" +
        '<div class="chips">' +
          chip("", runs.length + " stored runs") +
          chip("good", multi + " over five or more seeds") +
          chip(dirty ? "warn" : "good", dirty + " from a dirty tree") +
        "</div></div>" +
        '<div class="callout callout--choice"><div class="callout-title">One seed is not a result</div>' +
        "Anything quoted as a result of this project is a mean over at least five seeds, with its " +
        "standard deviation. The seed varies the train/validation split as well as the initialisation " +
        "and the batch order, so the spread is what a genuine repetition would show. Rows below with a " +
        "seed count under five are diagnostic runs, not results.</div>" +
        '<div class="table-wrap"><table class="data-table"><thead><tr>' +
        "<th>Finished (UTC)</th><th>Commit</th><th>Graph / normalisation</th><th>Seeds</th>" +
        "<th>Split</th><th>RMSE</th><th>NASA score</th><th>R²</th><th>ASUE</th>" +
        "</tr></thead><tbody>" + rows + "</tbody></table></div>";

      buildToc([]);
      document.title = "Stored runs — Thesis documentation";
    }).catch(function (error) {
      showError("The stored runs could not be read", error.message);
    });
  }

  function exploreAssumptions() {
    return loadResults().then(function (data) {
      var entries = data.entries;
      var assumptions = entries.filter(function (e) { return e.kind === "assumption"; });
      var findings = entries.filter(function (e) { return e.kind === "finding"; });

      function cardFor(entry) {
        return '<div class="entry-card" data-kind="' + entry.kind + '" data-status="' + entry.status + '">' +
          "<header><span class='entry-id'>" + entry.identifier + "</span>" +
          "<h3>" + escapeHtml(entry.title) + "</h3>" +
          '<span class="status status--' + entry.status + '">' + entry.status + "</span></header>" +
          "<p>" + escapeHtml(entry.summary) + "</p>" +
          (entry.where ? '<div class="where">' + escapeHtml(entry.where) + "</div>" : "") +
          "<p style='margin:8px 0 0'><a href=\"#/doc/docs/ASSUMPTIONS.md#" + entry.anchor +
          "\">Read the full entry →</a></p></div>";
      }

      content.innerHTML =
        crumbs([{ label: "Home", href: "#/" }, { label: "Explore" }, { label: "Assumptions and findings" }]) +
        '<div class="page-head"><div class="eyebrow">Parsed from docs/ASSUMPTIONS.md</div>' +
        "<h1 style='font-family:var(--sans);font-size:31px;margin:0 0 8px;letter-spacing:-0.02em'>Assumptions and findings</h1>" +
        "<p style='color:var(--ink-soft);max-width:74ch;margin:0'>An <strong>assumption</strong> is a " +
        "choice this project had to make because the paper does not specify it. A <strong>finding</strong> " +
        "is something established about the paper from the data. They are kept apart on purpose: one is " +
        "ours, the other is a fact about the source.</p>" +
        '<div class="chips">' +
          chip("", assumptions.length + " assumptions") +
          chip("", findings.length + " findings") +
          chip("warn", entries.filter(function (e) { return e.status !== "active"; }).length +
               " withdrawn, reversed or superseded") +
          verificationChip(data.verification) +
        "</div></div>" +
        '<div class="section-head"><h2>Assumptions — choices we made</h2>' +
        "<p>Each appears in the code as a comment beginning <code>ASSUMPTION</code>, at the point where it is used.</p></div>" +
        '<div class="grid grid--2">' + assumptions.map(cardFor).join("") + "</div>" +
        '<div class="section-head"><h2>Findings — facts about the paper</h2>' +
        "<p>Established from the data, not chosen. Several of these correct an earlier claim of this project's own.</p></div>" +
        '<div class="grid grid--2">' + findings.map(cardFor).join("") + "</div>";

      buildToc([]);
      document.title = "Assumptions and findings — Thesis documentation";
    }).catch(function (error) {
      showError("The assumptions log could not be read", error.message);
    });
  }

  // scripts/verify_claims.py recomputes the figures the log quotes and fails
  // if one disagrees. Its last result is how far the log can be trusted, so
  // it belongs beside the entries rather than buried in reports/.
  function verificationChip(checked) {
    if (!checked) {
      return chip("warn", "verify_claims.py has not been run");
    }
    var when = checked.finished.slice(0, 8);
    return chip(
      checked.failed ? "warn" : "good",
      checked.failed
        ? checked.failed + " of " + checked.checked + " claims disagree (" + when + ")"
        : checked.checked + " claims re-checked, none disagree (" + when + ")"
    );
  }

  function loadResults() {
    if (results) { return Promise.resolve(results); }
    return get("/api/results").then(function (data) { results = data; return data; });
  }

  function viewEntry(identifier) {
    return loadResults().then(function (data) {
      var entry = data.entries.filter(function (e) { return e.identifier === identifier; })[0];
      if (!entry) {
        showError("No entry called " + identifier, "The log holds " + data.entries.length + " entries.");
        return;
      }
      window.location.replace("#/doc/docs/ASSUMPTIONS.md#" + entry.anchor);
    });
  }

  function viewFigures() {
    content.innerHTML =
      '<div class="page-head"><div class="eyebrow">Every figure</div>' +
      "<h1 style='font-family:var(--sans);font-size:31px;margin:0'>Figures</h1>" +
      "<p style='color:var(--ink-soft)'>Each is drawn from this repository when the page loads.</p></div>" +
      (state.figures || []).map(function (name) {
        return "<div class='section-head'><h2>" + escapeHtml(name) + "</h2></div>" +
          figureBlock(name, "");
      }).join("");
    fillFigures(content);
    return Promise.resolve();
  }

  // --------------------------------------------------------------- search

  function initSearch() {
    var input = document.getElementById("search-input");
    var panel = document.getElementById("search-results");
    var timer = null;

    function close() { panel.hidden = true; }

    input.addEventListener("input", function () {
      clearTimeout(timer);
      var query = input.value.trim();
      if (query.length < 2) { close(); return; }
      timer = setTimeout(function () {
        get("/api/search?q=" + encodeURIComponent(query)).then(function (data) {
          panel.innerHTML = "";
          if (!data.hits.length) {
            panel.appendChild(el("div", { class: "search-empty", text: "Nothing matches “" + query + "”." }));
          } else {
            var byDocument = {};
            data.hits.forEach(function (hit) {
              (byDocument[hit.path] = byDocument[hit.path] || []).push(hit);
            });
            Object.keys(byDocument).slice(0, 12).forEach(function (path) {
              byDocument[path].slice(0, 3).forEach(function (hit) {
                panel.appendChild(el("a", {
                  class: "search-hit",
                  href: "#/doc/" + hit.path,
                  html: "<span class='where'>" + escapeHtml(hit.title) + " · line " + hit.line +
                        "</span><span class='line'>" + mark(hit.context, query) + "</span>"
                }));
              });
            });
            panel.appendChild(el("div", {
              class: "search-empty",
              text: data.hits.length + " matches in " + Object.keys(byDocument).length + " documents."
            }));
          }
          panel.hidden = false;
        }).catch(function () { close(); });
      }, 180);
    });

    input.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { input.value = ""; close(); input.blur(); }
    });
    document.addEventListener("click", function (event) {
      if (!panel.contains(event.target) && event.target !== input) { close(); }
    });
    document.addEventListener("keydown", function (event) {
      if ((event.key === "/" || (event.key === "k" && (event.metaKey || event.ctrlKey)))
          && document.activeElement !== input) {
        event.preventDefault();
        input.focus();
        input.select();
      }
    });
  }

  function mark(text, query) {
    var safe = escapeHtml(text);
    var needle = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return safe.replace(new RegExp("(" + needle + ")", "ig"), "<mark>$1</mark>");
  }

  // -------------------------------------------------------------- Jupyter

  function initJupyter() {
    var button = document.getElementById("jupyter-button");
    var dot = document.getElementById("jupyter-dot");

    function paint(status) {
      if (status.running) { dot.dataset.state = "running"; button.title = "JupyterLab is running at " + status.url; }
      else if (status.installed) { dot.dataset.state = "ready"; button.title = "Start JupyterLab on this project"; }
      else { dot.dataset.state = "absent"; button.title = "JupyterLab is not installed"; }
    }

    get("/api/jupyter").then(paint).catch(function () { dot.dataset.state = "absent"; });
    button.addEventListener("click", function () { launchJupyter(null); });
  }

  function launchJupyter(notebookPath) {
    var dot = document.getElementById("jupyter-dot");
    var button = document.getElementById("jupyter-button");

    get("/api/jupyter").then(function (status) {
      if (status.running) { openLab(status.url, notebookPath); return; }
      if (!status.installed) {
        toast("JupyterLab is not installed in this environment. Install it with " +
              "pip install -e .[notebook], then start this site again. The notebooks " +
              "can still be read here in full.", 10000);
        return;
      }
      dot.dataset.state = "busy";
      button.disabled = true;
      toast("Starting JupyterLab. This takes a few seconds…", 20000);
      fetch("/api/jupyter", { method: "POST" })
        .then(function (response) { return response.json().then(function (d) {
          if (!response.ok) { throw new Error(d.error || response.statusText); }
          return d;
        }); })
        .then(function (started) {
          dot.dataset.state = "running";
          button.disabled = false;
          toast("JupyterLab is running.", 4000);
          openLab(started.url, notebookPath);
        })
        .catch(function (error) {
          dot.dataset.state = "ready";
          button.disabled = false;
          toast("JupyterLab did not start: " + error.message, 12000);
        });
    });
  }

  function openLab(url, notebookPath) {
    var target = url;
    if (notebookPath) {
      var separator = url.indexOf("?") >= 0 ? url.split("?") : [url, ""];
      target = separator[0].replace(/\/$/, "") + "/lab/tree/" + notebookPath +
               (separator[1] ? "?" + separator[1] : "");
    }
    window.open(target, "jupyterlab");
  }

  // ----------------------------------------------------------------- boot

  function readingOrder() {
    var paths = [];
    catalog.sections.forEach(function (section) {
      section.documents.forEach(function (doc) {
        if (doc.kind !== "explore") { paths.push(doc.path); }
      });
    });
    return paths;
  }

  initTheme();
  loadMath();

  Promise.all([get("/api/state"), get("/api/catalog")]).then(function (loaded) {
    state = loaded[0];
    catalog = loaded[1];
    order = readingOrder();
    buildSidebar();
    initSearch();
    initJupyter();
    document.getElementById("menu-button").addEventListener("click", function () {
      sidebar.classList.toggle("open");
    });
    window.addEventListener("hashchange", navigate);
    navigate();
  }).catch(function (error) {
    content.innerHTML = '<div class="error-box"><strong>The site could not start</strong>' +
      escapeHtml(error.message) + "</div>";
  });
})();
