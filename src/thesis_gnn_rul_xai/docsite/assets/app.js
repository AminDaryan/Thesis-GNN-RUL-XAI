/* The documentation website's behaviour: routing, fetching and drawing.
   README.md in this folder explains what each view is for. */

(function () {
  "use strict";

  var state = {};        // what this machine can do, from /api/state
  var catalog = null;    // every document, from /api/catalog
  var log = null;        // the assumptions log's entries and last re-check, fetched once

  var content = document.getElementById("content");
  var sidebar = document.getElementById("sidebar");
  var tocPanel = document.getElementById("toc");

  // The site's own name, as shell.html's <title> gives it, for a tab with no
  // page of its own to name.
  var BRAND = document.title;

  // The log whose page carries the result of the last re-check of its numbers.
  var LOG_PATH = "docs/ASSUMPTIONS.md";

  // Addresses of pages that were removed, and the page that now says the same
  // thing. A bookmark or an old link lands there instead of on an error.
  var MOVED = {
    "explore/graph": "#/doc/docs/learn/02_graphs.md",
    "figures": "#/doc/docs/learn/02_graphs.md",
    "explore/architecture": "#/doc/docs/learn/05_the_model.md",
    "explore/results": "#/explore/playground",
    "explore/assumptions": "#/doc/docs/ASSUMPTIONS.md"
  };

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

  // A whole number with a space between thousands, as the figures print it.
  function spaced(value) {
    return Number(value).toLocaleString("en").replace(/,/g, " ");
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

  function hideToast() {
    var box = document.getElementById("toast");
    clearTimeout(box._timer);
    box.hidden = true;
  }

  function fixed(value, places) {
    return value === null || value === undefined || isNaN(value)
      ? "—" : Number(value).toFixed(places === undefined ? 2 : places);
  }

  // The browser tab names the page alone, in the words the menu uses for it.
  function setTitle(name) {
    document.title = name || BRAND;
  }

  function chip(kind, label) {
    return '<span class="chip' + (kind ? " chip--" + kind : "") + '">' + escapeHtml(label) + "</span>";
  }

  // ----------------------------------------------------------- navigation

  // The navigation is open or closed, recorded on <html> as data-nav. On a
  // wide screen the choice is remembered; on a narrow one it is a drawer that
  // starts closed and closes again whenever a page is chosen. Narrow means
  // below 1200: the menu (272), the page's margins (80) and a full line of
  // text (848) need 1200 side by side. app.css and shell.html use the same width.
  var narrowScreen = window.matchMedia("(max-width: 1199px)");

  function navIsOpen() {
    return document.documentElement.getAttribute("data-nav") !== "closed";
  }

  function setNav(open) {
    document.documentElement.setAttribute("data-nav", open ? "open" : "closed");
    var button = document.getElementById("nav-button");
    button.setAttribute("aria-expanded", String(open));
    button.title = open ? "Hide the navigation" : "Show the navigation";
    if (!narrowScreen.matches) {
      try { localStorage.setItem("docsite-nav", open ? "open" : "closed"); } catch (e) { /* private mode */ }
    }
  }

  function closeDrawer() {
    if (narrowScreen.matches && navIsOpen()) { setNav(false); }
  }

  function initNav() {
    var remembered = "open";
    try { remembered = localStorage.getItem("docsite-nav") || "open"; } catch (e) { /* private mode */ }
    setNav(narrowScreen.matches ? false : remembered !== "closed");

    document.getElementById("nav-button").addEventListener("click", function () {
      setNav(!navIsOpen());
    });
    document.getElementById("scrim").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { closeDrawer(); }
    });
    // Crossing the width where the drawer takes over, or back, re-applies the
    // right state: a drawer never starts open, a wide screen keeps its choice.
    narrowScreen.addEventListener("change", function () {
      var kept = "open";
      try { kept = localStorage.getItem("docsite-nav") || "open"; } catch (e) { /* ignore */ }
      setNav(narrowScreen.matches ? false : kept !== "closed");
    });
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
      // The page may be drawn before MathJax has loaded; once it is ready it
      // typesets whatever the page holds then, so a first visit never keeps
      // its raw formulas.
      startup: {
        typeset: false,
        ready: function () {
          window.MathJax.startup.defaultReady();
          window.MathJax.startup.promise.then(function () {
            document.documentElement.classList.remove("no-math");
            typeset(content);
          });
        }
      }
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
    if (window._mathFailed || !window.MathJax || !window.MathJax.tex2chtmlPromise) { return Promise.resolve(); }
    var nodes = root.querySelectorAll(".math-inline, .math-display");
    return Promise.all(Array.prototype.map.call(nodes, function (node) {
      if (node.dataset.typeset) { return; }
      var display = node.classList.contains("math-display");
      return window.MathJax.tex2chtmlPromise(node.textContent, { display: display })
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
    }));
  }

  // ------------------------------------------------------------- sidebar

  // Which sections the reader has folded away. Kept per browser so the menu
  // opens the way they left it; a browser that refuses storage just gets the
  // default, which is every section folded.
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
    // The playground is what the site is for, so it is the one link above the
    // sections, and every section starts folded until the reader opens it.
    if (catalog.primary) {
      sidebar.appendChild(el("a", {
        class: "nav-link nav-primary", href: routeFor(catalog.primary),
        "data-path": catalog.primary.path,
        text: catalog.primary.title
      }));
    }

    catalog.sections.forEach(function (section) {
      var open = folded[section.key] === undefined ? false : !folded[section.key];

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
          "<span class='nav-toggle-title'>" + escapeHtml(section.title) + "</span>"
      });

      var body = el("div", {
        class: "nav-body",
        id: "nav-body-" + section.key,
        hidden: open ? null : "hidden"
      });

      section.documents.forEach(function (doc) {
        body.appendChild(el("a", {
          class: "nav-link" + (doc.depth ? " nav-link--sub" : ""),
          href: routeFor(doc),
          "data-path": doc.path,
          text: doc.title
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

  // The menu's name for a page, or nothing when the page is not in the menu.
  function menuName(path) {
    if (catalog.primary && catalog.primary.path === path) { return catalog.primary.title; }
    var found = null;
    catalog.sections.forEach(function (section) {
      section.documents.forEach(function (doc) { if (doc.path === path) { found = doc.title; } });
    });
    return found;
  }

  function markActive(path) {
    Array.prototype.forEach.call(sidebar.querySelectorAll(".nav-link"), function (link) {
      if (link.dataset.path === path) { link.setAttribute("aria-current", "page"); }
      else { link.removeAttribute("aria-current"); }
    });
    if (path) { revealSection(path); }
  }

  // ----------------------------------------------------------------- TOC

  // The page's own address, without the section a link may add after it.
  function pageRoute() {
    return window.location.hash.replace(/^(#\/[^#]*).*$/, "$1");
  }

  // Each link names the page and the section, "#/doc/…md#section", so it can
  // be copied and still lead there. Where the list beside the page does not
  // fit, the same links sit folded at the top of the page instead.
  function buildToc(headings) {
    tocPanel.innerHTML = "";
    var old = content.querySelector(".toc-inline");
    if (old) { old.parentNode.removeChild(old); }
    var useful = (headings || []).filter(function (h) {
      return h.level >= 2 && h.level <= 3 && h.text !== "Process";
    });
    if (useful.length < 2) { return; }
    tocPanel.appendChild(el("div", { class: "toc-title", text: "On this page" }));
    var folded = el("details", { class: "toc-inline" });
    folded.appendChild(el("summary", { text: "On this page" }));
    useful.forEach(function (heading) {
      var stage = heading.level === 2 && heading.text.match(/^(\d+) — (.+)$/);
      var kind = stage ? "stage" : "section";
      var link = {
        href: pageRoute() + "#" + heading.anchor,
        "data-level": heading.level,
        "data-anchor": heading.anchor,
        "data-kind": kind
      };
      link.text = stage ? stage[1] + ". " + stage[2] : heading.text;
      tocPanel.appendChild(el("a", link));
      folded.appendChild(el("a", link));
    });
    var title = content.querySelector(".prose > h1, .notebook-frame h1");
    if (title) { title.parentNode.insertBefore(folded, title.nextSibling); }
    spyOnScroll();
  }

  // A link to a section of the page on show scrolls to it and changes the
  // address in place, rather than loading the page again; a link written as
  // "#section" alone would otherwise be read as the address of a page that
  // does not exist.
  function initSectionLinks() {
    document.addEventListener("click", function (event) {
      var link = event.target.closest && event.target.closest('a[href^="#"]');
      if (!link || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey) { return; }
      var href = link.getAttribute("href");
      var cut = href.indexOf("#", 1);
      var route = href.charAt(1) === "/" ? (cut < 0 ? href : href.slice(0, cut)) : "";
      var anchor = href.charAt(1) === "/" ? (cut < 0 ? "" : href.slice(cut + 1)) : href.slice(1);
      if (!anchor || (route && route !== pageRoute())) { return; }
      var target = findSection(anchor);
      if (!target) { return; }
      event.preventDefault();
      history.replaceState(null, "", pageRoute() + "#" + anchor);
      // The folded list closes first: closed after, it would shorten the page
      // above the heading once the scroll had already been measured.
      var folded = link.closest(".toc-inline");
      if (folded) { folded.open = false; }
      target.scrollIntoView();
    });
  }

  // A notebook's heading ids hold their "%E2%80%94" as written, a document's
  // hold the "²" itself; the id is tried as it stands, then decoded.
  function findSection(anchor) {
    var target = document.getElementById(anchor);
    if (!target) {
      try { target = document.getElementById(decodeURIComponent(anchor)); } catch (e) { target = null; }
    }
    return target;
  }

  // A heading's own # names its page as well, so it can be copied or opened
  // in a new tab; "#section" alone is the address of no page.
  function fullSectionLinks(root) {
    var page = pageRoute();
    Array.prototype.forEach.call(root.querySelectorAll('a[href^="#"]:not([href^="#/"])'), function (link) {
      link.setAttribute("href", page + link.getAttribute("href"));
    });
  }

  var spyObserver = null;
  var desktopFrame = window.matchMedia("(min-width: 1200px)");
  desktopFrame.addEventListener("change", spyOnScroll);
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
          // The list scrolls on its own, so the section being read is kept in
          // its view; the page itself is never moved by this.
          var top = link.offsetTop;
          if (top < tocPanel.scrollTop + 40 || top > tocPanel.scrollTop + tocPanel.clientHeight - 60) {
            tocPanel.scrollTo({ top: top - tocPanel.clientHeight / 3 });
          }
        }
      });
    }, { root: desktopFrame.matches ? content : null,
      rootMargin: desktopFrame.matches ? "-16px 0px -75% 0px" : "-70px 0px -75% 0px", threshold: 0 });
    Object.keys(links).forEach(function (anchor) {
      var target = document.getElementById(anchor);
      if (target) { spyObserver.observe(target); }
    });
  }

  // -------------------------------------------------------------- figures

  // Only a placeholder that is a paragraph of its own is a figure; one written
  // inside code, to show the syntax, stays text.
  var FIGURE_PATTERN = /<p>\s*\{\{\s*figure:([a-z0-9-]+)\s*\}\}\s*<\/p>/gi;

  function expandFigures(html) {
    return html.replace(FIGURE_PATTERN, function (whole, name) {
      return '<figure class="figure" data-figure="' + name + '">' +
             '<div class="loading">Drawing the figure…</div></figure>';
    });
  }

  function fillFigures(root) {
    return Promise.all(Array.prototype.map.call(root.querySelectorAll("[data-figure]"), function (holder) {
      if (holder.dataset.filled) { return; }
      holder.dataset.filled = "1";
      if (holder.dataset.figure === "engine" && window.ThesisEngineView) {
        window.ThesisEngineView.mountGuide(holder);
        return;
      }
      return fetch("/api/figure?name=" + encodeURIComponent(holder.dataset.figure))
        .then(function (r) { return r.text(); })
        .then(function (svg) {
          holder.innerHTML = svg;
        })
        .catch(function (error) {
          holder.innerHTML = '<div class="figure-notice figure-notice--error">' +
            "<strong>This figure could not be loaded</strong><p>" +
            escapeHtml(error.message) + "</p></div>";
        });
    }));
  }

  // Every table gets a box of its own that scrolls sideways when the table is
  // wider than the page. A name in code inside a cell may break after a dot,
  // an underscore, a slash, a colon, a bracket, a comma or an equals sign,
  // rather than between any two letters, so a column is not squeezed into
  // fragments of a name.
  function wrapTables(root) {
    Array.prototype.forEach.call(root.querySelectorAll(".prose > table, .notebook-frame .text_cell_render > table"), function (table) {
      var wrapper = el("div", { class: "table-scroll" });
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
    Array.prototype.forEach.call(root.querySelectorAll(".table-scroll td code"), breakAtPunctuation);
  }

  function breakAtPunctuation(node) {
    Array.prototype.slice.call(node.childNodes).forEach(function (child) {
      if (child.nodeType !== 3 || !/[._\/:(,=]/.test(child.nodeValue)) { return; }
      // Split at each mark and put the mark back on the piece before it; a
      // lookbehind would do this in one step, but Safari before 16.4 cannot
      // read one, and then reads none of this file.
      var raw = child.nodeValue.split(/([._\/:(,=])/), parts = [];
      for (var k = 0; k < raw.length; k += 2) {
        var piece = raw[k] + (raw[k + 1] || "");
        if (piece) { parts.push(piece); }
      }
      var frag = document.createDocumentFragment();
      parts.forEach(function (part, i) {
        frag.appendChild(document.createTextNode(part));
        if (i < parts.length - 1) { frag.appendChild(document.createElement("wbr")); }
      });
      node.replaceChild(frag, child);
    });
  }

  // A file path may break only after a folder's slash.
  function breakablePath(path) {
    return escapeHtml(path).split("/").join("/<wbr>");
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

  var XAI_MOVED_SECTIONS = {
    "dividing-credit-fairly": "10_dynamic_shap.md#dividing-credit-fairly",
    "shap-the-same-idea-applied-to-a-model": "10_dynamic_shap.md#3--average-what-each-sensor-adds",
    "exact-not-sampled": "10_dynamic_shap.md#3--average-what-each-sensor-adds",
    "what-removing-a-sensor-has-to-mean": "10_dynamic_shap.md#1--choose-a-reference-window",
    "dynamic-the-same-calculation-at-every-cycle": "10_dynamic_shap.md#5--repeat-along-an-engines-history",
    "what-shap-says-about-the-two-value-sensor": "10_dynamic_shap.md#what-an-explanation-does-and-does-not-establish",
    "walks-the-routes-a-prediction-took-through-the-graph": "11_gnn_lrp.md#1--follow-the-graph-calculation"
  };

  var DATA_MOVED_SECTIONS = {
    "03_explainability.md#read-an-explanation-in-context": "03_explainability.md#2--read-an-explanation-in-context",
    "10_dynamic_shap.md#shap-the-same-idea-applied-to-a-model": "10_dynamic_shap.md#3--average-what-each-sensor-adds",
    "10_dynamic_shap.md#exact-not-sampled": "10_dynamic_shap.md#3--average-what-each-sensor-adds",
    "10_dynamic_shap.md#what-removing-a-sensor-has-to-mean": "10_dynamic_shap.md#1--choose-a-reference-window",
    "10_dynamic_shap.md#dynamic-the-same-calculation-at-every-cycle": "10_dynamic_shap.md#5--repeat-along-an-engines-history",
    "10_dynamic_shap.md#what-shap-says-about-the-two-value-sensor": "10_dynamic_shap.md#what-an-explanation-does-and-does-not-establish",
    "11_gnn_lrp.md#walks-the-routes-a-prediction-took-through-the-graph": "11_gnn_lrp.md#1--follow-the-graph-calculation",
    "11_gnn_lrp.md#the-authors-check-adding-and-removing-sensors": "11_gnn_lrp.md#4--check-by-adding-and-removing-sensors",
    "11_gnn_lrp.md#what-it-does-not-show": "11_gnn_lrp.md#what-the-explanation-does-and-does-not-show",
    "05_the_model.md#output": "05_the_model.md#6--predicted-life-fraction",
    "05_the_model.md#predicted-life-fraction": "05_the_model.md#6--predicted-life-fraction",
    "05_the_model.md#graph-layer-1": "05_the_model.md#1--graph-layer-1",
    "05_the_model.md#graph-layer-2": "05_the_model.md#2--graph-layer-2",
    "05_the_model.md#graph-layer-3": "05_the_model.md#3--graph-layer-3",
    "05_the_model.md#flatten": "05_the_model.md#4--flatten",
    "05_the_model.md#fully-connected": "05_the_model.md#5--fully-connected",
    "05_the_model.md#scale-to-cycles": "05_the_model.md#7--scale-to-cycles",
    "05_the_model.md#run-the-examples-on-this-page": "05_the_model.md#example-setup",
    "05_the_model.md#before-table-4-the-stage-the-paper-does-not-tabulate": "09_temporal.md#choose-a-route",
    "05_the_model.md#the-six-layers-and-how-many-numbers-each-learns": "05_the_model.md#shapes-and-the-default-architecture",
    "05_the_model.md#the-last-layer-and-what-it-makes-impossible": "05_the_model.md#6--predicted-life-fraction",
    "05_the_model.md#why-the-shapes-are-checked-so-aggressively": "05_the_model.md#shapes-and-the-default-architecture",
    "05_the_model.md#the-stage-left-out-by-default": "09_temporal.md#the-stage-left-out-by-default",
    "09_temporal.md#what-enters-and-leaves": "09_temporal.md#choose-a-route",
    "09_temporal.md#load-the-tutorial-example": "09_temporal.md#example-setup",
    "09_temporal.md#split-by-sensor": "09_temporal.md#1--split-by-sensor",
    "09_temporal.md#keep-the-last-hidden-state": "09_temporal.md#3--keep-the-last-hidden-state",
    "09_temporal.md#turn-into-numbers-per-sensor": "09_temporal.md#4--turn-into-numbers-per-sensor",
    "09_temporal.md#last-reading": "09_temporal.md#keep-the-last-reading",
    "09_temporal.md#average": "09_temporal.md#average-the-window",
    "02_graphs.md#sensor-graph": "02_graphs.md#1--sensor-graph",
    "02_graphs.md#rescale-the-graph": "02_graphs.md#2--rescale-the-graph",
    "02_graphs.md#a-graph-is-points-and-connections": "02_graphs.md#1--sensor-graph",
    "02_graphs.md#the-adjacency-matrix-is-the-graph-written-as-a-table": "02_graphs.md#1--sensor-graph",
    "02_graphs.md#the-graph-this-project-uses": "02_graphs.md#1--sensor-graph",
    "02_graphs.md#what-a-graph-neural-network-does-message-passing": "02_graphs.md#how-information-travels-through-layers",
    "02_graphs.md#one-graph-layer-as-a-formula": "05_the_model.md#1--graph-layer-1",
    "02_graphs.md#why-a-graph-at-all": "02_graphs.md#why-use-a-graph",
    "04_the_pipeline.md#4--cap-the-labels": "04_the_pipeline.md#1--cap-the-labels",
    "04_the_pipeline.md#5--smooth-each-sensor": "04_the_pipeline.md#2--smooth-each-sensor",
    "04_the_pipeline.md#6--hold-engines-back": "04_the_pipeline.md#3--hold-engines-back",
    "04_the_pipeline.md#7--put-readings-on-a-comparable-scale": "04_the_pipeline.md#4--put-readings-on-a-comparable-scale",
    "04_the_pipeline.md#8--cut-into-windows": "04_the_pipeline.md#5--cut-into-windows",
    "04_the_pipeline.md#9--assign-windows-to-their-sets": "04_the_pipeline.md#prepared-windows",
    "04_the_pipeline.md#6--assign-windows-to-their-sets": "04_the_pipeline.md#prepared-windows",
    "01_the_problem.md#only-14-of-the-21-sensors": "01_the_problem.md#3--select-14-sensor-columns-for-model-inputs",
    "01_the_problem.md#select-14-sensor-columns-for-model-inputs": "01_the_problem.md#3--select-14-sensor-columns-for-model-inputs",
    "01_the_problem.md#what-the-model-is-asked-to-predict": "04_the_pipeline.md#1--cap-the-labels",
    "04_the_pipeline.md#1--load": "01_the_problem.md#1--read-the-engine-histories",
    "04_the_pipeline.md#2--diagnose": "01_the_problem.md#2--check-which-sensors-vary",
    "04_the_pipeline.md#3--select": "01_the_problem.md#3--select-14-sensor-columns-for-model-inputs",
    "04_the_pipeline.md#cap-the-labels": "04_the_pipeline.md#1--cap-the-labels",
    "04_the_pipeline.md#5--smooth-leakage-critical": "04_the_pipeline.md#2--smooth-each-sensor",
    "04_the_pipeline.md#smooth-each-sensor": "04_the_pipeline.md#2--smooth-each-sensor",
    "04_the_pipeline.md#6--split-leakage-critical": "04_the_pipeline.md#3--hold-engines-back",
    "04_the_pipeline.md#hold-engines-back": "04_the_pipeline.md#3--hold-engines-back",
    "04_the_pipeline.md#7--rescale-leakage-critical": "04_the_pipeline.md#rescale-each-sensor",
    "04_the_pipeline.md#8--window": "04_the_pipeline.md#5--cut-into-windows",
    "04_the_pipeline.md#cut-into-windows": "04_the_pipeline.md#5--cut-into-windows",
    "04_the_pipeline.md#what-the-model-receives-one-number-per-sensor-not-thirty": "09_temporal.md#choose-a-route"
  };

  function navigate() {
    var requestedHash = window.location.hash;
    var route = currentRoute();
    var argument = route.argument;
    var fragment = "";
    var hashIndex = argument.indexOf("#");
    if (hashIndex >= 0) {
      fragment = argument.slice(hashIndex + 1);
      argument = argument.slice(0, hashIndex);
    }

    if (route.view === "doc" && argument === "docs/learn/03_explainability.md" &&
        XAI_MOVED_SECTIONS[fragment]) {
      window.location.replace("#/doc/docs/learn/" + XAI_MOVED_SECTIONS[fragment]);
      return;
    }

    var mergedSection = DATA_MOVED_SECTIONS[argument.replace(/^docs\/learn\//, "") + "#" + fragment];
    if (route.view === "doc" && mergedSection) {
      window.location.replace("#/doc/docs/learn/" + mergedSection);
      return;
    }

    // The site opens on the playground, and a removed page's address opens the
    // page that took its place. replace() rather than assigning, so the Back
    // button does not return to an address that only redirects.
    var moved = MOVED[route.view + (argument ? "/" + argument : "")];
    if (route.view === "home" || !route.view || moved) {
      window.location.replace(moved || "#/explore/playground");
      return;
    }

    // A menu link is keyed by its catalog path, which for the playground
    // keeps the "explore/" the address splits off.
    var path = route.view === "explore" ? "explore/" + argument : argument;

    // A new page starts at its top, and a section it is opened at is reached
    // at once rather than by the page's smooth scroll: a smooth scroll is
    // aimed where the heading stood when it began, and the figures and
    // formulas that arrive meanwhile move it down by hundreds of pixels.
    window.scrollTo({ top: 0, behavior: "instant" });
    content.scrollTo({ top: 0, behavior: "instant" });
    tocPanel.innerHTML = "";
    closeDrawer();
    setTitle(menuName(path));
    content.innerHTML = '<div class="loading">Loading…</div>';

    var done;
    if (route.view === "doc") { done = viewDocument(argument); }
    else if (route.view === "notebook") { done = viewNotebook(argument); }
    else if (route.view === "source") { done = viewSource(argument); }
    else if (route.view === "explore") { done = viewExplore(argument); }
    else if (route.view === "entry") { done = viewEntry(argument); }
    else { done = Promise.resolve(showError("No page at " + route.view, "")); }

    markActive(path);
    Promise.resolve(done).then(function () {
      if (window.location.hash !== requestedHash) { return; }
      if (fragment) {
        var target = findSection(fragment);
        if (target) { target.scrollIntoView({ behavior: "instant" }); }
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

  // ------------------------------------------------------- view: document

  function viewDocument(path) {
    var requestedHash = window.location.hash;
    return get("/api/doc?path=" + encodeURIComponent(path)).then(function (doc) {
      if (window.location.hash !== requestedHash) { return; }
      var meta = [];
      if (doc.minutes) { meta.push("<span>about " + doc.minutes + (doc.minutes === 1 ? " minute" : " minutes") + "</span>"); }
      meta.push("<code>" + breakablePath(doc.path) + "</code>");

      content.innerHTML =
        '<article class="prose">' + expandFigures(doc.html) + "</article>" +
        '<div class="page-meta">' + meta.join("") + "</div>";

      if (path === LOG_PATH) { showVerification(content.querySelector(".prose h1")); }
      fullSectionLinks(content.querySelector(".prose"));
      wrapTables(content);
      var figuresReady = fillFigures(content);
      buildToc(doc.headings);
      setTitle(doc.title);
      return Promise.all([figuresReady, typeset(content)]);
    }).catch(function (error) {
      showError("That document could not be shown", error.message);
    });
  }

  function loadLog() {
    if (log) { return Promise.resolve(log); }
    return get("/api/results").then(function (data) { log = data; return data; });
  }

  // scripts/verify_claims.py recomputes the numbers the log quotes and fails
  // if one disagrees. Its last result says how far the log can be trusted, so
  // it sits under the log's title.
  function showVerification(heading) {
    if (!heading) { return; }
    loadLog().then(function (data) {
      if (!heading.isConnected) { return; }
      heading.insertAdjacentHTML("afterend", '<div class="chips">' + verificationChip(data.verification) + "</div>");
    }).catch(function () { /* the log still reads without it */ });
  }

  function verificationChip(checked) {
    if (!checked) {
      return chip("warn", "The numbers in this log have not been re-checked");
    }
    var when = checked.finished.slice(0, 4) + "-" + checked.finished.slice(4, 6) + "-" +
      checked.finished.slice(6, 8);
    var claims = spaced(checked.checked) + (checked.checked === 1 ? " claim" : " claims");
    return chip(
      checked.failed ? "warn" : "good",
      checked.failed
        ? spaced(checked.failed) + " of " + claims + " disagree (" + when + ")"
        : claims + " re-checked, none disagree (" + when + ")"
    );
  }

  function viewEntry(identifier) {
    return loadLog().then(function (data) {
      var entry = data.entries.filter(function (e) { return e.identifier === identifier; })[0];
      if (!entry) {
        showError("No entry called " + identifier, "The log holds " + data.entries.length + " entries.");
        return;
      }
      window.location.replace("#/doc/" + LOG_PATH + "#" + entry.anchor);
    });
  }

  // ------------------------------------------------------- view: notebook

  function viewNotebook(path) {
    return get("/api/notebook?path=" + encodeURIComponent(path)).then(function (notebook) {
      var run = state.jupyter
        ? 'Saved outputs. <button class="button button--small" id="open-in-jupyter" type="button">Open in JupyterLab</button>'
        : "Saved outputs. To run these cells, install JupyterLab with <code>pip install -e .[notebook]</code>.";

      content.innerHTML =
        '<p class="notebook-bar">' + run + "</p>" +
        '<div class="notebook-frame">' + notebook.html + "</div>";

      var button = document.getElementById("open-in-jupyter");
      if (button) {
        button.addEventListener("click", function () { launchJupyter(path, button); });
      }
      fullSectionLinks(content.querySelector(".notebook-frame"));
      wrapTables(content);
      markMath(content);
      keepPicturesReadable(content);
      buildToc(headingsIn(content));
      typeset(content);
      setTitle(notebook.title);
    }).catch(function (error) {
      showError("That notebook could not be shown", error.message);
    });
  }

  // A notebook's text cells write their maths between dollar signs, which the
  // server leaves as they are. Each formula is put in the same element a
  // document's formulas arrive in, so the one typesetting step draws both.
  // Code and printed output are left alone.
  function markMath(root) {
    var pattern = /(\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$)/g;
    Array.prototype.forEach.call(root.querySelectorAll(".notebook-frame .text_cell_render"), function (cell) {
      var walker = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT, {
        acceptNode: function (node) {
          return node.parentNode.closest("code, pre") || node.nodeValue.indexOf("$") < 0
            ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
        }
      });
      var texts = [];
      while (walker.nextNode()) { texts.push(walker.currentNode); }
      texts.forEach(function (text) {
        var value = text.nodeValue, last = 0, frag = document.createDocumentFragment(), found = false, m;
        pattern.lastIndex = 0;
        while ((m = pattern.exec(value))) {
          if (m.index > 0 && value.charAt(m.index - 1) === "\\") { continue; }
          found = true;
          frag.appendChild(document.createTextNode(value.slice(last, m.index)));
          var span = document.createElement("span");
          span.className = m[2] !== undefined ? "math-display" : "math-inline";
          span.textContent = (m[2] !== undefined ? m[2] : m[3]).trim();
          frag.appendChild(span);
          last = m.index + m[0].length;
        }
        if (!found) { return; }
        frag.appendChild(document.createTextNode(value.slice(last)));
        text.parentNode.replaceChild(frag, text);
      });
    });
  }

  // A plot never shrinks below 65% of the size it was saved at, nor is held
  // wider than the 820 pixels a figure keeps, so its words stay legible; its
  // output box scrolls sideways instead. The picture would cover a shadow
  // drawn behind it, so the edges that hide more are marked on a box around
  // the output, and the stylesheet draws their shadows over the picture.
  function keepPicturesReadable(root) {
    Array.prototype.forEach.call(root.querySelectorAll(".notebook-frame .output_subarea > img"), function (img) {
      var set = function () { img.style.setProperty("--nb-min", Math.min(820, Math.round(img.naturalWidth * 0.65)) + "px"); };
      if (img.complete && img.naturalWidth) { set(); } else { img.addEventListener("load", set); }
      var box = img.parentNode;
      if (box.parentNode.classList.contains("pic-edges")) { return; }
      var edges = el("div", { class: "pic-edges" });
      box.parentNode.insertBefore(edges, box);
      edges.appendChild(box);
      var mark = function () {
        edges.classList.toggle("more-left", box.scrollLeft > 0);
        edges.classList.toggle("more-right", box.scrollLeft + box.clientWidth < box.scrollWidth - 1);
      };
      box.addEventListener("scroll", mark, { passive: true });
      if (window.ResizeObserver) { new ResizeObserver(mark).observe(box); } else { mark(); }
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
      // The path is the page's name, and its ending already says the file's type.
      content.innerHTML =
        '<div class="page-head"><h1 class="page-title"><code>' + breakablePath(file.path) + "</code></h1>" +
        '<div class="page-meta">' + spaced(file.lines) + " lines</div></div>" +
        '<article class="prose">' + file.html + "</article>";
      setTitle(file.title);
    }).catch(function (error) {
      showError("That file could not be shown", error.message);
    });
  }

  // -------------------------------------------------------- view: explore

  function viewExplore(name) {
    if (name === "playground" && window.thesisPlayground) {
      return window.thesisPlayground(content, {
        get: get, toast: toast, fixed: fixed, escapeHtml: escapeHtml, chip: chip,
        buildToc: buildToc
      });
    }
    showError("No page called " + name, "");
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
            panel.appendChild(el("div", { class: "search-note", text: "Nothing matches “" + query + "”." }));
          } else {
            var byDocument = {};
            data.hits.forEach(function (hit) {
              (byDocument[hit.path] = byDocument[hit.path] || []).push(hit);
            });
            // Each document is named once, over the lines that matched in it.
            Object.keys(byDocument).slice(0, 12).forEach(function (path) {
              var hits = byDocument[path];
              panel.appendChild(el("div", { class: "search-where", text: hits[0].title }));
              hits.slice(0, 3).forEach(function (hit) {
                panel.appendChild(el("a", {
                  class: "search-hit",
                  href: "#/doc/" + hit.path,
                  html: mark(hit.context, query)
                }));
              });
            });
            var documents = Object.keys(byDocument).length;
            panel.appendChild(el("div", {
              class: "search-note",
              text: spaced(data.hits.length) + (data.hits.length === 1 ? " match" : " matches") +
                " in " + spaced(documents) + (documents === 1 ? " document." : " documents.")
            }));
          }
          panel.hidden = false;
        }).catch(function () { close(); });
      }, 180);
    });

    input.addEventListener("keydown", function (event) {
      if (event.key === "Escape") { input.value = ""; close(); input.blur(); }
    });
    // A result chosen opens its page, and the panel, which covers the page on
    // a phone, closes with it.
    panel.addEventListener("click", function (event) {
      if (event.target.closest(".search-hit")) { close(); }
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

  // Opens one notebook in JupyterLab, starting JupyterLab first if this site
  // has not. The window that opens is the confirmation; a message says where
  // JupyterLab is only when the browser blocked that window.
  function launchJupyter(notebookPath, button) {
    button.disabled = true;
    get("/api/jupyter").then(function (status) {
      if (status.running) { return status; }
      if (!status.installed) {
        throw new Error("JupyterLab is not installed. Install it with pip install -e .[notebook].");
      }
      toast("Starting JupyterLab. This takes a few seconds…", 60000);
      return fetch("/api/jupyter", { method: "POST" }).then(function (response) {
        return response.json().then(function (started) {
          if (!response.ok) { throw new Error("JupyterLab did not start: " + (started.error || response.statusText)); }
          return started;
        });
      });
    }).then(function (running) {
      hideToast();
      var address = labAddress(running.url, notebookPath);
      if (!window.open(address, "jupyterlab")) {
        toast("JupyterLab is running at " + address, 30000);
      }
    }).catch(function (error) {
      toast(error.message, 12000);
    }).then(function () {
      button.disabled = false;
    });
  }

  // JupyterLab reports itself as ".../lab?token=...". To open one notebook the
  // address has to become ".../lab/tree/<path>?token=...", keeping the token:
  // without it JupyterLab asks for a password nobody has been told.
  // The path is rebuilt rather than appended to, because appending to an
  // address that already ends in /lab produced /lab/lab/tree/... and a 404.
  function labAddress(url, notebookPath) {
    try {
      var parsed = new URL(url);
      var base = parsed.pathname.replace(/\/lab\/?$/, "").replace(/\/$/, "");
      parsed.pathname = base + "/lab/tree/" + notebookPath;
      return parsed.toString();
    } catch (e) {
      return url;
    }
  }

  // ----------------------------------------------------------------- boot

  initTheme();
  loadMath();

  Promise.all([get("/api/state"), get("/api/catalog")]).then(function (loaded) {
    state = loaded[0];
    catalog = loaded[1];
    // A server too old to say when it started cannot run the check at all,
    // and is older than this page, which is the same thing.
    document.getElementById("stale").hidden = !(state.stale || state.started === undefined);
    buildSidebar();
    initNav();
    initSearch();
    initSectionLinks();
    window.addEventListener("hashchange", navigate);
    navigate();
  }).catch(function (error) {
    content.innerHTML = '<div class="error-box"><strong>The site could not start</strong>' +
      escapeHtml(error.message) + "</div>";
  });
})();
