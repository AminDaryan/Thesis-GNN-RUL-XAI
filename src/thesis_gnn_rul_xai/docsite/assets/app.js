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
  // starts closed and closes again whenever a page is chosen.
  var narrowScreen = window.matchMedia("(max-width: 900px)");

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
             '<div class="loading">Drawing the figure…</div></figure>';
    });
  }

  function fillFigures(root) {
    Array.prototype.forEach.call(root.querySelectorAll("[data-figure]"), function (holder) {
      if (holder.dataset.filled) { return; }
      holder.dataset.filled = "1";
      fetch("/api/figure?name=" + encodeURIComponent(holder.dataset.figure))
        .then(function (r) { return r.text(); })
        .then(function (svg) { holder.innerHTML = svg; })
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

    window.scrollTo(0, 0);
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

  // ------------------------------------------------------- view: document

  function viewDocument(path) {
    return get("/api/doc?path=" + encodeURIComponent(path)).then(function (doc) {
      var meta = [];
      if (doc.minutes) { meta.push("about " + doc.minutes + (doc.minutes === 1 ? " minute" : " minutes")); }
      meta.push("<code>" + escapeHtml(doc.path) + "</code>");

      content.innerHTML =
        '<article class="prose">' + expandFigures(doc.html) + "</article>" +
        '<div class="page-meta">' + meta.join(" · ") + "</div>";

      if (path === LOG_PATH) { showVerification(content.querySelector(".prose h1")); }
      wrapTables(content);
      fillFigures(content);
      buildToc(doc.headings);
      typeset(content);
      setTitle(doc.title);
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
      buildToc(headingsIn(content));
      typeset(content);
      setTitle(notebook.title);
    }).catch(function (error) {
      showError("That notebook could not be shown", error.message);
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
        '<div class="page-head"><h1 class="page-title"><code>' + escapeHtml(file.path) + "</code></h1>" +
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
    window.addEventListener("hashchange", navigate);
    navigate();
  }).catch(function (error) {
    content.innerHTML = '<div class="error-box"><strong>The site could not start</strong>' +
      escapeHtml(error.message) + "</div>";
  });
})();
