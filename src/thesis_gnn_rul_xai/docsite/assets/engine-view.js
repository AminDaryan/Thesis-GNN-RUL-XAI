/* An engine section and its explanation graph. Sources and choices are in the folder README. */
(function (host) {
  "use strict";

  // ASSUMPTION A-31: anchor variables to documented sections or systems; drawing coordinates are illustrative, not probe locations.
  var locations = {
    T24: { at: [340, 95], anchor: [350, 332], part: "lpc", kind: "station", name: "Temperature leaving the low-pressure compressor", note: "The marker points to the outlet region, where air leaves the first compressor." },
    T30: { at: [500, 75], anchor: [496, 330], part: "hpc", kind: "station", name: "Temperature leaving the high-pressure compressor", note: "This is the compressor outlet, before fuel is burned." },
    P30: { at: [560, 155], anchor: [496, 342], part: "hpc", kind: "station", name: "Total pressure leaving the high-pressure compressor", note: "Total pressure includes the effect of bringing the moving air to rest. It shares an outlet region with T30 and Ps30." },
    Ps30: { at: [435, 160], anchor: [496, 352], part: "hpc", kind: "station", name: "Static pressure leaving the high-pressure compressor", note: "Static pressure is the local pressure of the flowing air. It shares an outlet region with T30 and P30." },
    T50: { at: [1040, 130], anchor: [976, 340], part: "lpt", kind: "station", name: "Temperature leaving the low-pressure turbine", note: "The marker points to the turbine outlet, before the core exhaust nozzle." },
    P15: { at: [235, 100], anchor: [290, 249], part: "bypass", kind: "station", name: "Total pressure in the bypass duct", note: "The outer passage carries air around the compressor, combustor and turbines." },
    Nf: { at: [145, 525], anchor: [175, 390], part: "fan", kind: "shaft", name: "Physical fan speed", note: "A rotational speed of the fan shaft. The callout identifies the shaft, not a documented speed-probe mounting point." },
    NRf: { at: [280, 565], anchor: [310, 390], part: "fan", kind: "derived", name: "Corrected fan speed", note: "Fan speed adjusted for inlet conditions. This is a calculated quantity associated with the same fan shaft as Nf, not another probe." },
    Nc: { at: [460, 530], anchor: [510, 407], part: "core", kind: "shaft", name: "Physical core speed", note: "A rotational speed of the shaft joining the high-pressure compressor and turbine. Its exact measuring point is not supplied." },
    phi: { at: [665, 75], anchor: [620, 287], part: "burner", kind: "derived", name: "Fuel flow divided by Ps30", note: "A ratio involving the fuel supply and compressor-outlet static pressure. Its fuel-system callout is an illustration, not a separate probe." },
    BPR: { at: [95, 145], anchor: [265, 276], part: "bypass", kind: "derived", name: "Bypass ratio", note: "The mass flow around the core divided by the mass flow through it. This describes two air paths; it is not a measurement at one point." },
    htBleed: { at: [645, 555], anchor: [560, 300], part: "cooling", kind: "system", name: "Bleed enthalpy", note: "An energy quantity for air taken from the main flow. The source names the quantity, but does not give a unique probe station or detailed pipe route." },
    W31: { at: [795, 130], anchor: [765, 307], part: "hpt", kind: "system", name: "High-pressure turbine coolant bleed", note: "The cooling-air flow associated with the high-pressure turbine. The drawn cooling branch is schematic; its exact extraction and measuring points are not specified." },
    W32: { at: [915, 170], anchor: [898, 311], part: "lpt", kind: "system", name: "Low-pressure turbine coolant bleed", note: "The cooling-air flow associated with the low-pressure turbine, following the dataset paper's Table 2. Its exact extraction and measuring points are not specified." }
  };
  var kinds = { station: "Measurement region", shaft: "Shaft quantity", derived: "Calculated quantity", system: "System quantity" };
  var source = "https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf";

  function esc(value) { return String(value).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function number(value) { return typeof value === "number" && isFinite(value); }
  function signed(value) { return number(value) ? (value > 0 ? "+" : "") + value.toFixed(3) : "not stored"; }

  function edgesFor(data, sensors, fallback) {
    if (data.summary.method !== "walks") { return fallback || []; }
    var row = data.values[0];
    var pairs = [];
    sensors.forEach(function (a, i) {
      sensors.slice(i + 1).forEach(function (b) {
        if (Object.prototype.hasOwnProperty.call(row, "link_" + a + "_" + b) ||
            Object.prototype.hasOwnProperty.call(row, "link_" + b + "_" + a)) { pairs.push([a, b]); }
      });
    });
    return pairs;
  }

  function walkSteps(walk, sensors) {
    return String(walk || "").split(" -> ").map(function (step) {
      return step.split("/").filter(function (s) { return sensors.indexOf(s) >= 0; });
    });
  }

  function walkPairs(steps, edges) {
    return edges.filter(function (edge) {
      return steps.some(function (step, i) {
        var next = steps[i + 1] || [];
        return (step.indexOf(edge[0]) >= 0 && next.indexOf(edge[1]) >= 0) ||
          (step.indexOf(edge[1]) >= 0 && next.indexOf(edge[0]) >= 0);
      });
    });
  }

  function peaks(data, sensors, edges) {
    var nodes = 0, links = 0, walks = data.summary.method === "walks";
    data.values.forEach(function (r) {
      sensors.forEach(function (s) { var v = r[(walks ? "start_" : "") + s]; if (number(v)) { nodes = Math.max(nodes, Math.abs(v)); } });
      if (walks) { edges.forEach(function (e) { links = Math.max(links, Math.abs(linkValue(r, e))); }); }
    });
    return { nodes: nodes, links: links };
  }

  function linkValue(row, edge) {
    return (Number(row["link_" + edge[0] + "_" + edge[1]]) || 0) + (Number(row["link_" + edge[1] + "_" + edge[0]]) || 0);
  }

  function blades(x, count, width, top, bottom) {
    var out = "";
    for (var i = 0; i < count; i++) {
      var a = x + i * width;
      out += '<path d="M' + a + " " + top + " l9 7 -2 " + (bottom - top - 14) + ' -9 7Z"/>';
    }
    return out;
  }

  function cutaway(part) {
    var group = function (name, markup) { return '<g class="ev-part' + (part === name ? " is-active" : "") + '" data-engine-part="' + name + '">' + markup + "</g>"; };
    return '<g class="ev-machine" aria-hidden="true">' +
      '<path class="ev-casing" d="M68 236 Q90 178 182 177 Q244 179 317 204 L1066 237 L1100 260 L1060 269 L320 244 Q235 219 103 251Z"/>' +
      '<path class="ev-casing" d="M68 437 Q90 484 182 486 Q244 484 317 464 L1066 437 L1100 413 L1060 405 L320 424 Q235 450 103 420Z"/>' +
      group("bypass", '<path class="ev-air" d="M225 252 Q660 257 1060 280 M225 417 Q660 412 1060 391"/>' +
        '<path class="ev-splitter" d="M244 294 Q280 277 355 280 L486 307 M244 378 Q280 394 355 392 L486 366"/>') +
      '<path class="ev-casing" d="M360 280 L498 302 Q590 277 696 297 L980 310 L1100 327 L1100 344 L980 364 L696 376 Q590 394 498 370 L360 392 L360 378 L498 355 Q590 377 696 362 L980 350 L1055 336 L980 324 L696 311 Q590 291 498 319 L360 295Z"/>' +
      group("fan", '<path class="ev-shaft" d="M140 390H960"/>' + blades(160, 3, 15, 213, 450) + '<path d="M96 336 Q124 310 165 316 L216 323V351L165 358Q124 362 96 336Z"/>') +
      group("lpc", blades(270, 4, 19, 297, 375)) +
      group("hpc", blades(379, 6, 17, 315, 359)) +
      group("core", '<path class="ev-shaft" d="M386 407H794"/>') +
      group("burner", '<path d="M555 313 Q611 299 675 316L675 355Q611 371 555 356Z"/>' +
        '<path class="ev-flame" d="M575 331q17-13 52 5-27 18-52 4m40-9q20-11 39 5-20 13-39 6"/>' +
        '<path class="ev-fuel" d="M620 274V320m-8-9 8 9 8-9"/>') +
      group("hpt", blades(738, 3, 18, 313, 362)) +
      group("lpt", blades(838, 6, 20, 301, 374)) +
      group("cooling", '<path class="ev-cooling" d="M492 320V286H768V314 M492 353V385H900V371"/>') +
      '<path class="ev-air" d="M24 271H120m-10-5 10 5-10 5M25 400H120m-10-5 10 5-10 5M1030 336H1130m-10-5 10 5-10 5"/>' +
      '<g class="ev-component-label"><text x="182" y="202">Fan</text><text x="305" y="322">Low-pressure</text><text x="305" y="339">compressor</text>' +
      '<text x="424" y="438">High-pressure</text><text x="424" y="455">compressor</text>' +
      '<text x="615" y="418">Combustor</text><text x="765" y="435">High-pressure</text><text x="765" y="452">turbine</text>' +
      '<text x="903" y="438">Low-pressure</text><text x="903" y="455">turbine</text>' +
      '<text x="790" y="240">Bypass air →</text><text x="1050" y="375">Core exhaust</text><text x="620" y="262">Fuel</text></g></g>';
  }

  function graphPicture(data, opt, state, sensors, edges, limit) {
    var row = data.values[opt.index], walks = data.summary.method === "walks", selected = state.sensor;
    var chosenWalk = state.walk >= 0 && (data.top_walks || [])[state.walk];
    var steps = chosenWalk ? walkSteps(chosenWalk.walk, sensors) : [];
    var paths = walkPairs(steps, edges), highlighted = [];
    steps.forEach(function (step) { highlighted = highlighted.concat(step); });
    var at = function (s) {
      var p = opt.graph.positions[s];
      return state.layout === "graph" && p ? [75 + p[0] / opt.graph.width * 1000, 35 + p[1] / opt.graph.height * 545] : locations[s].at;
    };
    var out = '<svg viewBox="0 0 1160 620" class="ev-drawing" role="group" aria-label="' +
      (state.layout === "engine" ? "Engine cutaway with explanation markers" : "Explanation on the sensor graph") + '">';
    if (state.layout === "engine") { out += cutaway(selected && locations[selected].part); }
    if (state.overlay || chosenWalk) {
      edges.slice().sort(function (a, b) { return Math.abs(linkValue(row, a)) - Math.abs(linkValue(row, b)); }).forEach(function (e) {
        var active = chosenWalk ? paths.indexOf(e) >= 0 : !selected || e.indexOf(selected) >= 0;
        var a = at(e[0]), b = at(e[1]), v = linkValue(row, e);
        var colour = walks ? (v >= 0 ? "var(--fig-accent)" : "var(--fig-warn)") : "var(--fig-axis)";
        var w = walks ? 1.2 + 5 * Math.abs(v) / (limit.links || 1) : 1.6;
        var title = e.join(" ↔ ") + (walks ? ": net " + signed(v) + " of the score, both ways and all layers; " +
          e[0] + " → " + e[1] + ": " + signed(row["link_" + e[0] + "_" + e[1]]) + "; " +
          e[1] + " → " + e[0] + ": " + signed(row["link_" + e[1] + "_" + e[0]]) : ": connection used by the model");
        out += '<g class="ev-edge" opacity="' + (active ? 0.82 : 0.07) + '"><title>' + esc(title) + '</title><line x1="' + a[0] + '" y1="' + a[1] + '" x2="' + b[0] + '" y2="' + b[1] + '" stroke="' + colour + '" stroke-width="' + w.toFixed(2) + '"/>' +
          '<line x1="' + a[0] + '" y1="' + a[1] + '" x2="' + b[0] + '" y2="' + b[1] + '" stroke="transparent" stroke-width="12"/></g>';
      });
    }
    sensors.forEach(function (s) {
      var loc = locations[s], p = at(s), v = row[(walks ? "start_" : "") + s];
      var active = chosenWalk ? highlighted.indexOf(s) >= 0 : !selected || selected === s;
      if (state.layout === "engine") {
        out += '<path class="ev-leader" opacity="' + (active ? 0.8 : 0.2) + '" d="M' + p[0] + " " + p[1] + " L" + loc.anchor[0] + " " + loc.anchor[1] + '"/>' +
          '<circle class="ev-anchor" cx="' + loc.anchor[0] + '" cy="' + loc.anchor[1] + '" r="3"/>';
      }
      var title = s + ": " + loc.name + "; " + signed(v) + (walks ? " of the score" : " cycles");
      out += '<g role="button" tabindex="0" data-ev-sensor="' + s + '" aria-label="' + esc(title) + '" aria-pressed="' + String(selected === s) + '" class="ev-node' + (selected === s ? " is-selected" : "") + '" opacity="' + (active ? 1 : 0.38) + '"><title>' + esc(title) + '</title>';
      var shape = loc.kind === "station" ? '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="25"' :
        '<rect x="' + (p[0] - 38) + '" y="' + (p[1] - 21) + '" width="76" height="42" rx="9"';
      out += shape + ' class="ev-node-base"/>' + shape + ' fill="' + (v >= 0 ? "var(--fig-accent)" : "var(--fig-warn)") +
        '" fill-opacity="' + (number(v) ? 0.12 + 0.78 * Math.abs(v) / (limit.nodes || 1) : 0).toFixed(3) + '" class="ev-node-value"/>' +
        '<text x="' + p[0] + '" y="' + (p[1] + 5) + '" class="ev-symbol">' + s + '</text>' +
        '<text x="' + p[0] + '" y="' + (p[1] + 42) + '" class="ev-value">' + esc(signed(v)) + '</text></g>';
    });
    return out + "</svg>";
  }

  function selection(data, opt, state, sensors, edges) {
    var row = data.values[opt.index], walks = data.summary.method === "walks", out = "";
    if (state.walk >= 0 && (data.top_walks || [])[state.walk]) {
      var w = data.top_walks[state.walk], steps = walkSteps(w.walk, sensors);
      out += '<div class="ev-walk-steps" aria-label="Selected walk by network layer">' + steps.map(function (step, i) {
        return '<div><small>' + (i ? "After graph layer " + i : "Input to graph layers") + '</small><strong>' + esc(step.join(" / ")) + "</strong></div>";
      }).join('<span aria-hidden="true">→</span>') + '</div><p>This saved walk has an average signed share of <strong>' + signed(w.mean) +
        '</strong> and an average size of <strong>' + signed(w.mean_abs) + '</strong> across all ' +
        (data.summary.scope === "test" ? "test engines" : "stored windows") +
        '. Its share at the selected window was not stored. Slashes group sensor positions with identical outputs after a graph layer; their original readings can differ. All compatible connections are highlighted. Repeated groups can include staying on the same sensor.</p>' +
        '<p>The line widths still show the selected window’s total connection shares, not this walk’s individual share.</p>';
    }
    if (state.sensor) {
      var s = state.sensor, loc = locations[s], v = row[(walks ? "start_" : "") + s];
      var neighbours = edges.filter(function (e) { return e.indexOf(s) >= 0; }).map(function (e) { return e[0] === s ? e[1] : e[0]; });
      out += '<p><strong>' + esc(s + " — " + loc.name) + '</strong><br>' + esc(loc.note) + '</p><p>' + kinds[loc.kind] +
        ' · Attributed value: <strong>' + signed(v) + (walks ? " of the score before conversion to cycles" : " cycles") + '</strong>.' +
        (number(row["reading_" + s]) ? ' The input reading at the last cycle, after smoothing and rescaling, is ' + row["reading_" + s].toFixed(3) + "." : "") + '</p>' +
        (walks && number(row["link_" + s + "_" + s]) ? '<p>Along its connection to itself, across the graph layers: ' + signed(row["link_" + s + "_" + s]) + ' of the score.</p>' : "") +
        '<p>Graph neighbours: ' + (neighbours.length ? esc(neighbours.join(", ")) : "none shown") + '.</p>';
    }
    return out || '<p>Select a marker to see what the variable measures and where it belongs. Circles mark measurement regions; rounded boxes identify shaft, calculated or system quantities.</p>';
  }

  function mount(container, data, opt, state) {
    var walks = data.summary.method === "walks";
    var sensors = Object.keys(walks ? data.summary.mean_abs_start : data.summary.mean_abs).filter(function (s) { return !!locations[s]; });
    var edges = edgesFor(data, sensors, opt.edges);
    var limit = peaks(data, sensors, edges);
    state.layout = state.layout || "engine";
    if (state.overlay === undefined) { state.overlay = true; }
    if (state.walk === undefined) { state.walk = -1; }
    if (sensors.indexOf(state.sensor) < 0) { state.sensor = null; }

    function render(focus) {
      container.innerHTML = '<figcaption><strong>Explore the engine.</strong> A schematic cutaway of the C-MAPSS engine, with this prediction’s explanation placed by component.</figcaption>' +
        '<div class="ev-controls"><div class="ev-layout" role="group" aria-label="Explanation layout">' +
        ['engine', 'graph'].map(function (mode) { return '<button type="button" class="button" data-ev-layout="' + mode + '" aria-pressed="' + String(state.layout === mode) + '">' + (mode === "engine" ? "Engine cutaway" : "Graph layout") + '</button>'; }).join("") + '</div>' +
        '<label><input type="checkbox" data-ev-overlay' + (state.overlay ? " checked" : "") + '> Show graph connections</label>' +
        '<button type="button" class="button" data-ev-clear>Clear selection</button></div>' +
        (walks && data.top_walks && data.top_walks.length ? '<label class="ev-walk-picker">Highlight a saved walk <select data-ev-walk><option value="-1">No walk selected</option>' +
          data.top_walks.map(function (w, i) { return '<option value="' + i + '"' + (state.walk === i ? " selected" : "") + '>' + esc(w.walk.split(" -> ").join(" → ")) + '</option>'; }).join("") + '</select></label>' : "") +
        '<div class="ev-scroll" tabindex="0" aria-label="Engine diagram; scroll horizontally on narrow screens">' + graphPicture(data, opt, state, sensors, edges, limit) + '</div>' +
        '<div class="pg-legend"><span><i style="background:var(--fig-accent)"></i>positive share</span><span><i style="background:var(--fig-warn)"></i>negative share</span>' +
        '<span>Marker strength: |value|, fixed scale across these ' + (data.summary.scope === "test" ? "engines" : "windows") + ' (maximum ' + limit.nodes.toFixed(3) + (walks ? ' score units' : ' cycles') + ')</span></div>' +
        '<p class="pg-faint">' + (walks ? 'Line width: net share along a connection, both ways and all layers together (maximum ' + limit.links.toFixed(3) + '). Opposing shares can cancel. ' : 'Lines show model connections; their widths do not represent importance. ') +
        'Dashed leaders locate variables on the engine. Graph lines are model relationships, not wires, airflow or damage paths.</p>' +
        (!walks && opt.edges === null ? '<p class="pg-faint">The saved network’s graph could not be read. Sensor locations are shown without connections.</p>' : '') +
        '<div class="ev-details" aria-live="polite">' + selection(data, opt, state, sensors, edges) + '</div>' +
        '<p class="pg-faint">Placement is schematic, not to scale. A strong share explains the prediction; it does not locate a fault. ' +
        '<a href="' + source + '" target="_blank" rel="noopener">NASA: engine and variable definitions</a> · ' +
        '<a href="https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/enginesimu/" target="_blank" rel="noopener">NASA EngineSim: interactive reference</a> · ' +
        '<a href="#/entry/A-31">Placement choices and limits</a></p>';
      if (focus) { var next = container.querySelector(focus); if (next) { next.focus({ preventScroll: true }); } }
    }
    container.onclick = function (event) {
      var sensor = event.target.closest("[data-ev-sensor]"), mode = event.target.closest("[data-ev-layout]");
      if (sensor) { state.sensor = state.sensor === sensor.dataset.evSensor ? null : sensor.dataset.evSensor; state.walk = -1; render('[data-ev-sensor="' + sensor.dataset.evSensor + '"]'); }
      else if (mode) { state.layout = mode.dataset.evLayout; render('[data-ev-layout="' + state.layout + '"]'); }
      else if (event.target.closest("[data-ev-clear]")) { state.sensor = null; state.walk = -1; render("[data-ev-clear]"); }
    };
    container.onchange = function (event) {
      if (event.target.hasAttribute("data-ev-overlay")) { state.overlay = event.target.checked; state.walk = -1; render("[data-ev-overlay]"); }
      if (event.target.hasAttribute("data-ev-walk")) { state.walk = Number(event.target.value); state.sensor = null; render("[data-ev-walk]"); }
    };
    container.onkeydown = function (event) {
      var sensor = event.target.closest("[data-ev-sensor]");
      if (sensor && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); sensor.dispatchEvent(new MouseEvent("click", { bubbles: true })); }
    };
    render();
    return { update: function (index) {
      opt.index = index;
      container.querySelector(".ev-scroll").innerHTML = graphPicture(data, opt, state, sensors, edges, limit);
      container.querySelector(".ev-details").innerHTML = selection(data, opt, state, sensors, edges);
    } };
  }

  function mountGuide(container) {
    var selected = null;
    function render() {
      var out = '<figcaption>Select a variable to see what it measures. This is the same schematic used in Explain predictions, without explanation scores.</figcaption><div class="ev-scroll" tabindex="0" aria-label="Engine and sensor locations; scroll horizontally"><svg viewBox="0 0 1160 620" class="ev-drawing" role="group" aria-label="Engine sensor guide">' + cutaway(selected && locations[selected].part);
      Object.keys(locations).forEach(function (sensor) {
        var loc = locations[sensor], p = loc.at;
        out += '<path class="ev-leader" d="M' + p[0] + ' ' + p[1] + ' L' + loc.anchor[0] + ' ' + loc.anchor[1] + '"/>' +
          '<g role="button" tabindex="0" data-ev-sensor="' + sensor + '" aria-pressed="' + (selected === sensor) + '" aria-label="' + esc(sensor + ': ' + loc.name) + '" class="ev-node' + (selected === sensor ? ' is-selected' : '') + '">' +
          (loc.kind === 'station' ? '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="25"' : '<rect x="' + (p[0]-38) + '" y="' + (p[1]-21) + '" width="76" height="42" rx="9"') +
          ' class="ev-node-base"/><text x="' + p[0] + '" y="' + (p[1]+5) + '" class="ev-symbol">' + sensor + '</text></g>';
      });
      var item = selected && locations[selected];
      container.innerHTML = out + '</svg></div><div class="ev-details" aria-live="polite"><p>' + (item ? '<strong>' + esc(selected + ': ' + item.name) + '</strong><br>' + esc(item.note) + ' ' + kinds[item.kind] + '.' : 'Circles show measurement regions. Rounded boxes identify shaft, calculated or system quantities. The fourteen model inputs are shown, not all raw-file columns.') + '</p></div><p class="pg-faint">Schematic locations, not exact probe coordinates or fault locations. <a href="#/entry/A-31">Placement choices and sources</a></p>';
    }
    container.onclick = function (event) {
      var marker = event.target.closest('[data-ev-sensor]');
      if (!marker) { return; }
      var name = marker.dataset.evSensor;
      selected = selected === name ? null : name;
      render();
      container.querySelector('[data-ev-sensor="' + name + '"]').focus({ preventScroll: true });
    };
    container.onkeydown = function (event) {
      if (event.target.closest('[data-ev-sensor]') && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault(); event.target.dispatchEvent(new MouseEvent('click', { bubbles: true }));
      }
    };
    render();
  }

  host.ThesisEngineView = { mount: mount, mountGuide: mountGuide, locations: locations, edgesFor: edgesFor, walkSteps: walkSteps, walkPairs: walkPairs, peaks: peaks, graphPicture: graphPicture };
})(typeof window === "undefined" ? module.exports : window);
