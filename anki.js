<div lang="ja">
<script>
window.AnkiCanvasOptions = {
  frontCanvasSize: 300,
  frontLineWidth: 7,
  backCanvasSize: 300,
  backLineWidth: 7,

  // 'auto' is a special value that will automatically select either 'light' or
  // 'dark' depending on Anki's "Night Mode" status. If you wish to force a
  // colorScheme, you can pass it's name from the colorSchemes settings below.
  colorScheme: 'auto',

  // You can modify the default colorSchemes in the dictionary below, or even
  // add your own colorSchemes beyond light and dark.
  colorSchemes: {
    light: {
      brush: '#000',
      grid: '#dcdcdc',
      gridBg: '#fff',
      buttonIcon: '#464646',
      buttonBg: '#dcdcdc',
      frontBrushColorizer: 'none', // none | spectrum | contrast
      backBrushColorizer: 'spectrum',
    },
    dark: {
      brush: '#fff',
      grid: '#646464',
      gridBg: '#222222',
      buttonIcon: '#000',
      buttonBg: '#646464',
      frontBrushColorizer: 'none',
      backBrushColorizer: 'spectrum',
    },
  },
}
</script>

{{#ex_1_obsc}}<ruby class=example>{{ex_1_obsc}}<rt>{{ex_1_read}}</rt></ruby>{{/ex_1_obsc}}
{{#ex_2_obsc}}<ruby class=example>{{ex_2_obsc}}<rt>{{ex_2_read}}</rt></ruby>{{/ex_2_obsc}}
{{#ex_3_obsc}}<ruby class=example>{{ex_3_obsc}}<rt>{{ex_3_read}}</rt></ruby>{{/ex_3_obsc}}
{{#ex_4_obsc}}<ruby class=example>{{ex_4_obsc}}<rt>{{ex_4_read}}</rt></ruby>{{/ex_4_obsc}}
{{#ex_5_obsc}}<ruby class=example>{{ex_5_obsc}}<rt>{{ex_5_read}}</rt></ruby>{{/ex_5_obsc}}
{{#ex_6_obsc}}<ruby class=example>{{ex_6_obsc}}<rt>{{ex_6_read}}</rt></ruby>{{/ex_6_obsc}}

<div id="ac-front"></div>
<div><b>{{#disambiguation}}({{disambiguation}}){{/disambiguation}}</b></div>
<div>音: {{onyomi}}<br>訓: {{kunyomi}}<br></div>
</p>
<div>{{meaning_obscured}}</div>



<script>
(function() {
  // Module 0: Options Configuration
  const optionsModule = (function() {
    const defaultOptions = {
      frontCanvasSize: 300,
      frontLineWidth: 7,
      backCanvasSize: 150,
      backLineWidth: 3.5,
      colorScheme: "auto",
      colorSchemes: {
        light: {
          brush: "#000",
          grid: "#dcdcdc",
          gridBg: "#fff",
          buttonIcon: "#464646",
          buttonBg: "#dcdcdc",
          frontBrushColorizer: "none",
          backBrushColorizer: "spectrum"
        },
        dark: {
          brush: "#fff",
          grid: "#646464",
          gridBg: "#000",
          buttonIcon: "#000",
          buttonBg: "#646464",
          frontBrushColorizer: "none",
          backBrushColorizer: "spectrum"
        }
      }
    };

    const devicePixelRatio = window.devicePixelRatio || 2;

    function getOption(key) {
      const userOptions = window.AnkiCanvasOptions || {};
      const defaultValue = defaultOptions[key];
      if (typeof userOptions[key] === typeof defaultValue) {
        return userOptions[key];
      }
      return defaultValue;
    }

    const options = {
      frontCanvasSize: getOption("frontCanvasSize") || defaultOptions.frontCanvasSize,
      frontLineWidth: getOption("frontLineWidth") || defaultOptions.frontLineWidth,
      backCanvasSize: getOption("backCanvasSize") || defaultOptions.backCanvasSize,
      backLineWidth: getOption("backLineWidth") || defaultOptions.backLineWidth,
      colorScheme: (function() {
        const selectedScheme = getOption("colorScheme") || defaultOptions.colorScheme;
        const colorSchemes = Object.assign({}, defaultOptions.colorSchemes, getOption("colorSchemes"));
        const isNightMode = document.getElementsByClassName("night_mode").length > 0;
        const autoScheme = isNightMode ? colorSchemes.dark : colorSchemes.light;
        return selectedScheme === "auto" ? autoScheme : (colorSchemes[selectedScheme] || autoScheme);
      })(),
      hdpiFactor: devicePixelRatio
    };

    return { options };
  })();

  // Module 1: State Management
  const stateManagerModule = (function() {
    const isoModule = (function() {
      function identity(x) { return x; }
      return { unwrap: identity, wrap: identity };
    })();
    const storageModule = (function() {
      window.db = window.db || {};
      const db = window.db;
      function setItem(key, value) { db[key] = value; }
      function getItem(key) { return db[key]; }
      const storage = navigator.userAgent.includes("QtWebEngine") ? 
        { setItem, getItem } : 
        (window.localStorage || window.sessionStorage);
      return {
        defaultStorage: () => storage,
        dump: JSON.stringify,
        parse: JSON.parse
      };
    })();

    const iso = isoModule;
    const storage = storageModule.defaultStorage();

    function saveState(state) {
      storage.setItem("state", storageModule.dump(state));
    }

    function createEmptyState() {
      const state = { lines: [], drawing: [], dirty: true, down: false };
      saveState(state);
      return iso.wrap(state);
    }

    function addPointToDrawing(wrappedState, point) {
      const state = iso.unwrap(wrappedState);
      if (state.down) {
        state.drawing.push(point);
        state.dirty = true;
        saveState(state);
      }
    }

    return {
      saveunsafe: function(state) { storage.setItem("state", storageModule.dump(state)); },
      load: function() {
        const savedState = storage.getItem("state");
        return savedState == null ? createEmptyState() : 
          iso.wrap(Object.assign({}, storageModule.parse(savedState), { dirty: true }));
      },
      empty: createEmptyState,
      map: function(wrappedState, transform) {
        const state = iso.unwrap(wrappedState);
        const clonedState = storageModule.parse(storageModule.dump(state));
        clonedState.lines = clonedState.lines.map(line => line.map(transform));
        return iso.wrap(clonedState);
      },
      undo: function(wrappedState) {
        const state = iso.unwrap(wrappedState);
        state.lines.pop();
        state.dirty = true;
        saveState(state);
      },
      clear: function(wrappedState) {
        const state = iso.unwrap(wrappedState);
        state.lines = [];
        state.dirty = true;
        saveState(state);
      },
      addDrawingPoint: addPointToDrawing,
      addFirstDrawingPoint: function(wrappedState, point) {
        const state = iso.unwrap(wrappedState);
        state.down = true;
        addPointToDrawing(wrappedState, point);
      },
      addLastDrawingPoint: function(wrappedState, point) {
        const state = iso.unwrap(wrappedState);
        state.drawing.push(point);
        state.lines.push(state.drawing);
        state.drawing = [];
        state.dirty = true;
        state.down = false;
        saveState(state);
      },
      willdisplay: function(wrappedState, renderFunc) {
        const state = iso.unwrap(wrappedState);
        if (state.dirty) {
          const linesToRender = [...state.lines, state.drawing].filter(line => line.length > 0);
          const renderingSuccessful = renderFunc(linesToRender);
          state.dirty = !renderingSuccessful;
        }
      }
    };
  })();

  // Module 2: DOM Builder (simplified)
  const domBuilderModule = (function() {
    function isNode(obj) { return obj && obj.nodeName && obj.nodeType; }
    function createElement() {
      const args = Array.from(arguments);
      let element = null;
      function appendChild(child) {
        let node;
        if (child == null) return;
        if (typeof child === "string") {
          element.appendChild(node = document.createTextNode(child));
        } else if (Array.isArray(child)) {
          child.forEach(appendChild);
        } else if (isNode(child)) {
          element.appendChild(node = child);
        } else if (typeof child === "object") {
          for (const key in child) {
            if (key === "style" && typeof child[key] === "object") {
              Object.assign(element.style, child[key]);
            } else if (key.startsWith("on")) {
              element.addEventListener(key.substring(2), child[key], false);
            } else {
              element.setAttribute(key, child[key]);
            }
          }
        }
      }
      while (args.length) {
        const arg = args.shift();
        if (typeof arg === "string" && !element) {
          element = document.createElement(arg);
        } else {
          appendChild(arg);
        }
      }
      return element;
    }
    createElement.context = () => createElement;
    return createElement;
  })();

  // Module 7: Renderer
  const rendererModule = (function() {
    const defaultRenderOptions = { lineWidth: 18 };
    return {
      rendercanvas: function(canvas, state, options) {
        stateManagerModule.willdisplay(state, linesToRender => {
          const ctx = canvas.getContext("2d");
          if (!ctx) return false;
          const renderOptions = Object.assign({}, defaultRenderOptions, options);
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          function drawGrid(ctx, width, height, colorScheme) {
            const halfWidth = width / 2;
            const halfHeight = height / 2;
            const gridLines = [
              [0, 0, width, height],
              [width, 0, 0, height],
              [halfWidth, 0, halfWidth, height],
              [0, halfHeight, width, halfHeight]
            ];
            ctx.save();
            gridLines.forEach(line => {
              ctx.beginPath();
              ctx.setLineDash([width / 80, height / 80]);
              ctx.strokeStyle = colorScheme.grid;
              ctx.lineWidth = 1;
              ctx.moveTo(line[0], line[1]);
              ctx.lineTo(line[2], line[3]);
              ctx.stroke();
            });
            ctx.restore();
          }
          drawGrid(ctx, canvas.width, canvas.height, renderOptions.colorScheme);
          ctx.save();
          ctx.lineWidth = renderOptions.lineWidth;
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          linesToRender.forEach((line, index) => {
            ctx.beginPath();
            ctx.strokeStyle = renderOptions.colorizer(index, linesToRender.length);
            for (let i = 1; i < line.length; i++) {
              const prevPoint = line[i - 1];
              const point = line[i];
              ctx.moveTo(prevPoint.x, prevPoint.y);
              ctx.lineTo(point.x, point.y);
            }
            ctx.stroke();
          });
          ctx.restore();
          return true;
        });
      },
      renderdom: function(id, element) {
        const container = document.getElementById(id);
        if (container) {
          if (container.firstChild) container.removeChild(container.firstChild);
          container.appendChild(element);
        }
      }
    };
  })();

  // Module 10: Color Utilities
  const colorUtilsModule = (function() {
    function rgbToHex(color) {
      return "#" + [color.r, color.g, color.b].map(val => {
        const hex = val.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      }).join("");
    }
    function hsvToRgb(hsv) {
      let r = 0, g = 0, b = 0;
      const { h, s, v } = hsv;
      const i = Math.floor(h * 6);
      const f = h * 6 - i;
      const p = v * (1 - s);
      const q = v * (1 - f * s);
      const t = v * (1 - (1 - f) * s);
      switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
      }
      return { r: Math.floor(r * 255), g: Math.floor(g * 255), b: Math.floor(b * 255) };
    }
    function spectrumColorizer(index, total) {
      return rgbToHex(hsvToRgb({ h: index / total, s: 0.95, v: 0.75 }));
    }
    function contrastColorizer(index, total) {
      return rgbToHex(hsvToRgb({ h: index / 0.618033988749895, s: 0.95, v: 0.75 }));
    }
    return {
      spectrum: spectrumColorizer,
      contrast: contrastColorizer,
      none: colorScheme => () => colorScheme.brush,
      getColorizer: function(colorScheme, colorizerType) {
        switch (colorizerType) {
          case "none": return this.none(colorScheme);
          case "spectrum": return spectrumColorizer;
          case "contrast": return contrastColorizer;
          default: return this.none(colorScheme);
        }
      }
    };
  })();

  // Module 11: Styles
  const stylesModule = (function() {
    return {
      canvas: colorScheme => ({
        height: `${optionsModule.options.frontCanvasSize}px`,
        width: `${optionsModule.options.frontCanvasSize}px`,
        border: `2px solid ${colorScheme.grid}`,
        background: colorScheme.gridBg
      }),
      wrapper: () => ({ "text-align": "center" }),
      actions: () => ({}),
      action: colorScheme => ({
        "font-size": "22px",
        border: "none",
        "border-radius": "50%",
        outline: "none",
        background: colorScheme.buttonBg,
        color: colorScheme.buttonIcon,
        display: "inline-flex",
        width: "44px",
        height: "44px",
        padding: "0",
        "align-items": "center",
        "justify-content": "center",
        margin: "0 5px"
      })
    };
  })();

  // Module 13: Icons
  const iconsModule = {
    undo: '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 -960 960 960"><path d="M280-200v-80h284q63 0 109.5-40T720-420t-46.5-100T564-560H312l104 104-56 56-200-200 200-200 56 56-104 104h252q97 0 166.5 63T800-420t-69.5 157T564-200z"/></svg>',
    clear: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16"><path d="M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm5 13.59L15.59 17 12 13.41 8.41 17 7 15.59 10.59 12 7 8.41 8.41 7 12 10.59 15.59 7 17 8.41 13.41 12 17 15.59z" fill="currentColor"/><path d="M0 0h24v24H0z" fill="none"/></svg>'
  };

  // Module 12: Entry Point
  requestAnimationFrame(function() {
    const domBuilder = domBuilderModule.context();
    const colorScheme = optionsModule.options.colorScheme;
    const canvas = domBuilder("canvas", {
      style: stylesModule.canvas(colorScheme),
      width: optionsModule.options.frontCanvasSize * optionsModule.options.hdpiFactor,
      height: optionsModule.options.frontCanvasSize * optionsModule.options.hdpiFactor
    });
    const buttons = {
      undo: domBuilder("button", { style: stylesModule.action(colorScheme) }),
      clear: domBuilder("button", { style: stylesModule.action(colorScheme) })
    };
    const actionsDiv = domBuilder("div", { style: stylesModule.actions(colorScheme) }, Object.values(buttons));
    const wrapperDiv = domBuilder("div", { style: stylesModule.wrapper(colorScheme) }, [canvas, actionsDiv]);
    rendererModule.renderdom("ac-front", wrapperDiv);
    const state = stateManagerModule.empty();

    const eventHandlers = [
      ["touchstart", stateManagerModule.addFirstDrawingPoint],
      ["touchmove", stateManagerModule.addDrawingPoint],
      ["touchend", stateManagerModule.addLastDrawingPoint],
      ["mousedown", stateManagerModule.addFirstDrawingPoint],
      ["mousemove", stateManagerModule.addDrawingPoint],
      ["mouseup", stateManagerModule.addLastDrawingPoint]
    ];
    eventHandlers.forEach(([eventName, handler]) => {
      canvas.addEventListener(eventName, event => {
        event.preventDefault();
        if (!(event instanceof TouchEvent || event instanceof MouseEvent)) return;
        const touchOrMouse = event instanceof TouchEvent ? event.changedTouches[0] : event;
        const point = {
          x: (touchOrMouse.pageX - canvas.offsetLeft) * optionsModule.options.hdpiFactor,
          y: (touchOrMouse.pageY - canvas.offsetTop) * optionsModule.options.hdpiFactor
        };
        handler(state, point);
      }, false);
    });

    function renderLoop() {
      rendererModule.rendercanvas(canvas, state, {
        colorizer: colorUtilsModule.getColorizer(colorScheme, colorScheme.frontBrushColorizer),
        lineWidth: optionsModule.options.frontLineWidth * optionsModule.options.hdpiFactor,
        colorScheme: colorScheme
      });
      requestAnimationFrame(renderLoop);
    }
    renderLoop();

    canvas.addEventListener("click", event => event.preventDefault(), false);
    buttons.undo.addEventListener("click", () => stateManagerModule.undo(state), false);
    buttons.clear.addEventListener("click", () => stateManagerModule.clear(state), false);
    buttons.undo.innerHTML = iconsModule.undo;
    buttons.clear.innerHTML = iconsModule.clear;
  });
})();
</script>
</div>

<!-- Anki Code Highlighter (Addon 112228974) BEGIN -->
<link rel="stylesheet" href="_ch-pygments-solarized.css" class="anki-code-highlighter">
<link rel="stylesheet" href="_ch-hljs-solarized.css" class="anki-code-highlighter">
<script src="_ch-highlight.js" class="anki-code-highlighter"></script>
<script src="_ch-my-highlight.js" class="anki-code-highlighter"></script>
<!-- Anki Code Highlighter (Addon 112228974) END -->
