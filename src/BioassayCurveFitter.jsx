import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import {
  model4PL, model5PL, getModelFn, computeBiologicalEC50,
  fitModel, fitConstrainedModel,
  groupByConcentration, grubbsTest, runGrubbsAllGroups,
  WEIGHTING_TYPES,
} from "./fitting/index.js";

// ── Multi-Compound CSV Parser ─────────────────────────────────────
// Handles two formats:
//   Log10 format: first column header contains "log" (e.g. "Log(10) Concentration (M)")
//                 values like -5.00, -6.50 → stored as-is, treated as log10
//   Molar format: any other header (e.g. "Concentration (M)")
//                 values like 1.00E-5 → stored as-is (linear molar)
// Adjacent duplicate column headers = replicate columns for the same compound.
function parsePrismCSV(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("File has fewer than 2 rows.");
  const headers = lines[0].split(",").map(h => h.trim());
  const isLog10 = /log/i.test(headers[0]);

  const compoundMap = new Map();
  const compoundOrder = [];
  for (let i = 1; i < headers.length; i++) {
    const name = headers[i] || `Molecule ${i}`;
    if (!compoundMap.has(name)) { compoundMap.set(name, []); compoundOrder.push(name); }
    compoundMap.get(name).push(i);
  }
  if (compoundOrder.length === 0) throw new Error("No molecule columns found.");

  const rows = lines.slice(1).map(l => l.split(",").map(v => v.trim()));
  const compounds = compoundOrder.map(name => {
    const colIndices = compoundMap.get(name);
    const points = rows.map(row => {
      const raw = parseFloat(row[0]);
      // Normalise to log10 for internal storage
      const conc = isLog10 ? raw : Math.log10(raw);
      const reps = colIndices.map(ci => parseFloat(row[ci])).filter(v => !isNaN(v));
      const avg = reps.length ? reps.reduce((a, b) => a + b, 0) / reps.length : NaN;
      return { conc, avg, reps };
    }).filter(p => !isNaN(p.conc) && !isNaN(p.avg));
    return { name, nReplicates: colIndices.length, points };
  });
  return compounds;
}

// Convert a parsed compound back to CSV text for the existing textarea/parser
function compoundToCSV(compound) {
  const repCount = compound.nReplicates;
  const repHeaders = Array.from({ length: repCount }, (_, i) => `Rep${i + 1}`).join(",");
  const header = `Concentration,${repHeaders}`;
  const rows = compound.points.map(p => {
    const repVals = p.reps.map(v => v.toString()).join(",");
    // Concentrations are stored as log10, so raising them back reintroduces
    // binary-float noise: 0.03 round-trips to 0.029999999999999995. Trimming to
    // 12 significant digits restores the entered value without touching any
    // precision a plate reader could actually have produced.
    const conc = Number(Math.pow(10, p.conc).toPrecision(12));
    return `${conc},${repVals}`;
  });
  return [header, ...rows].join("\n");
}

// Download a blank multi-compound CSV template (Molarity concentrations, not log)
// Mirrors the structure of the test CSV: 3 compounds × 2 replicates, 8 concentrations
function downloadMultiTemplate() {
  const logConcs = [-5.00, -5.50, -6.00, -6.50, -7.00, -7.49, -8.00, -8.52];
  const header = "Concentration (M),Molecule A,Molecule A,Molecule B,Molecule B,Molecule C,Molecule C";
  const rows = logConcs.map(lc => {
    const molar = Math.pow(10, lc);
    // Write in scientific notation e.g. 1.00E-5 for clarity
    const formatted = molar.toExponential(2).toUpperCase();
    return `${formatted},,,,,,`;
  });
  const csv = [header, ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "multi_molecule_template.csv"; a.click();
  URL.revokeObjectURL(url);
}

// ── Overlay color palette (one per compound, cycles if >8) ────────
const OVERLAY_COLORS = [
  "#3b9eff", // blue
  "#ef4444", // red
  "#22c55e", // green
  "#a855f7", // purple
  "#f59e0b", // amber
  "#ec4899", // pink
  "#06b6d4", // cyan
  "#f97316", // orange
  "#10b981", // emerald
  "#d946ef", // fuchsia
  "#6366f1", // indigo
  "#84cc16", // lime
];

// ── Point shape helper ────────────────────────────────────────────
// Draws a filled shape at (cx, cy) with radius r. ctx.fillStyle must be set before calling.
function drawPoint(ctx, cx, cy, r, shape) {
  ctx.beginPath();
  switch (shape) {
    case "square":
      ctx.rect(cx - r, cy - r, r * 2, r * 2);
      break;
    case "triangle":
      ctx.moveTo(cx, cy - r * 1.3);
      ctx.lineTo(cx + r * 1.1, cy + r * 0.85);
      ctx.lineTo(cx - r * 1.1, cy + r * 0.85);
      ctx.closePath();
      break;
    case "diamond":
      ctx.moveTo(cx, cy - r * 1.3);
      ctx.lineTo(cx + r, cy);
      ctx.lineTo(cx, cy + r * 1.3);
      ctx.lineTo(cx - r, cy);
      ctx.closePath();
      break;
    default: // circle
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
  }
  ctx.fill();
}

// ── Resizable graph popup ─────────────────────────────────────────
function GraphPopup({
  onClose, parsedData, fitResult, activeModel, overlayMode, allFitResults,
  selectedCompounds, getCompoundStyle, overlayEditIndex, loadCompoundForOverlayEdit,
  chartOutlierIndices, excludedIndices,
  pointView, setPointView, errorBarType, setErrorBarType,
  xAxisLog, setXAxisLog, yAxisFormat, setYAxisFormat, yAxisDecimals, setYAxisDecimals,
  axisXMin, setAxisXMin, axisXMax, setAxisXMax, axisYMin, setAxisYMin, axisYMax, setAxisYMax,
  statsTableVisible, setStatsTableVisible, statsTablePos, setStatsTablePos,
  statsTableCols, setStatsTableCols, statsColPickerOpen, setStatsColPickerOpen,
  theme: t,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [pos, setPos] = useState({ x: Math.max(40, (window.innerWidth - 820) / 2), y: 40 });
  const dragState = useRef(null);
  const popupStatsDragRef = useRef({ dragging: false, startX: 0, startY: 0, startPosX: 0, startPosY: 0 });

  // Track popup stats drag via window events
  useEffect(() => {
    const onMove = (e) => {
      const d = popupStatsDragRef.current;
      if (!d.dragging) return;
      setStatsTablePos({ x: d.startPosX + (e.clientX - d.startX), y: d.startPosY + (e.clientY - d.startY) });
    };
    const onUp = () => { popupStatsDragRef.current.dragging = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [setStatsTablePos]);

  const ao = {
    xMin: axisXMin !== "" ? parseFloat(axisXMin) : null,
    xMax: axisXMax !== "" ? parseFloat(axisXMax) : null,
    yMin: axisYMin !== "" ? parseFloat(axisYMin) : null,
    yMax: axisYMax !== "" ? parseFloat(axisYMax) : null,
  };

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (overlayMode && allFitResults?.length > 0) {
      drawOverlayChart(
        canvas,
        allFitResults.map((r, i) => ({ ...r, ...getCompoundStyle(r.name, i) }))
                     .filter(r => !selectedCompounds || selectedCompounds.has(r.name)),
        { pointView, errorBarType, axisOverride: ao, yAxisFormat, yAxisDecimals, xAxisLog },
        t
      );
    } else if (parsedData) {
      drawChart(canvas, parsedData.xData, parsedData.yData, fitResult, activeModel,
        { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices,
          axisOverride: ao, yAxisFormat, yAxisDecimals, xAxisLog },
        t
      );
    }
  }, [parsedData, fitResult, activeModel, overlayMode, allFitResults, selectedCompounds,
      getCompoundStyle, pointView, errorBarType, chartOutlierIndices, excludedIndices,
      axisXMin, axisXMax, axisYMin, axisYMax, yAxisFormat, yAxisDecimals, xAxisLog, t]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { redraw(); }, [redraw]);

  useEffect(() => {
    const ro = new ResizeObserver(redraw);
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [redraw]);

  const startDrag = (e) => {
    if (e.button !== 0) return;
    dragState.current = { startX: e.clientX - pos.x, startY: e.clientY - pos.y };
    const onMove = (e) => {
      if (!dragState.current) return;
      setPos({ x: e.clientX - dragState.current.startX, y: e.clientY - dragState.current.startY });
    };
    const onUp = () => {
      dragState.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    e.preventDefault();
  };

  const exportCanvas = (format) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const mimeType = format === "jpeg" ? "image/jpeg" : "image/png";
    const quality = format === "jpeg" ? 0.95 : undefined;
    // Always composite into an offscreen canvas so we can draw the stats table on top
    const out = document.createElement("canvas");
    out.width = canvas.width; out.height = canvas.height;
    const ctx = out.getContext("2d");
    if (format === "jpeg") {
      ctx.fillStyle = t.canvas || "#0a0f1a";
      ctx.fillRect(0, 0, out.width, out.height);
    }
    ctx.drawImage(canvas, 0, 0);
    // Draw molecule stats table on top if visible
    if (overlayMode && statsTableVisible && allFitResults?.length > 0) {
      const dpr = canvas.width / canvas.getBoundingClientRect().width;
      drawStatsTableOnCanvas(ctx, statsTablePos, allFitResults, selectedCompounds, overlayEditIndex, getCompoundStyle, dpr, statsTableCols);
    }
    const a = document.createElement("a");
    a.href = out.toDataURL(mimeType, quality);
    a.download = `chart_export.${format}`;
    a.click();
  };

  const hdrBtn = {
    padding: "3px 9px", background: "rgba(59,158,255,0.12)", border: "1px solid rgba(59,158,255,0.3)",
    borderRadius: 4, color: t.blue || "#3b9eff", fontSize: 9, cursor: "pointer",
    fontFamily: "'JetBrains Mono', monospace", letterSpacing: 0.5,
  };
  const ctrlBtn = (active, activeColor) => ({
    padding: "2px 7px", background: active ? `${activeColor}18` : (t.btnInactive || "rgba(30,40,60,0.8)"),
    border: `1px solid ${active ? `${activeColor}55` : (t.panelBorder || "rgba(60,100,160,0.15)")}`,
    borderRadius: 4, color: active ? activeColor : (t.textMuted || "rgba(160,190,230,0.45)"),
    fontSize: 9, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace",
  });
  const inputStyle = {
    width: 54, padding: "2px 4px",
    background: t.input || "rgba(10,15,26,0.6)", border: `1px solid ${t.inputBorder || "rgba(60,100,160,0.2)"}`,
    borderRadius: 4, color: t.text || "#c8dcf0", fontSize: 9,
    fontFamily: "'JetBrains Mono', monospace", outline: "none",
  };
  const teal = t.teal || "#00e6b4";
  const hasAxisOverride = axisXMin || axisXMax || axisYMin || axisYMax;

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 1100, pointerEvents: "none" }}>
      <div style={{
        position: "absolute", left: pos.x, top: pos.y,
        background: t.panel, border: `1px solid ${t.panelBorder}`,
        borderRadius: 10, boxShadow: "0 12px 48px rgba(0,0,0,0.7)",
        pointerEvents: "all", display: "flex", flexDirection: "column",
        minWidth: 380, minHeight: 300,
      }}>
        {/* Drag header */}
        <div
          onMouseDown={startDrag}
          style={{
            cursor: "move", padding: "7px 12px", userSelect: "none",
            borderBottom: `1px solid ${t.panelBorder}`,
            display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexShrink: 0,
          }}
        >
          <span style={{ fontSize: 11, fontWeight: 700, color: t.label, fontFamily: "'Space Grotesk', sans-serif" }}>
            Graph Preview
          </span>
          <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
            {["PNG", "JPEG"].map(fmt => (
              <button key={fmt} onClick={() => exportCanvas(fmt.toLowerCase())} style={hdrBtn}>
                ↓ {fmt}
              </button>
            ))}
            <button onClick={onClose} style={{ ...hdrBtn, background: "none", border: "none", color: "rgba(160,190,230,0.5)", fontSize: 14, padding: "0 4px" }}>
              ✕
            </button>
          </div>
        </div>

        {/* Controls toolbar */}
        <div style={{
          padding: "6px 10px", borderBottom: `1px solid ${t.panelBorder}`,
          display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap", flexShrink: 0,
          background: "rgba(0,0,0,0.15)",
        }}>
          {/* Log/Linear X */}
          <button onClick={() => setXAxisLog(v => !v)} style={ctrlBtn(xAxisLog, teal)}>
            {xAxisLog ? "Log X" : "Linear X"}
          </button>

          {/* X range */}
          <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>X:</span>
          <input type="number" value={axisXMin} placeholder="auto" onChange={e => setAxisXMin(e.target.value)} style={inputStyle} />
          <input type="number" value={axisXMax} placeholder="auto" onChange={e => setAxisXMax(e.target.value)} style={inputStyle} />

          {/* Y range */}
          <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>Y:</span>
          <input type="number" value={axisYMin} placeholder="auto" onChange={e => setAxisYMin(e.target.value)} style={{ ...inputStyle, width: 62 }} />
          <input type="number" value={axisYMax} placeholder="auto" onChange={e => setAxisYMax(e.target.value)} style={{ ...inputStyle, width: 62 }} />

          {/* Auto reset */}
          {hasAxisOverride && (
            <button onClick={() => { setAxisXMin(""); setAxisXMax(""); setAxisYMin(""); setAxisYMax(""); }} style={ctrlBtn(false, teal)}>
              ↺ Auto
            </button>
          )}

          {/* Y format */}
          <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace", marginLeft: 4 }}>Y:</span>
          <button onClick={() => setYAxisFormat(f => f === "decimal" ? "scientific" : "decimal")} style={ctrlBtn(yAxisFormat === "scientific", teal)}>
            {yAxisFormat === "scientific" ? "Sci" : "Dec"}
          </button>
          <button onClick={() => setYAxisDecimals(d => Math.max(0, d - 1))} disabled={yAxisDecimals <= 0} style={ctrlBtn(false, teal)}>−</button>
          <span style={{ fontSize: 9, color: t.text, fontFamily: "'JetBrains Mono', monospace", minWidth: 10, textAlign: "center" }}>{yAxisDecimals}</span>
          <button onClick={() => setYAxisDecimals(d => Math.min(6, d + 1))} disabled={yAxisDecimals >= 6} style={ctrlBtn(false, teal)}>+</button>

          {/* Point view */}
          <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace", marginLeft: 4 }}>Pts:</span>
          {[{ key: "individual", label: "Indiv" }, { key: "errorbars", label: "±Err" }].map(opt => (
            <button key={opt.key} onClick={() => setPointView(opt.key)} style={ctrlBtn(pointView === opt.key, teal)}>
              {opt.label}
            </button>
          ))}
          {pointView === "errorbars" && (
            <>
              {[{ key: "sd", label: "SD" }, { key: "sem", label: "SEM" }].map(opt => (
                <button key={opt.key} onClick={() => setErrorBarType(opt.key)} style={ctrlBtn(errorBarType === opt.key, "#ffb432")}>
                  ±{opt.label}
                </button>
              ))}
            </>
          )}

          {/* Stats table toggle (overlay only) */}
          {overlayMode && allFitResults?.length > 0 && (
            <button onClick={() => setStatsTableVisible(v => !v)} style={ctrlBtn(statsTableVisible, teal)}>
              {statsTableVisible ? "Hide Stats" : "Stats"}
            </button>
          )}
        </div>

        {/* Resizable canvas area */}
        <div
          ref={containerRef}
          style={{ resize: "both", overflow: "hidden", width: 760, height: 480, minWidth: 320, minHeight: 200, flexShrink: 0, position: "relative" }}
        >
          <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />

          {/* Floating molecule stats table inside popup */}
          {overlayMode && allFitResults && statsTableVisible && (
            <div style={{
              position: "absolute",
              left: statsTablePos.x,
              top: statsTablePos.y,
              background: "rgba(10,15,26,0.88)",
              border: "1px solid rgba(140,170,210,0.2)",
              borderRadius: 8,
              zIndex: 20,
              minWidth: 220,
              backdropFilter: "blur(8px)",
              userSelect: "none",
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              <div
                onMouseDown={e => {
                  popupStatsDragRef.current = { dragging: true, startX: e.clientX, startY: e.clientY, startPosX: statsTablePos.x, startPosY: statsTablePos.y };
                  e.preventDefault();
                }}
                style={{
                  cursor: "grab", padding: "5px 10px",
                  borderBottom: "1px solid rgba(140,170,210,0.12)",
                  display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6,
                }}
              >
                <span style={{ fontSize: 9, fontWeight: 700, color: teal, letterSpacing: 1 }}>MOLECULE STATS</span>
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                  <div style={{ position: "relative" }}>
                    <button
                      onMouseDown={e => e.stopPropagation()}
                      onClick={() => setStatsColPickerOpen(v => !v)}
                      title="Choose columns"
                      style={{ background: "none", border: "none", color: statsColPickerOpen ? teal : "rgba(160,190,230,0.45)", cursor: "pointer", fontSize: 11, lineHeight: 1, padding: "0 3px" }}
                    >⊞</button>
                    {statsColPickerOpen && (
                      <div
                        onMouseDown={e => e.stopPropagation()}
                        style={{
                          position: "absolute", right: 0, top: "100%", marginTop: 4,
                          background: "rgba(12,18,32,0.97)", border: "1px solid rgba(140,170,210,0.25)",
                          borderRadius: 6, padding: "6px 0", zIndex: 50, minWidth: 130,
                          boxShadow: "0 8px 24px rgba(0,0,0,0.6)",
                        }}
                      >
                        {RB_STAT_COLUMNS.filter(c => c.key !== "name").map(col => (
                          <label
                            key={col.key}
                            style={{
                              display: "flex", alignItems: "center", gap: 7,
                              padding: "3px 10px", cursor: "pointer",
                              color: statsTableCols.includes(col.key) ? "rgba(200,220,250,0.9)" : "rgba(140,170,210,0.45)",
                              fontSize: 10,
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={statsTableCols.includes(col.key)}
                              onChange={() => setStatsTableCols(prev =>
                                prev.includes(col.key) ? prev.filter(k => k !== col.key) : [...prev, col.key]
                              )}
                              style={{ accentColor: teal, width: 11, height: 11 }}
                            />
                            {col.label}
                          </label>
                        ))}
                      </div>
                    )}
                  </div>
                  <button
                    onMouseDown={e => e.stopPropagation()}
                    onClick={() => setStatsTableVisible(false)}
                    style={{ background: "none", border: "none", color: "rgba(160,190,230,0.5)", cursor: "pointer", fontSize: 12, lineHeight: 1, padding: "0 2px" }}
                  >×</button>
                </div>
              </div>
              <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse", padding: "4px 0" }}>
                <thead>
                  <tr style={{ color: "rgba(160,190,230,0.45)", textAlign: "left" }}>
                    <th style={{ padding: "4px 8px", fontWeight: 600 }}>Molecule</th>
                    {statsTableCols.map(key => {
                      const col = RB_STAT_COLUMNS.find(c => c.key === key);
                      return col ? <th key={key} style={{ padding: "4px 8px", fontWeight: 600 }}>{col.label}</th> : null;
                    })}
                  </tr>
                </thead>
                <tbody>
                  {allFitResults.map((r, i) => ({ r, i })).filter(({ r }) => !selectedCompounds || selectedCompounds.has(r.name)).map(({ r, i }) => {
                    const s = getCompoundStyle(r.name, i);
                    return (
                      <tr
                        key={r.name}
                        onClick={() => loadCompoundForOverlayEdit && loadCompoundForOverlayEdit(i)}
                        style={{
                          borderTop: "1px solid rgba(140,170,210,0.07)",
                          cursor: "pointer",
                          background: overlayEditIndex === i ? "rgba(255,255,255,0.06)" : "transparent",
                        }}
                      >
                        <td style={{ padding: "3px 8px" }}>
                          <span style={{ color: s.color }}>●</span>{" "}
                          <span style={{ color: "rgba(200,220,250,0.8)" }}>{r.name.length > 12 ? r.name.slice(0, 11) + "…" : r.name}</span>
                        </td>
                        {statsTableCols.map(key => {
                          const col = RB_STAT_COLUMNS.find(c => c.key === key);
                          const isDim = key === "model" || key === "n";
                          return col ? (
                            <td key={key} style={{ padding: "3px 8px", color: isDim ? "rgba(160,190,230,0.4)" : "rgba(200,220,250,0.7)" }}>
                              {col.fmt(r)}
                            </td>
                          ) : null;
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div style={{ padding: "4px 10px", fontSize: 8, color: "rgba(140,170,210,0.35)", fontFamily: "'JetBrains Mono', monospace", flexShrink: 0 }}>
          Drag corner to resize · export captures current size
        </div>
      </div>
    </div>
  );
}

// Column width hints for canvas rendering (pixels before DPR scaling)
const STATS_COL_WIDTHS = {
  name: 100, ec50: 68, bio_ec50: 68, hill: 44, top: 50, bottom: 58,
  s_param: 44, r2: 48, rmse: 58, aicc: 48, bic: 44, ssr: 58, model: 44, n: 28,
};

// ── Molecule stats table canvas renderer (used for PNG/JPEG export) ───
function drawStatsTableOnCanvas(ctx, tablePos, allFitResults, selectedCompounds, overlayEditIndex, getCompoundStyle, dpr, activeCols) {
  const colKeys = ["name", ...(activeCols || ["ec50", "hill", "r2", "model"])];
  const s = dpr;
  const x = Math.round(tablePos.x * s);
  const y = Math.round(tablePos.y * s);
  const padX = 8 * s;
  const dragH = 22 * s;
  const colH = 18 * s;
  const rowH = 19 * s;
  const COLS = colKeys.map(key => {
    const def = RB_STAT_COLUMNS.find(c => c.key === key);
    return { key, label: def?.label ?? key, w: (STATS_COL_WIDTHS[key] ?? 60) * s, fmt: def?.fmt };
  });
  const tableW = COLS.reduce((sum, c) => sum + c.w, 0) + padX;
  const filteredRows = allFitResults.map((r, i) => ({ r, i })).filter(({ r }) => !selectedCompounds || selectedCompounds.has(r.name));
  const totalH = dragH + colH + filteredRows.length * rowH + 4 * s;

  // Rounded background
  const rad = 8 * s;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x + rad, y);
  ctx.lineTo(x + tableW - rad, y);
  ctx.arcTo(x + tableW, y, x + tableW, y + rad, rad);
  ctx.lineTo(x + tableW, y + totalH - rad);
  ctx.arcTo(x + tableW, y + totalH, x + tableW - rad, y + totalH, rad);
  ctx.lineTo(x + rad, y + totalH);
  ctx.arcTo(x, y + totalH, x, y + totalH - rad, rad);
  ctx.lineTo(x, y + rad);
  ctx.arcTo(x, y, x + rad, y, rad);
  ctx.closePath();
  ctx.fillStyle = "rgba(10,15,26,0.92)";
  ctx.fill();
  ctx.strokeStyle = "rgba(140,170,210,0.25)";
  ctx.lineWidth = s;
  ctx.stroke();

  // Header title
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillStyle = "#00e6b4";
  ctx.font = `bold ${7 * s}px 'JetBrains Mono', monospace`;
  ctx.fillText("MOLECULE STATS", x + padX, y + dragH / 2);

  // Header separator
  ctx.strokeStyle = "rgba(140,170,210,0.12)";
  ctx.lineWidth = s;
  ctx.beginPath(); ctx.moveTo(x, y + dragH); ctx.lineTo(x + tableW, y + dragH); ctx.stroke();

  // Column headers
  ctx.fillStyle = "rgba(160,190,230,0.5)";
  ctx.font = `${8 * s}px 'JetBrains Mono', monospace`;
  let cx = x + padX;
  COLS.forEach(col => { ctx.fillText(col.label, cx, y + dragH + colH / 2); cx += col.w; });

  // Data rows
  filteredRows.forEach(({ r, i: globalIdx }, rowIdx) => {
    const ry = y + dragH + colH + rowIdx * rowH;
    if (overlayEditIndex === globalIdx) {
      ctx.fillStyle = "rgba(255,255,255,0.06)";
      ctx.fillRect(x, ry, tableW, rowH);
    }
    ctx.strokeStyle = "rgba(140,170,210,0.07)";
    ctx.lineWidth = s;
    ctx.beginPath(); ctx.moveTo(x, ry); ctx.lineTo(x + tableW, ry); ctx.stroke();

    const style = getCompoundStyle(r.name, globalIdx);
    ctx.font = `${9 * s}px 'JetBrains Mono', monospace`;
    cx = x + padX;
    COLS.forEach((col) => {
      if (col.key === "name") {
        ctx.fillStyle = style.color;
        ctx.fillText("●", cx, ry + rowH / 2);
        const nameStr = r.name.length > 12 ? r.name.slice(0, 11) + "…" : r.name;
        ctx.fillStyle = "rgba(200,220,250,0.85)";
        ctx.fillText(nameStr, cx + 11 * s, ry + rowH / 2);
      } else {
        const val = col.fmt ? col.fmt(r) : "—";
        ctx.fillStyle = col.key === "model" || col.key === "n" ? "rgba(160,190,230,0.45)" : "rgba(200,220,250,0.75)";
        ctx.fillText(val, cx, ry + rowH / 2);
      }
      cx += col.w;
    });
  });
  ctx.restore();
}

// ── Chart drawing ─────────────────────────────────────────────────
function drawChart(canvas, xData, yData, fitResult, modelType, options = {}, theme = {}) {
  const { pointView = "individual", errorBarType = "sd", outlierIndices = null, excludedIndices: exclSet = null,
          compoundStyle = null, axisOverride = null, yAxisFormat = "decimal", yAxisDecimals = 2,
          xAxisLog = true } = options;
  const fmtY = v => yAxisFormat === "scientific" ? v.toExponential(yAxisDecimals) : v.toFixed(yAxisDecimals);
  const t = theme;
  const grouped = groupByConcentration(xData, yData);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  // For off-screen canvases (detached from DOM) rect is all-zeros; use the raw pixel
  // dimensions as logical size (effectively dpr=1) so text/points match on-screen charts.
  const onScreen = rect.width > 0;
  const W = onScreen ? rect.width  : canvas.width;
  const H = onScreen ? rect.height : canvas.height;
  const eDpr = onScreen ? dpr : 1;
  canvas.width  = W * eDpr;
  canvas.height = H * eDpr;
  ctx.scale(eDpr, eDpr);

  const pad = { top: 30, right: 30, bottom: 55, left: 70 };

  const posX = xData.filter(x => x > 0);
  const allY = [...yData];

  let xMin, xMax;
  if (xAxisLog) {
    const logX = posX.map(x => Math.log10(x));
    xMin = logX.length > 0 ? Math.floor(Math.min(...logX)) - 0.5 : -2;
    xMax = logX.length > 0 ? Math.ceil(Math.max(...logX)) + 0.5 : 4;
  } else {
    const rawMin = posX.length > 0 ? Math.min(...posX) : 0;
    const rawMax = posX.length > 0 ? Math.max(...posX) : 1;
    const xPadR = (rawMax - rawMin) * 0.1 || rawMax * 0.1 || 1;
    xMin = rawMin - xPadR;
    xMax = rawMax + xPadR;
  }

  // Generate curve points for Y range
  const curveY = [];
  if (fitResult) {
    const modelFn = getModelFn(modelType);
    for (let i = 0; i <= 200; i++) {
      const u = xMin + (xMax - xMin) * (i / 200);
      curveY.push(modelFn(xAxisLog ? Math.pow(10, u) : u, fitResult.params));
    }
  }
  const allYValues = [...allY, ...curveY];

  let yMin = Math.min(...allYValues);
  let yMax = Math.max(...allYValues);
  const yPad = (yMax - yMin) * 0.1 || 1;
  yMin -= yPad;
  yMax += yPad;

  // Apply manual axis overrides (empty string / NaN = auto)
  if (axisOverride) {
    if (axisOverride.xMin != null && !isNaN(axisOverride.xMin)) xMin = axisOverride.xMin;
    if (axisOverride.xMax != null && !isNaN(axisOverride.xMax)) xMax = axisOverride.xMax;
    if (axisOverride.yMin != null && !isNaN(axisOverride.yMin)) yMin = axisOverride.yMin;
    if (axisOverride.yMax != null && !isNaN(axisOverride.yMax)) yMax = axisOverride.yMax;
  }

  // Measure max Y-axis label width and expand left padding to prevent overlap with axis title
  ctx.font = "11px 'JetBrains Mono', monospace";
  let _maxYW = 0;
  for (let i = 0; i <= 6; i++) {
    const _y = yMin + (yMax - yMin) * (i / 6);
    _maxYW = Math.max(_maxYW, ctx.measureText(fmtY(_y)).width);
  }
  pad.left = Math.max(70, Math.ceil(_maxYW) + 38);

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const toCanvasX = (lx) => pad.left + ((lx - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (y) => pad.top + ((yMax - y) / (yMax - yMin)) * plotH;

  // Store coordinate metadata on canvas for tooltip hit-testing
  canvas._chartMeta = { pad, plotW, plotH, xMin, xMax, yMin, yMax, W, H, toCanvasX, toCanvasY, errorBarGroups: [], theme: t, xAxisLog };

  // Background
  ctx.fillStyle = t.canvas || "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = t.grid || "rgba(100,140,180,0.08)";
  ctx.lineWidth = 1;
  const xTicks = 6;
  if (xAxisLog) {
    for (let lx = Math.ceil(xMin); lx <= Math.floor(xMax); lx++) {
      const cx = toCanvasX(lx);
      ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
    }
  } else {
    for (let i = 0; i <= xTicks; i++) {
      const cx = toCanvasX(xMin + (xMax - xMin) * (i / xTicks));
      ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
    }
  }
  const yTicks = 6;
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + (yMax - yMin) * (i / yTicks);
    const cy = toCanvasY(y);
    ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + plotW, cy); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = t.axis || "rgba(140,170,210,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = t.axisLabel || "rgba(160,190,230,0.6)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  if (xAxisLog) {
    for (let lx = Math.ceil(xMin); lx <= Math.floor(xMax); lx++) {
      ctx.fillText(`10^${lx}`, toCanvasX(lx), pad.top + plotH + 18);
    }
  } else {
    for (let i = 0; i <= xTicks; i++) {
      const x = xMin + (xMax - xMin) * (i / xTicks);
      ctx.fillText(+x.toPrecision(3), toCanvasX(x), pad.top + plotH + 18);
    }
  }
  ctx.textAlign = "right";
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + (yMax - yMin) * (i / yTicks);
    ctx.fillText(fmtY(y), pad.left - 8, toCanvasY(y) + 4);
  }

  // Axis titles
  ctx.fillStyle = t.label || "rgba(180,210,240,0.7)";
  ctx.font = "12px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText(xAxisLog ? "Concentration (log scale)" : "Concentration", pad.left + plotW / 2, H - 8);
  ctx.save();
  ctx.translate(16, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Response", 0, 0);
  ctx.restore();

  // Fitted curve
  if (fitResult) {
    const modelFn = getModelFn(modelType);
    const curveColor = modelType === "5PL" ? (t.purple || "#a855f7") : (t.blue || "#3b9eff");
    const curveShadow = modelType === "5PL" ? (t.purpleBg || "rgba(168,85,247,0.4)") : (t.blueBg || "rgba(59,158,255,0.4)");
    ctx.strokeStyle = curveColor;
    ctx.lineWidth = 2.0;
    ctx.shadowColor = curveShadow;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const u = xMin + (xMax - xMin) * (i / 200);
      const xVal = xAxisLog ? Math.pow(10, u) : u;
      const y = modelFn(xVal, fitResult.params);
      const cx = toCanvasX(u), cy = toCanvasY(y);
      if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // EC50 line
    const ec50 = fitResult.params[2];
    if (ec50 > 0) {
      const u50 = xAxisLog ? Math.log10(ec50) : ec50;
      if (u50 >= xMin && u50 <= xMax) {
        const midY = modelFn(ec50, fitResult.params);
        const cx = toCanvasX(u50), cy = toCanvasY(midY);
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = t.orangeBorder || "rgba(255,180,50,0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + plotW, cy); ctx.stroke();
        ctx.setLineDash([]);

        // EC50 marker
        ctx.fillStyle = t.orange || "rgba(255,180,50,0.9)";
        ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = t.orange || "rgba(255,180,50,0.7)";
        ctx.font = "10px 'JetBrains Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`EC50: ${ec50.toExponential(3)}`, cx + 10, cy - 8);
      }
    }

    // Biological EC50 for 5PL
    if (modelType === "5PL" && fitResult.bioEC50) {
      const bioEc50 = fitResult.bioEC50;
      if (bioEc50 > 0) {
        const ubec50 = xAxisLog ? Math.log10(bioEc50) : bioEc50;
        if (ubec50 >= xMin && ubec50 <= xMax) {
          const midY = modelFn(bioEc50, fitResult.params);
          const bx = toCanvasX(ubec50), by = toCanvasY(midY);
          ctx.setLineDash([2, 3]);
          ctx.strokeStyle = (t.tealGlow || "rgba(0,230,180,") + "0.4)";
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(bx, pad.top); ctx.lineTo(bx, pad.top + plotH); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(pad.left, by); ctx.lineTo(pad.left + plotW, by); ctx.stroke();
          ctx.setLineDash([]);

          ctx.fillStyle = t.teal || "#00e6b4";
          ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
          ctx.font = "9px 'JetBrains Mono', monospace";
          ctx.textAlign = "left";
          ctx.fillText(`Bio EC50: ${bioEc50.toExponential(3)}`, bx + 10, by + 14);
        }
      }
    }
  }

  // Data points
  const errorBarGroups = []; // collected for tooltip hit-testing
  if (pointView === "errorbars") {
    // Error bar view: mean ± SD or SEM per concentration, excluding excluded points
    grouped.forEach((g) => {
      if (g.x <= 0) return;
      // Filter out excluded indices
      const activeVals = [];
      g.indices.forEach((idx, j) => {
        if (!exclSet || !exclSet.has(idx)) activeVals.push(g.values[j]);
      });
      if (activeVals.length === 0) return;
      const n = activeVals.length;
      const nTotal = g.n;
      const mean = activeVals.reduce((a, b) => a + b, 0) / n;
      const variance = n > 1 ? activeVals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
      const sd = Math.sqrt(variance);
      const sem = n > 1 ? sd / Math.sqrt(n) : 0;
      const cv = mean !== 0 ? (sd / Math.abs(mean)) * 100 : 0;

      const cx = toCanvasX(xAxisLog ? Math.log10(g.x) : g.x);
      const cyMean = toCanvasY(mean);
      const errVal = errorBarType === "sem" ? sem : sd;
      const cyHi = toCanvasY(mean + errVal);
      const cyLo = toCanvasY(mean - errVal);

      // Store for tooltip
      errorBarGroups.push({ x: g.x, cx, cyMean, mean, sd, sem, cv, n, nTotal, errVal, errorBarType });

      // Error bar line
      if (n > 1) {
        ctx.strokeStyle = (t.tealGlow || "rgba(0,230,180,") + "0.55)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx, cyHi); ctx.lineTo(cx, cyLo); ctx.stroke();
        // Caps
        const capW = 4;
        ctx.beginPath(); ctx.moveTo(cx - capW, cyHi); ctx.lineTo(cx + capW, cyHi); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx - capW, cyLo); ctx.lineTo(cx + capW, cyLo); ctx.stroke();
      }

      // Mean point — use compound style if available
      const ptColorEB = compoundStyle?.color ?? (t.teal || "#00e6b4");
      const ptShapeEB = compoundStyle?.shape ?? "circle";
      ctx.fillStyle = ptColorEB;
      drawPoint(ctx, cx, cyMean, 4, ptShapeEB);
    });
  } else {
    // Individual points view
    const ptColor = compoundStyle?.color ?? (t.teal || "#00e6b4");
    const ptShape = compoundStyle?.shape ?? "circle";
    xData.forEach((x, i) => {
      if (x <= 0) return;
      const cx = toCanvasX(xAxisLog ? Math.log10(x) : x), cy = toCanvasY(yData[i]);
      const isExcluded = exclSet && exclSet.has(i);
      const isOutlier = !isExcluded && outlierIndices && outlierIndices.has(i);

      if (isExcluded) {
        // Dimmed with strikethrough
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = t.red || "#ff506a";
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
        // Strikethrough line
        ctx.strokeStyle = (t.redGlow || "rgba(255,80,106,") + "0.5)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy); ctx.stroke();
        ctx.globalAlpha = 1;
      } else if (isOutlier) {
        // Outlier flagged (red + X)
        ctx.fillStyle = t.red || "#ff506a";
        ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
        const s = 6;
        ctx.strokeStyle = (t.redGlow || "rgba(255,80,106,") + "0.7)";
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(cx - s, cy - s); ctx.lineTo(cx + s, cy + s); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx + s, cy - s); ctx.lineTo(cx - s, cy + s); ctx.stroke();
      } else {
        // Normal point — use compound style
        ctx.fillStyle = ptColor;
        drawPoint(ctx, cx, cy, 3.5, ptShape);
      }
    });
  }

  // Attach error bar groups to meta for tooltip
  canvas._chartMeta.errorBarGroups = errorBarGroups;
}

// ── Overlay chart (all compounds on one canvas) ───────────────────
// overlayCompounds: [{ name, xData, yData, fitResult, modelType, color }, ...]
function drawOverlayChart(canvas, overlayCompounds, options = {}, theme = {}) {
  const { pointView = "individual", errorBarType = "sd", axisOverride = null, yAxisFormat = "decimal", yAxisDecimals = 2,
          xAxisLog = true } = options;
  const fmtY = v => yAxisFormat === "scientific" ? v.toExponential(yAxisDecimals) : v.toFixed(yAxisDecimals);
  const t = theme;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const onScreen = rect.width > 0;
  const W = onScreen ? rect.width  : canvas.width;
  const H = onScreen ? rect.height : canvas.height;
  const eDpr = onScreen ? dpr : 1;
  canvas.width  = W * eDpr;
  canvas.height = H * eDpr;
  ctx.scale(eDpr, eDpr);

  const pad = { top: 30, right: 30, bottom: 55, left: 70 };

  // Compute axis ranges across all compounds
  const allPosX = overlayCompounds.flatMap(c => c.xData.filter(x => x > 0));
  const allY = overlayCompounds.flatMap(c => c.yData);

  let xMin, xMax;
  if (xAxisLog) {
    const allLogX = allPosX.map(x => Math.log10(x));
    xMin = allLogX.length > 0 ? Math.floor(Math.min(...allLogX)) - 0.5 : -2;
    xMax = allLogX.length > 0 ? Math.ceil(Math.max(...allLogX)) + 0.5 : 4;
  } else {
    const rawMin = allPosX.length > 0 ? Math.min(...allPosX) : 0;
    const rawMax = allPosX.length > 0 ? Math.max(...allPosX) : 1;
    const xPadR = (rawMax - rawMin) * 0.1 || rawMax * 0.1 || 1;
    xMin = rawMin - xPadR;
    xMax = rawMax + xPadR;
  }

  // Include curve Y values in range
  const curveY = overlayCompounds.flatMap(c => {
    if (!c.fitResult) return [];
    const fn = getModelFn(c.modelType);
    const pts = [];
    for (let i = 0; i <= 100; i++) {
      const u = xMin + (xMax - xMin) * (i / 100);
      pts.push(fn(xAxisLog ? Math.pow(10, u) : u, c.fitResult.params));
    }
    return pts;
  });

  let yMin = Math.min(...allY, ...curveY);
  let yMax = Math.max(...allY, ...curveY);
  const yPad = (yMax - yMin) * 0.1 || 1;
  yMin -= yPad;
  yMax += yPad;

  if (axisOverride) {
    if (axisOverride.xMin != null && !isNaN(axisOverride.xMin)) xMin = axisOverride.xMin;
    if (axisOverride.xMax != null && !isNaN(axisOverride.xMax)) xMax = axisOverride.xMax;
    if (axisOverride.yMin != null && !isNaN(axisOverride.yMin)) yMin = axisOverride.yMin;
    if (axisOverride.yMax != null && !isNaN(axisOverride.yMax)) yMax = axisOverride.yMax;
  }

  // Measure max Y-axis label width and expand left padding to prevent overlap with axis title
  ctx.font = "11px 'JetBrains Mono', monospace";
  let _maxYW = 0;
  for (let i = 0; i <= 6; i++) {
    const _y = yMin + (yMax - yMin) * (i / 6);
    _maxYW = Math.max(_maxYW, ctx.measureText(fmtY(_y)).width);
  }
  pad.left = Math.max(70, Math.ceil(_maxYW) + 38);

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const toCanvasX = (lx) => pad.left + ((lx - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (y) => pad.top + ((yMax - y) / (yMax - yMin)) * plotH;

  canvas._chartMeta = { pad, plotW, plotH, xMin, xMax, yMin, yMax, W, H, toCanvasX, toCanvasY, errorBarGroups: [], theme: t, isOverlay: true, overlayCompounds, xAxisLog };

  // Background
  ctx.fillStyle = t.canvas || "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = t.grid || "rgba(100,140,180,0.08)";
  ctx.lineWidth = 1;
  const xTicks = 6;
  if (xAxisLog) {
    for (let lx = Math.ceil(xMin); lx <= Math.floor(xMax); lx++) {
      const cx = toCanvasX(lx);
      ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
    }
  } else {
    for (let i = 0; i <= xTicks; i++) {
      const cx = toCanvasX(xMin + (xMax - xMin) * (i / xTicks));
      ctx.beginPath(); ctx.moveTo(cx, pad.top); ctx.lineTo(cx, pad.top + plotH); ctx.stroke();
    }
  }
  const yTicks = 6;
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + (yMax - yMin) * (i / yTicks);
    const cy = toCanvasY(y);
    ctx.beginPath(); ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + plotW, cy); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = t.axis || "rgba(140,170,210,0.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + plotH);
  ctx.lineTo(pad.left + plotW, pad.top + plotH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = t.axisLabel || "rgba(160,190,230,0.6)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  if (xAxisLog) {
    for (let lx = Math.ceil(xMin); lx <= Math.floor(xMax); lx++) {
      ctx.fillText(`10^${lx}`, toCanvasX(lx), pad.top + plotH + 18);
    }
  } else {
    for (let i = 0; i <= xTicks; i++) {
      const x = xMin + (xMax - xMin) * (i / xTicks);
      ctx.fillText(+x.toPrecision(3), toCanvasX(x), pad.top + plotH + 18);
    }
  }
  ctx.textAlign = "right";
  for (let i = 0; i <= yTicks; i++) {
    const y = yMin + (yMax - yMin) * (i / yTicks);
    ctx.fillText(fmtY(y), pad.left - 8, toCanvasY(y) + 4);
  }

  // Axis titles
  ctx.fillStyle = t.label || "rgba(180,210,240,0.7)";
  ctx.font = "12px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText(xAxisLog ? "Concentration (log scale)" : "Concentration", pad.left + plotW / 2, H - 8);
  ctx.save();
  ctx.translate(16, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Response", 0, 0);
  ctx.restore();

  // Per-compound rendering
  overlayCompounds.forEach((compound) => {
    const { xData: cX, yData: cY, fitResult: cFit, modelType: cModel, color } = compound;

    // Parse color to rgba for alpha variants
    const hexToRgb = (hex) => {
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return `${r},${g},${b}`;
    };
    const rgb = hexToRgb(color);

    // Fitted curve
    if (cFit) {
      const modelFn = getModelFn(cModel);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.0;
      ctx.shadowColor = `rgba(${rgb},0.35)`;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      for (let i = 0; i <= 200; i++) {
        const u = xMin + (xMax - xMin) * (i / 200);
        const xVal = xAxisLog ? Math.pow(10, u) : u;
        const y = modelFn(xVal, cFit.params);
        const cx = toCanvasX(u), cy = toCanvasY(y);
        if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

    }

    // Data points
    if (pointView === "errorbars") {
      const grouped = groupByConcentration(cX, cY);
      grouped.forEach(g => {
        if (g.x <= 0) return;
        const n = g.n;
        const mean = g.values.reduce((a, b) => a + b, 0) / n;
        const variance = n > 1 ? g.values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1) : 0;
        const sd = Math.sqrt(variance);
        const sem = n > 1 ? sd / Math.sqrt(n) : 0;
        const errVal = errorBarType === "sem" ? sem : sd;
        const pcx = toCanvasX(xAxisLog ? Math.log10(g.x) : g.x);
        const cyMean = toCanvasY(mean);
        const cyHi = toCanvasY(mean + errVal);
        const cyLo = toCanvasY(mean - errVal);
        if (n > 1) {
          ctx.strokeStyle = `rgba(${rgb},0.55)`;
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(pcx, cyHi); ctx.lineTo(pcx, cyLo); ctx.stroke();
          const capW = 4;
          ctx.beginPath(); ctx.moveTo(pcx - capW, cyHi); ctx.lineTo(pcx + capW, cyHi); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(pcx - capW, cyLo); ctx.lineTo(pcx + capW, cyLo); ctx.stroke();
        }
        ctx.fillStyle = color;
        drawPoint(ctx, pcx, cyMean, 4, compound.shape ?? "circle");
      });
    } else {
      cX.forEach((x, i) => {
        if (x <= 0) return;
        const pcx = toCanvasX(xAxisLog ? Math.log10(x) : x), pcy = toCanvasY(cY[i]);
        ctx.fillStyle = `rgba(${rgb},0.75)`;
        drawPoint(ctx, pcx, pcy, 3.5, compound.shape ?? "circle");
      });
    }
  });

}

// ── Residuals chart ───────────────────────────────────────────────
function drawResiduals(canvas, xData, yData, fitResult, modelType, theme = {}) {
  const t = theme;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = { top: 20, right: 30, bottom: 40, left: 70 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const modelFn = getModelFn(modelType);
  const res = xData.map((x, i) => yData[i] - modelFn(x, fitResult.params));
  const logX = xData.filter(x => x > 0).map(x => Math.log10(x));

  let xMin = Math.floor(Math.min(...logX)) - 0.5;
  let xMax = Math.ceil(Math.max(...logX)) + 0.5;
  const rMax = Math.max(Math.abs(Math.min(...res)), Math.abs(Math.max(...res))) * 1.3 || 1;

  const toCanvasX = (lx) => pad.left + ((lx - xMin) / (xMax - xMin)) * plotW;
  const toCanvasY = (r) => pad.top + ((rMax - r) / (2 * rMax)) * plotH;

  ctx.fillStyle = t.canvas || "#0a0f1a";
  ctx.fillRect(0, 0, W, H);

  // Zero line
  ctx.strokeStyle = t.orangeBorder || "rgba(255,180,50,0.3)";
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  const zeroY = toCanvasY(0);
  ctx.beginPath(); ctx.moveTo(pad.left, zeroY); ctx.lineTo(pad.left + plotW, zeroY); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  xData.forEach((x, i) => {
    if (x <= 0) return;
    const lx = Math.log10(x);
    const cx = toCanvasX(lx), cy = toCanvasY(res[i]);
    const color = res[i] >= 0 ? (t.teal || "#00e6b4") : (t.red || "#ff6b8a");
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
    
    // Line to zero
    ctx.strokeStyle = color + "44";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, zeroY); ctx.stroke();
  });

  ctx.fillStyle = t.axisLabel || "rgba(160,190,230,0.6)";
  ctx.font = "11px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.fillText("Residuals", pad.left + plotW / 2, H - 6);
}

// ── Sample Data ───────────────────────────────────────────────────
const EXAMPLE_DATASETS = {
  "Full sigmoid (5 reps, outliers)": `Concentration,Rep1,Rep2,Rep3,Rep4,Rep5
0.001,0.12,0.15,0.10,0.13,0.11
0.003,0.14,0.17,0.13,0.15,0.16
0.01,0.18,0.22,0.19,0.20,0.21
0.03,0.30,0.35,0.28,0.32,0.31
0.1,0.58,0.65,0.55,0.61,0.59
0.3,1.35,1.48,1.30,1.40,1.38
1,3.10,3.28,3.05,3.18,3.12
3,4.20,4.35,4.15,4.28,4.22
10,4.62,4.78,4.55,4.70,6.25
30,4.80,4.92,4.76,4.85,4.82
100,4.88,4.98,4.84,4.90,4.86
300,4.92,5.02,4.88,3.15,4.94`,

  "Incomplete top plateau": `Concentration,Rep1,Rep2,Rep3
0.001,0.08,0.11,0.09
0.003,0.10,0.13,0.11
0.01,0.15,0.19,0.16
0.03,0.28,0.34,0.30
0.1,0.62,0.71,0.65
0.3,1.45,1.60,1.50
1,2.80,3.05,2.88
3,3.65,3.88,3.72`,

  "Incomplete bottom plateau": `Concentration,Rep1,Rep2,Rep3
0.3,1.40,1.55,1.45
1,2.85,3.02,2.90
3,3.90,4.10,3.95
10,4.50,4.68,4.55
30,4.78,4.92,4.82
100,4.90,5.02,4.94
300,4.95,5.06,4.98
1000,4.98,5.08,5.00`,

  "No plateaus (mid-curve only)": `Concentration,Rep1,Rep2,Rep3
0.1,0.55,0.63,0.58
0.3,1.30,1.45,1.35
1,2.70,2.90,2.78
3,3.60,3.80,3.68
10,4.15,4.35,4.22
30,4.45,4.60,4.50`,

  "Steep Hill slope (cooperative)": `Concentration,Rep1,Rep2,Rep3
0.001,0.10,0.12,0.11
0.003,0.11,0.13,0.12
0.01,0.12,0.15,0.13
0.03,0.14,0.18,0.15
0.1,0.20,0.25,0.22
0.3,0.85,1.10,0.95
1,4.50,4.70,4.58
3,4.85,5.00,4.90
10,4.90,5.05,4.95
30,4.92,5.06,4.96
100,4.93,5.07,4.97`,

  "Shallow Hill slope": `Concentration,Rep1,Rep2,Rep3
0.001,0.50,0.58,0.53
0.003,0.65,0.74,0.68
0.01,0.90,1.02,0.95
0.03,1.25,1.38,1.30
0.1,1.75,1.90,1.82
0.3,2.30,2.48,2.38
1,2.90,3.10,2.98
3,3.35,3.55,3.42
10,3.70,3.88,3.78
30,3.95,4.12,4.02
100,4.15,4.30,4.20
300,4.30,4.45,4.35`,

  "Decreasing response": `Concentration,Rep1,Rep2,Rep3
0.001,4.90,5.05,4.95
0.003,4.88,5.02,4.92
0.01,4.82,4.98,4.88
0.03,4.65,4.80,4.72
0.1,4.10,4.30,4.18
0.3,3.05,3.25,3.12
1,1.55,1.72,1.62
3,0.65,0.78,0.70
10,0.25,0.35,0.28
30,0.14,0.20,0.16
100,0.10,0.15,0.12`,

  "Asymmetric (5PL)": `Concentration,Rep1,Rep2,Rep3
0.001,0.22,0.25,0.20
0.003,0.32,0.36,0.30
0.01,0.52,0.58,0.48
0.03,0.85,0.94,0.80
0.1,1.55,1.68,1.48
0.3,2.60,2.75,2.50
1,3.85,3.98,3.78
3,4.52,4.65,4.45
10,4.82,4.95,4.78
30,4.92,5.02,4.88
100,4.96,5.05,4.92
300,4.98,5.06,4.94`,

  "High variability": `Concentration,Rep1,Rep2,Rep3,Rep4,Rep5
0.001,0.05,0.22,0.12,0.18,0.08
0.01,0.10,0.30,0.18,0.25,0.14
0.1,0.40,0.85,0.60,0.72,0.48
1,2.10,3.40,2.70,3.15,2.35
10,4.00,5.20,4.55,4.90,4.15
100,4.50,5.40,4.90,5.15,4.65
1000,4.60,5.50,5.00,5.20,4.75`,

  "Simple (no replicates)": `Concentration,Response
0.001,0.10
0.01,0.15
0.1,0.55
1,2.80
10,4.55
100,4.90
1000,4.98`,
};

// ── Two-curve examples ────────────────────────────────────────────
// Prism-format multi-molecule CSVs, so these load through the same path as an
// uploaded multi-compound file rather than into the single-curve textarea.
//
// Relative potency is a ratio, and a ratio only exists if the two curves have
// the same SHAPE -- otherwise no single factor maps one dose axis onto the
// other. These two pairs are the contrast that makes that concrete. Both share
// the same reference curve and both have the test article about 4x more potent;
// they differ only in whether the test curve's Hill slope matches.
//
// Generated from a 4PL (bottom 6, top 100, EC50 10 nM / 2.5 nM) with seeded
// Gaussian noise at SD 2, then checked through the engine so the stated
// conclusions are the ones the fitter actually reaches.
const EXAMPLE_PAIRS = {
  // Slopes 1.25 vs 1.25. Equivalence testing passes, and relative potency is
  // reportable: RP 3.92, 95% CI 3.61-4.25.
  "Parallel pair (4x potency)": `Concentration (nM),Reference,Reference,Reference,Test,Test,Test
0.03,8,6.1,8.31,8.3,6.41,8.62
0.1,6.03,6.2,6.79,7.38,7.55,8.15
0.3,6.8,5.21,9.77,11.84,10.26,14.81
1,10.92,8.5,11.18,28.6,26.18,28.86
3,19.81,25.34,24.97,55.06,60.59,60.22
10,55.76,55.18,53.98,88.64,88.06,86.86
30,81.3,78.29,82.15,96.26,93.26,97.12
100,97.94,90.49,97.21,102.02,94.57,101.29
300,99.3,96.72,99.54,100.38,97.81,100.63`,

  // Same reference, same ~4x shift, but the test curve is far steeper (fitted
  // Hill 3.23 against 1.19). Equivalence fails on the slope criterion, so the
  // potency ratio -- which still computes, and still looks plausible at 3.97 --
  // is not reportable. That is the failure this example exists to show: the
  // number looks fine, and means nothing.
  "Non-parallel pair (slope differs)": `Concentration (nM),Reference,Reference,Reference,Test,Test,Test
0.03,8,6.1,8.31,7.93,6.03,8.25
0.1,6.03,6.2,6.79,5.74,5.91,6.51
0.3,6.8,5.21,9.77,5.84,4.25,8.81
1,10.92,8.5,11.18,12.08,9.66,12.34
3,19.81,25.34,24.97,61.87,67.4,67.03
10,55.76,55.18,53.98,101.1,100.52,99.33
30,81.3,78.29,82.15,100.22,97.22,101.08
100,97.94,90.49,97.21,102.95,95.49,102.21
300,99.3,96.72,99.54,100.62,98.04,100.86`,
};

const SAMPLE_DATA = EXAMPLE_DATASETS["Full sigmoid (5 reps, outliers)"];

// ── Report Builder ─────────────────────────────────────────────────
// A WYSIWYG report editor: drag/resize/edit items, then export PNG or PDF.

const RB_DRAFT_KEY = "rb-draft";

const RB_PAGE_SIZES = {
  letter: { w: 816, h: 1056 },
  a4:     { w: 794, h: 1123 },
};

function rbFormatParam(v) {
  if (v == null) return "--";
  return Math.abs(v) < 0.01 || Math.abs(v) > 10000 ? v.toExponential(4) : v.toFixed(4);
}

// A confidence interval as a display string. Returns null rather than a
// placeholder so callers can tell "no interval" from "an interval of --",
// which is what decides whether a table grows a third column at all.
function rbFormatCI(ci, dash = "\u2013") {
  if (!ci || ci.lo == null || ci.hi == null) return null;
  if (!isFinite(ci.lo) || !isFinite(ci.hi)) return null;
  return `${rbFormatParam(ci.lo)} ${dash} ${rbFormatParam(ci.hi)}`;
}

// A p-value small enough that its digits are noise reads better as a bound.
function rbFormatP(p) {
  if (p == null || !isFinite(p)) return "--";
  return p < 0.0001 ? "<0.0001" : p.toFixed(4);
}

function rbGetFitParamsRows(fitResult, modelType) {
  if (!fitResult) return [];
  const labels = modelType === "5PL"
    ? ["Bottom", "Hill", "EC50", "Top", "S (asymmetry)"]
    : ["A (min)", "B (slope)", "C (EC50)", "D (max)"];
  // Third element is the interval, which the renderers turn into a column when
  // any row carries one. A fixed parameter has no interval -- it was asserted
  // rather than estimated -- and correctly comes through as null.
  const rows = labels.map((l, i) => [
    l, rbFormatParam(fitResult.params[i]), rbFormatCI(fitResult.ci?.[i]),
  ]);
  if (modelType === "5PL" && fitResult.bioEC50) rows.push(["Bio EC50", rbFormatParam(fitResult.bioEC50), null]);
  return rows;
}

function rbGetGoFRows(fitResult) {
  if (!fitResult) return [];
  const rows = [
    ["R²",          fitResult.r2.toFixed(6)],
    ["RMSE",        fitResult.rmse.toFixed(6)],
    ["SSR",         fitResult.ssr.toExponential(4)],
  ];
  // Sy.x is the residual standard error, in the units of the response, which
  // makes it the one spread figure a reader can compare against their own
  // knowledge of the assay.
  if (fitResult.syx != null) rows.push(["Sy.x", fitResult.syx.toFixed(6)]);
  rows.push(
    ["AICc",        fitResult.aicc?.toFixed(2) ?? "--"],
    ["BIC",         fitResult.bic?.toFixed(2) ?? "--"],
    ["n (data pts)",String(fitResult.n)],
    ["k (params)",  String(fitResult.k)],
  );
  if (fitResult.dof != null) rows.push(["DOF", String(fitResult.dof)]);
  // Whether the model is adequate, which is a different question from how
  // tightly it fits and is not answered by any of the rows above.
  const lof = fitResult.lackOfFit;
  if (lof?.applicable) {
    rows.push(["Lack of fit F", lof.F.toFixed(3)]);
    rows.push(["Lack of fit p", rbFormatP(lof.pValue)]);
  }
  rows.push(["Converged",   fitResult.converged ? "Yes" : "No"]);
  return rows;
}

// Render all report items to an offscreen canvas (used for PNG/PDF export)
async function rbDrawItemsToCanvas(ctx, items, page) {
  for (const item of items) {
    ctx.save();
    const { x, y, width: w, height: h } = item;
    switch (item.type) {
      case "chart-image":
      case "overlay-chart":
      case "mol-chart": {
        if (!item.dataUrl) break;
        await new Promise(resolve => {
          const img = new Image();
          img.onload = () => { ctx.drawImage(img, x, y, w, h); resolve(); };
          img.onerror = resolve;
          img.src = item.dataUrl;
        });
        if (item.caption) {
          ctx.fillStyle = "#555";
          ctx.font = "italic 10px Arial, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(item.caption, x + w / 2, y + h + 14);
        }
        break;
      }
      case "text":
      case "heading": {
        const fs = item.fontSize || 12;
        const weight = item.fontWeight === "bold" ? "bold" : "normal";
        const style  = item.fontStyle  === "italic" ? "italic" : "normal";
        ctx.font = `${style} ${weight} ${fs}px Arial, sans-serif`;
        ctx.fillStyle = item.color || "#1a2a40";
        ctx.textAlign = item.align || "left";
        const tx = item.align === "center" ? x + w / 2 : item.align === "right" ? x + w - 4 : x + 4;
        const words = (item.content || "").split(" ");
        let line = "", ly = y + fs;
        const maxW = w - 8;
        for (const word of words) {
          const test = line ? line + " " + word : word;
          if (line && ctx.measureText(test).width > maxW) {
            ctx.fillText(line, tx, ly); line = word; ly += fs * 1.4;
          } else { line = test; }
        }
        if (line) ctx.fillText(line, tx, ly);
        break;
      }
      case "divider": {
        ctx.strokeStyle = item.color || "#cccccc";
        ctx.lineWidth = item.thickness || 1;
        ctx.beginPath();
        ctx.moveTo(x, y + h / 2);
        ctx.lineTo(x + w, y + h / 2);
        ctx.stroke();
        break;
      }
      case "fit-params":
        rbDrawTableToCanvas(ctx, item, rbGetFitParamsRows(item.fitResult, item.modelType), item.title);
        break;
      case "gof-table":
        rbDrawTableToCanvas(ctx, item, rbGetGoFRows(item.fitResult), item.title);
        break;
      case "stats-table":
        rbDrawStatsTableToCanvas(ctx, item);
        break;
      case "mol-fit-params":
        rbDrawTableToCanvas(ctx, item, rbGetFitParamsRows(item.fitResult, item.modelType), item.title);
        break;
      case "mol-gof":
        rbDrawTableToCanvas(ctx, item, rbGetGoFRows(item.fitResult), item.title);
        break;
      case "mol-raw-data":
        rbDrawRawDataToCanvas(ctx, item);
        break;
      default: break;
    }
    ctx.restore();
  }
}

// Column proportions for a table that carries confidence intervals. Defined
// once because the canvas exporter and the DOM preview must lay out
// identically -- the editor is a WYSIWYG view of the exported page, so the two
// drifting apart is a bug the user only discovers after exporting.
// The interval column is the widest: it holds two numbers and a separator.
const RB_CI_COLS = { label: 0.32, value: 0.26, ci: 0.42 };

function rbDrawTableToCanvas(ctx, item, rows, title) {
  const { x, y, width: w, height: h } = item;
  const titleH = title ? 28 : 0;
  const pad    = 8;
  // A third column appears only if something in this table has an interval to
  // put in it, so a goodness-of-fit table keeps its two-column layout.
  const hasCI  = rows.some(r => r[2] != null);
  const headH  = hasCI ? 14 : 0;
  const lines  = rows.length + (hasCI ? 1 : 0);
  const rowH   = Math.min(22, lines ? (h - titleH) / lines : 22);

  ctx.fillStyle = "#f8fafd"; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#c8d8ec"; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
  if (title) {
    ctx.fillStyle = "#1a3a6a"; ctx.fillRect(x, y, w, titleH);
    ctx.fillStyle = "#fff"; ctx.font = "bold 11px Arial, sans-serif";
    ctx.textAlign = "left"; ctx.fillText(title, x + pad, y + titleH - 9);
  }

  const inner = w - 2 * pad;
  const valueRight = x + pad + inner * (RB_CI_COLS.label + RB_CI_COLS.value);
  const headerH = hasCI ? Math.min(headH, rowH) : 0;

  if (hasCI) {
    // Without a header the third column is an unexplained pair of numbers.
    const hy = y + titleH;
    ctx.fillStyle = "rgba(40,80,140,0.08)"; ctx.fillRect(x, hy, w, headerH);
    ctx.fillStyle = "#1a3a6a"; ctx.font = "bold 8px Arial, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("Value", valueRight, hy + headerH - 4);
    ctx.fillText("95% CI", x + w - pad, hy + headerH - 4);
  }

  rows.forEach(([label, val, ci], i) => {
    const ry = y + titleH + headerH + i * rowH;
    if (i % 2 === 0) { ctx.fillStyle = "rgba(200,220,240,0.18)"; ctx.fillRect(x, ry, w, rowH); }
    ctx.strokeStyle = "rgba(100,140,180,0.12)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(x, ry + rowH); ctx.lineTo(x + w, ry + rowH); ctx.stroke();
    ctx.fillStyle = "#446688"; ctx.font = "10px Arial, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(label, x + pad, ry + rowH - 6);
    ctx.fillStyle = "#1a2a3a"; ctx.font = "bold 10px Arial, sans-serif"; ctx.textAlign = "right";
    ctx.fillText(String(val), hasCI ? valueRight : x + w - pad, ry + rowH - 6);
    if (hasCI) {
      // Dimmer and unbolded: the interval qualifies the estimate beside it
      // rather than competing with it for attention.
      ctx.fillStyle = "#5a7a9a"; ctx.font = "9px Arial, sans-serif";
      ctx.fillText(ci ?? "--", x + w - pad, ry + rowH - 6);
    }
  });
}

// ── Compound summary table column definitions ─────────────────────
// format(r) receives one entry from allFitResults: {name, fitResult, modelType}
const RB_STAT_COLUMNS = [
  { key: "name",     label: "Molecule",   fmt: r => (r.name || "--").slice(0, 20) },
  { key: "ec50",     label: "EC50",       fmt: r => r.fitResult?.params?.[2] > 0 ? r.fitResult.params[2].toExponential(2) : "--" },
  { key: "ec50_ci",  label: "EC50 95% CI",fmt: r => {
      const ci = r.fitResult?.ci?.[2];
      if (!ci || !isFinite(ci.lo) || !isFinite(ci.hi)) return "--";
      return `${ci.lo.toExponential(1)}\u2013${ci.hi.toExponential(1)}`;
    } },
  { key: "bio_ec50", label: "Bio EC50",   fmt: r => r.fitResult?.bioEC50  > 0 ? r.fitResult.bioEC50.toExponential(2) : "--" },
  { key: "hill",     label: "Hill",       fmt: r => r.fitResult?.params?.[1] != null ? r.fitResult.params[1].toFixed(3) : "--" },
  { key: "hill_ci",  label: "Hill 95% CI",fmt: r => {
      const ci = r.fitResult?.ci?.[1];
      if (!ci || !isFinite(ci.lo) || !isFinite(ci.hi)) return "--";
      return `${ci.lo.toFixed(2)}\u2013${ci.hi.toFixed(2)}`;
    } },
  { key: "top",      label: "Top (D)",    fmt: r => r.fitResult?.params?.[3] != null ? r.fitResult.params[3].toFixed(3) : "--" },
  { key: "bottom",   label: "Bottom (A)", fmt: r => r.fitResult?.params?.[0] != null ? r.fitResult.params[0].toFixed(3) : "--" },
  { key: "s_param",  label: "S (asym)",   fmt: r => r.fitResult?.params?.length >= 5 ? r.fitResult.params[4].toFixed(3) : "--" },
  { key: "r2",       label: "R²",         fmt: r => r.fitResult?.r2   != null ? r.fitResult.r2.toFixed(4)   : "--" },
  { key: "rmse",     label: "RMSE",       fmt: r => r.fitResult?.rmse != null ? r.fitResult.rmse.toFixed(4) : "--" },
  { key: "aicc",     label: "AICc",       fmt: r => r.fitResult?.aicc != null ? r.fitResult.aicc.toFixed(2) : "--" },
  { key: "bic",      label: "BIC",        fmt: r => r.fitResult?.bic  != null ? r.fitResult.bic.toFixed(2)  : "--" },
  { key: "ssr",      label: "SSR",        fmt: r => r.fitResult?.ssr  != null ? r.fitResult.ssr.toExponential(2)  : "--" },
  { key: "syx",      label: "Sy.x",       fmt: r => r.fitResult?.syx  != null ? r.fitResult.syx.toPrecision(3)   : "--" },
  { key: "lof_p",    label: "LoF p",      fmt: r => r.fitResult?.lackOfFit?.applicable
      ? (r.fitResult.lackOfFit.pValue < 0.0001 ? "<0.0001" : r.fitResult.lackOfFit.pValue.toFixed(4))
      : "--" },
  { key: "model",    label: "Model",      fmt: r => r.modelType || "--" },
  { key: "n",        label: "n",          fmt: r => r.fitResult?.n != null ? String(r.fitResult.n) : "--" },
];
const RB_DEFAULT_COLUMNS = ["name", "ec50", "hill", "r2"];

// Compute fractional column widths for a given set of column keys.
// "name" (first) gets a larger share; the rest split evenly.
function rbColWidths(keys) {
  const n = keys.length;
  if (!n) return [];
  const nameW = Math.max(0.18, 0.32 - Math.max(0, n - 3) * 0.018);
  const restW  = n > 1 ? (1 - nameW) / (n - 1) : 0;
  return keys.map((k, i) => (i === 0 && k === "name") ? nameW : restW);
}

function rbDrawStatsTableToCanvas(ctx, item) {
  if (!item.data?.length) return;
  const { x, y, width: w, height: h, data, title } = item;
  const colKeys = item.columns?.length ? item.columns : RB_DEFAULT_COLUMNS;
  const colDefs = colKeys.map(k => RB_STAT_COLUMNS.find(c => c.key === k)).filter(Boolean);
  const widths  = rbColWidths(colDefs.map(c => c.key));
  const colW    = widths.map(fw => fw * w);
  const pad = 6, titleH = title ? 28 : 0, colHeaderH = 18;
  const rowH = Math.min(20, (h - titleH - colHeaderH) / Math.max(data.length, 1));

  ctx.fillStyle = "#f8fafd"; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#c8d8ec"; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
  if (title) {
    ctx.fillStyle = "#1a3a6a"; ctx.fillRect(x, y, w, titleH);
    ctx.fillStyle = "#fff"; ctx.font = "bold 11px Arial, sans-serif";
    ctx.textAlign = "left"; ctx.fillText(title, x + pad, y + titleH - 9);
  }
  let cy = y + titleH;
  ctx.fillStyle = "rgba(40,80,140,0.1)"; ctx.fillRect(x, cy, w, colHeaderH);
  let cx = x;
  colDefs.forEach((col, ci) => {
    ctx.fillStyle = "#1a3a6a"; ctx.font = "bold 9px Arial, sans-serif";
    ctx.textAlign = ci === 0 ? "left" : "right";
    ctx.fillText(col.label, ci === 0 ? cx + pad : cx + colW[ci] - pad, cy + colHeaderH - 5);
    cx += colW[ci];
  });
  cy += colHeaderH;
  data.forEach((r, i) => {
    if (i % 2 === 0) { ctx.fillStyle = "rgba(200,220,240,0.18)"; ctx.fillRect(x, cy, w, rowH); }
    cx = x;
    colDefs.forEach((col, ci) => {
      ctx.fillStyle = "#1a2a3a"; ctx.font = "9px Arial, sans-serif";
      ctx.textAlign = ci === 0 ? "left" : "right";
      ctx.fillText(col.fmt(r), ci === 0 ? cx + pad : cx + colW[ci] - pad, cy + rowH - 5);
      cx += colW[ci];
    });
    ctx.strokeStyle = "rgba(100,140,180,0.1)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(x, cy + rowH); ctx.lineTo(x + w, cy + rowH); ctx.stroke();
    cy += rowH;
  });
}

// ── Per-molecule raw data rows ────────────────────────────────────
function rbGetRawDataRows(molFit, molMulti) {
  if (!molFit?.xData?.length) return { columns: [], rows: [] };
  const grouped = groupByConcentration(molFit.xData, molFit.yData);
  const columns = ["Conc (M)", "Mean", "SD", "SEM", "%CV", "n"];
  const fmtNum = (v) => (v == null || isNaN(v)) ? "--" : (Math.abs(v) < 0.01 || Math.abs(v) > 99999) ? v.toExponential(3) : v.toFixed(4);
  const rows = grouped.map(g => {
    const cv = g.mean !== 0 && g.n > 1 ? ((g.sd / Math.abs(g.mean)) * 100).toFixed(1) + "%" : "--";
    return [g.x.toExponential(2), fmtNum(g.mean), g.n > 1 ? fmtNum(g.sd) : "--", g.n > 1 ? fmtNum(g.sem) : "--", cv, String(g.n)];
  });
  return { columns, rows };
}

function rbDrawRawDataToCanvas(ctx, item) {
  const { columns, rows } = item.rawData || { columns: [], rows: [] };
  if (!rows?.length) return;
  const { x, y, width: w, height: h, title } = item;
  const pad = 4, titleH = title ? 28 : 0, colHeaderH = 16;
  const rowH = Math.min(16, (h - titleH - colHeaderH) / Math.max(rows.length, 1));
  const colW = w / columns.length;
  ctx.fillStyle = "#f8fafd"; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "#c8d8ec"; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h);
  if (title) {
    ctx.fillStyle = "#1a3a6a"; ctx.fillRect(x, y, w, titleH);
    ctx.fillStyle = "#fff"; ctx.font = "bold 11px Arial, sans-serif";
    ctx.textAlign = "left"; ctx.fillText(title, x + pad, y + titleH - 9);
  }
  let cy = y + titleH;
  ctx.fillStyle = "rgba(40,80,140,0.1)"; ctx.fillRect(x, cy, w, colHeaderH);
  columns.forEach((col, ci) => {
    ctx.fillStyle = "#1a3a6a"; ctx.font = "bold 7px Arial, sans-serif";
    ctx.textAlign = ci === 0 ? "left" : "right";
    const cx = x + ci * colW;
    ctx.fillText(col, ci === 0 ? cx + pad : cx + colW - pad, cy + colHeaderH - 4);
  });
  cy += colHeaderH;
  rows.forEach((row, ri) => {
    if (ri % 2 === 0) { ctx.fillStyle = "rgba(200,220,240,0.18)"; ctx.fillRect(x, cy, w, rowH); }
    row.forEach((val, ci) => {
      ctx.fillStyle = "#1a2a3a"; ctx.font = "7px Arial, sans-serif";
      ctx.textAlign = ci === 0 ? "left" : "right";
      const cx = x + ci * colW;
      ctx.fillText(String(val), ci === 0 ? cx + pad : cx + colW - pad, cy + rowH - 4);
    });
    ctx.strokeStyle = "rgba(100,140,180,0.1)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(x, cy + rowH); ctx.lineTo(x + w, cy + rowH); ctx.stroke();
    cy += rowH;
  });
}

// ── RBReportItem: one draggable/resizable element on the page ─────
function RBReportItem({ item, isSelected, isEditing, onSelect, onStartEdit, onStopEdit, onChange, onStartDrag, getCompoundStyle }) {
  const HANDLE_POS = {
    nw: { top: -4, left: -4 },
    n:  { top: -4, left: "50%", transform: "translateX(-50%)" },
    ne: { top: -4, right: -4 },
    e:  { top: "50%", right: -4, transform: "translateY(-50%)" },
    se: { bottom: -4, right: -4 },
    s:  { bottom: -4, left: "50%", transform: "translateX(-50%)" },
    sw: { bottom: -4, left: -4 },
    w:  { top: "50%", left: -4, transform: "translateY(-50%)" },
  };
  const CURSORS = { nw: "nw-resize", n: "n-resize", ne: "ne-resize", e: "e-resize", se: "se-resize", s: "s-resize", sw: "sw-resize", w: "w-resize" };

  return (
    <div
      style={{ position: "absolute", left: item.x, top: item.y, width: item.width, height: item.height,
        outline: isSelected ? "2px solid #3b9eff" : "1px dashed rgba(100,140,180,0.12)",
        cursor: "move", boxSizing: "border-box", userSelect: "none" }}
      onMouseDown={e => { if (!isEditing) { onSelect(); onStartDrag(e, item, null); } }}
      onClick={e => { e.stopPropagation(); onSelect(); }}
      onDoubleClick={e => { e.stopPropagation(); if (item.type === "text" || item.type === "heading") onStartEdit(); }}
    >
      <RBItemContent item={item} isEditing={isEditing} onChange={onChange} onStopEdit={onStopEdit} getCompoundStyle={getCompoundStyle} />
      {isSelected && Object.entries(HANDLE_POS).map(([handle, pos]) => (
        <div key={handle}
          style={{ position: "absolute", ...pos, width: 8, height: 8, background: "#fff", border: "2px solid #3b9eff", borderRadius: 2, cursor: CURSORS[handle], zIndex: 10 }}
          onMouseDown={e => { e.stopPropagation(); onStartDrag(e, item, handle); }}
        />
      ))}
    </div>
  );
}

function RBItemContent({ item, isEditing, onChange, onStopEdit, getCompoundStyle }) {
  switch (item.type) {
    case "chart-image":
    case "overlay-chart":
    case "mol-chart":
      return (
        <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {item.dataUrl
            ? <img src={item.dataUrl} style={{ width: "100%", flex: 1, objectFit: "contain", display: "block", minHeight: 0 }} draggable={false} alt="" />
            : <div style={{ flex: 1, background: "#eee", display: "flex", alignItems: "center", justifyContent: "center", color: "#999", fontSize: 12 }}>No chart</div>
          }
          {item.caption && <div style={{ fontSize: 10, color: "#666", textAlign: "center", fontStyle: "italic", padding: "2px 4px", flexShrink: 0 }}>{item.caption}</div>}
        </div>
      );

    case "text":
    case "heading":
      if (isEditing) {
        return (
          <textarea autoFocus value={item.content || ""} onChange={e => onChange({ content: e.target.value })}
            onBlur={onStopEdit} onKeyDown={e => { if (e.key === "Escape") onStopEdit(); e.stopPropagation(); }}
            onClick={e => e.stopPropagation()}
            style={{ width: "100%", height: "100%", resize: "none", border: "none", outline: "none",
              fontSize: item.fontSize || 12, fontWeight: item.fontWeight || "normal",
              fontStyle: item.fontStyle || "normal", color: item.color || "#1a2a40",
              textAlign: item.align || "left", background: "rgba(230,240,255,0.92)",
              padding: 4, boxSizing: "border-box", fontFamily: "Arial, sans-serif" }}
          />
        );
      }
      return (
        <div style={{ width: "100%", height: "100%", overflow: "hidden",
          fontSize: item.fontSize || 12, fontWeight: item.fontWeight || "normal",
          fontStyle: item.fontStyle || "normal", color: item.color || "#1a2a40",
          textAlign: item.align || "left", whiteSpace: "pre-wrap", wordBreak: "break-word",
          padding: 4, fontFamily: "Arial, sans-serif", lineHeight: 1.4 }}>
          {item.content || <span style={{ opacity: 0.3 }}>Double-click to edit</span>}
        </div>
      );

    case "divider":
      return (
        <div style={{ width: "100%", height: "100%", display: "flex", alignItems: "center" }}>
          <div style={{ width: "100%", height: item.thickness || 1, background: item.color || "#cccccc" }} />
        </div>
      );

    case "fit-params":
      return <RBDataTable rows={rbGetFitParamsRows(item.fitResult, item.modelType)} title={item.title} />;
    case "gof-table":
      return <RBDataTable rows={rbGetGoFRows(item.fitResult)} title={item.title} />;
    case "stats-table":
      return <RBStatsTable item={item} getCompoundStyle={getCompoundStyle} />;
    case "mol-fit-params":
      return <RBDataTable rows={rbGetFitParamsRows(item.fitResult, item.modelType)} title={item.title} />;
    case "mol-gof":
      return <RBDataTable rows={rbGetGoFRows(item.fitResult)} title={item.title} />;
    case "mol-raw-data":
      return <RBRawDataTable columns={item.rawData?.columns || []} rows={item.rawData?.rows || []} title={item.title} />;
    default:
      return <div style={{ width: "100%", height: "100%", background: "#eee", display: "flex", alignItems: "center", justifyContent: "center", color: "#999", fontSize: 11 }}>{item.type}</div>;
  }
}

function RBDataTable({ rows, title }) {
  // Mirrors rbDrawTableToCanvas exactly: the editor is a WYSIWYG preview of
  // the exported page, so the two renderers have to make the same layout
  // decision from the same data.
  const hasCI = rows.some(r => r[2] != null);
  const grid = hasCI
    ? `${RB_CI_COLS.label * 100}% ${RB_CI_COLS.value * 100}% ${RB_CI_COLS.ci * 100}%`
    : null;

  return (
    <div style={{ width: "100%", height: "100%", background: "#f8fafd", border: "1px solid #c8d8ec", overflow: "hidden", fontFamily: "Arial, sans-serif", boxSizing: "border-box" }}>
      {title && <div style={{ background: "#1a3a6a", color: "#fff", padding: "5px 8px", fontSize: 11, fontWeight: "bold" }}>{title}</div>}
      {hasCI && (
        <div style={{ display: "grid", gridTemplateColumns: grid, background: "rgba(40,80,140,0.08)", borderBottom: "1px solid rgba(100,140,180,0.15)", padding: "2px 8px" }}>
          <span />
          <span style={{ fontSize: 8, fontWeight: "bold", color: "#1a3a6a", textAlign: "right" }}>Value</span>
          <span style={{ fontSize: 8, fontWeight: "bold", color: "#1a3a6a", textAlign: "right" }}>95% CI</span>
        </div>
      )}
      <div>
        {rows.map(([label, val, ci], i) => (
          <div key={i} style={{
            display: hasCI ? "grid" : "flex",
            gridTemplateColumns: grid ?? undefined,
            justifyContent: hasCI ? undefined : "space-between",
            alignItems: "center", padding: "3px 8px", fontSize: 10,
            background: i % 2 === 0 ? "rgba(200,220,240,0.18)" : "transparent", borderBottom: "1px solid rgba(100,140,180,0.1)" }}>
            <span style={{ color: "#446688" }}>{label}</span>
            <span style={{ color: "#1a2a3a", fontWeight: "bold", fontFamily: "monospace", textAlign: hasCI ? "right" : undefined }}>{val}</span>
            {hasCI && (
              <span style={{ color: "#5a7a9a", fontFamily: "monospace", fontSize: 9, textAlign: "right" }}>{ci ?? "--"}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function RBStatsTable({ item, getCompoundStyle }) {
  const { data, title, columns } = item;
  if (!data) return null;
  const colKeys = columns?.length ? columns : RB_DEFAULT_COLUMNS;
  const colDefs = colKeys.map(k => RB_STAT_COLUMNS.find(c => c.key === k)).filter(Boolean);
  const widths  = rbColWidths(colDefs.map(c => c.key));
  const gridCols = widths.map(w => `${(w * 100).toFixed(1)}%`).join(" ");

  return (
    <div style={{ width: "100%", height: "100%", background: "#f8fafd", border: "1px solid #c8d8ec", overflow: "hidden", fontFamily: "Arial, sans-serif", boxSizing: "border-box" }}>
      {title && <div style={{ background: "#1a3a6a", color: "#fff", padding: "5px 8px", fontSize: 11, fontWeight: "bold" }}>{title}</div>}
      {/* Column headers */}
      <div style={{ display: "grid", gridTemplateColumns: gridCols, background: "rgba(40,80,140,0.08)", borderBottom: "1px solid rgba(100,140,180,0.15)" }}>
        {colDefs.map((col, ci) => (
          <span key={col.key} style={{ fontSize: 9, fontWeight: "bold", color: "#1a3a6a", padding: "3px 6px", textAlign: ci === 0 ? "left" : "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{col.label}</span>
        ))}
      </div>
      {/* Data rows */}
      {data.map((r, ri) => {
        const cs = getCompoundStyle ? getCompoundStyle(r.name, ri) : { color: OVERLAY_COLORS[ri % OVERLAY_COLORS.length] };
        return (
          <div key={ri} style={{ display: "grid", gridTemplateColumns: gridCols,
            background: ri % 2 === 0 ? "rgba(200,220,240,0.15)" : "transparent",
            borderBottom: "1px solid rgba(100,140,180,0.08)" }}>
            {colDefs.map((col, ci) => (
              <span key={col.key} style={{ fontSize: 9, padding: "2px 6px", textAlign: ci === 0 ? "left" : "right", fontFamily: ci === 0 ? "inherit" : "monospace",
                color: ci === 0 ? cs.color : "#1a2a3a",
                fontWeight: ci === 0 ? "bold" : "normal",
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {col.fmt(r)}
              </span>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function RBRawDataTable({ columns, rows, title }) {
  if (!rows?.length) return <div style={{ width: "100%", height: "100%", background: "#f8fafd", border: "1px solid #c8d8ec", display: "flex", alignItems: "center", justifyContent: "center", color: "#999", fontSize: 11 }}>No data</div>;
  const gridCols = columns.map(() => "1fr").join(" ");
  return (
    <div style={{ width: "100%", height: "100%", background: "#f8fafd", border: "1px solid #c8d8ec", overflow: "hidden", fontFamily: "Arial, sans-serif", boxSizing: "border-box" }}>
      {title && <div style={{ background: "#1a3a6a", color: "#fff", padding: "5px 8px", fontSize: 11, fontWeight: "bold" }}>{title}</div>}
      <div style={{ display: "grid", gridTemplateColumns: gridCols, background: "rgba(40,80,140,0.08)", borderBottom: "1px solid rgba(100,140,180,0.15)" }}>
        {columns.map((col, ci) => (
          <span key={ci} style={{ fontSize: 8, fontWeight: "bold", color: "#1a3a6a", padding: "3px 4px", textAlign: ci === 0 ? "left" : "right", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{col}</span>
        ))}
      </div>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: "grid", gridTemplateColumns: gridCols, background: ri % 2 === 0 ? "rgba(200,220,240,0.15)" : "transparent", borderBottom: "1px solid rgba(100,140,180,0.08)" }}>
          {row.map((val, ci) => (
            <span key={ci} style={{ fontSize: 8, padding: "2px 4px", textAlign: ci === 0 ? "left" : "right", fontFamily: "monospace", color: "#1a2a3a", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{val}</span>
          ))}
        </div>
      ))}
    </div>
  );
}

// ── Properties Panel ──────────────────────────────────────────────
function RBPropertiesPanel({ item, onChange, onDelete, onDuplicate, totalPages }) {
  const lbl = { fontSize: 9, color: "rgba(160,190,230,0.5)", display: "block", marginBottom: 2 };
  const inp = { width: "100%", padding: "4px 6px", background: "#1a2035", border: "1px solid rgba(60,100,160,0.22)", borderRadius: 4, color: "#c8daf0", fontSize: 10, fontFamily: "'JetBrains Mono', monospace", outline: "none", boxSizing: "border-box" };
  const sec = (label, children) => (
    <div style={{ marginBottom: 12 }}>
      <div style={{ fontSize: 9, fontWeight: 700, color: "#3b9eff", textTransform: "uppercase", letterSpacing: 1, marginBottom: 5, borderBottom: "1px solid rgba(59,158,255,0.1)", paddingBottom: 3 }}>{label}</div>
      {children}
    </div>
  );

  return (
    <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#c8daf0", textTransform: "uppercase", letterSpacing: 1 }}>Properties</span>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={onDuplicate} style={{ padding: "3px 7px", background: "rgba(59,158,255,0.1)", border: "1px solid rgba(59,158,255,0.25)", borderRadius: 3, color: "rgba(160,190,230,0.6)", fontSize: 9, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace" }}>⧉</button>
          <button onClick={onDelete} style={{ padding: "3px 7px", background: "rgba(255,80,80,0.1)", border: "1px solid rgba(255,80,80,0.25)", borderRadius: 3, color: "rgba(255,100,100,0.7)", fontSize: 11, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace" }}>🗑</button>
        </div>
      </div>

      {sec("Position & Size",
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
          {[["X", "x"], ["Y", "y"], ["W", "width"], ["H", "height"]].map(([lb, key]) => (
            <div key={key}>
              <label style={lbl}>{lb}</label>
              <input type="number" value={Math.round(item[key] ?? 0)} style={inp} onChange={e => onChange({ [key]: Number(e.target.value) })} />
            </div>
          ))}
          {totalPages > 1 && (
            <div style={{ gridColumn: "1 / -1" }}>
              <label style={lbl}>Page</label>
              <select value={item.page ?? 0} onChange={e => onChange({ page: Number(e.target.value) })}
                style={{ ...inp, padding: "3px 5px" }}>
                {Array.from({ length: totalPages }, (_, i) => (
                  <option key={i} value={i}>Page {i + 1}</option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}

      {(item.type === "text" || item.type === "heading") && sec("Text", <>
        <label style={lbl}>Content</label>
        <textarea value={item.content || ""} style={{ ...inp, height: 60, resize: "vertical", marginBottom: 6 }} onChange={e => onChange({ content: e.target.value })} />
        <label style={lbl}>Font size: {item.fontSize || 12}px</label>
        <input type="range" min={7} max={60} value={item.fontSize || 12} onChange={e => onChange({ fontSize: Number(e.target.value) })} style={{ width: "100%", marginBottom: 4, accentColor: "#3b9eff" }} />
        <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
          {[["B", "fontWeight", "bold", "normal"], ["I", "fontStyle", "italic", "normal"]].map(([icon, key, onVal, offVal]) => (
            <button key={key} onClick={() => onChange({ [key]: item[key] === onVal ? offVal : onVal })}
              style={{ flex: 1, padding: "4px 0", background: item[key] === onVal ? "rgba(59,158,255,0.25)" : "rgba(20,30,55,0.7)", border: "1px solid rgba(59,158,255,0.3)", borderRadius: 3, color: "#c8daf0", fontSize: 11, cursor: "pointer", fontStyle: key === "fontStyle" ? "italic" : "normal", fontWeight: key === "fontWeight" ? "bold" : "normal" }}>{icon}</button>
          ))}
          {["left", "center", "right"].map(a => (
            <button key={a} onClick={() => onChange({ align: a })}
              style={{ flex: 1, padding: "4px 0", background: item.align === a ? "rgba(59,158,255,0.25)" : "rgba(20,30,55,0.7)", border: "1px solid rgba(59,158,255,0.3)", borderRadius: 3, color: "#c8daf0", fontSize: 9, cursor: "pointer" }}>
              {a === "left" ? "⬅" : a === "center" ? "⬆" : "➡"}
            </button>
          ))}
        </div>
        <label style={lbl}>Color</label>
        <input type="color" value={item.color || "#1a2a40"} onChange={e => onChange({ color: e.target.value })} style={{ width: "100%", height: 28, border: "none", background: "transparent", cursor: "pointer" }} />
      </>)}

      {(item.type === "chart-image" || item.type === "overlay-chart" || item.type === "mol-chart") && sec("Caption", <>
        <textarea value={item.caption || ""} placeholder="Optional caption…" style={{ ...inp, height: 48, resize: "vertical" }} onChange={e => onChange({ caption: e.target.value })} />
      </>)}

      {(item.type === "fit-params" || item.type === "gof-table") && sec("Table", <>
        <label style={lbl}>Title</label>
        <input type="text" value={item.title || ""} style={inp} onChange={e => onChange({ title: e.target.value })} />
      </>)}

      {(item.type === "mol-fit-params" || item.type === "mol-gof" || item.type === "mol-raw-data") && sec("Molecule Item", <>
        <label style={lbl}>Molecule</label>
        <div style={{ ...inp, background: "rgba(168,85,247,0.1)", border: "1px solid rgba(168,85,247,0.25)", marginBottom: 6, cursor: "default" }}>{item.moleculeName || "--"}</div>
        <label style={lbl}>Title</label>
        <input type="text" value={item.title || ""} style={inp} onChange={e => onChange({ title: e.target.value })} />
      </>)}

      {item.type === "stats-table" && sec("Molecule Table", (() => {
        const colKeys = item.columns?.length ? item.columns : RB_DEFAULT_COLUMNS;
        const usedKeys = new Set(colKeys);
        const available = RB_STAT_COLUMNS.filter(c => !usedKeys.has(c.key));

        const moveCol = (idx, dir) => {
          const next = [...colKeys];
          const swapIdx = idx + dir;
          if (swapIdx < 0 || swapIdx >= next.length) return;
          [next[idx], next[swapIdx]] = [next[swapIdx], next[idx]];
          onChange({ columns: next });
        };
        const removeCol = (key) => {
          if (key === "name") return; // name is mandatory
          onChange({ columns: colKeys.filter(k => k !== key) });
        };
        const addCol = (key) => {
          if (!key || usedKeys.has(key)) return;
          onChange({ columns: [...colKeys, key] });
        };

        const rowSt = { display: "flex", alignItems: "center", gap: 4, marginBottom: 3, padding: "3px 5px", background: "rgba(20,30,60,0.5)", border: "1px solid rgba(60,100,160,0.15)", borderRadius: 3 };
        const iconBtn = (label, onClick, disabled, color = "rgba(160,190,230,0.5)") => (
          <button onClick={onClick} disabled={disabled} style={{ padding: "1px 5px", background: "none", border: "none", color: disabled ? "rgba(80,100,140,0.3)" : color, fontSize: 11, cursor: disabled ? "default" : "pointer", lineHeight: 1 }}>{label}</button>
        );

        return <>
          <label style={lbl}>Title</label>
          <input type="text" value={item.title || ""} style={{ ...inp, marginBottom: 8 }} onChange={e => onChange({ title: e.target.value })} />

          <label style={{ ...lbl, marginBottom: 4 }}>Columns (drag to reorder)</label>
          {colKeys.map((key, idx) => {
            const def = RB_STAT_COLUMNS.find(c => c.key === key);
            if (!def) return null;
            return (
              <div key={key} style={rowSt}>
                <span style={{ flex: 1, fontSize: 9, color: "#c8daf0" }}>{def.label}</span>
                {iconBtn("↑", () => moveCol(idx, -1), idx === 0)}
                {iconBtn("↓", () => moveCol(idx,  1), idx === colKeys.length - 1)}
                {iconBtn("🗑", () => removeCol(key), key === "name", "rgba(255,100,100,0.6)")}
              </div>
            );
          })}

          {available.length > 0 && (
            <div style={{ marginTop: 6 }}>
              <label style={lbl}>Add column</label>
              <select defaultValue="" onChange={e => { addCol(e.target.value); e.target.value = ""; }}
                style={{ ...inp, padding: "3px 5px" }}>
                <option value="">— select —</option>
                {available.map(c => <option key={c.key} value={c.key}>{c.label}</option>)}
              </select>
            </div>
          )}
        </>;
      })())}

      {item.type === "divider" && sec("Divider", <>
        <label style={lbl}>Color</label>
        <input type="color" value={item.color || "#cccccc"} onChange={e => onChange({ color: e.target.value })} style={{ width: "100%", height: 28, border: "none", background: "transparent", cursor: "pointer", marginBottom: 6 }} />
        <label style={lbl}>Thickness: {item.thickness || 1}px</label>
        <input type="range" min={1} max={8} value={item.thickness || 1} onChange={e => onChange({ thickness: Number(e.target.value) })} style={{ width: "100%", accentColor: "#3b9eff" }} />
      </>)}
    </div>
  );
}

// ── Main ReportBuilder component ──────────────────────────────────
function ReportBuilder({ onClose, chartDataUrl, overlayChartDataUrl, fitResult, activeModel, allFitResults, getCompoundStyle, multiData, renderMolChart }) {
  const idRef = useRef(1000);
  const genId = () => `rb-${idRef.current++}`;

  const [pageKey, setPageKey] = useState("letter");
  const page = RB_PAGE_SIZES[pageKey];
  const [zoom, setZoom] = useState(0.72);
  const [exporting, setExporting] = useState(false);
  const [selectedId, setSelectedId] = useState(null);
  const [editingId, setEditingId] = useState(null);
  const [totalPages, setTotalPages] = useState(1);
  const [activePage, setActivePage] = useState(0);

  // Build default layout when component first mounts
  const [items, setItems] = useState(() => {
    const id = () => `rb-${idRef.current++}`;
    const init = [];
    const pw = RB_PAGE_SIZES.letter.w;
    const now = new Date().toLocaleDateString();
    init.push({ id: id(), type: "text", page: 0, x: 40, y: 28, width: pw - 80, height: 44, content: "Bioassay Report", fontSize: 22, fontWeight: "bold", fontStyle: "normal", color: "#0a1a3a", align: "left" });
    init.push({ id: id(), type: "text", page: 0, x: 40, y: 74, width: pw - 80, height: 20, content: `Generated: ${now}  |  Model: ${activeModel || "N/A"}`, fontSize: 10, fontWeight: "normal", fontStyle: "italic", color: "#667788", align: "left" });
    init.push({ id: id(), type: "divider", page: 0, x: 40, y: 100, width: pw - 80, height: 6, color: "#2a4a8a", thickness: 1 });
    if (chartDataUrl) {
      // 388 wide, not 478: the parameters table beside it grew a
      // confidence-interval column and the two would otherwise overlap.
      init.push({ id: id(), type: "chart-image", page: 0, x: 40, y: 114, width: 388, height: 320, dataUrl: chartDataUrl, caption: "" });
    }
    if (fitResult && activeModel) {
      // Size both tables from the rows they will actually contain rather than
      // from a guess at the model's parameter count. The row sets are no longer
      // fixed -- a fit may or may not carry intervals, a lack-of-fit test or a
      // biological EC50 -- and a stale estimate shows up as either a band of
      // empty table or rows squeezed to illegibility.
      const paramRows = rbGetFitParamsRows(fitResult, activeModel).length;
      const gofRows   = rbGetGoFRows(fitResult).length;
      // Rows + the interval column's header + the title bar, plus a little slack.
      const paramsH = paramRows * 22 + 50;
      const gofH    = gofRows * 20 + 34;
      // Wider than the other auto-placed tables: the parameters table carries a
      // confidence-interval column, which does not fit the 238px default.
      init.push({ id: id(), type: "fit-params", page: 0, x: 438, y: 114, width: 338, height: paramsH, fitResult, modelType: activeModel, title: `${activeModel} Parameters` });
      init.push({ id: id(), type: "gof-table", page: 0, x: 538, y: 114 + paramsH + 12, width: 238, height: gofH, fitResult, modelType: activeModel, title: "Goodness of Fit" });
    }
    if (allFitResults?.length > 0) {
      const tableY = chartDataUrl ? 448 : 114;
      init.push({ id: id(), type: "stats-table", page: 0, x: 40, y: tableY, width: pw - 80, height: Math.min(320, 46 + allFitResults.length * 22), data: allFitResults, title: "Molecule Summary", columns: [...RB_DEFAULT_COLUMNS] });
    }
    return init;
  });

  // ── Auto-save drafts ──
  const [draftBanner, setDraftBanner] = useState(null);

  // Check for saved draft on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(RB_DRAFT_KEY);
      if (!raw) return;
      const draft = JSON.parse(raw);
      if (draft?.items?.length > 0 && draft.timestamp) {
        setDraftBanner({ timestamp: draft.timestamp, items: draft.items, pageKey: draft.pageKey, totalPages: draft.totalPages || 1 });
      }
    } catch { /* ignore corrupt data */ }
  }, []);

  // Save every 60 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      try {
        const draft = {
          items: items.map(it => {
            // Strip large dataUrls to stay within localStorage ~5MB
            if ((it.type === "chart-image" || it.type === "overlay-chart" || it.type === "mol-chart") && it.dataUrl) {
              return { ...it, dataUrl: null, _hadDataUrl: true };
            }
            return it;
          }),
          pageKey,
          totalPages,
          timestamp: Date.now(),
        };
        localStorage.setItem(RB_DRAFT_KEY, JSON.stringify(draft));
      } catch { /* quota exceeded or unavailable */ }
    }, 60000);
    return () => clearInterval(interval);
  }, [items, pageKey, totalPages]);

  const handleClose = useCallback(() => {
    try { localStorage.removeItem(RB_DRAFT_KEY); } catch {}
    onClose();
  }, [onClose]);

  const updateItem = useCallback((id, changes) =>
    setItems(prev => prev.map(it => it.id === id ? { ...it, ...changes } : it)), []);

  const deleteItem = useCallback((id) => {
    setItems(prev => prev.filter(it => it.id !== id));
    setSelectedId(s => s === id ? null : s);
  }, []);

  const duplicateItem = useCallback((id) => {
    setItems(prev => {
      const orig = prev.find(it => it.id === id);
      if (!orig) return prev;
      return [...prev, { ...orig, id: genId(), x: orig.x + 20, y: orig.y + 20 }];
    });
  }, []);

  const addItem = useCallback((type) => {
    const newId = genId();
    const pw = page.w;
    const baseDefaults = {
      "chart-image":    { x: 40, y: 40, width: 480, height: 320, dataUrl: chartDataUrl, caption: "" },
      "overlay-chart":  { x: 40, y: 40, width: 480, height: 320, dataUrl: overlayChartDataUrl, caption: "" },
      "text":           { x: 40, y: 40, width: 320, height: 80,  content: "New text block", fontSize: 12, fontWeight: "normal", fontStyle: "normal", color: "#1a2a40", align: "left" },
      "heading":        { x: 40, y: 40, width: 420, height: 50,  content: "Section Heading", fontSize: 18, fontWeight: "bold",   fontStyle: "normal", color: "#0a1a3a", align: "left" },
      "fit-params":     { x: 40, y: 40, width: 340, height: rbGetFitParamsRows(fitResult, activeModel).length * 22 + 50, fitResult, modelType: activeModel, title: `${activeModel || "Fit"} Parameters` },
      "gof-table":      { x: 40, y: 40, width: 260, height: rbGetGoFRows(fitResult).length * 20 + 34, fitResult, modelType: activeModel, title: "Goodness of Fit" },
      "stats-table":    { x: 40, y: 40, width: pw - 80, height: 200, data: allFitResults, title: "Molecule Summary", columns: [...RB_DEFAULT_COLUMNS] },
      "divider":        { x: 40, y: 200, width: pw - 80, height: 6, color: "#cccccc", thickness: 1 },
    };
    setItems(prev => [...prev, { id: newId, type, page: activePage, ...baseDefaults[type] }]);
    setSelectedId(newId);
  }, [chartDataUrl, overlayChartDataUrl, fitResult, activeModel, allFitResults, page.w, activePage]);

  // ── Per-molecule item helpers ──
  const [molPickerOpen, setMolPickerOpen] = useState(false);

  const addMolItem = useCallback((type, molIndex) => {
    const r = allFitResults[molIndex];
    if (!r) return;
    const id = genId();
    const name = r.name || `Molecule ${molIndex + 1}`;
    const pw = page.w;
    let data;
    switch (type) {
      case "mol-chart": {
        const dataUrl = renderMolChart ? renderMolChart(molIndex) : null;
        data = { x: 40, y: 40, width: 480, height: 320, dataUrl, caption: name, moleculeIndex: molIndex, moleculeName: name };
        break;
      }
      case "mol-fit-params": {
        data = { x: 40, y: 40, width: 340, height: rbGetFitParamsRows(r.fitResult, r.modelType).length * 22 + 50,
          fitResult: r.fitResult, modelType: r.modelType,
          title: `${name} \u2014 ${r.modelType || "Fit"} Parameters`, moleculeIndex: molIndex, moleculeName: name };
        break;
      }
      case "mol-gof": {
        data = { x: 40, y: 40, width: 260, height: rbGetGoFRows(r.fitResult).length * 20 + 34,
          fitResult: r.fitResult, modelType: r.modelType,
          title: `${name} \u2014 Goodness of Fit`, moleculeIndex: molIndex, moleculeName: name };
        break;
      }
      case "mol-raw-data": {
        const molMulti = multiData ? multiData[molIndex] : null;
        const rawData = rbGetRawDataRows(r, molMulti);
        const nCols = rawData.columns.length;
        data = { x: 40, y: 40, width: Math.min(pw - 80, nCols * 60 + 40), height: Math.min(400, 46 + rawData.rows.length * 18),
          rawData, title: `${name} \u2014 Raw Data`, moleculeIndex: molIndex, moleculeName: name };
        break;
      }
      default: return;
    }
    setItems(prev => [...prev, { id, type, page: activePage, ...data }]);
    setSelectedId(id);
  }, [allFitResults, multiData, renderMolChart, page.w, activePage]);

  const addAllMolItems = useCallback((molIndex) => {
    const r = allFitResults[molIndex];
    if (!r) return;
    const name = r.name || `Molecule ${molIndex + 1}`;
    const pw = page.w;
    const batch = [];
    let yPos = 40;
    // Chart
    const dataUrl = renderMolChart ? renderMolChart(molIndex) : null;
    batch.push({ id: genId(), type: "mol-chart", page: activePage, x: 40, y: yPos, width: 388, height: 320,
      dataUrl, caption: name, moleculeIndex: molIndex, moleculeName: name });
    // Fit params (right of chart) — sized from the rows it will hold, which now
    // include a confidence-interval column.
    const paramsH = rbGetFitParamsRows(r.fitResult, r.modelType).length * 22 + 50;
    const gofH    = rbGetGoFRows(r.fitResult).length * 20 + 34;
    batch.push({ id: genId(), type: "mol-fit-params", page: activePage, x: 438, y: yPos, width: 338, height: paramsH,
      fitResult: r.fitResult, modelType: r.modelType,
      title: `${name} \u2014 ${r.modelType || "Fit"} Parameters`, moleculeIndex: molIndex, moleculeName: name });
    // GoF (below fit params)
    batch.push({ id: genId(), type: "mol-gof", page: activePage, x: 538, y: yPos + paramsH + 12, width: 238, height: gofH,
      fitResult: r.fitResult, modelType: r.modelType,
      title: `${name} \u2014 Goodness of Fit`, moleculeIndex: molIndex, moleculeName: name });
    yPos += 330;
    // Raw data (below chart)
    const molMulti = multiData ? multiData[molIndex] : null;
    const rawData = rbGetRawDataRows(r, molMulti);
    const nCols = rawData.columns.length;
    batch.push({ id: genId(), type: "mol-raw-data", page: activePage, x: 40, y: yPos,
      width: Math.min(pw - 80, nCols * 60 + 40), height: Math.min(300, 46 + rawData.rows.length * 18),
      rawData, title: `${name} \u2014 Raw Data`, moleculeIndex: molIndex, moleculeName: name });
    setItems(prev => [...prev, ...batch]);
  }, [allFitResults, multiData, renderMolChart, page.w, activePage]);

  // Close molecule picker on outside click
  useEffect(() => {
    if (!molPickerOpen) return;
    const handler = () => setMolPickerOpen(false);
    const timer = setTimeout(() => window.addEventListener("click", handler), 0);
    return () => { clearTimeout(timer); window.removeEventListener("click", handler); };
  }, [molPickerOpen]);

  // Drag & resize via mouse events
  const dragRef = useRef({ active: false, type: null, itemId: null, startX: 0, startY: 0, origX: 0, origY: 0, origW: 0, origH: 0, handle: "" });

  const startDrag = useCallback((e, item, handle) => {
    if (editingId) return;
    e.stopPropagation(); e.preventDefault();
    setSelectedId(item.id);
    dragRef.current = { active: true, type: handle ? "resize" : "move", itemId: item.id, handle: handle || "",
      startX: e.clientX, startY: e.clientY, origX: item.x, origY: item.y, origW: item.width, origH: item.height };
  }, [editingId]);

  useEffect(() => {
    const onMove = (e) => {
      const d = dragRef.current;
      if (!d.active) return;
      const dx = (e.clientX - d.startX) / zoom;
      const dy = (e.clientY - d.startY) / zoom;
      if (d.type === "move") {
        updateItem(d.itemId, {
          x: Math.max(0, Math.min(page.w - d.origW, d.origX + dx)),
          y: Math.max(0, Math.min(page.h - d.origH, d.origY + dy)),
        });
      } else {
        let nx = d.origX, ny = d.origY, nw = d.origW, nh = d.origH;
        const minW = 50, minH = 20;
        if (d.handle.includes("e")) nw = Math.max(minW, d.origW + dx);
        if (d.handle.includes("s")) nh = Math.max(minH, d.origH + dy);
        if (d.handle.includes("w")) { nw = Math.max(minW, d.origW - dx); nx = d.origX + d.origW - nw; }
        if (d.handle.includes("n")) { nh = Math.max(minH, d.origH - dy); ny = d.origY + d.origH - nh; }
        updateItem(d.itemId, { x: nx, y: ny, width: nw, height: nh });
      }
    };
    const onUp = () => { dragRef.current.active = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [zoom, updateItem, page.w, page.h]);

  useEffect(() => {
    const handler = (e) => {
      if (editingId) return;
      if ((e.key === "Delete" || e.key === "Backspace") && selectedId && !["INPUT","TEXTAREA","SELECT"].includes(document.activeElement?.tagName)) {
        deleteItem(selectedId);
      }
      if (e.key === "Escape") { setSelectedId(null); setEditingId(null); }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedId, editingId, deleteItem]);

  // Build offscreen canvases — one per page — for export
  const buildExportCanvases = async () => {
    const scale = 2;
    const canvases = [];
    for (let p = 0; p < totalPages; p++) {
      const canvas = document.createElement("canvas");
      canvas.width = page.w * scale;
      canvas.height = page.h * scale;
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, page.w, page.h);
      const pageItems = items.filter(it => (it.page ?? 0) === p);
      await rbDrawItemsToCanvas(ctx, pageItems, page);
      canvases.push(canvas);
    }
    return canvases;
  };

  const exportPNG = async () => {
    setExporting(true); setSelectedId(null); setEditingId(null);
    await new Promise(r => setTimeout(r, 80));
    try {
      const canvases = await buildExportCanvases();
      const scale = 2;
      const gap = 4 * scale;
      const totalH = canvases.reduce((sum, c) => sum + c.height, 0) + gap * (canvases.length - 1);
      const merged = document.createElement("canvas");
      merged.width = page.w * scale;
      merged.height = totalH;
      const mctx = merged.getContext("2d");
      mctx.fillStyle = "#e0e0e0";
      mctx.fillRect(0, 0, merged.width, merged.height);
      let yOff = 0;
      for (const c of canvases) {
        mctx.drawImage(c, 0, yOff);
        yOff += c.height + gap;
      }
      const a = document.createElement("a");
      a.href = merged.toDataURL("image/png"); a.download = "bioassay_report.png"; a.click();
      try { localStorage.removeItem(RB_DRAFT_KEY); } catch {}
    } finally { setExporting(false); }
  };

  const exportPDF = async () => {
    setExporting(true); setSelectedId(null); setEditingId(null);
    await new Promise(r => setTimeout(r, 80));
    try {
      const canvases = await buildExportCanvases();
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ orientation: "portrait", unit: "px", format: [page.w, page.h] });
      for (let i = 0; i < canvases.length; i++) {
        if (i > 0) doc.addPage([page.w, page.h], "portrait");
        doc.addImage(canvases[i].toDataURL("image/png"), "PNG", 0, 0, page.w, page.h);
      }
      doc.save("bioassay_report.pdf");
      try { localStorage.removeItem(RB_DRAFT_KEY); } catch {}
    } finally { setExporting(false); }
  };

  const selectedItem = items.find(it => it.id === selectedId);

  const tbStyle = (bg, bc) => ({
    padding: "4px 10px", background: bg, border: `1px solid ${bc}`, borderRadius: 4,
    color: "rgba(200,220,255,0.8)", fontSize: 10, cursor: "pointer",
    fontFamily: "'JetBrains Mono', monospace", whiteSpace: "nowrap",
  });

  return (
    <div style={{ position: "fixed", inset: 0, background: "#12172a", zIndex: 2000, display: "flex", flexDirection: "column", fontFamily: "'JetBrains Mono', monospace" }}>

      {/* ── Toolbar ── */}
      <div style={{ height: 46, background: "#080c18", borderBottom: "1px solid rgba(60,100,160,0.25)", display: "flex", alignItems: "center", gap: 5, padding: "0 12px", flexShrink: 0, overflow: "visible" }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: "#3b9eff", marginRight: 6, whiteSpace: "nowrap" }}>Report Builder</span>
        <div style={{ width: 1, height: 24, background: "rgba(60,100,160,0.3)", flexShrink: 0 }} />

        {/* Add-item buttons */}
        {[
          ["text",         "+ Text"],
          ["heading",      "+ Heading"],
          chartDataUrl                                                  && ["chart-image",   "+ Chart"],
          overlayChartDataUrl && overlayChartDataUrl !== chartDataUrl   && ["overlay-chart", "+ Overlay"],
          fitResult                                                     && ["fit-params",    "+ Fit Params"],
          fitResult                                                     && ["gof-table",     "+ GoF Stats"],
          allFitResults?.length > 0                                     && ["stats-table",   "+ Molecule Table"],
          ["divider", "+ Divider"],
        ].filter(Boolean).map(([type, label]) => (
          <button key={type} onClick={() => addItem(type)} style={tbStyle("rgba(25,38,72,0.8)", "rgba(60,100,160,0.3)")}>{label}</button>
        ))}

        {/* Per-molecule dropdown */}
        {allFitResults?.length > 0 && (
          <div style={{ position: "relative" }} onClick={e => e.stopPropagation()}>
            <button onClick={() => setMolPickerOpen(v => !v)}
              style={tbStyle("rgba(168,85,247,0.12)", "rgba(168,85,247,0.35)")}>
              Per Molecule {molPickerOpen ? "\u25B2" : "\u25BC"}
            </button>
            {molPickerOpen && (
              <div style={{ position: "absolute", top: "100%", left: 0, marginTop: 4,
                background: "#12172a", border: "1px solid rgba(168,85,247,0.3)",
                borderRadius: 6, padding: 6, zIndex: 100, width: 300,
                maxHeight: 340, overflowY: "auto",
                boxShadow: "0 8px 32px rgba(0,0,0,0.5)" }}>
                {allFitResults.map((r, idx) => {
                  const cs = getCompoundStyle(r.name, idx);
                  const mbtn = (label, tip, type) => (
                    <button key={type} onClick={() => { addMolItem(type, idx); setMolPickerOpen(false); }}
                      title={tip} style={{ padding: "2px 6px", background: "none",
                        border: "1px solid rgba(60,100,160,0.2)", borderRadius: 3,
                        color: "rgba(160,190,230,0.6)", fontSize: 9, cursor: "pointer",
                        fontFamily: "'JetBrains Mono', monospace" }}>{label}</button>
                  );
                  return (
                    <div key={idx} style={{ display: "flex", alignItems: "center", gap: 4,
                      padding: "4px 6px", borderBottom: "1px solid rgba(60,100,160,0.1)" }}>
                      <span style={{ width: 8, height: 8, borderRadius: "50%", background: cs.color, flexShrink: 0 }} />
                      <span style={{ flex: 1, fontSize: 9, color: "#c8daf0", overflow: "hidden",
                        textOverflow: "ellipsis", whiteSpace: "nowrap", minWidth: 0 }}>{r.name}</span>
                      {mbtn("Plot", "Add plot", "mol-chart")}
                      {mbtn("Model", "Add model params", "mol-fit-params")}
                      {mbtn("GoF", "Add GoF stats", "mol-gof")}
                      {mbtn("Raw", "Add raw data", "mol-raw-data")}
                      <button onClick={() => { addAllMolItems(idx); setMolPickerOpen(false); }}
                        title="Add all" style={{ padding: "2px 6px", background: "rgba(168,85,247,0.1)",
                          border: "1px solid rgba(168,85,247,0.3)", borderRadius: 3,
                          color: "rgba(168,85,247,0.8)", fontSize: 9, cursor: "pointer",
                          fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>ALL</button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        <div style={{ width: 1, height: 24, background: "rgba(60,100,160,0.3)", flexShrink: 0, margin: "0 2px" }} />

        {/* Page size & zoom */}
        <select value={pageKey} onChange={e => setPageKey(e.target.value)} style={{ ...tbStyle("rgba(8,12,24,0.9)", "rgba(60,100,160,0.25)"), padding: "3px 6px" }}>
          <option value="letter">Letter</option>
          <option value="a4">A4</option>
        </select>
        <select value={zoom} onChange={e => setZoom(Number(e.target.value))} style={{ ...tbStyle("rgba(8,12,24,0.9)", "rgba(60,100,160,0.25)"), padding: "3px 6px" }}>
          {[0.5, 0.6, 0.72, 0.85, 1.0, 1.25].map(z => <option key={z} value={z}>{Math.round(z * 100)}%</option>)}
        </select>

        <div style={{ width: 1, height: 24, background: "rgba(60,100,160,0.3)", flexShrink: 0, margin: "0 2px" }} />
        <span style={{ fontSize: 9, color: "rgba(160,190,230,0.6)", whiteSpace: "nowrap", fontFamily: "'JetBrains Mono', monospace" }}>
          Pg {activePage + 1}/{totalPages}
        </span>
        <button onClick={() => { setTotalPages(n => n + 1); setActivePage(totalPages); }}
          style={tbStyle("rgba(0,230,180,0.08)", "rgba(0,230,180,0.3)")}>+ Page</button>
        <button onClick={() => {
          if (totalPages <= 1) return;
          const lastPage = totalPages - 1;
          const hasItems = items.some(it => (it.page ?? 0) === lastPage);
          if (hasItems && !window.confirm(`Page ${lastPage + 1} has items. Remove it?`)) return;
          setItems(prev => prev.filter(it => (it.page ?? 0) !== lastPage));
          setTotalPages(n => n - 1);
          if (activePage >= lastPage) setActivePage(lastPage - 1);
        }} disabled={totalPages <= 1}
          style={tbStyle("rgba(255,70,70,0.08)", "rgba(255,70,70,0.25)")}>- Page</button>

        {selectedId && <>
          <div style={{ width: 1, height: 24, background: "rgba(60,100,160,0.3)", flexShrink: 0, margin: "0 2px" }} />
          <button onClick={() => deleteItem(selectedId)} style={tbStyle("rgba(255,70,70,0.1)", "rgba(255,70,70,0.3)")}>🗑 Delete</button>
          <button onClick={() => duplicateItem(selectedId)} style={tbStyle("rgba(59,158,255,0.1)", "rgba(59,158,255,0.3)")}>⧉ Duplicate</button>
        </>}

        <div style={{ flex: 1 }} />
        <button onClick={exportPNG} disabled={exporting} style={tbStyle("rgba(0,230,180,0.12)", "rgba(0,230,180,0.4)")}>{exporting ? "Exporting…" : "↓ PNG"}</button>
        <button onClick={exportPDF} disabled={exporting} style={tbStyle("rgba(59,158,255,0.12)", "rgba(59,158,255,0.4)")}>{exporting ? "Exporting…" : "↓ PDF"}</button>
        <button onClick={handleClose} style={tbStyle("rgba(50,65,110,0.4)", "rgba(100,130,180,0.3)")}>✕ Close</button>
      </div>

      {/* ── Draft restore banner ── */}
      {draftBanner && (
        <div style={{ padding: "8px 16px", background: "rgba(255,180,50,0.1)",
          borderBottom: "1px solid rgba(255,180,50,0.25)",
          display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
          <span style={{ fontSize: 10, color: "rgba(255,200,100,0.9)", flex: 1,
            fontFamily: "'JetBrains Mono', monospace" }}>
            Draft from {new Date(draftBanner.timestamp).toLocaleString()} available
          </span>
          <button onClick={() => {
            setItems(draftBanner.items.map(it => it.page == null ? { ...it, page: 0 } : it));
            if (draftBanner.pageKey) setPageKey(draftBanner.pageKey);
            setTotalPages(draftBanner.totalPages || 1);
            setActivePage(0);
            setDraftBanner(null);
          }} style={{ padding: "4px 12px", background: "rgba(59,158,255,0.15)",
            border: "1px solid rgba(59,158,255,0.4)", borderRadius: 4,
            color: "#3b9eff", fontSize: 10, cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace" }}>Restore</button>
          <button onClick={() => {
            setDraftBanner(null);
            try { localStorage.removeItem(RB_DRAFT_KEY); } catch {}
          }} style={{ padding: "4px 12px", background: "rgba(255,70,70,0.08)",
            border: "1px solid rgba(255,70,70,0.25)", borderRadius: 4,
            color: "rgba(255,100,100,0.7)", fontSize: 10, cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace" }}>Dismiss</button>
        </div>
      )}

      {/* ── Body: page canvas + optional properties panel ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Scrollable page area */}
        <div style={{ flex: 1, overflow: "auto", display: "flex", justifyContent: "center", padding: 32, background: "#191e2e" }}
          onClick={() => { setSelectedId(null); setEditingId(null); }}>

          {/* Vertical stack of pages */}
          <div style={{ display: "flex", flexDirection: "column", gap: 24, alignItems: "center" }}>
            {Array.from({ length: totalPages }, (_, pageIndex) => {
              const pageItems = items.filter(it => (it.page ?? 0) === pageIndex);
              const isActive = pageIndex === activePage;
              return (
                <div key={pageIndex}
                  style={{ width: page.w * zoom, height: page.h * zoom, background: "white", position: "relative", flexShrink: 0,
                    boxShadow: isActive ? "0 0 0 2px #3b9eff, 0 8px 48px rgba(0,0,0,0.6)" : "0 8px 48px rgba(0,0,0,0.6)" }}
                  onClick={e => { e.stopPropagation(); setActivePage(pageIndex); setSelectedId(null); setEditingId(null); }}>
                  {/* Inner page at natural scale, scaled via CSS transform */}
                  <div style={{ width: page.w, height: page.h, position: "absolute", top: 0, left: 0, transform: `scale(${zoom})`, transformOrigin: "top left" }}>
                    {pageItems.map(item => (
                      <RBReportItem
                        key={item.id}
                        item={item}
                        isSelected={selectedId === item.id}
                        isEditing={editingId === item.id}
                        onSelect={() => { setSelectedId(item.id); setActivePage(pageIndex); }}
                        onStartEdit={() => setEditingId(item.id)}
                        onStopEdit={() => setEditingId(null)}
                        onChange={changes => updateItem(item.id, changes)}
                        onStartDrag={startDrag}
                        getCompoundStyle={getCompoundStyle}
                      />
                    ))}
                  </div>
                  {/* Page number */}
                  <div style={{ position: "absolute", bottom: 4 * zoom, left: 0, right: 0, textAlign: "center",
                    fontSize: 9 * zoom, color: "rgba(0,0,0,0.2)", pointerEvents: "none", fontFamily: "Arial, sans-serif" }}>
                    {pageIndex + 1}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Properties panel */}
        {selectedItem && (
          <div style={{ width: 218, background: "#080c18", borderLeft: "1px solid rgba(60,100,160,0.2)", padding: 12, overflowY: "auto", flexShrink: 0 }}>
            <RBPropertiesPanel
              item={selectedItem}
              onChange={changes => updateItem(selectedItem.id, changes)}
              onDelete={() => deleteItem(selectedItem.id)}
              onDuplicate={() => duplicateItem(selectedItem.id)}
              totalPages={totalPages}
            />
          </div>
        )}
      </div>

      {/* Exporting overlay */}
      {exporting && (
        <div style={{ position: "absolute", inset: 0, background: "rgba(8,12,24,0.75)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 10 }}>
          <span style={{ color: "#00e6b4", fontSize: 14, fontFamily: "'JetBrains Mono', monospace" }}>Rendering export…</span>
        </div>
      )}
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────
export default function BioassayCurveFitter() {
  const [rawData, setRawData] = useState(SAMPLE_DATA);
  const [modelType, setModelType] = useState("Auto");
  const [normalize, setNormalize] = useState(false);
  const [fixedMin, setFixedMin] = useState("");
  const [fixedMax, setFixedMax] = useState("");
  const [fixedHill, setFixedHill] = useState("1");
  const [showPdfModal, setShowPdfModal] = useState(false);
  const [showReportBuilder, setShowReportBuilder] = useState(false);
  const [showGraphPopup, setShowGraphPopup] = useState(false);
  const [rbChartUrl, setRbChartUrl] = useState(null);

  const [pdfSections, setPdfSections] = useState({
    modelInfo: true,
    modelParams: true,
    plot: true,
    rawData: true,
    fitParams: true,
    paramR2: true,
    paramRMSE: true,
    paramSSR: true,
    paramAIC: false,
    paramAICc: true,
    paramBIC: true,
    paramEC50: true,
    paramBioEC50: true,
    paramConverged: true,
    paramNK: true,
    modelComparison: true,
    outlierResults: true,
    backgroundInfo: true,
    normalizationInfo: true,
  });
  const [pdfGenerating, setPdfGenerating] = useState(false);

  // Pre-populate fixed param fields from data when switching to constrained models
  useEffect(() => {
    if (!["1PL", "2PL", "3PL"].includes(modelType)) return;
    // Try to estimate from parsedData, or fall back to raw data parsing
    let yVals = null;
    if (parsedData && parsedData.yData && parsedData.yData.length > 0) {
      yVals = parsedData.yData;
    } else {
      try {
        const { yData } = parseData(rawData);
        if (yData && yData.length > 0) yVals = yData;
      } catch (e) { /* ignore */ }
    }
    if (!yVals) return;

    if (modelType === "1PL" || modelType === "2PL") {
      // Only pre-populate if fields are empty
      if (fixedMin === "") {
        const sorted = [...yVals].sort((a, b) => a - b);
        const lowN = Math.max(1, Math.floor(sorted.length * 0.2));
        const estMin = sorted.slice(0, lowN).reduce((a, b) => a + b, 0) / lowN;
        setFixedMin(estMin.toPrecision(4));
      }
      if (fixedMax === "") {
        const sorted = [...yVals].sort((a, b) => a - b);
        const lowN = Math.max(1, Math.floor(sorted.length * 0.2));
        const estMax = sorted.slice(-lowN).reduce((a, b) => a + b, 0) / lowN;
        setFixedMax(estMax.toPrecision(4));
      }
    }
  }, [modelType]); // only on model switch, not on every data change
  const [fitResult, setFitResult] = useState(null);
  const [error, setError] = useState(null);
  const [parsedData, setParsedData] = useState(null);
  const [showResiduals, setShowResiduals] = useState(false);
  const [weightsType, setWeightsType] = useState("none");

  // Options threaded into every fitModel call. Relative weighting is suppressed
  // on normalised data: min-max scaling puts the lower asymptote at exactly 0,
  // where 1/Y and 1/Y^2 are undefined, and rescaling has destroyed the variance
  // structure those schemes assume in the first place.
  const fitOpts = useMemo(() => ({
    weighting: normalize && weightsType !== "none" && weightsType !== "1/SD^2"
      ? "none"
      : weightsType,
  }), [weightsType, normalize]);

  const [comparison, setComparison] = useState(null); // { fit4PL, fit5PL, selected, reason }
  const [activeModel, setActiveModel] = useState("4PL"); // which model is currently displayed
  const [pointView, setPointView] = useState("individual"); // "individual" or "errorbars"
  const [errorBarType, setErrorBarType] = useState("sd"); // "sd" or "sem"
  const [grubbsAlpha, setGrubbsAlpha] = useState(0.05);
  const [grubbsResults, setGrubbsResults] = useState(null);
  const [showOutliers, setShowOutliers] = useState(false);
  const [excludedIndices, setExcludedIndices] = useState(new Set()); // manually excluded data point indices
  const [selectedGrubbsGroup, setSelectedGrubbsGroup] = useState(null); // concentration key for expanded view
  const [bgRawData, setBgRawData] = useState("");
  const [bgEnabled, setBgEnabled] = useState(false);
  const [bgStats, setBgStats] = useState(null); // { mean, sd, n, values }
  const [theme, setTheme] = useState("dark"); // "dark" or "light"
  const [isMobile, setIsMobile] = useState(false);

  // Responsive breakpoint
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768);
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  // Theme color palettes
  const t = useMemo(() => {
    const dark = {
      bg: "linear-gradient(160deg, #060a12 0%, #0c1324 40%, #0a1020 100%)",
      text: "#c8daf0",
      textMuted: "rgba(160,190,230,0.5)",
      textDim: "rgba(140,170,210,0.4)",
      textFaint: "rgba(140,170,210,0.3)",
      label: "rgba(160,190,230,0.7)",
      labelDim: "rgba(160,190,230,0.6)",
      panel: "rgba(12,20,40,0.8)",
      panelBorder: "rgba(60,100,160,0.15)",
      panelBorderLight: "rgba(60,100,160,0.08)",
      input: "rgba(6,10,20,0.8)",
      inputBorder: "rgba(60,100,160,0.2)",
      canvas: "#0a0f1a",
      grid: "rgba(100,140,180,0.08)",
      axis: "rgba(140,170,210,0.3)",
      axisLabel: "rgba(160,190,230,0.6)",
      teal: "#00e6b4",
      tealBg: "rgba(0,230,180,0.12)",
      tealBorder: "rgba(0,230,180,0.3)",
      tealGlow: "rgba(0,230,180,",
      blue: "#3b9eff",
      blueBg: "rgba(59,158,255,0.15)",
      blueBorder: "rgba(59,158,255,0.3)",
      purple: "#a855f7",
      purpleBg: "rgba(168,85,247,0.15)",
      purpleBorder: "rgba(168,85,247,0.3)",
      orange: "#ffb432",
      orangeBg: "rgba(255,180,50,0.15)",
      orangeBorder: "rgba(255,180,50,0.3)",
      red: "#ff6b8a",
      redBg: "rgba(255,80,106,0.12)",
      redBorder: "rgba(255,80,106,0.3)",
      redGlow: "rgba(255,80,100,",
      tooltip: "rgba(10,16,30,0.92)",
      tooltipBorder: "rgba(80,120,180,0.25)",
      scrollTrack: "rgba(0,0,0,0.2)",
      scrollThumb: "rgba(100,140,200,0.3)",
      btnInactive: "rgba(6,10,20,0.5)",
      btnInactiveBorder: "rgba(60,100,160,0.1)",
      btnInactiveText: "rgba(160,190,230,0.4)",
    };
    const light = {
      bg: "linear-gradient(160deg, #f0f4f8 0%, #e8edf4 40%, #f2f5fa 100%)",
      text: "#1a2a40",
      textMuted: "rgba(60,80,110,0.6)",
      textDim: "rgba(60,80,110,0.5)",
      textFaint: "rgba(60,80,110,0.35)",
      label: "rgba(40,60,90,0.75)",
      labelDim: "rgba(40,60,90,0.6)",
      panel: "rgba(255,255,255,0.85)",
      panelBorder: "rgba(60,100,160,0.15)",
      panelBorderLight: "rgba(60,100,160,0.08)",
      input: "rgba(245,248,252,0.9)",
      inputBorder: "rgba(60,100,160,0.2)",
      canvas: "#f8fafd",
      grid: "rgba(60,100,160,0.08)",
      axis: "rgba(60,80,120,0.25)",
      axisLabel: "rgba(40,60,100,0.55)",
      teal: "#009e7e",
      tealBg: "rgba(0,158,126,0.1)",
      tealBorder: "rgba(0,158,126,0.3)",
      tealGlow: "rgba(0,158,126,",
      blue: "#2563eb",
      blueBg: "rgba(37,99,235,0.1)",
      blueBorder: "rgba(37,99,235,0.3)",
      purple: "#7c3aed",
      purpleBg: "rgba(124,58,237,0.1)",
      purpleBorder: "rgba(124,58,237,0.3)",
      orange: "#d97706",
      orangeBg: "rgba(217,119,6,0.1)",
      orangeBorder: "rgba(217,119,6,0.3)",
      red: "#e11d48",
      redBg: "rgba(225,29,72,0.08)",
      redBorder: "rgba(225,29,72,0.25)",
      redGlow: "rgba(225,29,72,",
      tooltip: "rgba(255,255,255,0.95)",
      tooltipBorder: "rgba(60,100,160,0.2)",
      scrollTrack: "rgba(0,0,0,0.05)",
      scrollThumb: "rgba(60,100,160,0.2)",
      btnInactive: "rgba(240,243,248,0.8)",
      btnInactiveBorder: "rgba(60,100,160,0.12)",
      btnInactiveText: "rgba(60,80,110,0.45)",
    };
    return theme === "dark" ? dark : light;
  }, [theme]);

  const mainCanvasRef = useRef(null);
  const residCanvasRef = useRef(null);
  const tooltipRef = useRef(null);
  const chartContainerRef = useRef(null);

  const parseData = useCallback((text) => {
    const lines = text.trim().split("\n").filter(l => l.trim());
    const xData = [], yData = [];
    let startIdx = 0;

    // Detect format: is it tab-delimited with comma-formatted numbers?
    // e.g. "1000.000000\t47,189.7\t44,534.5"
    const firstLine = lines[0] || "";
    const hasTabDelim = firstLine.includes("\t");

    // Check if header row (first token is non-numeric)
    const firstToken = hasTabDelim
      ? firstLine.split("\t")[0].trim()
      : firstLine.split(/[,\t]/)[0].trim();
    if (firstToken && isNaN(parseFloat(firstToken.replace(/,/g, "")))) startIdx = 1;

    // Helper: parse a number that may have thousands commas (e.g. "47,189.7" → 47189.7)
    const parseNum = (s) => {
      if (!s) return NaN;
      s = s.trim();
      // If the string has commas and a decimal point, treat commas as thousands separators
      // e.g. "47,189.7" → "47189.7"
      // But also handle plain comma-separated values like "0.01,0.5"
      if (s.includes(",") && s.includes(".")) {
        // Thousands-separator pattern: digits,digits with optional decimal
        s = s.replace(/,/g, "");
      }
      return Number(s);
    };

    for (let i = startIdx; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;

      if (hasTabDelim) {
        // Tab-delimited: first column = concentration, remaining = replicate responses
        // Each value may contain commas as thousands separators
        const cols = line.split("\t").map(s => s.trim()).filter(s => s);
        if (cols.length < 2) continue;
        const conc = parseNum(cols[0]);
        if (isNaN(conc)) continue;

        for (let j = 1; j < cols.length; j++) {
          const resp = parseNum(cols[j]);
          if (!isNaN(resp)) {
            xData.push(conc);
            yData.push(resp);
          }
        }
      } else {
        // CSV or space-delimited: try to detect multi-column replicates
        // First, try splitting on comma (standard CSV)
        let parts = line.split(",").map(s => s.trim());
        
        // Check if we have simple two-column CSV (no thousands commas)
        // vs. multi-column with plain numbers
        if (parts.length >= 2) {
          const allNumeric = parts.every(p => !isNaN(Number(p)));
          if (allNumeric) {
            // Simple CSV with multiple columns: col 0 = conc, rest = replicates
            const conc = Number(parts[0]);
            if (!isNaN(conc)) {
              for (let j = 1; j < parts.length; j++) {
                const resp = Number(parts[j]);
                if (!isNaN(resp)) {
                  xData.push(conc);
                  yData.push(resp);
                }
              }
            }
            continue;
          }
        }

        // Fallback: space-delimited
        parts = line.split(/\s+/).map(s => parseNum(s));
        if (parts.length >= 2 && !isNaN(parts[0])) {
          const conc = parts[0];
          for (let j = 1; j < parts.length; j++) {
            if (!isNaN(parts[j])) {
              xData.push(conc);
              yData.push(parts[j]);
            }
          }
        }
      }
    }
    return { xData, yData };
  }, []);

  // Parse background values: accepts a flat list of numbers (any delimiter)
  const parseBgValues = useCallback((text) => {
    if (!text.trim()) return null;
    const values = [];
    // Handle same formats as main parser: tab, comma, space, newline delimited
    // Also handle comma-formatted thousands separators
    const tokens = text.replace(/\n/g, "\t").split(/[\t]+/);
    for (const token of tokens) {
      // Each token might contain comma-separated values or comma-formatted numbers
      const parts = token.split(",").map(s => s.trim()).filter(s => s);
      // Check if it looks like thousands-separated: "47,189.7" → single number
      // vs comma-delimited: "100,200,300" → multiple numbers
      const rejoined = parts.join(",");
      if (parts.length >= 2 && rejoined.includes(".")) {
        // Could be thousands-formatted; try parsing as single number
        const asOne = Number(rejoined.replace(/,/g, ""));
        if (!isNaN(asOne)) { values.push(asOne); continue; }
      }
      // Otherwise treat each comma-part as separate
      for (const p of parts) {
        const v = Number(p.replace(/,/g, ""));
        if (!isNaN(v) && p.trim()) values.push(v);
      }
    }
    if (values.length === 0) return null;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sd = values.length > 1
      ? Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1))
      : 0;
    return { mean, sd, n: values.length, values };
  }, []);

  const runFit = useCallback(() => {
    try {
      setError(null);
      setComparison(null);
      const { xData, yData: yRaw } = parseData(rawData);
      if (xData.length < 4) { setError("Need at least 4 data points"); return; }

      // Background subtraction
      let bgSub = null;
      let yData = yRaw;
      if (bgEnabled && bgRawData.trim()) {
        bgSub = parseBgValues(bgRawData);
        if (!bgSub) { setError("Could not parse background values"); return; }
        setBgStats(bgSub);
        yData = yRaw.map(y => y - bgSub.mean);
      } else {
        setBgStats(null);
      }

      // Normalization: scale to 0-100% using raw min/max
      let normMin = 0, normMax = 1, normalized = false;
      if (normalize) {
        normMin = Math.min(...yData);
        normMax = Math.max(...yData);
        const range = normMax - normMin;
        if (range > 0) {
          yData = yData.map(y => ((y - normMin) / range) * 100);
          normalized = true;
        }
      }

      setParsedData({ xData, yData, yRaw, bgSubtracted: bgSub ? bgSub.mean : 0, normMin, normMax, normalized });

      if (modelType === "Auto") {
        // Fit 4PL and 5PL, compare
        const fit4 = fitModel(xData, yData, model4PL, false, fitOpts);
        const fit5 = xData.length >= 5 ? fitModel(xData, yData, model5PL, true, fitOpts) : null;

        if (!fit4 && !fit5) { setError("Fitting failed for both models. Check your data."); return; }
        if (!fit5) {
          setComparison({ fit4PL: fit4, fit5PL: null, selected: "4PL", reason: "Too few points for 5PL" });
          setActiveModel("4PL");
          setFitResult(fit4);
          return;
        }

        // Compare using AICc (preferred for small n) and BIC
        const deltaAICc = fit4.aicc - fit5.aicc; // positive => 5PL better
        const deltaBIC = fit4.bic - fit5.bic;
        const eParam = fit5.params[4];
        const eNearOne = Math.abs(eParam - 1) < 0.05;

        let selected, reason;
        // A model whose parameters are not separately identifiable must not be
        // recommended, however well it scores. Information criteria reward the
        // 5PL for fitting the residuals more closely even when the asymmetry
        // parameter has run away to a meaningless value -- weighting makes this
        // markedly more likely -- and a null covariance is exactly that signal.
        const fit5Degenerate = fit5.cov == null;
        if (fit5Degenerate) {
          selected = "4PL";
          reason = `4PL preferred: 5PL parameters are not separately identifiable ` +
            `from this data (S=${eParam.toPrecision(4)}), so its ΔAICc=${deltaAICc.toFixed(1)} ` +
            `advantage is not meaningful`;
        } else if (eNearOne) {
          selected = "4PL";
          reason = `5PL asymmetry parameter S≈${eParam.toFixed(3)} (near 1.0); extra parameter not justified`;
        } else if (deltaAICc > 2 && deltaBIC > 0) {
          selected = "5PL";
          reason = `5PL preferred: ΔAICc=${deltaAICc.toFixed(1)} (>2 threshold), ΔBIC=${deltaBIC.toFixed(1)}`;
        } else if (deltaAICc < -2) {
          selected = "4PL";
          reason = `4PL preferred: ΔAICc=${deltaAICc.toFixed(1)} favors simpler model`;
        } else {
          selected = "4PL";
          reason = `Models within ΔAICc=${deltaAICc.toFixed(1)}; 4PL preferred by parsimony`;
        }

        setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason });
        setActiveModel(selected);
        setFitResult(selected === "4PL" ? fit4 : fit5);
      } else if (modelType === "1PL") {
        // Fix A, B=fixedHill, D; fit only C (EC50)
        const aVal = parseFloat(fixedMin), dVal = parseFloat(fixedMax), hVal = parseFloat(fixedHill);
        if (isNaN(aVal) || isNaN(dVal)) { setError("1PL requires min and max asymptote values"); return; }
        if (isNaN(hVal) || hVal === 0) { setError("1PL requires a non-zero Hill slope value"); return; }
        const result = fitConstrainedModel(xData, yData, { 0: aVal, 1: hVal, 3: dVal });
        if (!result) { setError("1PL fitting failed. Check your data."); return; }
        setActiveModel("1PL");
        setFitResult(result);
      } else if (modelType === "2PL") {
        // Fix A and D; fit B and C
        const aVal = parseFloat(fixedMin), dVal = parseFloat(fixedMax);
        if (isNaN(aVal) || isNaN(dVal)) { setError("2PL requires min and max asymptote values"); return; }
        const result = fitConstrainedModel(xData, yData, { 0: aVal, 3: dVal });
        if (!result) { setError("2PL fitting failed. Check your data."); return; }
        setActiveModel("2PL");
        setFitResult(result);
      } else if (modelType === "3PL") {
        // 4PL with B fixed to fixedHill; fit A, C, D
        const hVal = parseFloat(fixedHill);
        if (isNaN(hVal) || hVal === 0) { setError("3PL requires a non-zero Hill slope value"); return; }
        const result = fitConstrainedModel(xData, yData, { 1: hVal });
        if (!result) { setError("3PL fitting failed. Check your data."); return; }
        setActiveModel("3PL");
        setFitResult(result);
      } else {
        // Manual 4PL or 5PL
        if (modelType === "5PL" && xData.length < 5) { setError("Need at least 5 data points for 5PL"); return; }
        const modelFn = getModelFn(modelType);
        const result = fitModel(xData, yData, modelFn, modelType === "5PL", fitOpts);
        if (!result) { setError("Fitting failed to converge. Check your data."); return; }
        setActiveModel(modelType);
        setFitResult(result);
        setComparison(null);
      }
    } catch (e) {
      setError("Error: " + e.message);
    }
  }, [rawData, modelType, parseData, bgEnabled, bgRawData, parseBgValues, normalize, fixedMin, fixedMax, fixedHill, fitOpts]);

  // Merged set of outlier + excluded indices for chart display
  const chartOutlierIndices = useMemo(() => {
    const s = new Set(excludedIndices);
    if (showOutliers && grubbsResults) {
      for (const idx of grubbsResults.outlierIndices) s.add(idx);
    }
    return s.size > 0 ? s : null;
  }, [excludedIndices, showOutliers, grubbsResults]);

  // ── Multi-compound + overlay state (declared early — referenced in useEffect dep arrays below) ──
  const [multiData, setMultiData] = useState(null);
  const [multiIndex, setMultiIndex] = useState(0);
  const [multiCsvError, setMultiCsvError] = useState(null);
  const multiCsvRef = useRef(null);
  const [overlayMode, setOverlayMode] = useState(false);
  const [allFitResults, setAllFitResults] = useState(null);
  const [selectedCompounds, setSelectedCompounds] = useState(null); // null = all selected
  const [overlayEditIndex, setOverlayEditIndex] = useState(null); // index of compound being edited in left panel
  const [navBarCollapsed, setNavBarCollapsed] = useState(false);

  // ── Per-compound styles: { [compoundName]: { color, shape } } ──
  const [compoundStyles, setCompoundStyles] = useState({});

  // ── Manual axis range overrides (empty string = auto) ──
  const [axisXMin, setAxisXMin] = useState("");
  const [axisXMax, setAxisXMax] = useState("");
  const [axisYMin, setAxisYMin] = useState("");
  const [axisYMax, setAxisYMax] = useState("");
  const [yAxisFormat, setYAxisFormat] = useState("decimal"); // "decimal" | "scientific"
  const [yAxisDecimals, setYAxisDecimals] = useState(2);
  const [xAxisLog, setXAxisLog] = useState(true);

  // ── Floating stats table state ──
  const [statsTablePos, setStatsTablePos] = useState({ x: 16, y: 16 });
  const [statsTableVisible, setStatsTableVisible] = useState(true);
  const [statsTableCols, setStatsTableCols] = useState(["ec50", "hill", "r2", "model"]);
  const [statsColPickerOpen, setStatsColPickerOpen] = useState(false);
  const dragRef = useRef({ dragging: false, startX: 0, startY: 0, startPosX: 0, startPosY: 0 });

  // Returns { color, shape } for a compound, falling back to defaults
  const getCompoundStyle = useCallback((name, idx) => ({
    color: compoundStyles[name]?.color ?? OVERLAY_COLORS[idx % OVERLAY_COLORS.length],
    shape: compoundStyles[name]?.shape ?? "circle",
  }), [compoundStyles]);

  // Render a per-molecule chart to a data URL for the Report Builder
  const renderMolChart = useCallback((molIndex) => {
    if (!allFitResults || !allFitResults[molIndex]) return null;
    const r = allFitResults[molIndex];
    if (!r.xData?.length || !r.fitResult) return null;
    const canvas = document.createElement("canvas");
    canvas.width = 960; canvas.height = 640;
    const cStyle = getCompoundStyle(r.name, molIndex);
    drawChart(canvas, r.xData, r.yData, r.fitResult, r.modelType,
      { pointView, errorBarType, compoundStyle: cStyle, yAxisFormat, yAxisDecimals, xAxisLog }, t);
    return canvas.toDataURL("image/png");
  }, [allFitResults, t, pointView, errorBarType, yAxisFormat, yAxisDecimals, xAxisLog, getCompoundStyle]);

  // Global drag tracking for the floating stats table
  useEffect(() => {
    const onMove = (e) => {
      const d = dragRef.current;
      if (!d.dragging) return;
      setStatsTablePos({ x: d.startPosX + (e.clientX - d.startX), y: d.startPosY + (e.clientY - d.startY) });
    };
    const onUp = () => { dragRef.current.dragging = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, []); // empty deps — accesses only refs and setters

  // Draw charts whenever data changes
  useEffect(() => {
    if (!mainCanvasRef.current) return;
    const ao = {
      xMin: axisXMin !== "" ? parseFloat(axisXMin) : null,
      xMax: axisXMax !== "" ? parseFloat(axisXMax) : null,
      yMin: axisYMin !== "" ? parseFloat(axisYMin) : null,
      yMax: axisYMax !== "" ? parseFloat(axisYMax) : null,
    };
    if (overlayMode && allFitResults && allFitResults.length > 0) {
      drawOverlayChart(
        mainCanvasRef.current,
        allFitResults.map((r, i) => ({ ...r, ...getCompoundStyle(r.name, i) })).filter(r => !selectedCompounds || selectedCompounds.has(r.name)),
        { pointView, errorBarType, axisOverride: ao, yAxisFormat, yAxisDecimals, xAxisLog },
        t
      );
    } else if (parsedData) {
      const cStyle = multiData ? getCompoundStyle(multiData[multiIndex].name, multiIndex) : null;
      drawChart(mainCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices, compoundStyle: cStyle, axisOverride: ao, yAxisFormat, yAxisDecimals, xAxisLog }, t);
    } else {
      const canvas = mainCanvasRef.current;
      const ctx = canvas.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      canvas.width = canvas.getBoundingClientRect().width * dpr;
      canvas.height = canvas.getBoundingClientRect().height * dpr;
      ctx.scale(dpr, dpr);
      ctx.fillStyle = t.canvas || "#0a0f1a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [parsedData, fitResult, activeModel, pointView, errorBarType, chartOutlierIndices, excludedIndices, t, overlayMode, allFitResults, compoundStyles, axisXMin, axisXMax, axisYMin, axisYMax, yAxisFormat, yAxisDecimals, xAxisLog, getCompoundStyle, multiData, multiIndex, selectedCompounds]);

  useEffect(() => {
    if (residCanvasRef.current && parsedData && fitResult && showResiduals) {
      drawResiduals(residCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, t);
    }
  }, [parsedData, fitResult, activeModel, showResiduals, t]);

  // Resize handler
  useEffect(() => {
    const handleResize = () => {
      const ao = {
        xMin: axisXMin !== "" ? parseFloat(axisXMin) : null,
        xMax: axisXMax !== "" ? parseFloat(axisXMax) : null,
        yMin: axisYMin !== "" ? parseFloat(axisYMin) : null,
        yMax: axisYMax !== "" ? parseFloat(axisYMax) : null,
      };
      if (mainCanvasRef.current) {
        if (overlayMode && allFitResults && allFitResults.length > 0) {
          drawOverlayChart(
            mainCanvasRef.current,
            allFitResults.map((r, i) => ({ ...r, ...getCompoundStyle(r.name, i) })).filter(r => !selectedCompounds || selectedCompounds.has(r.name)),
            { pointView, errorBarType, axisOverride: ao, yAxisFormat, yAxisDecimals, xAxisLog },
            t
          );
        } else if (parsedData) {
          const cStyle = multiData ? getCompoundStyle(multiData[multiIndex].name, multiIndex) : null;
          drawChart(mainCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, { pointView, errorBarType, outlierIndices: chartOutlierIndices, excludedIndices, compoundStyle: cStyle, axisOverride: ao, yAxisFormat, yAxisDecimals, xAxisLog }, t);
        }
      }
      if (residCanvasRef.current && parsedData && fitResult && showResiduals) {
        drawResiduals(residCanvasRef.current, parsedData.xData, parsedData.yData, fitResult, activeModel, t);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [parsedData, fitResult, activeModel, showResiduals, pointView, errorBarType, chartOutlierIndices, excludedIndices, t, overlayMode, allFitResults, compoundStyles, axisXMin, axisXMax, axisYMin, axisYMax, yAxisFormat, yAxisDecimals, xAxisLog, getCompoundStyle, multiData, multiIndex, selectedCompounds]);

  // Load a compound's CURRENT stored fit into the left panel (no re-fitting).
  // Used in overlay mode to select a compound for editing via the model panel.
  // Declared before the tooltip/click useEffect to avoid TDZ in its dependency array.
  const loadCompoundForOverlayEdit = useCallback((idx) => {
    if (!allFitResults || !multiData || idx == null) return;
    const r = allFitResults[idx];
    const csv = compoundToCSV(multiData[idx]);
    setRawData(csv);
    setComparison(null);
    setError(null);
    setGrubbsResults(null);
    setShowOutliers(false);
    setSelectedGrubbsGroup(null);
    setExcludedIndices(new Set());
    setBgStats(null);
    setParsedData({ xData: r.xData, yData: r.yData, yRaw: r.yData,
                    bgSubtracted: 0, normMin: 0, normMax: 1, normalized: false });
    setFitResult(r.fitResult);
    setActiveModel(r.modelType ?? "4PL");
    setModelType(r.modelType ?? "4PL");
    setOverlayEditIndex(idx);
  }, [allFitResults, multiData]);

  // Tooltip handler for main chart
  useEffect(() => {
    const canvas = mainCanvasRef.current;
    const tooltip = tooltipRef.current;
    if (!canvas || !tooltip) return;

    const handleMouseMove = (e) => {
      const meta = canvas._chartMeta;
      if (!meta || (!parsedData && !meta.isOverlay)) { tooltip.style.display = "none"; return; }

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const { pad, plotW, plotH, xMin, xMax, yMin, yMax } = meta;
      const mt = meta.theme || {};

      // Check if mouse is within plot area
      if (mx < pad.left || mx > pad.left + plotW || my < pad.top || my > pad.top + plotH) {
        tooltip.style.display = "none";
        return;
      }

      // Convert mouse position to data coordinates
      const logXMouse = xMin + (mx - pad.left) / plotW * (xMax - xMin);
      const yMouse = yMax - (my - pad.top) / plotH * (yMax - yMin);

      // ── Overlay mode: find nearest curve or data point among all compounds ──
      if (meta.isOverlay && meta.overlayCompounds) {
        const containerRect = chartContainerRef.current ? chartContainerRef.current.getBoundingClientRect() : rect;
        const xVal = Math.pow(10, logXMouse);
        let bestDist = 20; // px threshold
        let bestCompound = null;
        for (const compound of meta.overlayCompounds) {
          // Data point proximity (euclidean, wins at ≤15 px)
          if (compound.xData) {
            for (let i = 0; i < compound.xData.length; i++) {
              const x = compound.xData[i];
              if (x <= 0) continue;
              const dist = Math.sqrt((mx - meta.toCanvasX(Math.log10(x))) ** 2 + (my - meta.toCanvasY(compound.yData[i])) ** 2);
              if (dist < 15 && dist < bestDist) { bestDist = dist; bestCompound = compound; }
            }
          }
          // Curve proximity (vertical)
          if (!compound.fitResult) continue;
          const modelFn = getModelFn(compound.modelType);
          const yFit = modelFn(xVal, compound.fitResult.params);
          const cyFit = meta.toCanvasY(yFit);
          const dist = Math.abs(my - cyFit);
          if (dist < bestDist) { bestDist = dist; bestCompound = compound; }
        }
        if (bestCompound) {
          const ec50 = bestCompound.fitResult.params[2];
          const ec50Str = ec50 > 0 ? ec50.toExponential(3) : "N/A";
          const yFit = getModelFn(bestCompound.modelType)(xVal, bestCompound.fitResult.params);
          const fmtY = Math.abs(yFit) < 0.01 || Math.abs(yFit) >= 100000 ? yFit.toExponential(3) : yFit.toFixed(1);
          tooltip.innerHTML = `<span style="color:${bestCompound.color};font-weight:600">${bestCompound.name}</span>&nbsp; y: ${fmtY}&nbsp; EC50: ${ec50Str}`;
          let tx = e.clientX - containerRect.left + 14;
          let ty = e.clientY - containerRect.top - 28;
          const tw = tooltip.offsetWidth || 180;
          if (tx + tw > containerRect.width - 8) tx = e.clientX - containerRect.left - tw - 10;
          if (ty < 4) ty = 4;
          tooltip.style.left = tx + "px";
          tooltip.style.top = ty + "px";
          tooltip.style.display = "block";
          canvas.style.cursor = "pointer";
        } else {
          tooltip.style.display = "none";
          canvas.style.cursor = "crosshair";
        }
        return;
      }

      let nearestDist = Infinity;
      let nearestInfo = null;
      const hitRadius = 12;

      // In error bar mode, check proximity to aggregated group points
      if (pointView === "errorbars" && meta.errorBarGroups && meta.errorBarGroups.length > 0) {
        for (const g of meta.errorBarGroups) {
          const dist = Math.sqrt((mx - g.cx) ** 2 + (my - g.cyMean) ** 2);
          if (dist < 16 && dist < nearestDist) {
            nearestDist = dist;
            nearestInfo = { type: "errorbar", group: g };
          }
        }
      } else {
        // Check proximity to individual data points (within 12px)
        parsedData.xData.forEach((x, i) => {
          if (x <= 0) return;
          const lx = Math.log10(x);
          const cx = meta.toCanvasX(lx);
          const cy = meta.toCanvasY(parsedData.yData[i]);
          const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
          if (dist < hitRadius && dist < nearestDist) {
            nearestDist = dist;
            nearestInfo = { type: "point", x, y: parsedData.yData[i], index: i };
          }
        });
      }

      // If no data point nearby, show curve value
      if (!nearestInfo && fitResult) {
        const modelFn = getModelFn(activeModel);
        const xVal = Math.pow(10, logXMouse);
        const yFit = modelFn(xVal, fitResult.params);
        const cyFit = meta.toCanvasY(yFit);
        if (Math.abs(my - cyFit) < 20) {
          nearestInfo = { type: "curve", x: xVal, y: yFit };
        }
      }

      if (nearestInfo) {
        const containerRect = chartContainerRef.current ? chartContainerRef.current.getBoundingClientRect() : rect;

        if (nearestInfo.type === "errorbar") {
          const g = nearestInfo.group;
          const fmtX = g.x < 0.01 || g.x >= 10000 ? g.x.toExponential(3) : g.x.toPrecision(4);
          const fmtMean = Math.abs(g.mean) < 0.01 || Math.abs(g.mean) >= 100000 ? g.mean.toExponential(4) : g.mean.toFixed(1);
          const fmtSD = g.sd < 0.01 ? g.sd.toExponential(2) : g.sd < 100 ? g.sd.toFixed(2) : g.sd.toFixed(0);
          const fmtSEM = g.sem < 0.01 ? g.sem.toExponential(2) : g.sem < 100 ? g.sem.toFixed(2) : g.sem.toFixed(0);
          const cvColor = g.cv > 20 ? (mt.red || "#ff6b8a") : g.cv > 10 ? (mt.orange || "#ffb432") : (mt.teal || "#00e6b4");
          const nLabel = g.n < g.nTotal ? `${g.n}/${g.nTotal}` : `${g.n}`;

          tooltip.innerHTML = [
            `<div style="color:${mt.teal || '#00e6b4'};font-weight:600;margin-bottom:3px">Conc: ${fmtX}</div>`,
            `<div style="display:grid;grid-template-columns:auto auto;gap:1px 10px;font-size:9px">`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">Mean</span><span>${fmtMean}</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">SD</span><span>±${fmtSD}</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">SEM</span><span>±${fmtSEM}</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">%CV</span><span style="color:${cvColor}">${g.cv.toFixed(1)}%</span>`,
            `<span style="color:${mt.textMuted || 'rgba(160,190,230,0.5)'}">n</span><span>${nLabel}</span>`,
            `</div>`,
          ].join("");

          let tx = e.clientX - containerRect.left + 14;
          let ty = e.clientY - containerRect.top - 80;
          const tw = tooltip.offsetWidth || 160;
          if (tx + tw > containerRect.width - 8) tx = e.clientX - containerRect.left - tw - 10;
          if (ty < 4) ty = 4;
          tooltip.style.left = tx + "px";
          tooltip.style.top = ty + "px";
          tooltip.style.display = "block";
        } else {
          const fmtX = nearestInfo.x < 0.01 || nearestInfo.x >= 10000
            ? nearestInfo.x.toExponential(3) : nearestInfo.x.toPrecision(4);
          const fmtY = Math.abs(nearestInfo.y) < 0.01 || Math.abs(nearestInfo.y) >= 100000
            ? nearestInfo.y.toExponential(3) : nearestInfo.y.toFixed(1);

          let label = nearestInfo.type === "curve" ? "Fit" : "Data";
          tooltip.innerHTML = `<span style="color:${nearestInfo.type === "curve" ? (activeModel === "4PL" ? (mt.blue || "#3b9eff") : (mt.purple || "#a855f7")) : (mt.teal || "#00e6b4")}">${label}</span>&nbsp; x: ${fmtX}&nbsp; y: ${fmtY}`;
          
          let tx = e.clientX - containerRect.left + 14;
          let ty = e.clientY - containerRect.top - 28;
          const tw = tooltip.offsetWidth || 160;
          if (tx + tw > containerRect.width - 8) tx = e.clientX - containerRect.left - tw - 10;
          if (ty < 4) ty = 4;
          tooltip.style.left = tx + "px";
          tooltip.style.top = ty + "px";
          tooltip.style.display = "block";
        }
      } else {
        tooltip.style.display = "none";
      }
    };

    const handleMouseLeave = () => {
      tooltip.style.display = "none";
      canvas.style.cursor = "crosshair";
    };

    // Click on overlay canvas: select the nearest compound (curve or data point) for left-panel editing
    const handleClick = (e) => {
      const meta = canvas._chartMeta;
      if (!meta || !meta.isOverlay || !meta.overlayCompounds) return;

      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const { pad, plotW, plotH, xMin, xMax } = meta;

      if (mx < pad.left || mx > pad.left + plotW || my < pad.top || my > pad.top + plotH) return;

      const logXMouse = xMin + (mx - pad.left) / plotW * (xMax - xMin);
      const xVal = Math.pow(10, logXMouse);

      let bestDist = Infinity;
      let bestCompound = null;

      for (const compound of meta.overlayCompounds) {
        // Check data point proximity (euclidean, 15 px)
        if (compound.xData) {
          for (let i = 0; i < compound.xData.length; i++) {
            const x = compound.xData[i];
            if (x <= 0) continue;
            const px = meta.toCanvasX(Math.log10(x));
            const py = meta.toCanvasY(compound.yData[i]);
            const dist = Math.sqrt((mx - px) ** 2 + (my - py) ** 2);
            if (dist < 15 && dist < bestDist) { bestDist = dist; bestCompound = compound; }
          }
        }
        // Check curve proximity (vertical, 12 px)
        if (compound.fitResult) {
          const yFit = getModelFn(compound.modelType)(xVal, compound.fitResult.params);
          const dist = Math.abs(my - meta.toCanvasY(yFit));
          if (dist < 12 && dist < bestDist) { bestDist = dist; bestCompound = compound; }
        }
      }

      if (bestCompound && allFitResults) {
        const idx = allFitResults.findIndex(r => r.name === bestCompound.name);
        if (idx !== -1) loadCompoundForOverlayEdit(idx);
      }
    };

    canvas.addEventListener("mousemove", handleMouseMove);
    canvas.addEventListener("mouseleave", handleMouseLeave);
    canvas.addEventListener("click", handleClick);
    return () => {
      canvas.removeEventListener("mousemove", handleMouseMove);
      canvas.removeEventListener("mouseleave", handleMouseLeave);
      canvas.removeEventListener("click", handleClick);
    };
  }, [parsedData, fitResult, activeModel, pointView, overlayMode, allFitResults, loadCompoundForOverlayEdit]);

  const paramLabels = activeModel === "5PL"
    ? ["Bottom", "Hill", "EC50", "Top", "S (asymmetry)"]
    : ["A (min)", "B (slope)", "C (EC50)", "D (max)"];

  // What the replicates say about the variance structure, so the user can see
  // whether the weighting they picked matches their data. Reported for every
  // fit, weighted or not; deliberately a hint, not an override, since with a
  // handful of doses the exponent is imprecise.
  const varianceHint = useMemo(() => {
    const v = fitResult?.weighting?.variance;
    if (!v) return null;
    if (v.theta == null) return null;
    const label = { none: "no weighting", "1/Y": "1/Y", "1/Y^2": "1/Y^2" }[v.recommended];
    const agrees = v.recommended === (fitResult.weighting.applied || "none");
    return `Replicates give a variance exponent of ${v.theta.toFixed(2)} ± ${v.se.toFixed(2)} ` +
      `across ${v.groups} concentrations, which suggests ${label}` +
      (agrees ? " — matching the current choice." : ".");
  }, [fitResult]);

  // Compact fixed/exponential formatting, shared by the value and its interval
  // so an estimate and its bounds never render in different notations.
  const fmtParam = (v) => (v == null || !isFinite(v)) ? "--"
    : (Math.abs(v) < 0.01 || Math.abs(v) > 10000) ? v.toExponential(3) : v.toFixed(4);

  // Which params are fixed for constrained models (indices into 4PL param vector)
  const fixedParams = activeModel === "1PL" ? new Set([0, 1, 3])
    : activeModel === "2PL" ? new Set([0, 3])
    : activeModel === "3PL" ? new Set([1])
    : new Set();

  const hasReplicates = useMemo(() => {
    if (!parsedData) return false;
    const uniqueConc = new Set(parsedData.xData).size;
    return parsedData.xData.length > uniqueConc;
  }, [parsedData]);

  // Grouped data with Grubbs results per concentration
  const groupedData = useMemo(() => {
    if (!parsedData) return [];
    return groupByConcentration(parsedData.xData, parsedData.yData);
  }, [parsedData]);

  // Run Grubbs test on all groups
  const runGrubbs = useCallback(() => {
    if (!parsedData) return;
    const result = runGrubbsAllGroups(parsedData.xData, parsedData.yData, grubbsAlpha);
    setGrubbsResults(result);
    setShowOutliers(true);
    // Auto-select the first group that has outliers, or the first group
    const firstOutlierGroup = result.groupResults.find(g => g.outlierCount > 0);
    setSelectedGrubbsGroup(firstOutlierGroup ? firstOutlierGroup.x.toString() : (groupedData[0] ? groupedData[0].x.toString() : null));
  }, [parsedData, grubbsAlpha, groupedData]);

  // Toggle exclusion of a specific data point index
  const toggleExclusion = useCallback((idx) => {
    setExcludedIndices(prev => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }, []);

  // Exclude all detected outliers at once
  const excludeAllOutliers = useCallback(() => {
    if (!grubbsResults) return;
    setExcludedIndices(prev => {
      const next = new Set(prev);
      for (const idx of grubbsResults.outlierIndices) next.add(idx);
      return next;
    });
  }, [grubbsResults]);

  // Clear all exclusions
  const clearExclusions = useCallback(() => {
    setExcludedIndices(new Set());
  }, []);

  // Refit model using only non-excluded data
  const refitWithoutExcluded = useCallback(() => {
    if (!parsedData || excludedIndices.size === 0) return;
    const xFiltered = [], yFiltered = [];
    parsedData.xData.forEach((x, i) => {
      if (!excludedIndices.has(i)) { xFiltered.push(x); yFiltered.push(parsedData.yData[i]); }
    });
    if (xFiltered.length < 4) { setError("Too few points remaining after exclusion"); return; }
    setError(null);

    if (modelType === "Auto") {
      const fit4 = fitModel(xFiltered, yFiltered, model4PL, false, fitOpts);
      const fit5 = xFiltered.length >= 5 ? fitModel(xFiltered, yFiltered, model5PL, true, fitOpts) : null;
      if (!fit4 && !fit5) { setError("Fitting failed"); return; }
      if (!fit5) { setActiveModel("4PL"); setFitResult(fit4); return; }
      const deltaAICc = fit4.aicc - fit5.aicc;
      const eNearOne = Math.abs(fit5.params[4] - 1) < 0.05;
      const selected = (eNearOne || deltaAICc <= 2) ? "4PL" : "5PL";
      setActiveModel(selected);
      setFitResult(selected === "4PL" ? fit4 : fit5);
      setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason: "Refit after exclusion" });
    } else if (["1PL", "2PL", "3PL"].includes(modelType)) {
      const aVal = parseFloat(fixedMin), dVal = parseFloat(fixedMax), hVal = parseFloat(fixedHill);
      if (["1PL", "2PL"].includes(modelType) && (isNaN(aVal) || isNaN(dVal))) {
        setError(`${modelType} requires min and max asymptote values`); return;
      }
      if (["1PL", "3PL"].includes(modelType) && (isNaN(hVal) || hVal === 0)) {
        setError(`${modelType} requires a non-zero Hill slope value`); return;
      }
      const fixedMap = modelType === "1PL" ? { 0: aVal, 1: hVal, 3: dVal }
        : modelType === "2PL" ? { 0: aVal, 3: dVal }
        : { 1: hVal };
      const result = fitConstrainedModel(xFiltered, yFiltered, fixedMap);
      if (!result) { setError("Fitting failed"); return; }
      setActiveModel(modelType);
      setFitResult(result);
    } else {
      const modelFn = getModelFn(modelType);
      const result = fitModel(xFiltered, yFiltered, modelFn, modelType === "5PL", fitOpts);
      if (!result) { setError("Fitting failed"); return; }
      setActiveModel(modelType);
      setFitResult(result);
    }
  }, [parsedData, excludedIndices, modelType, fixedMin, fixedMax, fixedHill]);

  const exportCSV = useCallback(() => {
    if (!parsedData || !fitResult) return;
    const modelFn = getModelFn(activeModel);
    const hasBg = parsedData.bgSubtracted > 0;
    let csv = "";
    csv += `# Model: ${activeModel}\n`;
    if (hasBg) csv += `# Background subtracted: ${parsedData.bgSubtracted.toFixed(2)}\n`;
    if (parsedData.normalized) csv += `# Normalized: 0-100% (min=${parsedData.normMin.toFixed(4)}, max=${parsedData.normMax.toFixed(4)})\n`;
    csv += hasBg
      ? "Concentration,Raw,BgSubtracted,Fitted,Residual\n"
      : "Concentration,Observed,Fitted,Residual\n";
    parsedData.xData.forEach((x, i) => {
      const fitted = modelFn(x, fitResult.params);
      const resid = parsedData.yData[i] - fitted;
      if (hasBg && parsedData.yRaw) {
        csv += `${x},${parsedData.yRaw[i]},${parsedData.yData[i].toFixed(6)},${fitted.toFixed(6)},${resid.toFixed(6)}\n`;
      } else {
        csv += `${x},${parsedData.yData[i]},${fitted.toFixed(6)},${resid.toFixed(6)}\n`;
      }
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `bioassay_${activeModel}_fit.csv`; a.click();
    URL.revokeObjectURL(url);
  }, [parsedData, fitResult, activeModel]);

  const exportImage = useCallback((format) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const mimeType = format === "jpeg" ? "image/jpeg" : "image/png";
    const quality = format === "jpeg" ? 0.95 : undefined;
    const needsStatsTable = overlayMode && statsTableVisible && allFitResults?.length > 0;
    const needsOffscreen = format === "jpeg" || needsStatsTable;
    let exportCanvas = canvas;
    if (needsOffscreen) {
      exportCanvas = document.createElement("canvas");
      exportCanvas.width = canvas.width;
      exportCanvas.height = canvas.height;
      const ctx = exportCanvas.getContext("2d");
      if (format === "jpeg") {
        ctx.fillStyle = t.canvas || "#0a0f1a";
        ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      }
      ctx.drawImage(canvas, 0, 0);
      if (needsStatsTable) {
        const dpr = canvas.width / canvas.getBoundingClientRect().width;
        drawStatsTableOnCanvas(ctx, statsTablePos, allFitResults, selectedCompounds, overlayEditIndex, getCompoundStyle, dpr, statsTableCols);
      }
    }
    const dataUrl = exportCanvas.toDataURL(mimeType, quality);
    const a = document.createElement("a");
    a.href = dataUrl;
    a.download = `bioassay_${activeModel}_fit.${format}`;
    a.click();
  }, [activeModel, t, overlayMode, statsTableVisible, allFitResults, selectedCompounds, statsTablePos, overlayEditIndex, getCompoundStyle]);

  // PDF Report Generation
  const generatePdfReport = useCallback(async () => {
    const isMulti = allFitResults?.length > 0;
    if (!isMulti && (!parsedData || !fitResult)) return;
    setPdfGenerating(true);
    try {
      const { jsPDF } = await import("jspdf");
      const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
      const pw = 210, margin = 15;
      const cw = pw - 2 * margin;
      let y = margin;
      const gray = [100, 100, 100];
      const dark = [30, 30, 30];
      const accent = [0, 150, 120];
      const lineH = 5;

      const addTitle = (text) => { doc.setFontSize(16); doc.setFont("helvetica","bold"); doc.setTextColor(...dark); doc.text(text, margin, y); y += 8; };
      const addSection = (text) => { if (y > 260) { doc.addPage(); y = margin; } doc.setFontSize(11); doc.setFont("helvetica","bold"); doc.setTextColor(...accent); doc.text(text, margin, y); y += 2; doc.setDrawColor(...accent); doc.setLineWidth(0.3); doc.line(margin, y, margin + cw, y); y += 5; doc.setFontSize(9); doc.setFont("helvetica","normal"); doc.setTextColor(...gray); };
      const addText = (text, opts = {}) => { const sz = opts.size || 9; const clr = opts.color || gray; doc.setFontSize(sz); doc.setFont("helvetica", opts.bold ? "bold" : "normal"); doc.setTextColor(...clr); const lines = doc.splitTextToSize(text, cw); if (y + lines.length * (sz * 0.4) > 280) { doc.addPage(); y = margin; } doc.text(lines, margin, y); y += lines.length * (sz * 0.4) + 2; doc.setFontSize(9); doc.setFont("helvetica","normal"); doc.setTextColor(...gray); };
      const addKeyVal = (key, val) => { doc.setFontSize(9); doc.setFont("helvetica","bold"); doc.setTextColor(...dark); doc.text(key + ":", margin, y); doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.text(String(val), margin + 35, y); y += lineH; };
      // Sanitize text for jsPDF (replace unsupported Unicode with ASCII)
      const sanitize = (s) => String(s).replace(/\u0394/g,"Delta ").replace(/\u2248/g,"~").replace(/--/g,"--").replace(/\u2019/g,"'").replace(/\u03B1/g,"alpha").replace(/\u00B2/g,"2").replace(/\u2265/g,">=").replace(/\u2264/g,"<=");

      // ── Multi-molecule report ──────────────────────────────────────
      if (isMulti) {
        // Formatter using RB_STAT_COLUMNS for consistent sig figs across report
        const fmtStat = (key, r) => {
          const col = RB_STAT_COLUMNS.find(c => c.key === key);
          return col ? sanitize(col.fmt(r)) : "--";
        };
        // Numeric formatter for raw data values (consistent across columns)
        const fmtNum = (v) => (v == null || isNaN(v)) ? "--" : (Math.abs(v) < 0.01 || Math.abs(v) > 99999) ? v.toExponential(3) : v.toFixed(4);

        // Header
        addTitle("Bioassay Curve Fitting Report");
        doc.setFontSize(8); doc.setTextColor(...gray); doc.setFont("helvetica","normal");
        doc.text("Generated " + new Date().toLocaleString() + "  |  assaycurvefit.com  |  " + (typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "dev") + "  |  " + allFitResults.length + " molecules", margin, y); y += 8;

        // ── Overlay chart ──
        if (mainCanvasRef.current) {
          addSection("Overlay View");
          const oCanvas = mainCanvasRef.current;
          const imgData = oCanvas.toDataURL("image/png");
          const imgW = cw;
          const imgH = Math.min((oCanvas.height / oCanvas.width) * imgW, 110);
          if (y + imgH > 280) { doc.addPage(); y = margin; }
          doc.addImage(imgData, "PNG", margin, y, imgW, imgH);
          y += imgH + 4;
        }

        // ── Summary stats table ──
        addSection("Molecule Summary");
        const sumColKeys = ["name", ...statsTableCols];
        // Column widths: name=42mm, rest split equally
        const restCols = sumColKeys.filter(k => k !== "name");
        const nameW = 42, restW = restCols.length > 0 ? (cw - nameW) / restCols.length : 0;
        const colXs = sumColKeys.map((k, ci) => margin + (ci === 0 ? 0 : nameW + (ci - 1) * restW));

        // Header row
        doc.setFontSize(7.5); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        sumColKeys.forEach((key, ci) => {
          const col = RB_STAT_COLUMNS.find(c => c.key === key);
          const label = col ? col.label : key;
          doc.text(label, colXs[ci], y);
        });
        y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFontSize(7); doc.setFont("helvetica","normal");

        allFitResults.forEach((r, ri) => {
          if (y > 275) { doc.addPage(); y = margin; }
          const cs = getCompoundStyle(r.name, ri);
          // Parse hex color → RGB for jsPDF
          const hexToRgb = (hex) => { const h = hex.replace("#",""); return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; };
          const [cr, cg, cb] = hexToRgb(cs.color || "#3b9eff");
          sumColKeys.forEach((key, ci) => {
            if (key === "name") {
              // Colored bullet + name
              doc.setTextColor(cr, cg, cb); doc.setFont("helvetica","bold");
              doc.text("*", colXs[ci], y);
              doc.setTextColor(...dark); doc.setFont("helvetica","normal");
              const nameStr = r.name.length > 18 ? r.name.slice(0,17) + "…" : r.name;
              doc.text(nameStr, colXs[ci] + 4, y);
            } else {
              doc.setTextColor(...gray); doc.setFont("helvetica","normal");
              doc.text(fmtStat(key, r), colXs[ci], y);
            }
          });
          y += 4;
        });
        y += 4;

        // ── Per-molecule sections ──
        allFitResults.forEach((r, ri) => {
          // Each molecule starts on a new page
          doc.addPage(); y = margin;

          // Molecule header banner
          const cs = getCompoundStyle(r.name, ri);
          const hexToRgb = (hex) => { const h = hex.replace("#",""); return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; };
          const [cr, cg, cb] = hexToRgb(cs.color || "#3b9eff");
          doc.setFillColor(cr, cg, cb);
          doc.rect(margin, y, 4, 12, "F");
          doc.setFontSize(13); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
          doc.text("Molecule: " + r.name, margin + 7, y + 8);
          y += 16;

          if (!r.fitResult) {
            doc.setFontSize(9); doc.setTextColor(...gray); doc.setFont("helvetica","italic");
            doc.text("Fit failed or insufficient data points.", margin, y); y += 8;
            return; // continue to next molecule
          }

          // Model parameters
          addSection("Model");
          addKeyVal("Model Type", r.modelType || "--");
          const pLabels = r.modelType === "5PL" ? ["Bottom","Hill","EC50","Top","S"] : ["A (min)","B (slope)","C (EC50)","D (max)"];
          pLabels.forEach((label, pi) => {
            const val = r.fitResult.params[pi];
            if (val == null) return;
            const ci = r.fitResult.ci?.[pi];
            const suffix = ci && isFinite(ci.lo) && isFinite(ci.hi)
              ? `   [95% CI ${ci.lo.toExponential(3)} to ${ci.hi.toExponential(3)}]`
              : "";
            addKeyVal(label, val.toExponential(4) + suffix);
          });
          if (r.modelType === "5PL" && r.fitResult.bioEC50) {
            addKeyVal("Bio EC50", r.fitResult.bioEC50.toExponential(4));
          }
          y += 2;

          // Goodness of fit — use same sig figs as summary table via fmtStat
          addSection("Goodness of Fit");
          addKeyVal("R^2",    fmtStat("r2",   r));
          addKeyVal("RMSE",   fmtStat("rmse", r));
          addKeyVal("SSR",    fmtStat("ssr",  r));
          addKeyVal("AICc",   fmtStat("aicc", r));
          addKeyVal("BIC",    fmtStat("bic",  r));
          if (r.fitResult.syx != null) addKeyVal("Sy.x", r.fitResult.syx.toPrecision(4));
          if (r.fitResult.n != null)  addKeyVal("N data pts", String(r.fitResult.n));
          if (r.fitResult.k != null)  addKeyVal("K params",   String(r.fitResult.k));
          if (r.fitResult.dof != null) addKeyVal("DOF",       String(r.fitResult.dof));
          if (r.fitResult.lackOfFit?.applicable) {
            const lof = r.fitResult.lackOfFit;
            addKeyVal("Lack of fit F", lof.F.toFixed(3));
            addKeyVal("Lack of fit p", lof.pValue < 0.0001 ? "<0.0001" : lof.pValue.toFixed(4));
          }
          addKeyVal("Converged", r.fitResult.converged ? "Yes" : "No");
          y += 2;

          // Individual chart (offscreen canvas)
          addSection("Fitted Curve");
          const molCanvas = document.createElement("canvas");
          molCanvas.width = 900; molCanvas.height = 500;
          drawChart(molCanvas, r.xData, r.yData, r.fitResult, r.modelType,
            { pointView, errorBarType, xAxisLog, yAxisFormat, yAxisDecimals }, t);
          const molImgW = cw;
          const molImgH = Math.min((molCanvas.height / molCanvas.width) * molImgW, 95);
          if (y + molImgH > 280) { doc.addPage(); y = margin; }
          doc.addImage(molCanvas.toDataURL("image/png"), "PNG", margin, y, molImgW, molImgH);
          y += molImgH + 5;

          // Raw data — pull from multiData[ri] for individual replicate values
          const molData = multiData ? multiData[ri] : null;
          if (molData?.points?.length > 0) {
            if (y > 210) { doc.addPage(); y = margin; }
            addSection("Raw Data");
            doc.setFontSize(7); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
            const cX = margin, cMean = margin + 24, cSD = margin + 50, cSEM = margin + 72, cCV = margin + 93, cN = margin + 114, cVals = margin + 122;
            doc.text("Conc (M)", cX, y); doc.text("Mean", cMean, y); doc.text("SD", cSD, y); doc.text("SEM", cSEM, y); doc.text("%CV", cCV, y); doc.text("n", cN, y); doc.text("Values", cVals, y);
            y += 1; doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
            doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.setFontSize(6.5);
            for (const pt of molData.points) {
              if (y > 275) { doc.addPage(); y = margin; }
              const reps = pt.reps;
              const n = reps.length;
              const mean = reps.reduce((a, b) => a + b, 0) / n;
              const sd = n > 1 ? Math.sqrt(reps.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
              const sem = n > 1 ? sd / Math.sqrt(n) : 0;
              const cv = mean !== 0 ? (sd / Math.abs(mean)) * 100 : 0;
              const concM = Math.pow(10, pt.conc);
              doc.text(concM.toExponential(3), cX, y);
              doc.text(fmtNum(mean), cMean, y);
              doc.text(n > 1 ? fmtNum(sd)  : "--", cSD,  y);
              doc.text(n > 1 ? fmtNum(sem) : "--", cSEM, y);
              doc.text(n > 1 ? cv.toFixed(1) + "%" : "--", cCV, y);
              doc.text(String(n), cN, y);
              const valStr = reps.map(v => fmtNum(v)).join(", ");
              doc.text(valStr.length > 38 ? valStr.slice(0, 35) + "..." : valStr, cVals, y);
              y += 3.5;
            }
            y += 3;
          }

          // Outlier analysis — run grubbsTest per concentration group
          if (molData?.points?.length > 0) {
            const grubbsGroups = molData.points.map(pt => {
              const reps = pt.reps;
              const concM = Math.pow(10, pt.conc);
              if (reps.length < 3) return { conc: concM, n: reps.length, tested: false };
              const result = grubbsTest(reps, grubbsAlpha);
              const outlierCount = result.outliers.length;
              return { conc: concM, n: reps.length, tested: true, result, outlierCount };
            });
            const hasAnyTested = grubbsGroups.some(g => g.tested);
            if (hasAnyTested) {
              if (y > 220) { doc.addPage(); y = margin; }
              const totalOutliers = grubbsGroups.reduce((s, g) => s + (g.outlierCount || 0), 0);
              addSection("Outlier Analysis (Grubbs, a=" + grubbsAlpha + ")");
              addKeyVal("Total outliers", totalOutliers);
              y += 2;
              doc.setFontSize(7); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
              doc.text("Conc (M)", margin, y); doc.text("n", margin + 26, y); doc.text("Outliers", margin + 36, y); doc.text("G stat", margin + 58, y); doc.text("G crit", margin + 78, y); doc.text("Result", margin + 98, y);
              y += 1; doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
              doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.setFontSize(7);
              for (const gr of grubbsGroups) {
                if (y > 275) { doc.addPage(); y = margin; }
                const hasOutlier = gr.tested && gr.outlierCount > 0;
                if (hasOutlier) {
                  doc.setFillColor(255, 235, 235);
                  doc.rect(margin - 1, y - 2.5, cw + 2, 3.5, "F");
                  doc.setTextColor(180, 40, 40); doc.setFont("helvetica","bold");
                } else {
                  doc.setTextColor(...gray); doc.setFont("helvetica","normal");
                }
                doc.text(gr.conc.toExponential(3), margin, y);
                doc.text(String(gr.n), margin + 26, y);
                if (!gr.tested) {
                  doc.setTextColor(...gray); doc.setFont("helvetica","normal");
                  doc.text("n<3, skipped", margin + 36, y);
                } else {
                  const maxG = gr.result.details ? Math.max(...gr.result.details.map(d => d.g)) : 0;
                  doc.text(String(gr.outlierCount), margin + 36, y);
                  doc.text(maxG.toFixed(4), margin + 58, y);
                  doc.text(gr.result.gCrit ? gr.result.gCrit.toFixed(4) : "--", margin + 78, y);
                  doc.text(hasOutlier ? "OUTLIER" : "Pass", margin + 98, y);
                }
                y += 3.5;
                if (hasOutlier && gr.result.outliers) {
                  doc.setFontSize(6); doc.setTextColor(180, 40, 40); doc.setFont("helvetica","italic");
                  const olVals = gr.result.outliers.map(o => fmtNum(o.value) + " (G=" + o.g.toFixed(3) + ")").join(", ");
                  doc.text("  Flagged: " + olVals, margin + 4, y);
                  y += 3; doc.setFontSize(7);
                }
              }
              doc.setTextColor(...gray); doc.setFont("helvetica","normal");
              y += 3;
            }
          }
        });

        // Footer
        const pages = doc.getNumberOfPages();
        for (let i = 1; i <= pages; i++) {
          doc.setPage(i);
          doc.setFontSize(7); doc.setTextColor(160,160,160); doc.setFont("helvetica","normal");
          doc.text("assaycurvefit.com  |  Page " + i + " of " + pages, pw / 2, 290, { align: "center" });
        }
        doc.save("bioassay_multi_molecule_report.pdf");
        return;
      }

      // ── Single-compound report (unchanged) ────────────────────────

      // Header
      addTitle("Bioassay Curve Fitting Report");
      doc.setFontSize(8); doc.setTextColor(...gray); doc.setFont("helvetica","normal");
      doc.text("Generated " + new Date().toLocaleString() + "  |  assaycurvefit.com  |  " + (typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "dev"), margin, y); y += 8;

      // Model info
      if (pdfSections.modelInfo) {
        addSection("Model");
        addKeyVal("Model Type", activeModel);
        if (pdfSections.modelParams) {
          const pLabels = activeModel === "5PL" ? ["Bottom","Hill","EC50","Top","S"] : ["A (min)","B (slope)","C (EC50)","D (max)"];
          // An estimate without its interval invites the reader to treat it as
          // exact, which is the whole reason the engine computes one. Spelled
          // "to" rather than an en-dash because jsPDF's core fonts are Latin-1.
          const ciText = (i) => {
            const ci = fitResult.ci?.[i];
            if (!ci || !isFinite(ci.lo) || !isFinite(ci.hi)) return "";
            return `   [95% CI ${ci.lo.toExponential(3)} to ${ci.hi.toExponential(3)}]`;
          };
          pLabels.forEach((label, i) => {
            const val = fitResult.params[i];
            const suffix = fixedParams.has(i) ? "  [fixed]" : ciText(i);
            addKeyVal(label, val.toExponential(6) + suffix);
          });
        }
        if (pdfSections.paramEC50) {
          addKeyVal("EC50", fitResult.params[activeModel === "5PL" ? 2 : 2].toExponential(6));
        }
        if (pdfSections.paramBioEC50 && activeModel === "5PL" && fitResult.bioEC50) {
          addKeyVal("Biological EC50", fitResult.bioEC50.toExponential(6));
        }
        y += 2;
      }

      // Fit statistics (granular)
      if (pdfSections.fitParams) {
        addSection("Goodness of Fit");
        if (pdfSections.paramR2) addKeyVal("R^2", fitResult.r2.toFixed(8));
        if (pdfSections.paramRMSE) addKeyVal("RMSE", fitResult.rmse.toFixed(6));
        if (pdfSections.paramSSR) addKeyVal("SSR", fitResult.ssr.toExponential(4));
        if (pdfSections.paramAIC) addKeyVal("AIC", fitResult.aic.toFixed(2));
        if (pdfSections.paramAICc) addKeyVal("AICc", fitResult.aicc.toFixed(2));
        if (pdfSections.paramBIC) addKeyVal("BIC", fitResult.bic.toFixed(2));
        if (fitResult.syx != null) addKeyVal("Sy.x", fitResult.syx.toPrecision(6));
        if (fitResult.dof != null) addKeyVal("DOF", String(fitResult.dof));
        if (pdfSections.paramConverged) addKeyVal("Converged", fitResult.converged ? "Yes" : "No");
        if (pdfSections.paramNK) { addKeyVal("N data points", fitResult.n); addKeyVal("K parameters", fitResult.k); }
        y += 2;
      }

      // Lack of fit — a different question from the statistics above: not how
      // closely the curve tracks the data, but whether its deviations exceed
      // what the replicates say measurement error can explain.
      if (pdfSections.fitParams && fitResult.lackOfFit?.applicable) {
        const lof = fitResult.lackOfFit;
        addSection("Lack of Fit (replicate-based F-test)");
        addKeyVal("F", lof.F.toFixed(4));
        addKeyVal("p-value", lof.pValue < 0.0001 ? "<0.0001" : lof.pValue.toFixed(4));
        addKeyVal("DF (LoF, pure)", `${lof.dfLackOfFit}, ${lof.dfPureError}`);
        addKeyVal("Pure error SD", lof.sdPureError.toPrecision(6));
        addText(lof.significant
          ? "Significant: the curve deviates from the replicate means by more than "
            + "measurement error, so the model is missing real structure."
          : "Not significant: scatter about the curve is consistent with the "
            + "assay's own replicate noise.");
        y += 2;
      }

      // Model comparison
      if (pdfSections.modelComparison && comparison && comparison.fit5PL) {
        addSection("Model Comparison (Auto Selection)");
        const headers = ["Metric", "4PL", "5PL"];
        const rows = [
          ["R^2", comparison.fit4PL.r2.toFixed(6), comparison.fit5PL.r2.toFixed(6)],
          ["AICc", comparison.fit4PL.aicc.toFixed(2), comparison.fit5PL.aicc.toFixed(2)],
          ["BIC", comparison.fit4PL.bic.toFixed(2), comparison.fit5PL.bic.toFixed(2)],
          ["SSR", comparison.fit4PL.ssr.toExponential(3), comparison.fit5PL.ssr.toExponential(3)],
          ["S param", "--", comparison.fit5PL.params[4].toFixed(4)],
        ];
        // Table header
        doc.setFontSize(8); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        doc.text(headers[0], margin, y); doc.text(headers[1], margin + 50, y); doc.text(headers[2], margin + 90, y); y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFont("helvetica","normal"); doc.setTextColor(...gray);
        for (const row of rows) {
          doc.text(row[0], margin, y); doc.text(row[1], margin + 50, y); doc.text(row[2], margin + 90, y); y += lineH;
        }
        y += 1;
        addText("Selected: " + comparison.selected + " -- " + sanitize(comparison.reason), { size: 8, color: accent, bold: true }); y += 2;
      }

      // Background info
      if (pdfSections.backgroundInfo && parsedData.bgSubtracted > 0) {
        addSection("Background Subtraction");
        addKeyVal("Background Mean", parsedData.bgSubtracted.toFixed(4));
        y += 2;
      }

      // Normalization info
      if (pdfSections.normalizationInfo && parsedData.normalized) {
        addSection("Normalization");
        addText("Data normalized to 0-100% using raw min=" + parsedData.normMin.toFixed(4) + ", max=" + parsedData.normMax.toFixed(4), { size: 9 });
        y += 2;
      }

      // Plot
      if (pdfSections.plot && mainCanvasRef.current) {
        if (y > 160) { doc.addPage(); y = margin; }
        addSection("Fitted Curve");
        const canvas = mainCanvasRef.current;
        const imgData = canvas.toDataURL("image/png");
        const imgW = cw;
        const imgH = (canvas.height / canvas.width) * imgW;
        if (y + imgH > 280) { doc.addPage(); y = margin; }
        doc.addImage(imgData, "PNG", margin, y, imgW, Math.min(imgH, 100));
        y += Math.min(imgH, 100) + 5;
      }

      // Raw data table
      if (pdfSections.rawData) {
        if (y > 200) { doc.addPage(); y = margin; }
        addSection("Raw Data");
        const groups = groupByConcentration(parsedData.xData, parsedData.yData);
        // Table header
        doc.setFontSize(7); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        const colX = margin, colMean = margin + 22, colSD = margin + 48, colSEM = margin + 70, colCV = margin + 90, colN = margin + 112, colVals = margin + 122;
        doc.text("Conc", colX, y); doc.text("Mean", colMean, y); doc.text("SD", colSD, y); doc.text("SEM", colSEM, y); doc.text("%CV", colCV, y); doc.text("n", colN, y); doc.text("Values", colVals, y);
        y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.setFontSize(6.5);
        for (const g of groups) {
          if (y > 275) { doc.addPage(); y = margin; }
          const vals = g.values;
          const n = vals.length;
          const mean = vals.reduce((a, b) => a + b, 0) / n;
          const sd = n > 1 ? Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
          const sem = n > 1 ? sd / Math.sqrt(n) : 0;
          const cv = mean !== 0 ? (sd / Math.abs(mean)) * 100 : 0;
          doc.text(g.x.toString(), colX, y);
          doc.text(mean.toFixed(4), colMean, y);
          doc.text(n > 1 ? sd.toFixed(4) : "--", colSD, y);
          doc.text(n > 1 ? sem.toFixed(4) : "--", colSEM, y);
          doc.text(n > 1 ? cv.toFixed(1) + "%" : "--", colCV, y);
          doc.text(String(n), colN, y);
          const valStr = vals.map(v => v.toFixed(3)).join(", ");
          const trimmed = valStr.length > 40 ? valStr.substring(0, 37) + "..." : valStr;
          doc.text(trimmed, colVals, y);
          y += 3.5;
        }
        y += 3;
      }

      // Outlier test results
      if (pdfSections.outlierResults && grubbsResults) {
        if (y > 220) { doc.addPage(); y = margin; }
        addSection("Grubbs' Outlier Test (a=" + grubbsAlpha + ")");
        addKeyVal("Total outliers", grubbsResults.totalOutliers);
        y += 2;
        doc.setFontSize(7); doc.setFont("helvetica","bold"); doc.setTextColor(...dark);
        doc.text("Conc", margin, y); doc.text("n", margin + 22, y); doc.text("Outliers", margin + 32, y); doc.text("G stat", margin + 56, y); doc.text("G crit", margin + 76, y); doc.text("Result", margin + 96, y);
        y += 1;
        doc.setDrawColor(180,180,180); doc.setLineWidth(0.2); doc.line(margin, y, margin + cw, y); y += 3;
        doc.setFont("helvetica","normal"); doc.setTextColor(...gray); doc.setFontSize(7);
        for (const gr of grubbsResults.groupResults) {
          if (y > 275) { doc.addPage(); y = margin; }
          const hasOutlier = gr.tested && gr.outlierCount > 0;
          if (hasOutlier) {
            // Highlight row background
            doc.setFillColor(255, 235, 235);
            doc.rect(margin - 1, y - 2.5, cw + 2, 3.5, "F");
            doc.setTextColor(180, 40, 40);
            doc.setFont("helvetica","bold");
          } else {
            doc.setTextColor(...gray);
            doc.setFont("helvetica","normal");
          }
          doc.text(gr.x.toString(), margin, y);
          doc.text(String(gr.n), margin + 22, y);
          if (!gr.tested) {
            doc.setTextColor(...gray); doc.setFont("helvetica","normal");
            doc.text("n<3, skipped", margin + 32, y);
          } else {
            const maxG = gr.result.details ? Math.max(...gr.result.details.map(d => d.g)) : 0;
            doc.text(String(gr.outlierCount), margin + 32, y);
            doc.text(maxG.toFixed(4), margin + 56, y);
            doc.text(gr.result.gCrit ? gr.result.gCrit.toFixed(4) : "--", margin + 76, y);
            doc.text(hasOutlier ? "OUTLIER" : "Pass", margin + 96, y);
          }
          y += 3.5;
          // List individual outlier values below the row
          if (hasOutlier && gr.result.outliers) {
            doc.setFontSize(6); doc.setTextColor(180, 40, 40); doc.setFont("helvetica","italic");
            const olVals = gr.result.outliers.map(o => o.value.toFixed(4) + " (G=" + o.g.toFixed(3) + ")").join(", ");
            doc.text("  Flagged: " + olVals, margin + 4, y);
            y += 3;
            doc.setFontSize(7);
          }
        }
        // Reset
        doc.setTextColor(...gray); doc.setFont("helvetica","normal");
        y += 3;
      }

      // Footer on last page
      const pages = doc.getNumberOfPages();
      for (let i = 1; i <= pages; i++) {
        doc.setPage(i);
        doc.setFontSize(7); doc.setTextColor(160,160,160); doc.setFont("helvetica","normal");
        doc.text("assaycurvefit.com  |  Page " + i + " of " + pages, pw / 2, 290, { align: "center" });
      }

      doc.save("bioassay_" + activeModel + "_report.pdf");
    } catch (e) {
      console.error("PDF generation failed:", e);
      setError("PDF generation failed: " + e.message);
    } finally {
      setPdfGenerating(false);
      setShowPdfModal(false);
    }
  }, [parsedData, fitResult, activeModel, comparison, grubbsResults, grubbsAlpha, pdfSections, fixedParams, t, allFitResults, multiData, statsTableCols, getCompoundStyle, pointView, errorBarType, xAxisLog, yAxisFormat, yAxisDecimals]);

  const interpolate = useCallback((targetY) => {
    if (!fitResult) return null;
    const [A, B, C, D] = fitResult.params;
    if (activeModel === "4PL") {
      const ratio = (A - targetY) / (targetY - D);
      if (ratio <= 0) return null;
      return C * Math.pow(ratio, 1 / B);
    }
    // 5PL: numerical inverse via bisection in log-space
    const modelFn = model5PL;
    let lo = 1e-15, hi = 1e15;
    // Determine curve direction: evaluate at low and high x
    const yLo = modelFn(lo, fitResult.params);
    const yHi = modelFn(hi, fitResult.params);
    const increasing = yHi > yLo;
    for (let i = 0; i < 100; i++) {
      const mid = Math.sqrt(lo * hi);
      const yMid = modelFn(mid, fitResult.params);
      if ((increasing && yMid > targetY) || (!increasing && yMid < targetY)) hi = mid;
      else lo = mid;
    }
    return Math.sqrt(lo * hi);
  }, [fitResult, activeModel]);

  const [interpY, setInterpY] = useState("");
  const [interpResult, setInterpResult] = useState(null);

  // Helper: reset all fit-related state, load a compound's data, and auto-fit
  const loadCompound = useCallback((compound) => {
    const csv = compoundToCSV(compound);
    setRawData(csv);
    setParsedData(null);
    setFitResult(null);
    setComparison(null);
    setError(null);
    setGrubbsResults(null);
    setShowOutliers(false);
    setSelectedGrubbsGroup(null);
    setExcludedIndices(new Set());
    setBgStats(null);
    // Run fit inline with the csv string directly — avoids stale rawData state
    try {
      const { xData, yData } = parseData(csv);
      if (xData.length < 4) return;
      const fit4 = fitModel(xData, yData, model4PL, false, fitOpts);
      const fit5 = xData.length >= 5 ? fitModel(xData, yData, model5PL, true, fitOpts) : null;
      setParsedData({ xData, yData, yRaw: yData, bgSubtracted: 0, normMin: 0, normMax: 1, normalized: false });
      if (!fit4 && !fit5) return;
      if (!fit5 || Math.abs(fit5.params[4] - 1) < 0.05) {
        setActiveModel("4PL"); setFitResult(fit4);
        setComparison({ fit4PL: fit4, fit5PL: fit5 || null, selected: "4PL", reason: "Auto" });
      } else {
        const deltaAICc = (fit4?.aicc ?? Infinity) - fit5.aicc;
        const selected = (deltaAICc > 2 && fit4.bic - fit5.bic > 0) ? "5PL" : "4PL";
        setActiveModel(selected);
        setFitResult(selected === "5PL" ? fit5 : fit4);
        setComparison({ fit4PL: fit4, fit5PL: fit5, selected, reason: "Auto" });
      }
    } catch { /* silently skip on malformed data */ }
  }, [parseData]);

  // When multiIndex changes, swap the textarea content
  useEffect(() => {
    if (!multiData) return;
    loadCompound(multiData[multiIndex]);
    if (!overlayMode) setOverlayEditIndex(null);
  }, [multiIndex, multiData, overlayMode]); // intentionally not including loadCompound to avoid stale closure loop

  // Pre-fit all compounds for overlay mode whenever multiData changes
  useEffect(() => {
    if (!multiData) { setAllFitResults(null); setOverlayMode(false); return; }
    const results = multiData.map(compound => {
      try {
        const csv = compoundToCSV(compound);
        const { xData, yData } = parseData(csv);
        if (xData.length < 4) return { name: compound.name, xData: [], yData: [], fitResult: null, modelType: null };
        const fit4 = fitModel(xData, yData, model4PL, false, fitOpts);
        const fit5 = xData.length >= 5 ? fitModel(xData, yData, model5PL, true, fitOpts) : null;
        let fitResult, modelType;
        if (!fit5 || Math.abs(fit5.params[4] - 1) < 0.05) {
          modelType = "4PL"; fitResult = fit4;
        } else {
          const deltaAICc = (fit4?.aicc ?? Infinity) - fit5.aicc;
          modelType = (deltaAICc > 2 && fit4.bic - fit5.bic > 0) ? "5PL" : "4PL";
          fitResult = modelType === "5PL" ? fit5 : fit4;
        }
        return { name: compound.name, xData, yData, fitResult, modelType };
      } catch {
        return { name: compound.name, xData: [], yData: [], fitResult: null, modelType: null };
      }
    });
    setAllFitResults(results);
    setSelectedCompounds(null); // reset to all-selected when new CSV loaded
    setOverlayEditIndex(null);  // reset overlay editing on new CSV
  }, [multiData, parseData]);

  // Auto-select first molecule when entering overlay mode or loading a new CSV while overlay is active
  useEffect(() => {
    if (overlayMode && allFitResults?.length > 0 && overlayEditIndex === null) {
      loadCompoundForOverlayEdit(0);
    }
  }, [allFitResults, overlayMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // While a compound is selected for overlay editing, propagate left-panel fit changes back to allFitResults.
  // This makes the overlay canvas update live when the user re-fits or switches model (View 4PL / View 5PL).
  useEffect(() => {
    if (!overlayMode || overlayEditIndex === null || !fitResult || !activeModel) return;
    setAllFitResults(prev => prev
      ? prev.map((r, i) => i === overlayEditIndex ? { ...r, fitResult, modelType: activeModel } : r)
      : prev
    );
  }, [fitResult, activeModel, overlayMode, overlayEditIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleMultiCsv = useCallback((file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const compounds = parsePrismCSV(e.target.result);
        setMultiData(compounds);
        setMultiIndex(0);
        setMultiCsvError(null);
        // Immediately load first compound
        loadCompound(compounds[0]);
      } catch (err) {
        setMultiCsvError(err.message);
        setMultiData(null);
      }
    };
    reader.readAsText(file);
  }, [loadCompound]);

  // Load a named example from either bank. Two-curve examples go through the
  // same path as an uploaded multi-compound CSV, and open in overlay mode --
  // a parallelism example that shows one curve at a time buries its own point.
  const loadExample = useCallback((name) => {
    setFixedMin("");
    setFixedMax("");
    if (EXAMPLE_PAIRS[name]) {
      try {
        const compounds = parsePrismCSV(EXAMPLE_PAIRS[name]);
        setMultiData(compounds);
        setMultiIndex(0);
        setMultiCsvError(null);
        setOverlayMode(true);
        loadCompound(compounds[0]);
      } catch (err) {
        setMultiCsvError(err.message);
        setMultiData(null);
      }
      return;
    }
    const csv = EXAMPLE_DATASETS[name];
    if (!csv) return;
    // Clearing any loaded pair matters: otherwise picking a single-curve
    // example leaves the app in multi-compound mode, showing a compound
    // switcher and an overlay built from the previous example's data.
    setMultiData(null);
    setMultiIndex(0);
    setMultiCsvError(null);
    setRawData(csv);
    setParsedData(null);
    setFitResult(null);
    setComparison(null);
    setError(null);
    setGrubbsResults(null);
    setShowOutliers(false);
    setSelectedGrubbsGroup(null);
    setExcludedIndices(new Set());
    setBgStats(null);
  }, [loadCompound]);

  useEffect(() => {
    if (!multiData) return;
    const handler = (e) => {
      if (e.key === "ArrowLeft") setMultiIndex(i => Math.max(0, i - 1));
      if (e.key === "ArrowRight") setMultiIndex(i => Math.min(multiData.length - 1, i + 1));
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [multiData]);

  return (
    <div style={{
      minHeight: "100vh",
      background: t.bg,
      color: t.text,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'SF Mono', monospace",
      padding: isMobile ? "12px" : "24px",
      transition: "background 0.3s, color 0.3s",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        textarea { font-family: 'JetBrains Mono', monospace; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: ${t.scrollTrack}; }
        ::-webkit-scrollbar-thumb { background: ${t.scrollThumb}; border-radius: 3px; }
        html, body { overflow-x: hidden; }
        @media (max-width: 767px) {
          textarea { font-size: 11px !important; }
          select { font-size: 11px !important; }
        }
      `}</style>

      {/* Header */}
      <div style={{ margin: "0 0 24px", padding: "0 20px", display: "flex", justifyContent: "space-between", alignItems: isMobile ? "center" : "flex-start", flexWrap: "wrap", gap: 12 }}>
        <div>
          <h1 style={{
            fontFamily: "'Space Grotesk', sans-serif",
            fontSize: isMobile ? 22 : 28,
            fontWeight: 700,
            background: "linear-gradient(135deg, #3b9eff, #a855f7, #00e6b4)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "-0.5px",
            display: "flex",
            alignItems: "baseline",
            gap: 8,
          }}>
            Bioassay Curve Fitter
            <span style={{ fontSize: 10, fontWeight: 400, opacity: 0.5, WebkitTextFillColor: t.textDim, background: "none" }}>
              {typeof __APP_VERSION__ !== "undefined" ? __APP_VERSION__ : "dev"}
            </span>
          </h1>
          <p style={{ fontSize: 12, color: t.textMuted, marginTop: 4 }}>
            4-Parameter & 5-Parameter Logistic Regression | Levenberg-Marquardt Optimization
          </p>
        </div>
        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          style={{
            padding: "6px 12px",
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 6,
            color: t.textMuted,
            fontSize: 10,
            cursor: "pointer",
            fontFamily: "'JetBrains Mono', monospace",
            display: "flex",
            alignItems: "center",
            gap: 6,
            transition: "all 0.2s",
          }}
        >
          <span style={{ fontSize: 14 }}>{theme === "dark" ? "☀️" : "🌙"}</span>
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>

      <div style={{ margin: "0", padding: "0 20px", display: "grid", gridTemplateColumns: isMobile ? "1fr" : "340px 1fr", gap: 20, alignItems: "start" }}>
        {/* Left Panel */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, order: isMobile ? 2 : 1 }}>
          {/* Data Input */}
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 10,
            padding: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 8, textTransform: "uppercase", letterSpacing: 1, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 6 }}>
              <span style={{ whiteSpace: "nowrap" }}>Data Input</span>
              <div style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
              <select
                onChange={(e) => {
                  if (e.target.value) loadExample(e.target.value);
                  e.target.value = "";
                }}
                style={{
                  padding: "2px 4px",
                  background: t.input,
                  border: `1px solid ${t.inputBorder}`,
                  borderRadius: 4,
                  color: t.textDim,
                  fontSize: 9,
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: "none",
                  cursor: "pointer",
                  width: 82,
                }}
              >
                <option value="">Examples…</option>
                <optgroup label="Single curve">
                  {Object.keys(EXAMPLE_DATASETS).map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </optgroup>
                <optgroup label="Two curves (parallelism)">
                  {Object.keys(EXAMPLE_PAIRS).map(name => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </optgroup>
              </select>
              {/* Multi-Compound CSV upload button */}
              <button
                onClick={() => multiCsvRef.current && multiCsvRef.current.click()}
                title="Load a Prism-style multi-molecule CSV (duplicate headers = replicates)"
                style={{
                  padding: "2px 8px",
                  background: multiData ? "rgba(0,230,180,0.12)" : t.btnInactive,
                  border: `1px solid ${multiData ? "rgba(0,230,180,0.3)" : t.inputBorder}`,
                  borderRadius: 4,
                  color: multiData ? (t.teal || "#00e6b4") : t.textDim,
                  fontSize: 9,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  whiteSpace: "nowrap",
                  fontWeight: multiData ? 700 : 400,
                }}
              >
                {multiData ? `Multi CSV ✓` : "Multi CSV"}
              </button>
              {/* Template download button */}
              <button
                onClick={downloadMultiTemplate}
                title="Download a blank multi-molecule CSV template"
                style={{
                  padding: "2px 8px",
                  background: t.btnInactive,
                  border: `1px solid ${t.inputBorder}`,
                  borderRadius: 4,
                  color: t.textDim,
                  fontSize: 9,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  whiteSpace: "nowrap",
                }}
              >
                Template
              </button>
              <input
                ref={multiCsvRef}
                type="file"
                accept=".csv"
                style={{ display: "none" }}
                onChange={e => { handleMultiCsv(e.target.files[0]); e.target.value = ""; }}
              />
              </div>
            </div>
            <textarea
              value={rawData}
              onChange={(e) => {
                setRawData(e.target.value);
                setParsedData(null);
                setFitResult(null);
                setComparison(null);
                setError(null);
                setGrubbsResults(null);
                setShowOutliers(false);
                setSelectedGrubbsGroup(null);
                setExcludedIndices(new Set());
                setBgStats(null);
              }}
              placeholder="Concentration,Response&#10;0.01,0.5&#10;0.1,1.2&#10;..."
              style={{
                width: "100%",
                height: 200,
                background: t.input,
                border: `1px solid ${t.inputBorder}`,
                borderRadius: 6,
                color: t.text,
                fontSize: 11,
                padding: 10,
                resize: "vertical",
                outline: "none",
              }}
            />
            <p style={{ fontSize: 9, color: t.textDim, marginTop: 6 }}>
              CSV/TSV format. First column = concentration, additional columns = replicates. Comma-formatted numbers (e.g. 47,189.7) supported.
            </p>
            {parsedData && (
              <p style={{ fontSize: 9, color: t.teal, marginTop: 4 }}>
                Parsed: {parsedData.xData.length} data points across {new Set(parsedData.xData).size} concentrations
                {parsedData.bgSubtracted ? ` (bg: −${parsedData.bgSubtracted.toFixed(1)})` : ""}
                {parsedData.normalized ? " (normalized 0-100%)" : ""}
              </p>
            )}
          </div>

          {multiCsvError && (
            <div style={{
              padding: "10px 12px",
              background: "rgba(255,80,80,0.08)",
              border: "1px solid rgba(255,80,80,0.25)",
              borderRadius: 6,
              color: t.red,
              fontSize: 10,
            }}>
              Multi CSV: {multiCsvError}
            </div>
          )}

          {/* Background Subtraction */}
          <div style={{
            background: t.panel,
            border: `1px solid ${bgEnabled ? "rgba(168,85,247,0.2)" : "rgba(60,100,160,0.15)"}`,
            borderRadius: 10,
            padding: bgEnabled ? 16 : 0,
            overflow: "hidden",
            transition: "all 0.2s",
          }}>
            <button
              onClick={() => setBgEnabled(!bgEnabled)}
              style={{
                width: "100%",
                padding: bgEnabled ? "0 0 10px 0" : "12px 16px",
                background: "transparent",
                border: "none",
                color: bgEnabled ? "rgba(168,85,247,0.9)" : "rgba(160,190,230,0.5)",
                fontSize: 11,
                fontWeight: 600,
                cursor: "pointer",
                fontFamily: "'JetBrains Mono', monospace",
                textTransform: "uppercase",
                letterSpacing: 1,
                textAlign: "left",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <span>Background Subtraction</span>
              <span style={{ fontSize: 10, opacity: 0.6 }}>{bgEnabled ? "▾" : "▸"}</span>
            </button>
            {bgEnabled && (
              <>
                <textarea
                  value={bgRawData}
                  onChange={(e) => setBgRawData(e.target.value)}
                  placeholder={"Paste background response values\ne.g. 2150.3  2089.1  2201.5\nor one per line"}
                  style={{
                    width: "100%",
                    height: 60,
                    background: t.input,
                    border: "1px solid rgba(168,85,247,0.15)",
                    borderRadius: 6,
                    color: t.text,
                    fontSize: 11,
                    padding: 10,
                    resize: "vertical",
                    outline: "none",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                />
                <p style={{ fontSize: 9, color: t.textDim, marginTop: 6 }}>
                  Response values only (no concentrations). Mean is subtracted from all data before fitting.
                </p>
                {bgStats && (
                  <div style={{
                    marginTop: 8,
                    padding: "6px 10px",
                    background: "rgba(168,85,247,0.06)",
                    border: "1px solid rgba(168,85,247,0.12)",
                    borderRadius: 6,
                    fontSize: 10,
                    color: "rgba(190,170,230,0.7)",
                    display: "flex",
                    gap: 12,
                  }}>
                    <span>n={bgStats.n}</span>
                    <span>Mean: <span style={{ color: t.purple, fontWeight: 600 }}>{bgStats.mean.toFixed(1)}</span></span>
                    {bgStats.n > 1 && <span>SD: {bgStats.sd.toFixed(1)}</span>}
                    {bgStats.n > 1 && <span>%CV: {(bgStats.sd / bgStats.mean * 100).toFixed(1)}%</span>}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Model Selection */}
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 10,
            padding: 16,
          }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 10, textTransform: "uppercase", letterSpacing: 1 }}>
              Model
            </div>
            <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
              {["Auto", "1PL", "2PL", "3PL", "4PL", "5PL"].map(m => {
                const colors = {
                  Auto: { active: t.teal, bg: t.tealBg, border: t.tealBorder },
                  "1PL": { active: t.orange, bg: t.orangeBg, border: t.orangeBorder },
                  "2PL": { active: t.orange, bg: t.orangeBg, border: t.orangeBorder },
                  "3PL": { active: t.blue, bg: t.blueBg, border: t.blueBorder },
                  "4PL": { active: t.blue, bg: t.blueBg, border: t.blueBorder },
                  "5PL": { active: t.purple, bg: t.purpleBg, border: t.purpleBorder },
                };
                const c = colors[m];
                return (
                  <button
                    key={m}
                    onClick={() => setModelType(m)}
                    style={{
                      flex: modelType === m ? 2 : 1,
                      minWidth: 40,
                      padding: "8px 0",
                      background: modelType === m ? c.bg : t.btnInactive,
                      border: `1px solid ${modelType === m ? c.border : "rgba(60,100,160,0.15)"}`,
                      borderRadius: 6,
                      color: modelType === m ? c.active : "rgba(160,190,230,0.5)",
                      fontSize: 12,
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      transition: "all 0.2s",
                    }}
                  >
                    {m}
                  </button>
                );
              })}
            </div>
            <div style={{ marginTop: 10, fontSize: 10, color: t.textDim, lineHeight: 1.6 }}>
              {modelType === "Auto"
                ? "Fits 4PL and 5PL; selects best via AICc with parsimony preference"
                : modelType === "1PL"
                  ? "Fix asymptotes and Hill slope; fit EC50 only (1 free parameter)"
                  : modelType === "2PL"
                    ? "Fix asymptotes; fit Hill slope and EC50 (2 free parameters)"
                    : modelType === "3PL"
                      ? "Fix Hill slope; fit asymptotes and EC50 (3 free parameters)"
                      : modelType === "4PL"
                        ? "y = D + (A−D) / (1 + (x/C)^B)"
                        : "y = Bot + (Top−Bot) / (1 + (EC50/x)^Hill)^S"}
            </div>

            {/* Constraint inputs for 1PL, 2PL, 3PL */}
            {["1PL", "2PL", "3PL"].includes(modelType) && (
              <div style={{
                marginTop: 10,
                padding: "10px 12px",
                background: "rgba(255,180,50,0.06)",
                border: "1px solid rgba(255,180,50,0.15)",
                borderRadius: 6,
              }}>
                <div style={{ fontSize: 9, color: t.orange, fontWeight: 600, marginBottom: 8, textTransform: "uppercase", letterSpacing: 0.5 }}>
                  Fixed Parameters
                </div>
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  {(modelType === "1PL" || modelType === "2PL") && (
                    <>
                      <div style={{ flex: 1, minWidth: 60 }}>
                        <label style={{ fontSize: 8, color: t.textDim, display: "block", marginBottom: 3 }}>Min (A)</label>
                        <input
                          type="number"
                          value={fixedMin}
                          onChange={(e) => setFixedMin(e.target.value)}
                          placeholder="e.g. 0"
                          style={{
                            width: "100%",
                            padding: "5px 8px",
                            background: t.input,
                            border: `1px solid ${fixedMin === "" ? "rgba(255,180,50,0.3)" : t.inputBorder}`,
                            borderRadius: 4,
                            color: t.text,
                            fontSize: 11,
                            fontFamily: "'JetBrains Mono', monospace",
                            outline: "none",
                          }}
                        />
                      </div>
                      <div style={{ flex: 1, minWidth: 60 }}>
                        <label style={{ fontSize: 8, color: t.textDim, display: "block", marginBottom: 3 }}>Max (D)</label>
                        <input
                          type="number"
                          value={fixedMax}
                          onChange={(e) => setFixedMax(e.target.value)}
                          placeholder="e.g. 100"
                          style={{
                            width: "100%",
                            padding: "5px 8px",
                            background: t.input,
                            border: `1px solid ${fixedMax === "" ? "rgba(255,180,50,0.3)" : t.inputBorder}`,
                            borderRadius: 4,
                            color: t.text,
                            fontSize: 11,
                            fontFamily: "'JetBrains Mono', monospace",
                            outline: "none",
                          }}
                        />
                      </div>
                    </>
                  )}
                  {(modelType === "1PL" || modelType === "3PL") && (
                    <div style={{ flex: 1, minWidth: 60 }}>
                      <label style={{ fontSize: 8, color: t.textDim, display: "block", marginBottom: 3 }}>Hill slope (B)</label>
                      <input
                        type="number"
                        value={fixedHill}
                        onChange={(e) => setFixedHill(e.target.value)}
                        placeholder="1"
                        style={{
                          width: "100%",
                          padding: "5px 8px",
                          background: t.input,
                          border: `1px solid ${t.inputBorder}`,
                          borderRadius: 4,
                          color: t.text,
                          fontSize: 11,
                          fontFamily: "'JetBrains Mono', monospace",
                          outline: "none",
                        }}
                      />
                    </div>
                  )}
                </div>
                <p style={{ fontSize: 8, color: t.textDim, marginTop: 6 }}>
                  {modelType === "3PL" ? "Hill slope is fixed; asymptotes are fitted from data"
                    : modelType === "2PL" ? "Asymptotes are fixed; Hill slope and EC50 are fitted"
                    : "All parameters fixed except EC50"}
                </p>
              </div>
            )}

            {/* Normalize toggle */}
            <div style={{
              marginTop: 10,
              paddingTop: 10,
              borderTop: `1px solid ${t.panelBorder}`,
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}>
              <button
                onClick={() => setNormalize(!normalize)}
                style={{
                  width: 36, height: 20,
                  borderRadius: 10,
                  border: `1px solid ${normalize ? t.tealBorder : "rgba(60,100,160,0.15)"}`,
                  background: normalize ? t.tealBg : t.btnInactive,
                  cursor: "pointer",
                  position: "relative",
                  transition: "all 0.2s",
                  padding: 0,
                }}
              >
                <div style={{
                  width: 14, height: 14,
                  borderRadius: 7,
                  background: normalize ? (t.teal || "#00e6b4") : "rgba(140,170,210,0.3)",
                  position: "absolute",
                  top: 2,
                  left: normalize ? 18 : 2,
                  transition: "all 0.2s",
                }} />
              </button>
              <span style={{ fontSize: 10, color: normalize ? t.teal : t.textDim }}>
                Normalize (0-100%)
              </span>
            </div>
            {normalize && (
              <p style={{ fontSize: 8, color: t.textDim, marginTop: 4 }}>
                Responses scaled to 0-100% using raw min/max before fitting
              </p>
            )}

            {/* Weighting. Assay response is usually heteroscedastic, so an
                unweighted fit lets the top of the curve dominate. Which scheme
                is right depends on the assay's variance structure, which the
                engine measures from the replicates and reports below. */}
            <div style={{ marginTop: 10, paddingTop: 10, borderTop: `1px solid ${t.panelBorder}` }}>
              <div style={{ fontSize: 10, color: t.labelDim, marginBottom: 6 }}>Weighting</div>
              <div style={{ display: "flex", gap: 4 }}>
                {WEIGHTING_TYPES.map(w => (
                  <button
                    key={w}
                    onClick={() => setWeightsType(w)}
                    disabled={normalize && w !== "none" && w !== "1/SD^2"}
                    title={normalize && w !== "none" && w !== "1/SD^2"
                      ? "Relative weighting is not valid on normalised data"
                      : undefined}
                    style={{
                      flex: 1, padding: "4px 0", fontSize: 9,
                      fontFamily: "'JetBrains Mono', monospace",
                      borderRadius: 4,
                      border: `1px solid ${weightsType === w ? t.tealBorder : "rgba(60,100,160,0.15)"}`,
                      background: weightsType === w ? t.tealBg : t.btnInactive,
                      color: normalize && w !== "none" && w !== "1/SD^2" ? t.textFaint
                        : weightsType === w ? t.teal : t.textDim,
                      cursor: normalize && w !== "none" && w !== "1/SD^2" ? "not-allowed" : "pointer",
                    }}
                  >{w === "none" ? "None" : w}</button>
                ))}
              </div>
              {varianceHint && (
                <p style={{ fontSize: 8, color: t.textDim, marginTop: 5, lineHeight: 1.5 }}>
                  {varianceHint}
                </p>
              )}
              {fitResult?.weighting?.warning && (
                <p style={{
                  fontSize: 8, color: t.orange, marginTop: 5, lineHeight: 1.5,
                  padding: "5px 6px", borderRadius: 4,
                  background: "rgba(255,180,50,0.08)",
                  border: "1px solid rgba(255,180,50,0.2)",
                }}>
                  {fitResult.weighting.warning}
                </p>
              )}
            </div>
          </div>

          {/* Model Comparison Panel (Auto mode) */}
          {comparison && comparison.fit5PL && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 10, textTransform: "uppercase", letterSpacing: 1, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span>Model Comparison</span>
                {overlayMode && overlayEditIndex !== null && allFitResults?.[overlayEditIndex] && (() => {
                  const c = getCompoundStyle(allFitResults[overlayEditIndex].name, overlayEditIndex).color;
                  return <span style={{ fontSize: 9, color: c, fontWeight: 700, textTransform: "none", letterSpacing: 0, padding: "2px 6px", border: `1px solid ${c}55`, borderRadius: 4, background: `${c}18` }}>{allFitResults[overlayEditIndex].name}</span>;
                })()}
              </div>
              {/* Information criteria are computed from the minimised objective,
                  which the weights rescale. 4PL vs 5PL here is a fair comparison
                  (same weighting), but the absolute values are not comparable to
                  a run with different weighting. */}
              {fitResult?.weighting?.applied && fitResult.weighting.applied !== "none" && (
                <p style={{ fontSize: 8, color: t.textDim, marginBottom: 8, lineHeight: 1.5 }}>
                  Fitted with {fitResult.weighting.applied} weighting. AICc/BIC/SSR are on
                  the weighted scale — compare 4PL against 5PL here, but not against a fit
                  using different weighting.
                </p>
              )}
              
              {/* Comparison table */}
              <div style={{ fontSize: 10 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 4, marginBottom: 4 }}>
                  <span style={{ color: t.textDim }}></span>
                  <span style={{ color: t.blue, fontWeight: 600, textAlign: "center" }}>4PL</span>
                  <span style={{ color: t.purple, fontWeight: 600, textAlign: "center" }}>5PL</span>
                </div>
                {[
                  { label: "R²", v4: comparison.fit4PL.r2.toFixed(6), v5: comparison.fit5PL.r2.toFixed(6) },
                  { label: "AICc", v4: comparison.fit4PL.aicc.toFixed(1), v5: comparison.fit5PL.aicc.toFixed(1) },
                  { label: "BIC", v4: comparison.fit4PL.bic.toFixed(1), v5: comparison.fit5PL.bic.toFixed(1) },
                  { label: "SSR", v4: comparison.fit4PL.ssr.toExponential(3), v5: comparison.fit5PL.ssr.toExponential(3) },
                ].map((row, idx) => (
                  <div key={idx} style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: 4,
                    padding: "4px 0",
                    borderTop: "1px solid rgba(60,100,160,0.06)",
                  }}>
                    <span style={{ color: t.textMuted }}>{row.label}</span>
                    <span style={{ textAlign: "center", color: comparison.selected === "4PL" && row.label === "AICc" ? "#00e6b4" : "#c8daf0" }}>{row.v4}</span>
                    <span style={{ textAlign: "center", color: comparison.selected === "5PL" && row.label === "AICc" ? "#00e6b4" : "#c8daf0" }}>{row.v5}</span>
                  </div>
                ))}
                {comparison.fit5PL && (
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    gap: 4,
                    padding: "4px 0",
                    borderTop: "1px solid rgba(60,100,160,0.06)",
                  }}>
                    <span style={{ color: t.textMuted }}>S param</span>
                    <span style={{ textAlign: "center", color: t.textFaint }}>—</span>
                    <span style={{ textAlign: "center", color: Math.abs(comparison.fit5PL.params[4] - 1) < 0.05 ? "#ffb432" : "#c8daf0" }}>
                      {comparison.fit5PL.params[4].toFixed(4)}
                    </span>
                  </div>
                )}
              </div>

              {/* Selection result */}
              <div style={{
                marginTop: 10,
                padding: "8px 10px",
                background: comparison.selected === "4PL" ? "rgba(59,158,255,0.08)" : "rgba(168,85,247,0.08)",
                border: `1px solid ${comparison.selected === "4PL" ? "rgba(59,158,255,0.2)" : "rgba(168,85,247,0.2)"}`,
                borderRadius: 6,
              }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: comparison.selected === "4PL" ? t.blue : t.purple, marginBottom: 2 }}>
                  ▸ {comparison.selected} Selected
                </div>
                <div style={{ fontSize: 9, color: t.textMuted, lineHeight: 1.5 }}>
                  {comparison.reason}
                </div>
              </div>

              {/* Toggle to view the other model */}
              <div style={{ display: "flex", gap: 6, marginTop: 8 }}>
                {["4PL", "5PL"].map(m => (
                  <button
                    key={m}
                    onClick={() => {
                      const fit = m === "4PL" ? comparison.fit4PL : comparison.fit5PL;
                      if (fit) { setActiveModel(m); setFitResult(fit); }
                    }}
                    style={{
                      flex: 1,
                      padding: "6px 0",
                      background: activeModel === m ? "rgba(0,230,180,0.1)" : t.btnInactive,
                      border: `1px solid ${activeModel === m ? "rgba(0,230,180,0.3)" : "rgba(60,100,160,0.1)"}`,
                      borderRadius: 4,
                      color: activeModel === m ? "#00e6b4" : "rgba(160,190,230,0.4)",
                      fontSize: 9,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      textTransform: "uppercase",
                      letterSpacing: 0.5,
                    }}
                  >
                    View {m}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Overlay molecule indicator */}
          {overlayMode && (() => {
            if (overlayEditIndex === null || !allFitResults?.[overlayEditIndex]) {
              return (
                <div style={{
                  padding: "8px 12px",
                  background: "rgba(60,100,160,0.07)",
                  border: "1px dashed rgba(140,170,210,0.25)",
                  borderRadius: 6,
                  fontSize: 10,
                  color: "rgba(160,190,230,0.5)",
                  fontFamily: "'JetBrains Mono', monospace",
                  textAlign: "center",
                }}>
                  ← Click a molecule name in the list to view its fit details
                </div>
              );
            }
            const editColor = getCompoundStyle(allFitResults[overlayEditIndex].name, overlayEditIndex).color;
            return (
              <div style={{
                padding: "8px 12px",
                background: `${editColor}18`,
                border: `1px solid ${editColor}55`,
                borderLeft: `4px solid ${editColor}`,
                borderRadius: 6,
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <span style={{ fontSize: 9, color: editColor, textTransform: "uppercase", letterSpacing: 1, fontFamily: "'JetBrains Mono', monospace", flexShrink: 0 }}>Viewing</span>
                <span style={{ fontSize: 13, fontWeight: 700, color: editColor, fontFamily: "'Space Grotesk', sans-serif", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{allFitResults[overlayEditIndex].name}</span>
              </div>
            );
          })()}

          {/* Fit Button */}
          <button
            onClick={runFit}
            style={{
              padding: "14px 0",
              background: "linear-gradient(135deg, rgba(59,158,255,0.2), rgba(0,230,180,0.2))",
              border: "1px solid rgba(59,158,255,0.3)",
              borderRadius: 8,
              color: t.blue,
              fontSize: 14,
              fontWeight: 700,
              cursor: "pointer",
              fontFamily: "'Space Grotesk', sans-serif",
              letterSpacing: 0.5,
              transition: "all 0.2s",
            }}
            onMouseOver={(e) => {
              e.target.style.background = "linear-gradient(135deg, rgba(59,158,255,0.3), rgba(0,230,180,0.3))";
              e.target.style.borderColor = "rgba(59,158,255,0.5)";
            }}
            onMouseOut={(e) => {
              e.target.style.background = "linear-gradient(135deg, rgba(59,158,255,0.2), rgba(0,230,180,0.2))";
              e.target.style.borderColor = "rgba(59,158,255,0.3)";
            }}
          >
            FIT MODEL
          </button>

          {error && (
            <div style={{
              padding: 12,
              background: "rgba(255,80,80,0.1)",
              border: "1px solid rgba(255,80,80,0.3)",
              borderRadius: 6,
              color: t.red,
              fontSize: 11,
            }}>
              {error}
            </div>
          )}

          {/* Results */}
          {fitResult && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: t.label, marginBottom: 12, textTransform: "uppercase", letterSpacing: 1, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span>Fit Parameters</span>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  {overlayMode && overlayEditIndex !== null && allFitResults?.[overlayEditIndex] && (() => {
                    const c = getCompoundStyle(allFitResults[overlayEditIndex].name, overlayEditIndex).color;
                    return <span style={{ fontSize: 9, color: c, fontWeight: 700, textTransform: "none", letterSpacing: 0, padding: "2px 6px", border: `1px solid ${c}55`, borderRadius: 4, background: `${c}18` }}>{allFitResults[overlayEditIndex].name}</span>;
                  })()}
                  <span style={{ color: activeModel === "5PL" ? t.purple : ["1PL","2PL"].includes(activeModel) ? t.orange : t.blue, fontSize: 10 }}>{activeModel}</span>
                </div>
              </div>
              {paramLabels.map((label, i) => (
                <div key={i} style={{
                  padding: "6px 0",
                  borderBottom: i < paramLabels.length - 1 ? "1px solid rgba(60,100,160,0.08)" : "none",
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 11, color: t.labelDim, display: "flex", alignItems: "center", gap: 4 }}>
                      {label}
                      {fixedParams.has(i) && (
                        <span style={{
                          fontSize: 7, padding: "1px 4px", borderRadius: 3,
                          background: "rgba(255,180,50,0.12)", border: "1px solid rgba(255,180,50,0.25)",
                          color: t.orange, fontWeight: 600, textTransform: "uppercase",
                        }}>fixed</span>
                      )}
                    </span>
                    <span style={{
                      fontSize: 12,
                      fontWeight: 600,
                      color: i === 2 ? t.orange : t.teal,
                    }}>
                      {fmtParam(fitResult.params[i])}
                    </span>
                  </div>
                  {/* Standard error and 95% CI. The EC50 interval is computed in
                      log space and back-transformed, so it is asymmetric. */}
                  {fitResult.ci?.[i] && (
                    <div style={{
                      display: "flex", justifyContent: "space-between", alignItems: "baseline",
                      marginTop: 2, fontSize: 9, color: t.textDim,
                      fontFamily: "'JetBrains Mono', monospace",
                    }}>
                      <span>{fitResult.se?.[i] != null ? `± ${fmtParam(fitResult.se[i])}` : ""}</span>
                      <span title="95% confidence interval">
                        95% CI {fmtParam(fitResult.ci[i].lo)} – {fmtParam(fitResult.ci[i].hi)}
                      </span>
                    </div>
                  )}
                </div>
              ))}

              {/* Why no intervals, when there are none to show. */}
              {fitResult.ci == null && (
                <p style={{ fontSize: 8, color: t.textDim, marginTop: 6, lineHeight: 1.5 }}>
                  No confidence intervals: {fitResult.n <= fitResult.k
                    ? `${fitResult.n} points cannot support ${fitResult.k} parameters.`
                    : "the parameters are not separately identifiable from this data."}
                </p>
              )}

              {/* Biological EC50 for 5PL */}
              {activeModel === "5PL" && fitResult.bioEC50 && (
                <div style={{
                  marginTop: 8,
                  padding: "8px 0",
                  borderTop: `1px solid ${t.panelBorder}`,
                }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 10, color: t.labelDim }}>Parametric EC50</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color: t.orange }}>
                      {fitResult.params[2] < 0.01 || fitResult.params[2] > 10000
                        ? fitResult.params[2].toExponential(3)
                        : fitResult.params[2].toPrecision(4)}
                    </span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ fontSize: 10, color: t.labelDim }}>Biological EC50</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color: t.teal }}>
                      {fitResult.bioEC50 < 0.01 || fitResult.bioEC50 > 10000
                        ? fitResult.bioEC50.toExponential(3)
                        : fitResult.bioEC50.toPrecision(4)}
                    </span>
                  </div>
                  <p style={{ fontSize: 8, color: t.textDim, marginTop: 4 }}>
                    Biological EC50 = concentration at half-maximal response
                  </p>
                </div>
              )}

              <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${t.panelBorder}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: t.labelDim }}>R²</span>
                  <span style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: fitResult.r2 > 0.99 ? t.teal : fitResult.r2 > 0.95 ? t.orange : t.red,
                  }}>
                    {fitResult.r2.toFixed(6)}
                  </span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                  <span style={{ fontSize: 11, color: t.labelDim }}>RMSE</span>
                  <span style={{ fontSize: 12, color: t.text }}>{fitResult.rmse.toFixed(6)}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontSize: 11, color: t.labelDim }}>SSR</span>
                  <span style={{ fontSize: 12, color: t.text }}>{fitResult.ssr.toFixed(6)}</span>
                </div>
                {fitResult.aicc !== undefined && (
                  <>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: t.labelDim }}>AICc</span>
                      <span style={{ fontSize: 12, color: t.text }}>{fitResult.aicc.toFixed(2)}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                      <span style={{ fontSize: 11, color: t.labelDim }}>BIC</span>
                      <span style={{ fontSize: 12, color: t.text }}>{fitResult.bic.toFixed(2)}</span>
                    </div>
                  </>
                )}
              </div>

              {/* Interpolation */}
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${t.panelBorder}` }}>
                <div style={{ fontSize: 10, color: t.textMuted, marginBottom: 6 }}>
                  INTERPOLATE: Response → Concentration
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <input
                    type="number"
                    value={interpY}
                    onChange={(e) => setInterpY(e.target.value)}
                    placeholder="Response value"
                    style={{
                      flex: 1,
                      padding: "6px 8px",
                      background: t.input,
                      border: `1px solid ${t.inputBorder}`,
                      borderRadius: 4,
                      color: t.text,
                      fontSize: 11,
                      fontFamily: "'JetBrains Mono', monospace",
                      outline: "none",
                    }}
                  />
                  <button
                    onClick={() => {
                      const val = parseFloat(interpY);
                      if (!isNaN(val)) setInterpResult(interpolate(val));
                    }}
                    style={{
                      padding: "6px 12px",
                      background: "rgba(255,180,50,0.15)",
                      border: "1px solid rgba(255,180,50,0.3)",
                      borderRadius: 4,
                      color: t.orange,
                      fontSize: 10,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                    }}
                  >
                    CALC
                  </button>
                </div>
                {interpResult !== null && interpResult !== undefined && (
                  <div style={{ marginTop: 6, fontSize: 11, color: t.orange }}>
                    Concentration: {interpResult.toExponential(4)}
                  </div>
                )}
              </div>

              {/* Actions */}
              <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
                <button
                  onClick={() => setShowResiduals(!showResiduals)}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: showResiduals ? "rgba(0,230,180,0.1)" : t.btnInactive,
                    border: `1px solid ${showResiduals ? "rgba(0,230,180,0.3)" : "rgba(60,100,160,0.15)"}`,
                    borderRadius: 4,
                    color: showResiduals ? "#00e6b4" : "rgba(160,190,230,0.5)",
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  Residuals
                </button>
                <button
                  onClick={exportCSV}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: t.btnInactive,
                    border: `1px solid ${t.panelBorder}`,
                    borderRadius: 4,
                    color: t.textMuted,
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  Export CSV
                </button>
                <button
                  onClick={() => setShowPdfModal(true)}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: t.btnInactive,
                    border: `1px solid ${t.panelBorder}`,
                    borderRadius: 4,
                    color: t.textMuted,
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  PDF Report
                </button>
                <button
                  onClick={() => {
                    const url = mainCanvasRef.current ? mainCanvasRef.current.toDataURL("image/png") : null;
                    setRbChartUrl(url);
                    setShowReportBuilder(true);
                  }}
                  style={{
                    flex: 1,
                    padding: "8px 0",
                    background: "rgba(59,158,255,0.08)",
                    border: `1px solid rgba(59,158,255,0.25)`,
                    borderRadius: 4,
                    color: t.blue,
                    fontSize: 9,
                    cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  PDF Builder
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Charts */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16, order: isMobile ? 1 : 2 }}>
          {/* Multi-Compound nav bar — shown above chart when a multi-compound CSV is loaded */}
          {multiData && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.tealBorder || "rgba(0,230,180,0.25)"}`,
              borderRadius: 10,
              padding: navBarCollapsed ? "8px 16px" : "12px 16px",
              display: "flex",
              alignItems: "center",
              gap: 8,
              flexWrap: "wrap",
            }}>
              {/* Always-visible: label + current compound name */}
              <span style={{ fontSize: 11, fontWeight: 700, color: t.teal, textTransform: "uppercase", letterSpacing: 1, whiteSpace: "nowrap" }}>
                Multi CSV
              </span>
              <span style={{ fontSize: 11, fontWeight: 600, color: t.text, whiteSpace: "nowrap" }}>
                {multiData[multiIndex].name}
              </span>

              {navBarCollapsed ? (
                /* Collapsed state: just spacer + Expand button */
                <>
                  <div style={{ flex: 1, minWidth: 8 }} />
                  <button
                    onClick={() => setNavBarCollapsed(false)}
                    style={{
                      padding: "4px 10px",
                      background: t.btnInactive,
                      border: `1px solid ${t.panelBorder}`,
                      borderRadius: 5,
                      color: t.textMuted,
                      fontSize: 9,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      whiteSpace: "nowrap",
                    }}
                  >▼ Expand</button>
                </>
              ) : (
                /* Expanded state: full controls */
                <>
                  <span style={{ fontSize: 10, color: t.textDim, whiteSpace: "nowrap" }}>
                    n={multiData[multiIndex].nReplicates} · {multiData[multiIndex].points.length} conc
                  </span>

                  {/* Spacer */}
                  <div style={{ flex: 1, minWidth: 8 }} />

                  {/* Prev button */}
                  <button
                    onClick={() => setMultiIndex(i => Math.max(0, i - 1))}
                    disabled={multiIndex === 0}
                    style={{
                      width: 28, height: 28, flexShrink: 0,
                      background: t.btnInactive,
                      border: `1px solid ${t.panelBorder}`,
                      borderRadius: 5,
                      color: multiIndex === 0 ? t.textFaint : t.textMuted,
                      fontSize: 14,
                      cursor: multiIndex === 0 ? "not-allowed" : "pointer",
                      fontFamily: "monospace",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      lineHeight: 1,
                    }}
                  >←</button>

                  {/* Dropdown */}
                  <select
                    value={multiIndex}
                    onChange={e => setMultiIndex(Number(e.target.value))}
                    style={{
                      padding: "4px 8px",
                      background: t.input,
                      border: `1px solid ${t.inputBorder}`,
                      borderRadius: 5,
                      color: t.text,
                      fontSize: 11,
                      fontFamily: "'JetBrains Mono', monospace",
                      outline: "none",
                      cursor: "pointer",
                      maxWidth: 160,
                    }}
                  >
                    {multiData.map((c, i) => (
                      <option key={i} value={i}>{c.name}</option>
                    ))}
                  </select>

                  {/* Counter */}
                  <span style={{ fontSize: 10, color: t.textDim, whiteSpace: "nowrap", minWidth: 36 }}>
                    {multiIndex + 1}/{multiData.length}
                  </span>

                  {/* Next button */}
                  <button
                    onClick={() => setMultiIndex(i => Math.min(multiData.length - 1, i + 1))}
                    disabled={multiIndex === multiData.length - 1}
                    style={{
                      width: 28, height: 28, flexShrink: 0,
                      background: t.btnInactive,
                      border: `1px solid ${t.panelBorder}`,
                      borderRadius: 5,
                      color: multiIndex === multiData.length - 1 ? t.textFaint : t.textMuted,
                      fontSize: 14,
                      cursor: multiIndex === multiData.length - 1 ? "not-allowed" : "pointer",
                      fontFamily: "monospace",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      lineHeight: 1,
                    }}
                  >→</button>

                  {/* Overlay toggle */}
                  <button
                    onClick={() => { setOverlayMode(m => !m); setOverlayEditIndex(null); }}
                    style={{
                      padding: "4px 10px",
                      background: overlayMode ? (t.tealBg || "rgba(0,230,180,0.12)") : t.btnInactive,
                      border: `1px solid ${overlayMode ? (t.teal || "#00e6b4") : t.panelBorder}`,
                      borderRadius: 5,
                      color: overlayMode ? (t.teal || "#00e6b4") : t.textMuted,
                      fontSize: 9,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      whiteSpace: "nowrap",
                      fontWeight: overlayMode ? 700 : 400,
                    }}
                  >
                    {overlayMode ? "⊞ Single View" : "⊞ Overlay"}
                  </button>

                  {/* Stats table toggle — only in overlay mode */}
                  {overlayMode && (
                    <button
                      onClick={() => setStatsTableVisible(v => !v)}
                      style={{
                        padding: "4px 10px",
                        background: statsTableVisible ? (t.tealBg || "rgba(0,230,180,0.12)") : t.btnInactive,
                        border: `1px solid ${statsTableVisible ? (t.teal || "#00e6b4") : t.panelBorder}`,
                        borderRadius: 5,
                        color: statsTableVisible ? (t.teal || "#00e6b4") : t.textMuted,
                        fontSize: 9,
                        cursor: "pointer",
                        fontFamily: "'JetBrains Mono', monospace",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {statsTableVisible ? "Hide Stats" : "📊 Stats"}
                    </button>
                  )}

                  {/* Collapse button */}
                  <button
                    onClick={() => setNavBarCollapsed(true)}
                    style={{
                      padding: "4px 10px",
                      background: t.btnInactive,
                      border: `1px solid ${t.panelBorder}`,
                      borderRadius: 5,
                      color: t.textMuted,
                      fontSize: 9,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      whiteSpace: "nowrap",
                    }}
                  >▲ Collapse</button>

                  {/* Per-compound color + shape pickers — only in overlay mode */}
                  {overlayMode && (
                    <div style={{
                      width: "100%",
                      borderTop: `1px solid rgba(140,170,210,0.1)`,
                      paddingTop: 8,
                      marginTop: 4,
                      display: "flex",
                      flexDirection: "column",
                      gap: 5,
                    }}>
                      {multiData.map((c, i) => {
                        const s = getCompoundStyle(c.name, i);
                        const isSelected = !selectedCompounds || selectedCompounds.has(c.name);
                        const toggleSelected = () => setSelectedCompounds(prev => {
                          const base = prev ?? new Set(multiData.map(x => x.name));
                          const next = new Set(base);
                          if (next.has(c.name)) { next.delete(c.name); } else { next.add(c.name); }
                          return next.size === multiData.length ? null : next; // null = all selected
                        });
                        return (
                          <div key={c.name} style={{ display: "flex", alignItems: "center", gap: 6, opacity: isSelected ? 1 : 0.4 }}>
                            {/* Visibility toggle */}
                            <button
                              onClick={toggleSelected}
                              style={{
                                width: 18, height: 18, flexShrink: 0,
                                background: isSelected ? s.color : "transparent",
                                border: `1px solid ${isSelected ? s.color : "rgba(140,170,210,0.3)"}`,
                                borderRadius: 3, cursor: "pointer", fontSize: 9,
                                color: isSelected ? "#000" : "rgba(160,190,230,0.4)",
                                display: "flex", alignItems: "center", justifyContent: "center",
                              }}
                              title={isSelected ? "Hide molecule" : "Show molecule"}
                            >{isSelected ? "✓" : ""}</button>
                            {/* Color picker */}
                            <input
                              type="color"
                              value={s.color}
                              onChange={e => setCompoundStyles(prev => ({ ...prev, [c.name]: { ...prev[c.name], color: e.target.value } }))}
                              style={{ width: 22, height: 22, border: "none", padding: 0, borderRadius: 3, cursor: "pointer", background: "transparent", flexShrink: 0 }}
                              title="Pick color"
                            />
                            {/* Shape picker */}
                            {[["circle","●"],["square","■"],["triangle","▲"],["diamond","◆"]].map(([sh, icon]) => (
                              <button
                                key={sh}
                                onClick={() => setCompoundStyles(prev => ({ ...prev, [c.name]: { ...prev[c.name], shape: sh } }))}
                                style={{
                                  width: 20, height: 20, flexShrink: 0,
                                  background: s.shape === sh ? s.color : "transparent",
                                  border: `1px solid ${s.shape === sh ? s.color : "rgba(140,170,210,0.25)"}`,
                                  borderRadius: 3,
                                  cursor: "pointer",
                                  fontSize: 10,
                                  color: s.shape === sh ? "#000" : "rgba(160,190,230,0.55)",
                                  lineHeight: 1,
                                  display: "flex", alignItems: "center", justifyContent: "center",
                                }}
                                title={sh}
                              >{icon}</button>
                            ))}
                            <span
                              role="button"
                              tabIndex={0}
                              onClick={() => loadCompoundForOverlayEdit(i)}
                              onKeyDown={e => e.key === "Enter" && loadCompoundForOverlayEdit(i)}
                              title="Click to edit this molecule's fit in the left panel"
                              style={{
                                fontSize: 10,
                                fontFamily: "'JetBrains Mono', monospace",
                                cursor: "pointer",
                                padding: "1px 5px",
                                borderRadius: 3,
                                color: overlayEditIndex === i ? "#0a0f1a" : s.color,
                                background: overlayEditIndex === i ? s.color : "transparent",
                                border: `1px solid ${overlayEditIndex === i ? s.color : "transparent"}`,
                                flexShrink: 0,
                              }}
                            >{c.name}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Chart panel */}
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 10,
            padding: 16,
          }}>
            {/* Point view toggle + export buttons */}
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10, flexWrap: "wrap", gap: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                {parsedData && hasReplicates && (
                  <>
                    <div style={{ display: "flex", gap: 4 }}>
                      {[
                        { key: "individual", label: "Individual Points" },
                        { key: "errorbars", label: "Error Bars" },
                      ].map(opt => (
                        <button
                          key={opt.key}
                          onClick={() => setPointView(opt.key)}
                          style={{
                            padding: "5px 10px",
                            background: pointView === opt.key ? "rgba(0,230,180,0.12)" : t.btnInactive,
                            border: `1px solid ${pointView === opt.key ? "rgba(0,230,180,0.3)" : "rgba(60,100,160,0.1)"}`,
                            borderRadius: 4,
                            color: pointView === opt.key ? "#00e6b4" : "rgba(160,190,230,0.4)",
                            fontSize: 9,
                            fontWeight: 600,
                            cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace",
                            textTransform: "uppercase",
                            letterSpacing: 0.5,
                            transition: "all 0.15s",
                          }}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                    {pointView === "errorbars" && (
                      <div style={{ display: "flex", gap: 4 }}>
                        {[
                          { key: "sd", label: "±SD" },
                          { key: "sem", label: "±SEM" },
                        ].map(opt => (
                          <button
                            key={opt.key}
                            onClick={() => setErrorBarType(opt.key)}
                            style={{
                              padding: "5px 8px",
                              background: errorBarType === opt.key ? "rgba(255,180,50,0.12)" : t.btnInactive,
                              border: `1px solid ${errorBarType === opt.key ? "rgba(255,180,50,0.3)" : "rgba(60,100,160,0.1)"}`,
                              borderRadius: 4,
                              color: errorBarType === opt.key ? "#ffb432" : "rgba(160,190,230,0.4)",
                              fontSize: 9,
                              fontWeight: 600,
                              cursor: "pointer",
                              fontFamily: "'JetBrains Mono', monospace",
                              transition: "all 0.15s",
                            }}
                          >
                            {opt.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
              {fitResult && (
                <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                  <button
                    onClick={() => setShowGraphPopup(true)}
                    style={{
                      padding: "4px 8px",
                      background: t.btnInactive,
                      border: `1px solid rgba(60,100,160,0.1)`,
                      borderRadius: 4,
                      color: "rgba(160,190,230,0.55)",
                      fontSize: 8,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      letterSpacing: 0.5,
                      transition: "all 0.15s",
                    }}
                    title="Open resizable graph preview for export"
                  >⤢ Preview</button>
                  {["PNG", "JPEG"].map(fmt => (
                    <button
                      key={fmt}
                      onClick={() => exportImage(fmt.toLowerCase())}
                      style={{
                        padding: "4px 8px",
                        background: t.btnInactive,
                        border: `1px solid rgba(60,100,160,0.1)`,
                        borderRadius: 4,
                        color: "rgba(160,190,230,0.4)",
                        fontSize: 8,
                        cursor: "pointer",
                        fontFamily: "'JetBrains Mono', monospace",
                        textTransform: "uppercase",
                        letterSpacing: 0.5,
                        transition: "all 0.15s",
                      }}
                    >
                      {fmt}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div ref={chartContainerRef} style={{ position: "relative" }}>
              <canvas
                ref={mainCanvasRef}
                style={{
                  width: "100%",
                  height: isMobile ? (showResiduals ? 240 : 320) : (showResiduals ? 340 : 480),
                  borderRadius: 6,
                  cursor: "crosshair",
                }}
              />
              <div
                ref={tooltipRef}
                style={{
                  display: "none",
                  position: "absolute",
                  top: 0,
                  left: 0,
                  padding: "4px 8px",
                  background: t.tooltip,
                  border: `1px solid ${t.tooltipBorder}`,
                  borderRadius: 4,
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono', monospace",
                  color: t.text,
                  pointerEvents: "none",
                  whiteSpace: "nowrap",
                  zIndex: 10,
                  backdropFilter: "blur(4px)",
                }}
              />

              {/* Draggable stats table — floats inside chart area in overlay mode */}
              {overlayMode && allFitResults && statsTableVisible && (
                <div style={{
                  position: "absolute",
                  left: statsTablePos.x,
                  top: statsTablePos.y,
                  background: "rgba(10,15,26,0.88)",
                  border: "1px solid rgba(140,170,210,0.2)",
                  borderRadius: 8,
                  zIndex: 20,
                  minWidth: 220,
                  backdropFilter: "blur(8px)",
                  userSelect: "none",
                  fontFamily: "'JetBrains Mono', monospace",
                }}>
                  {/* Drag handle */}
                  <div
                    onMouseDown={e => {
                      dragRef.current = { dragging: true, startX: e.clientX, startY: e.clientY, startPosX: statsTablePos.x, startPosY: statsTablePos.y };
                      e.preventDefault();
                    }}
                    style={{
                      cursor: "grab",
                      padding: "5px 10px",
                      borderBottom: "1px solid rgba(140,170,210,0.12)",
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    <span style={{ fontSize: 9, fontWeight: 700, color: t.teal || "#00e6b4", letterSpacing: 1 }}>MOLECULE STATS</span>
                    <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
                      {/* Column picker toggle */}
                      <div style={{ position: "relative" }}>
                        <button
                          onMouseDown={e => e.stopPropagation()}
                          onClick={() => setStatsColPickerOpen(v => !v)}
                          title="Choose columns"
                          style={{ background: "none", border: "none", color: statsColPickerOpen ? (t.teal || "#00e6b4") : "rgba(160,190,230,0.45)", cursor: "pointer", fontSize: 11, lineHeight: 1, padding: "0 3px" }}
                        >⊞</button>
                        {statsColPickerOpen && (
                          <div
                            onMouseDown={e => e.stopPropagation()}
                            style={{
                              position: "absolute", right: 0, top: "100%", marginTop: 4,
                              background: "rgba(12,18,32,0.97)", border: "1px solid rgba(140,170,210,0.25)",
                              borderRadius: 6, padding: "6px 0", zIndex: 50, minWidth: 130,
                              boxShadow: "0 8px 24px rgba(0,0,0,0.6)",
                            }}
                          >
                            {RB_STAT_COLUMNS.filter(c => c.key !== "name").map(col => (
                              <label
                                key={col.key}
                                style={{
                                  display: "flex", alignItems: "center", gap: 7,
                                  padding: "3px 10px", cursor: "pointer",
                                  color: statsTableCols.includes(col.key) ? "rgba(200,220,250,0.9)" : "rgba(140,170,210,0.45)",
                                  fontSize: 10,
                                }}
                              >
                                <input
                                  type="checkbox"
                                  checked={statsTableCols.includes(col.key)}
                                  onChange={() => setStatsTableCols(prev =>
                                    prev.includes(col.key) ? prev.filter(k => k !== col.key) : [...prev, col.key]
                                  )}
                                  style={{ accentColor: t.teal || "#00e6b4", width: 11, height: 11 }}
                                />
                                {col.label}
                              </label>
                            ))}
                          </div>
                        )}
                      </div>
                      <button
                        onMouseDown={e => e.stopPropagation()}
                        onClick={() => setStatsTableVisible(false)}
                        style={{ background: "none", border: "none", color: "rgba(160,190,230,0.5)", cursor: "pointer", fontSize: 12, lineHeight: 1, padding: "0 2px" }}
                      >×</button>
                    </div>
                  </div>
                  {/* Stats table */}
                  <table style={{ width: "100%", fontSize: 10, borderCollapse: "collapse", padding: "4px 0" }}>
                    <thead>
                      <tr style={{ color: "rgba(160,190,230,0.45)", textAlign: "left" }}>
                        <th style={{ padding: "4px 8px", fontWeight: 600 }}>Molecule</th>
                        {statsTableCols.map(key => {
                          const col = RB_STAT_COLUMNS.find(c => c.key === key);
                          return col ? <th key={key} style={{ padding: "4px 8px", fontWeight: 600 }}>{col.label}</th> : null;
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {allFitResults.map((r, i) => ({ r, i })).filter(({ r }) => !selectedCompounds || selectedCompounds.has(r.name)).map(({ r, i }) => {
                        const s = getCompoundStyle(r.name, i);
                        return (
                          <tr
                            key={r.name}
                            onClick={() => loadCompoundForOverlayEdit(i)}
                            style={{
                              borderTop: "1px solid rgba(140,170,210,0.07)",
                              cursor: "pointer",
                              background: overlayEditIndex === i ? "rgba(255,255,255,0.06)" : "transparent",
                            }}
                          >
                            <td style={{ padding: "3px 8px" }}>
                              <span style={{ color: s.color }}>●</span>{" "}
                              <span style={{ color: "rgba(200,220,250,0.8)" }}>{r.name.length > 12 ? r.name.slice(0, 11) + "…" : r.name}</span>
                            </td>
                            {statsTableCols.map(key => {
                              const col = RB_STAT_COLUMNS.find(c => c.key === key);
                              const isDim = key === "model" || key === "n";
                              return col ? (
                                <td key={key} style={{ padding: "3px 8px", color: isDim ? "rgba(160,190,230,0.4)" : "rgba(200,220,250,0.7)" }}>
                                  {col.fmt(r)}
                                </td>
                              ) : null;
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Axis range override inputs */}
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
              <button
                onClick={() => setXAxisLog(v => !v)}
                style={{
                  padding: "2px 7px", background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`, borderRadius: 4,
                  color: xAxisLog ? t.teal || "#00e6b4" : t.textMuted,
                  fontSize: 9, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace",
                }}
              >{xAxisLog ? "Log X" : "Linear X"}</button>
              <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace" }}>{xAxisLog ? "X (log₁₀):" : "X range:"}</span>
              {[["xMin", axisXMin, setAxisXMin, "auto"], ["xMax", axisXMax, setAxisXMax, "auto"]].map(([key, val, setter, ph]) => (
                <input
                  key={key}
                  type="number"
                  value={val}
                  placeholder={ph}
                  onChange={e => setter(e.target.value)}
                  style={{
                    width: 58, padding: "2px 5px",
                    background: t.input, border: `1px solid ${t.inputBorder}`,
                    borderRadius: 4, color: t.text, fontSize: 10,
                    fontFamily: "'JetBrains Mono', monospace", outline: "none",
                  }}
                />
              ))}
              <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace", marginLeft: 6 }}>Y axis:</span>
              {[["yMin", axisYMin, setAxisYMin, "auto"], ["yMax", axisYMax, setAxisYMax, "auto"]].map(([key, val, setter, ph]) => (
                <input
                  key={key}
                  type="number"
                  value={val}
                  placeholder={ph}
                  onChange={e => setter(e.target.value)}
                  style={{
                    width: 68, padding: "2px 5px",
                    background: t.input, border: `1px solid ${t.inputBorder}`,
                    borderRadius: 4, color: t.text, fontSize: 10,
                    fontFamily: "'JetBrains Mono', monospace", outline: "none",
                  }}
                />
              ))}
              {(axisXMin || axisXMax || axisYMin || axisYMax) && (
                <button
                  onClick={() => { setAxisXMin(""); setAxisXMax(""); setAxisYMin(""); setAxisYMax(""); }}
                  style={{
                    padding: "2px 8px", background: t.btnInactive,
                    border: `1px solid ${t.panelBorder}`, borderRadius: 4,
                    color: t.textMuted, fontSize: 9, cursor: "pointer",
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >↺ Auto</button>
              )}
              <span style={{ fontSize: 9, color: t.textMuted, fontFamily: "'JetBrains Mono', monospace", marginLeft: 6 }}>Y fmt:</span>
              <button
                onClick={() => setYAxisFormat(f => f === "decimal" ? "scientific" : "decimal")}
                style={{
                  padding: "2px 7px", background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`, borderRadius: 4,
                  color: yAxisFormat === "scientific" ? t.teal || "#00e6b4" : t.textMuted,
                  fontSize: 9, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace",
                }}
              >{yAxisFormat === "scientific" ? "Sci" : "Dec"}</button>
              <button
                onClick={() => setYAxisDecimals(d => Math.max(0, d - 1))}
                disabled={yAxisDecimals <= 0}
                style={{
                  padding: "2px 5px", background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`, borderRadius: 4,
                  color: t.textMuted, fontSize: 9, cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >−</button>
              <span style={{ fontSize: 9, color: t.text, fontFamily: "'JetBrains Mono', monospace", minWidth: 10, textAlign: "center" }}>{yAxisDecimals}</span>
              <button
                onClick={() => setYAxisDecimals(d => Math.min(6, d + 1))}
                disabled={yAxisDecimals >= 6}
                style={{
                  padding: "2px 5px", background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`, borderRadius: 4,
                  color: t.textMuted, fontSize: 9, cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >+</button>
            </div>
          </div>

          {showResiduals && fitResult && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <canvas
                ref={residCanvasRef}
                style={{
                  width: "100%",
                  height: isMobile ? 100 : 140,
                  borderRadius: 6,
                }}
              />
            </div>
          )}

          {/* Grubbs Outlier Test Panel */}
          {parsedData && hasReplicates && (
            <div style={{
              background: t.panel,
              border: `1px solid ${t.panelBorder}`,
              borderRadius: 10,
              padding: 16,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 600, color: t.label, textTransform: "uppercase", letterSpacing: 1, display: "flex", alignItems: "center", gap: 8 }}>
                  <span>Grubbs' Outlier Test</span>
                  {overlayMode && overlayEditIndex !== null && allFitResults?.[overlayEditIndex] && (() => {
                    const c = getCompoundStyle(allFitResults[overlayEditIndex].name, overlayEditIndex).color;
                    return <span style={{ fontSize: 9, color: c, fontWeight: 700, textTransform: "none", letterSpacing: 0, padding: "2px 6px", border: `1px solid ${c}55`, borderRadius: 4, background: `${c}18` }}>{allFitResults[overlayEditIndex].name}</span>;
                  })()}
                </div>
                <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <span style={{ fontSize: 9, color: t.textDim }}>α =</span>
                  <select
                    value={grubbsAlpha}
                    onChange={(e) => setGrubbsAlpha(parseFloat(e.target.value))}
                    style={{
                      padding: "3px 6px",
                      background: t.input,
                      border: `1px solid ${t.inputBorder}`,
                      borderRadius: 4,
                      color: t.text,
                      fontSize: 10,
                      fontFamily: "'JetBrains Mono', monospace",
                      outline: "none",
                    }}
                  >
                    <option value={0.01}>0.01</option>
                    <option value={0.05}>0.05</option>
                    <option value={0.10}>0.10</option>
                  </select>
                  <button
                    onClick={runGrubbs}
                    style={{
                      padding: "5px 12px",
                      background: "rgba(255,80,106,0.12)",
                      border: "1px solid rgba(255,80,106,0.3)",
                      borderRadius: 4,
                      color: t.red,
                      fontSize: 9,
                      fontWeight: 600,
                      cursor: "pointer",
                      fontFamily: "'JetBrains Mono', monospace",
                      textTransform: "uppercase",
                      letterSpacing: 0.5,
                    }}
                  >
                    Run Test
                  </button>
                </div>
              </div>

              {grubbsResults && (
                <>
                  {/* Summary bar */}
                  <div style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "6px 10px", marginBottom: 10, borderRadius: 6,
                    background: grubbsResults.totalOutliers > 0 ? "rgba(255,80,106,0.08)" : "rgba(0,230,180,0.06)",
                    border: `1px solid ${grubbsResults.totalOutliers > 0 ? "rgba(255,80,106,0.15)" : "rgba(0,230,180,0.15)"}`,
                    flexWrap: "wrap", gap: 6,
                  }}>
                    <span style={{ fontSize: 10, color: grubbsResults.totalOutliers > 0 ? "#ff6b8a" : "#00e6b4" }}>
                      {grubbsResults.totalOutliers > 0
                        ? `${grubbsResults.totalOutliers} outlier${grubbsResults.totalOutliers > 1 ? "s" : ""} detected across ${grubbsResults.groupResults.filter(g => g.outlierCount > 0).length} group${grubbsResults.groupResults.filter(g => g.outlierCount > 0).length > 1 ? "s" : ""}`
                        : "No outliers detected at α=" + grubbsAlpha}
                    </span>
                    <div style={{ display: "flex", gap: 4 }}>
                      {grubbsResults.totalOutliers > 0 && (
                        <button onClick={excludeAllOutliers} style={{
                          padding: "3px 8px", background: "rgba(255,80,106,0.15)", border: "1px solid rgba(255,80,106,0.25)",
                          borderRadius: 3, color: t.red, fontSize: 8, cursor: "pointer",
                          fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                        }}>Exclude All</button>
                      )}
                      {excludedIndices.size > 0 && (
                        <>
                          <button onClick={clearExclusions} style={{
                            padding: "3px 8px", background: "rgba(140,170,210,0.08)", border: "1px solid rgba(140,170,210,0.15)",
                            borderRadius: 3, color: t.labelDim, fontSize: 8, cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                          }}>Clear All</button>
                          <button onClick={refitWithoutExcluded} style={{
                            padding: "3px 8px", background: "rgba(59,158,255,0.15)", border: "1px solid rgba(59,158,255,0.3)",
                            borderRadius: 3, color: t.blue, fontSize: 8, fontWeight: 700, cursor: "pointer",
                            fontFamily: "'JetBrains Mono', monospace", textTransform: "uppercase",
                          }}>Refit ({excludedIndices.size} excl.)</button>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Concentration group list */}
                  <div style={{ display: "flex", flexDirection: isMobile ? "column" : "row", gap: 10 }}>
                    {/* Left: clickable concentration list */}
                    <div style={{ minWidth: isMobile ? "auto" : 120, maxHeight: isMobile ? 150 : 260, overflowY: "auto", display: "flex", flexDirection: isMobile ? "row" : "column", flexWrap: isMobile ? "wrap" : "nowrap", gap: 2 }}>
                      {grubbsResults.groupResults.map((g, gi) => {
                        const key = g.x.toString();
                        const isSelected = selectedGrubbsGroup === key;
                        const hasOutlier = g.outlierCount > 0;
                        const groupExcluded = g.indices ? g.indices.some(idx => excludedIndices.has(idx)) : false;
                        return (
                          <button
                            key={gi}
                            onClick={() => setSelectedGrubbsGroup(isSelected ? null : key)}
                            style={{
                              padding: "5px 8px",
                              background: isSelected ? "rgba(59,158,255,0.12)" : t.btnInactive,
                              border: `1px solid ${isSelected ? "rgba(59,158,255,0.3)" : hasOutlier ? "rgba(255,80,106,0.15)" : "rgba(60,100,160,0.08)"}`,
                              borderRadius: 4,
                              color: hasOutlier ? "#ff6b8a" : groupExcluded ? "rgba(255,180,50,0.7)" : "rgba(160,190,230,0.6)",
                              fontSize: 9,
                              cursor: "pointer",
                              fontFamily: "'JetBrains Mono', monospace",
                              textAlign: "left",
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              gap: 6,
                              transition: "all 0.1s",
                            }}
                          >
                            <span>{g.x < 0.01 || g.x >= 10000 ? g.x.toExponential(2) : g.x.toPrecision(4)}</span>
                            <span style={{ fontSize: 8, opacity: 0.6 }}>
                              n={g.n}
                              {hasOutlier && ` ⚠${g.outlierCount}`}
                            </span>
                          </button>
                        );
                      })}
                    </div>

                    {/* Right: detail view for selected group */}
                    <div style={{ flex: 1, minHeight: 80 }}>
                      {selectedGrubbsGroup ? (() => {
                        const gResult = grubbsResults.groupResults.find(g => g.x.toString() === selectedGrubbsGroup);
                        if (!gResult) return null;
                        const grouped = groupedData.find(g => g.x.toString() === selectedGrubbsGroup);
                        if (!grouped) return null;

                        return (
                          <div>
                            <div style={{ fontSize: 10, color: t.labelDim, marginBottom: 6 }}>
                              Conc: <span style={{ color: t.text, fontWeight: 600 }}>{gResult.x < 0.01 || gResult.x >= 10000 ? gResult.x.toExponential(3) : gResult.x.toPrecision(5)}</span>
                              {gResult.tested && gResult.result && (
                                <span style={{ marginLeft: 10 }}>
                                  G<sub>crit</sub>: <span style={{ color: t.orange }}>{gResult.result.gCrit.toFixed(3)}</span>
                                </span>
                              )}
                            </div>

                            {!gResult.tested && (
                              <div style={{ fontSize: 9, color: t.textDim, fontStyle: "italic" }}>
                                n={gResult.n}: need n≥3 for Grubbs' test
                              </div>
                            )}

                            {gResult.tested && gResult.result && (
                              <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                                {/* Column header */}
                                <div style={{ display: "grid", gridTemplateColumns: "28px 1fr 60px 60px 40px", gap: 4, padding: "2px 0", borderBottom: "1px solid rgba(60,100,160,0.1)" }}>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)" }}></span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)" }}>Value</span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)", textAlign: "right" }}>G stat</span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)", textAlign: "right" }}>Deviation</span>
                                  <span style={{ fontSize: 8, color: "rgba(140,170,210,0.35)", textAlign: "center" }}>Flag</span>
                                </div>
                                {gResult.result.details.map((d, di) => {
                                  const globalIdx = grouped.indices[d.index];
                                  const isExcl = excludedIndices.has(globalIdx);
                                  return (
                                    <div
                                      key={di}
                                      onClick={() => toggleExclusion(globalIdx)}
                                      style={{
                                        display: "grid",
                                        gridTemplateColumns: "28px 1fr 60px 60px 40px",
                                        gap: 4,
                                        padding: "4px 0",
                                        borderBottom: "1px solid rgba(60,100,160,0.04)",
                                        cursor: "pointer",
                                        opacity: isExcl ? 0.4 : 1,
                                        textDecoration: isExcl ? "line-through" : "none",
                                        transition: "opacity 0.15s",
                                      }}
                                    >
                                      <span style={{
                                        width: 18, height: 18, borderRadius: 3, display: "flex", alignItems: "center", justifyContent: "center",
                                        background: isExcl ? "rgba(255,80,106,0.2)" : "rgba(0,230,180,0.1)",
                                        border: `1px solid ${isExcl ? "rgba(255,80,106,0.3)" : "rgba(0,230,180,0.2)"}`,
                                        fontSize: 10,
                                      }}>
                                        {isExcl ? "✕" : "✓"}
                                      </span>
                                      <span style={{ fontSize: 11, color: t.text, fontFamily: "'JetBrains Mono', monospace" }}>
                                        {d.value.toFixed(1)}
                                      </span>
                                      <span style={{
                                        fontSize: 10, textAlign: "right",
                                        color: d.isOutlier ? "#ff6b8a" : "rgba(160,190,230,0.5)",
                                        fontWeight: d.isOutlier ? 700 : 400,
                                      }}>
                                        {d.g.toFixed(3)}
                                      </span>
                                      <span style={{
                                        fontSize: 10, textAlign: "right",
                                        color: d.deviation > 0 ? "rgba(0,230,180,0.6)" : "rgba(255,140,180,0.6)",
                                      }}>
                                        {d.deviation > 0 ? "+" : ""}{d.deviation.toFixed(1)}
                                      </span>
                                      <span style={{ fontSize: 9, textAlign: "center", color: d.isOutlier ? "#ff6b8a" : "rgba(100,140,180,0.3)" }}>
                                        {d.isOutlier ? "OUT" : "—"}
                                      </span>
                                    </div>
                                  );
                                })}
                                {/* Group stats */}
                                <div style={{ marginTop: 6, paddingTop: 6, borderTop: "1px solid rgba(60,100,160,0.1)", fontSize: 9, color: t.textDim, display: "flex", gap: 12 }}>
                                  <span>Mean: <span style={{ color: t.text }}>{gResult.result.mean.toFixed(1)}</span></span>
                                  <span>SD: <span style={{ color: t.text }}>{gResult.result.sd.toFixed(1)}</span></span>
                                  <span>%CV: <span style={{ color: gResult.result.sd / gResult.result.mean * 100 > 20 ? "#ffb432" : "#c8daf0" }}>{(gResult.result.sd / gResult.result.mean * 100).toFixed(1)}%</span></span>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })() : (
                        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: 80, color: "rgba(140,170,210,0.25)", fontSize: 10 }}>
                          Click a concentration to inspect replicates
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {!parsedData && (
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: 300,
              color: t.textFaint,
              fontSize: 13,
            }}>
              Enter data and click FIT MODEL to begin
            </div>
          )}
        </div>
      </div>

      {/* PDF Report Modal */}
      {showPdfModal && (
        <div style={{
          position: "fixed", top: 0, left: 0, right: 0, bottom: 0,
          background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
          display: "flex", alignItems: "center", justifyContent: "center",
          zIndex: 1000,
        }} onClick={() => setShowPdfModal(false)}>
          <div style={{
            background: t.panel,
            border: `1px solid ${t.panelBorder}`,
            borderRadius: 12,
            padding: 24,
            width: 380,
            maxWidth: "90vw",
            maxHeight: "80vh",
            overflowY: "auto",
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{ fontSize: 14, fontWeight: 700, color: t.text, marginBottom: 16 }}>
              PDF Report
            </div>
            <p style={{ fontSize: 9, color: t.textDim, marginBottom: 14, lineHeight: 1.5 }}>
              Select sections to include in the exported report.
            </p>

            {/* Sections */}
            {[
              { key: "modelInfo", label: "Model & Parameters", bold: true },
            ].map(({ key, label, bold }) => (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections[key]}
                  onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text, fontWeight: bold ? 600 : 400 }}>{label}</span>
              </label>
            ))}
            {pdfSections.modelInfo && (
              <div style={{ marginLeft: 24, marginBottom: 8 }}>
                {[
                  { key: "modelParams", label: "Parameter values (A, B, C, D / Bottom, Hill, EC50, Top, S)" },
                  { key: "paramEC50", label: "EC50" },
                  ...(activeModel === "5PL" && fitResult && fitResult.bioEC50 ? [{ key: "paramBioEC50", label: "Biological EC50" }] : []),
                ].map(({ key, label }) => (
                  <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={pdfSections[key]}
                      onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                      style={{ accentColor: t.teal }}
                    />
                    <span style={{ fontSize: 9, color: t.textMuted }}>{label}</span>
                  </label>
                ))}
              </div>
            )}

            {[
              { key: "plot", label: "Fitted Curve Plot" },
              { key: "rawData", label: "Raw Data Table" },
            ].map(({ key, label }) => (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections[key]}
                  onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>{label}</span>
              </label>
            ))}

            {/* Fit Parameters - expandable */}
            <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={pdfSections.fitParams}
                onChange={() => setPdfSections(prev => ({ ...prev, fitParams: !prev.fitParams }))}
                style={{ accentColor: t.teal }}
              />
              <span style={{ fontSize: 10, color: t.text, fontWeight: 600 }}>Goodness of Fit Statistics</span>
            </label>
            {pdfSections.fitParams && (
              <div style={{ marginLeft: 24, marginBottom: 8 }}>
                {[
                  { key: "paramR2", label: "R\u00B2" },
                  { key: "paramRMSE", label: "RMSE" },
                  { key: "paramSSR", label: "SSR" },
                  { key: "paramAIC", label: "AIC" },
                  { key: "paramAICc", label: "AICc" },
                  { key: "paramBIC", label: "BIC" },
                  { key: "paramConverged", label: "Converged / N / K" },
                  { key: "paramNK", label: "N data points / K parameters" },
                ].map(({ key, label }) => (
                  <label key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, cursor: "pointer" }}>
                    <input
                      type="checkbox"
                      checked={pdfSections[key]}
                      onChange={() => setPdfSections(prev => ({ ...prev, [key]: !prev[key] }))}
                      style={{ accentColor: t.teal }}
                    />
                    <span style={{ fontSize: 9, color: t.textMuted }}>{label}</span>
                  </label>
                ))}
              </div>
            )}

            {/* Model comparison */}
            {comparison && comparison.fit5PL && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.modelComparison}
                  onChange={() => setPdfSections(prev => ({ ...prev, modelComparison: !prev.modelComparison }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Model Comparison (4PL vs 5PL)</span>
              </label>
            )}

            {/* Background */}
            {parsedData && parsedData.bgSubtracted > 0 && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.backgroundInfo}
                  onChange={() => setPdfSections(prev => ({ ...prev, backgroundInfo: !prev.backgroundInfo }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Background Subtraction</span>
              </label>
            )}

            {/* Normalization */}
            {parsedData && parsedData.normalized && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.normalizationInfo}
                  onChange={() => setPdfSections(prev => ({ ...prev, normalizationInfo: !prev.normalizationInfo }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Normalization Info</span>
              </label>
            )}

            {/* Outlier results */}
            {grubbsResults && (
              <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={pdfSections.outlierResults}
                  onChange={() => setPdfSections(prev => ({ ...prev, outlierResults: !prev.outlierResults }))}
                  style={{ accentColor: t.teal }}
                />
                <span style={{ fontSize: 10, color: t.text }}>Grubbs' Outlier Test Results</span>
              </label>
            )}

            {/* Actions */}
            <div style={{ display: "flex", gap: 8, marginTop: 18 }}>
              <button
                onClick={() => setShowPdfModal(false)}
                style={{
                  flex: 1, padding: "10px 0",
                  background: t.btnInactive,
                  border: `1px solid ${t.panelBorder}`,
                  borderRadius: 6,
                  color: t.textMuted,
                  fontSize: 10, fontWeight: 600,
                  cursor: "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                }}
              >
                Cancel
              </button>
              <button
                onClick={generatePdfReport}
                disabled={pdfGenerating}
                style={{
                  flex: 2, padding: "10px 0",
                  background: pdfGenerating ? "rgba(0,230,180,0.05)" : "rgba(0,230,180,0.12)",
                  border: `1px solid rgba(0,230,180,0.3)`,
                  borderRadius: 6,
                  color: pdfGenerating ? t.textDim : "#00e6b4",
                  fontSize: 10, fontWeight: 700,
                  cursor: pdfGenerating ? "wait" : "pointer",
                  fontFamily: "'JetBrains Mono', monospace",
                  textTransform: "uppercase",
                  letterSpacing: 0.5,
                }}
              >
                {pdfGenerating ? "Generating..." : "Generate PDF"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Graph Preview Popup */}
      {showGraphPopup && (
        <GraphPopup
          onClose={() => setShowGraphPopup(false)}
          parsedData={parsedData}
          fitResult={fitResult}
          activeModel={activeModel}
          overlayMode={overlayMode}
          allFitResults={allFitResults}
          selectedCompounds={selectedCompounds}
          getCompoundStyle={getCompoundStyle}
          overlayEditIndex={overlayEditIndex}
          loadCompoundForOverlayEdit={loadCompoundForOverlayEdit}
          chartOutlierIndices={chartOutlierIndices}
          excludedIndices={excludedIndices}
          pointView={pointView} setPointView={setPointView}
          errorBarType={errorBarType} setErrorBarType={setErrorBarType}
          xAxisLog={xAxisLog} setXAxisLog={setXAxisLog}
          yAxisFormat={yAxisFormat} setYAxisFormat={setYAxisFormat}
          yAxisDecimals={yAxisDecimals} setYAxisDecimals={setYAxisDecimals}
          axisXMin={axisXMin} setAxisXMin={setAxisXMin}
          axisXMax={axisXMax} setAxisXMax={setAxisXMax}
          axisYMin={axisYMin} setAxisYMin={setAxisYMin}
          axisYMax={axisYMax} setAxisYMax={setAxisYMax}
          statsTableVisible={statsTableVisible} setStatsTableVisible={setStatsTableVisible}
          statsTablePos={statsTablePos} setStatsTablePos={setStatsTablePos}
          statsTableCols={statsTableCols} setStatsTableCols={setStatsTableCols}
          statsColPickerOpen={statsColPickerOpen} setStatsColPickerOpen={setStatsColPickerOpen}
          theme={t}
        />
      )}

      {/* Report Builder */}
      {showReportBuilder && (
        <ReportBuilder
          onClose={() => setShowReportBuilder(false)}
          chartDataUrl={rbChartUrl}
          overlayChartDataUrl={overlayMode && rbChartUrl ? rbChartUrl : null}
          fitResult={fitResult}
          activeModel={activeModel}
          allFitResults={allFitResults}
          getCompoundStyle={getCompoundStyle}
          multiData={multiData}
          renderMolChart={renderMolChart}
        />
      )}
    </div>
  );
}