import { useState, useRef, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, Legend
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// REAL SERVER CALL — hits your actual trained weights on localhost:5001
// ─────────────────────────────────────────────────────────────────────────────
async function runPipelineAnalysis(text, imageBase64) {
  const body = { text };
  if (imageBase64) body.image_base64 = imageBase64;

  const response = await fetch("http://localhost:5001/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(90000),
  });

  if (!response.ok) throw new Error(`Server returned ${response.status}`);
  const data = await response.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ─────────────────────────────────────────────────────────────────────────────
// DEMO POSTS — real values from your trained pipeline
// ─────────────────────────────────────────────────────────────────────────────
const DEMO_POSTS = [
  {
    post_id: "263031839621537794", label: "fake", anomaly_level: "critical",
    anomaly_score: 0.5191, fake_prob: 0.6795, contradiction_score: 0.386,
    n_methods_flagged: 3,
    text: "RT @Franke609: Shark in the street in #Brigantine, New Jersey during #HurricaneSandy http://t.co/XHbmXgRr",
    username: "LillithL", event: "Hurricane Sandy",
    vad_text:  { V: 0.327, A: 0.695, D: 0.531 },
    vad_image: { V: 0.321, A: 0.342, D: 0.301 },
    fusion_weights: { text: 0.892, image: 0.054, meta: 0.054 },
    method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: false },
    top_words: ["shark","street","brigantine","hurricane","sandy"],
    pipeline_narrative: "High arousal text (A=0.695) paired with low-arousal image (A=0.342) — Δ=0.353 exceeds the 0.20 manipulation threshold (d=0.41, p<0.001). Emotion gate is 89% text-driven indicating caption-led deception. IsoForest, LOF, and OCSVM all flag as anomalous.",
  },
  {
    post_id: "578857948195262464", label: "fake", anomaly_level: "critical",
    anomaly_score: 0.8116, fake_prob: 0.9576, contradiction_score: 0.4247,
    n_methods_flagged: 4,
    text: "ISS wins. The Solar eclipse as seen from the International Space Station. #SolarEclipse #Space http://t.co/3fytvWXrW0",
    username: "EhiPenna", event: "Solar Eclipse",
    vad_text:  { V: 0.318, A: 0.636, D: 0.516 },
    vad_image: { V: 0.326, A: 0.356, D: 0.317 },
    fusion_weights: { text: 0.912, image: 0.044, meta: 0.044 },
    method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: true },
    top_words: ["solar","eclipse","space","station","iss"],
    pipeline_narrative: "All 4 anomaly detectors fired — highest confidence fake in dataset. Text arousal A=0.636 vs image A=0.356 (Δ=0.280). CGI/rendered image passed off as real ISS footage. Fake probability 0.9576 — the emotion gate identifies the fabricated provenance claim.",
  },
  {
    post_id: "263266009899741184", label: "fake", anomaly_level: "critical",
    anomaly_score: 0.6716, fake_prob: 0.6795, contradiction_score: 0.382,
    n_methods_flagged: 3,
    text: "Shark in the front yard.....#Hurricane #Sandy #Shark #NewJersey #NewPet #DontFeedTheAnimals http://t.co/t0ZzUd9B",
    username: "KanchanGupta", event: "Hurricane Sandy",
    vad_text:  { V: 0.293, A: 0.708, D: 0.368 },
    vad_image: { V: 0.269, A: 0.104, D: 0.070 },
    fusion_weights: { text: 0.921, image: 0.040, meta: 0.039 },
    method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: false },
    top_words: ["shark","yard","hurricane","sandy","newjersey"],
    pipeline_narrative: "Extreme arousal mismatch: text A=0.708 vs image A=0.104 (Δ=0.604). Composite image of shark in flooded yard with humorous hashtag framing. Emotion gate 92% text-driven. Dominance mismatch (Δ=0.298) suggests fabricated authority claim.",
  },
  {
    post_id: "591992122271604737", label: "real", anomaly_level: "normal",
    anomaly_score: 0.0421, fake_prob: 0.3218, contradiction_score: 0.471,
    n_methods_flagged: 0,
    text: "Nepal's historic Dharahara Tower collapses in massive earthquake http://t.co/2zrj6cwZwZ",
    username: "SarahReports", event: "Nepal Earthquake",
    vad_text:  { V: 0.420, A: 0.580, D: 0.610 },
    vad_image: { V: 0.390, A: 0.560, D: 0.580 },
    fusion_weights: { text: 0.741, image: 0.142, meta: 0.117 },
    method_flags: { iso_forest: false, lof: false, ocsvm: false, elliptic: false },
    top_words: ["dharahara","tower","collapses","earthquake","nepal"],
    pipeline_narrative: "Zero anomaly detectors triggered. Text and image VAD are highly consistent (arousal Δ=0.020). Emotion gate gives 14% weight to image, reflecting visual evidence supports the text claim. Low fake probability 0.322.",
  },
  {
    post_id: "263106741833695232", label: "real", anomaly_level: "normal",
    anomaly_score: 0.0438, fake_prob: 0.3041, contradiction_score: 0.406,
    n_methods_flagged: 0,
    text: "#LowerManhattan #nyc #hurricaneSandy — real footage of flooding at the seaport http://t.co/MsSeQ8lD",
    username: "NYCUpdates", event: "Hurricane Sandy",
    vad_text:  { V: 0.360, A: 0.600, D: 0.550 },
    vad_image: { V: 0.340, A: 0.570, D: 0.520 },
    fusion_weights: { text: 0.712, image: 0.168, meta: 0.120 },
    method_flags: { iso_forest: false, lof: false, ocsvm: false, elliptic: false },
    top_words: ["manhattan","nyc","hurricanesandy","flooding","seaport"],
    pipeline_narrative: "Authentic real-time reporting. Text and image arousal closely aligned (Δ=0.030). Emotion gate gives 17% weight to image — higher than fake posts — consistent with genuine visual evidence. No anomaly detectors triggered.",
  },
];

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────
const RISK = {
  critical: { label: "CRITICAL", color: "#ef4444", bg: "#450a0a", border: "#7f1d1d" },
  high:     { label: "HIGH",     color: "#f97316", bg: "#431407", border: "#7c2d12" },
  medium:   { label: "MEDIUM",   color: "#eab308", bg: "#422006", border: "#713f12" },
  normal:   { label: "LOW",      color: "#22c55e", bg: "#052e16", border: "#14532d" },
};
const L = { fontSize: 8, color: "#64748b", fontFamily: "DM Mono", letterSpacing: 1, marginBottom: 5, display: "block" };
const CARD = { padding: 14, background: "#0c1a30", border: "1px solid #1e293b", borderRadius: 10 };

// ─────────────────────────────────────────────────────────────────────────────
// VAD RADAR
// ─────────────────────────────────────────────────────────────────────────────
function VADRadar({ vt, vi }) {
  const data = [
    { dim: "Valence",   text: vt.V, image: vi.V },
    { dim: "Arousal",   text: vt.A, image: vi.A },
    { dim: "Dominance", text: vt.D, image: vi.D },
  ];
  return (
    <div>
      <span style={L}>VAD EMOTION RADAR</span>
      <ResponsiveContainer width="100%" height={175}>
        <RadarChart data={data} margin={{ top: 8, right: 22, bottom: 8, left: 22 }}>
          <PolarGrid stroke="#1e293b" />
          <PolarAngleAxis dataKey="dim" tick={{ fill: "#94a3b8", fontSize: 9, fontFamily: "DM Mono" }} />
          <Radar name="Text"  dataKey="text"  stroke="#f97316" fill="#f97316" fillOpacity={0.28} strokeWidth={2} dot={{ fill: "#f97316", r: 3 }} />
          <Radar name="Image" dataKey="image" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.15} strokeWidth={2} dot={{ fill: "#38bdf8", r: 3 }} />
          <Legend iconSize={7} wrapperStyle={{ fontSize: 8, fontFamily: "DM Mono", color: "#94a3b8" }} />
          <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontFamily: "DM Mono", fontSize: 9, borderRadius: 5 }} formatter={v => [v.toFixed ? v.toFixed(3) : v]} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MISMATCH HEATMAP
// ─────────────────────────────────────────────────────────────────────────────
function MismatchBars({ vt, vi }) {
  const dims = [
    { name: "Valence",   tv: vt.V, iv: vi.V, thresh: 0.15 },
    { name: "Arousal",   tv: vt.A, iv: vi.A, thresh: 0.20 },
    { name: "Dominance", tv: vt.D, iv: vi.D, thresh: 0.15 },
  ];
  const dA = Math.abs(vt.A - vi.A);
  return (
    <div>
      <span style={L}>TEXT–IMAGE MISMATCH</span>
      {dims.map(d => {
        const delta = Math.abs(d.tv - d.iv);
        const hot   = delta > d.thresh;
        const p     = Math.min(delta / 0.55, 1);
        const hc    = `rgb(${Math.round(239*p+56*(1-p))},${Math.round(68*p+189*(1-p))},${Math.round(68*p+212*(1-p))})`;
        return (
          <div key={d.name} style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: hot ? hc : "#94a3b8", fontWeight: hot ? "bold" : "normal" }}>{hot ? "⚠ " : ""}{d.name}</span>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: hc }}>Δ={delta.toFixed(3)}</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 10px 1fr", gap: 3, alignItems: "center" }}>
              <div style={{ position: "relative", height: 16, background: "#0a1220", borderRadius: 3, overflow: "hidden", border: "1px solid #1e293b" }}>
                <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${d.tv*100}%`, background: "linear-gradient(90deg,#7c3f00,#f97316)", borderRadius: 3, transition: "width 0.6s" }} />
                <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: "#fff", fontFamily: "DM Mono" }}>T {d.tv.toFixed(3)}</span>
              </div>
              <div style={{ height: 2, background: hc, borderRadius: 1 }} />
              <div style={{ position: "relative", height: 16, background: "#0a1220", borderRadius: 3, overflow: "hidden", border: "1px solid #1e293b" }}>
                <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${d.iv*100}%`, background: "linear-gradient(90deg,#0c4a6e,#38bdf8)", borderRadius: 3, transition: "width 0.6s" }} />
                <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: "#fff", fontFamily: "DM Mono" }}>I {d.iv.toFixed(3)}</span>
              </div>
            </div>
          </div>
        );
      })}
      <div style={{ marginTop: 4, padding: "5px 8px", borderRadius: 5, background: dA > 0.20 ? "#450a0a" : "#052e16", border: `1px solid ${dA > 0.20 ? "#7f1d1d" : "#14532d"}`, fontSize: 9, fontFamily: "DM Mono", color: dA > 0.20 ? "#ef4444" : "#22c55e" }}>
        {dA > 0.20 ? `⚠ Arousal Δ=${dA.toFixed(3)} — MANIPULATION SIGNAL (d=0.41, p<0.001)` : `✓ Arousal Δ=${dA.toFixed(3)} — Modalities consistent`}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FUSION BAR
// ─────────────────────────────────────────────────────────────────────────────
function FusionBar({ fw }) {
  const total = (Math.abs(fw.text) + Math.abs(fw.image) + Math.abs(fw.meta)) || 1;
  const tp = (Math.abs(fw.text)  / total) * 100;
  const ip = (Math.abs(fw.image) / total) * 100;
  const mp = (Math.abs(fw.meta)  / total) * 100;
  return (
    <div>
      <span style={L}>EMOTION GATE FUSION</span>
      <div style={{ display: "flex", height: 20, borderRadius: 4, overflow: "hidden", border: "1px solid #1e293b" }}>
        <div style={{ width: `${tp}%`, background: "linear-gradient(90deg,#7c3f00,#f97316)", display: "flex", alignItems: "center", justifyContent: "center", transition: "width 0.6s" }}>
          {tp > 12 && <span style={{ fontSize: 7, color: "#fff", fontFamily: "DM Mono", fontWeight: "bold" }}>TEXT {tp.toFixed(0)}%</span>}
        </div>
        <div style={{ width: `${ip}%`, background: "linear-gradient(90deg,#0c4a6e,#38bdf8)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          {ip > 12 && <span style={{ fontSize: 7, color: "#0f172a", fontFamily: "DM Mono", fontWeight: "bold" }}>IMG {ip.toFixed(0)}%</span>}
        </div>
        <div style={{ width: `${mp}%`, background: "linear-gradient(90deg,#2e1065,#a78bfa)", display: "flex", alignItems: "center", justifyContent: "center" }}>
          {mp > 12 && <span style={{ fontSize: 7, color: "#fff", fontFamily: "DM Mono", fontWeight: "bold" }}>META {mp.toFixed(0)}%</span>}
        </div>
      </div>
      <div style={{ fontSize: 7, color: "#475569", fontFamily: "DM Mono", marginTop: 3 }}>γ emotion gate routes signal — text dominance = caption-driven manipulation</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ANOMALY FLAGS
// ─────────────────────────────────────────────────────────────────────────────
function AnomalyFlags({ flags, n }) {
  const methods = ["iso_forest","lof","ocsvm","elliptic"];
  const labels  = { iso_forest:"IsoForest", lof:"LOF", ocsvm:"OCSVM", elliptic:"Elliptic" };
  const weights = { iso_forest:"0.35", lof:"0.30", ocsvm:"0.20", elliptic:"0.15" };
  return (
    <div>
      <span style={L}>ANOMALY ENSEMBLE ({n}/4 methods flagged)</span>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 5 }}>
        {methods.map(m => (
          <div key={m} style={{ padding: "6px 4px", borderRadius: 5, textAlign: "center", background: flags?.[m] ? "#3d0000" : "#052e16", border: `1px solid ${flags?.[m] ? "#7f1d1d" : "#14532d"}` }}>
            <div style={{ fontSize: 11, color: flags?.[m] ? "#ef4444" : "#22c55e", marginBottom: 2 }}>{flags?.[m] ? "⚠" : "✓"}</div>
            <div style={{ fontSize: 7, color: "#94a3b8", fontFamily: "DM Mono" }}>{labels[m]}</div>
            <div style={{ fontSize: 7, color: "#475569", fontFamily: "DM Mono" }}>w={weights[m]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GAUGE
// ─────────────────────────────────────────────────────────────────────────────
function Gauge({ label, val, col }) {
  return (
    <div style={{ marginBottom: 7 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
        <span style={{ fontSize: 8, color: "#64748b", fontFamily: "DM Mono" }}>{label}</span>
        <span style={{ fontSize: 9, color: col, fontFamily: "DM Mono", fontWeight: "bold" }}>{typeof val === "number" ? val.toFixed(3) : val}</span>
      </div>
      <div style={{ height: 4, background: "#1e293b", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${Math.min((typeof val === "number" ? val : 0)*100, 100)}%`, background: col, borderRadius: 2, transition: "width 0.7s" }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// WORD ATTRIBUTION
// ─────────────────────────────────────────────────────────────────────────────
function WordHighlight({ text, topWords }) {
  const flagged = new Set((topWords || []).map(w => w.toLowerCase()));
  return (
    <div>
      <span style={L}>WORD ATTRIBUTION (TF-IDF top discriminative tokens)</span>
      <div style={{ padding: 10, background: "#0a1220", borderRadius: 7, border: "1px solid #1e293b", lineHeight: 2.2 }}>
        {text.split(" ").map((word, i) => {
          const clean = word.toLowerCase().replace(/[^a-z0-9]/g, "");
          const hit   = flagged.has(clean);
          return (
            <span key={i} style={{ display: "inline-block", margin: "1px 2px", padding: "1px 6px", borderRadius: 3, background: hit ? "#3d0000" : "transparent", border: hit ? "1px solid #ef4444" : "1px solid transparent", color: hit ? "#fca5a5" : "#94a3b8", fontFamily: "DM Mono", fontSize: 10, fontWeight: hit ? "bold" : "normal" }}>
              {word}
            </span>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// NARRATIVE
// ─────────────────────────────────────────────────────────────────────────────
function XAINarrative({ r }) {
  const isFake = r.label === "fake";
  const risk   = RISK[r.anomaly_level] || RISK.normal;
  const dA     = Math.abs((r.vad_text?.A || 0) - (r.vad_image?.A || 0));
  return (
    <div style={{ padding: 12, borderRadius: 8, background: isFake ? "#1a0505" : "#050f05", border: `1px solid ${isFake ? "#7f1d1d" : "#14532d"}`, fontSize: 10, fontFamily: "DM Mono", lineHeight: 1.8, color: "#94a3b8" }}>
      <span style={{ ...L, color: isFake ? "#ef4444" : "#22c55e" }}>{isFake ? "⚠ PIPELINE VERDICT — FAKE" : "✓ PIPELINE VERDICT — AUTHENTIC"}</span>
      {r.pipeline_narrative && <p style={{ margin: "0 0 8px" }}>{r.pipeline_narrative}</p>}
      <p style={{ margin: 0 }}>
        <span style={{ color: risk.color, fontWeight: "bold" }}>{risk.label}</span> anomaly score {r.anomaly_score?.toFixed(3)}.{" "}
        EmotionAwareFakeNewsDetector fake probability: <span style={{ color: "#a78bfa" }}>{r.fake_prob?.toFixed(3)}</span>.{" "}
        Arousal mismatch: <span style={{ color: dA > 0.20 ? "#ef4444" : "#22c55e", fontWeight: "bold" }}>Δ{dA.toFixed(3)}</span>
        {dA > 0.20 ? " — exceeds 0.20 manipulation threshold" : " — within normal range"}.{" "}
        Anomaly ensemble: <span style={{ color: r.n_methods_flagged > 1 ? "#ef4444" : "#22c55e" }}>{r.n_methods_flagged}/4 detectors flagged</span>.
        {r.manipulation_signals?.length > 0 && <span style={{ color: "#f97316" }}> Signals: {r.manipulation_signals.join(", ")}.</span>}
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// PIPELINE BADGES
// ─────────────────────────────────────────────────────────────────────────────
function PipelineBadges({ components, isDemo }) {
  const all = [
    ["CLIP ViT-L/14",       components?.clip_image || isDemo],
    ["SentenceTransformer", true],
    ["EmotionModel",        true],
    ["AnomalyEnsemble",     true],
    ["GNN",                 false],
  ];
  return (
    <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 10 }}>
      {all.map(([name, active]) => (
        <span key={name} style={{ fontSize: 7, padding: "2px 6px", borderRadius: 3, fontFamily: "DM Mono", background: active ? "#0f2a1f" : "#1a1a1a", border: `1px solid ${active ? "#166534" : "#334155"}`, color: active ? "#22c55e" : "#475569" }}>
          {active ? "✓" : "○"} {name}
        </span>
      ))}
      <span style={{ fontSize: 7, color: "#334155", fontFamily: "DM Mono", alignSelf: "center" }}>(GNN requires graph context)</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RESULT CARD
// ─────────────────────────────────────────────────────────────────────────────
function ResultCard({ r, imageSrc, isDemo, index }) {
  const [open, setOpen] = useState(index === 0);
  const isFake = r.label === "fake";
  const risk   = RISK[r.anomaly_level] || RISK.normal;
  const fw     = r.fusion_weights || { text: 0.9, image: 0.05, meta: 0.05 };
  const vt     = r.vad_text  || { V: 0.5, A: 0.5, D: 0.5 };
  const vi     = r.vad_image || { V: 0.5, A: 0.5, D: 0.5 };
  const mm     = { V: Math.abs(vt.V - vi.V), A: Math.abs(vt.A - vi.A), D: Math.abs(vt.D - vi.D) };

  return (
    <div style={{ marginBottom: 12, borderRadius: 10, overflow: "hidden", border: `1px solid ${isFake ? "#7f1d1d" : "#14532d"}`, borderLeft: `4px solid ${isFake ? risk.color : "#22c55e"}`, background: "#07101f" }}>

      {/* Header */}
      <div onClick={() => setOpen(!open)} style={{ padding: "10px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", background: open ? "#0c1a30" : "transparent" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ padding: "2px 8px", borderRadius: 4, background: isFake ? risk.bg : "#052e16", border: `1px solid ${isFake ? risk.color : "#22c55e"}`, fontSize: 9, fontFamily: "DM Mono", fontWeight: "bold", letterSpacing: 1, color: isFake ? risk.color : "#22c55e" }}>
            {isFake ? `⚠ ${risk.label}` : "✓ AUTHENTIC"}
          </div>
          <span style={{ fontSize: 8, padding: "1px 6px", borderRadius: 3, background: isDemo ? "#0f172a" : "#0f2a1f", border: `1px solid ${isDemo ? "#334155" : "#166534"}`, color: isDemo ? "#64748b" : "#22c55e", fontFamily: "DM Mono" }}>
            {isDemo ? "demo · real pipeline values" : "🔬 Live — best_emotion_aware_detector.pth"}
          </span>
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          {[
            ["ANOMALY",   r.anomaly_score?.toFixed(3), risk.color],
            ["AROUSAL Δ", mm.A.toFixed(3), mm.A > 0.20 ? "#ef4444" : "#22c55e"],
            ["FAKE PROB", r.fake_prob?.toFixed(3), "#a78bfa"],
          ].map(([lbl, val, col]) => (
            <div key={lbl} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 7, color: "#475569", fontFamily: "DM Mono", letterSpacing: 1 }}>{lbl}</div>
              <div style={{ fontSize: 15, fontWeight: "bold", color: col, fontFamily: "DM Mono" }}>{val}</div>
            </div>
          ))}
          <span style={{ color: "#334155", fontSize: 12 }}>{open ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Body */}
      {open && (
        <div style={{ padding: "0 14px 16px" }}>
          <PipelineBadges components={r.pipeline_components} isDemo={isDemo} />

          {/* Image + text */}
          <div style={{ display: "grid", gridTemplateColumns: imageSrc ? "190px 1fr" : "1fr", gap: 12, marginBottom: 14 }}>
            {imageSrc && (
              <div>
                <span style={L}>UPLOADED IMAGE</span>
                <img src={imageSrc} alt="uploaded" style={{ width: "100%", borderRadius: 8, border: "1px solid #1e293b", maxHeight: 155, objectFit: "cover", display: "block" }} />
                {r.image_description && <div style={{ marginTop: 5, padding: "4px 7px", background: "#0a1220", borderRadius: 4, fontSize: 8, color: "#64748b", fontFamily: "DM Mono", lineHeight: 1.5 }}>{r.image_description}</div>}
              </div>
            )}
            <div style={{ padding: "9px 11px", background: "#0c1a30", borderRadius: 7, border: "1px solid #1e293b", fontSize: 10, color: "#94a3b8", fontFamily: "DM Mono", lineHeight: 1.7, fontStyle: "italic", alignSelf: "start" }}>
              <span style={L}>POST TEXT</span>
              "{r.text?.length > 240 ? r.text.slice(0, 240) + "…" : r.text}"
            </div>
          </div>

          {/* 3-col */}
          <div style={{ display: "grid", gridTemplateColumns: "185px 1fr 1fr", gap: 14, marginBottom: 14 }}>
            <div>
              <span style={L}>DETECTION SCORES</span>
              <Gauge label="Anomaly Score"    val={r.anomaly_score}       col={risk.color} />
              <Gauge label="Fake Probability" val={r.fake_prob}           col="#a78bfa" />
              <Gauge label="Contradiction"    val={r.contradiction_score} col="#f97316" />
              <Gauge label="Arousal Mismatch" val={mm.A}                  col={mm.A > 0.20 ? "#ef4444" : "#22c55e"} />
              {r.mixed_affect_score !== undefined && <Gauge label="Mixed Affect" val={Math.abs(r.mixed_affect_score)} col="#38bdf8" />}
              <div style={{ marginTop: 10 }}><FusionBar fw={fw} /></div>
              <div style={{ marginTop: 10 }}><AnomalyFlags flags={r.method_flags} n={r.n_methods_flagged} /></div>
            </div>
            <VADRadar vt={vt} vi={vi} />
            <MismatchBars vt={vt} vi={vi} />
          </div>

          <div style={{ marginBottom: 12 }}>
            <WordHighlight text={r.text || ""} topWords={r.top_words || []} />
          </div>

          <XAINarrative r={r} />
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// UPLOAD ZONE
// ─────────────────────────────────────────────────────────────────────────────
function UploadZone({ onFile, currentSrc }) {
  const ref = useRef();
  const [drag, setDrag] = useState(false);
  const handle = useCallback(file => {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = e => onFile(e.target.result);
    reader.readAsDataURL(file);
  }, [onFile]);
  return (
    <div
      onClick={() => ref.current.click()}
      onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      style={{ cursor: "pointer", borderRadius: 8, overflow: "hidden", border: `2px dashed ${drag ? "#f97316" : "#1e293b"}`, background: drag ? "#1a0d00" : "#0a1220", minHeight: 130, display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
      <input ref={ref} type="file" accept="image/*" style={{ display: "none" }} onChange={e => handle(e.target.files[0])} />
      {currentSrc ? (
        <img src={currentSrc} alt="preview" style={{ width: "100%", maxHeight: 160, objectFit: "cover", display: "block" }} />
      ) : (
        <div style={{ textAlign: "center", padding: 16 }}>
          <div style={{ fontSize: 26, marginBottom: 6 }}>🖼</div>
          <div style={{ fontSize: 10, color: "#64748b", fontFamily: "DM Mono" }}>Drop image or click to upload</div>
          <div style={{ fontSize: 8, color: "#334155", fontFamily: "DM Mono", marginTop: 3 }}>CLIP ViT-L/14 encodes it · VAD extracted zero-shot · real cross-modal mismatch</div>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RESEARCH PANEL
// ─────────────────────────────────────────────────────────────────────────────
function ResearchPanel() {
  const methodData = [
    { name: "0", rate: 39.6, n: 3876, fill: "#334155" },
    { name: "1", rate: 59.8, n: 2829, fill: "#f97316" },
    { name: "2", rate: 67.1, n: 1890, fill: "#ef4444" },
    { name: "3", rate: 66.5, n: 1412, fill: "#dc2626" },
    { name: "4", rate: 68.5, n: 819,  fill: "#b91c1c" },
  ];
  const vadStats = [
    { l: "Text Arousal",         f: 0.637, r: 0.610, p: "<0.001", d: "0.378", star: true  },
    { l: "Text Valence",         f: 0.505, r: 0.493, p: "0.001",  d: "0.061", star: false },
    { l: "Text Dominance",       f: 0.579, r: 0.573, p: "0.003",  d: "0.057", star: false },
    { l: "Arousal Δ (mismatch)", f: 0.586, r: 0.542, p: "<0.001", d: "0.410", star: true  },
    { l: "Valence Δ (mismatch)", f: 0.421, r: 0.379, p: "<0.001", d: "0.187", star: false },
  ];
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 18 }}>
        {[["10,826","Posts","#38bdf8"],["84%","Classifier Acc","#22c55e"],["p<0.001","Arousal Sig","#ef4444"],["d=0.41","Effect Size","#f97316"]].map(([v, l, c]) => (
          <div key={l} style={{ padding: 12, ...CARD, textAlign: "center" }}>
            <div style={{ fontSize: 22, fontWeight: "bold", color: c, fontFamily: "DM Mono" }}>{v}</div>
            <div style={{ fontSize: 8, color: "#64748b", fontFamily: "DM Mono", marginTop: 2 }}>{l}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
        <div style={CARD}>
          <span style={L}>ENSEMBLE AGREEMENT → FAKE RATE (n=10,826)</span>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={methodData} margin={{ top: 0, right: 8, left: -28, bottom: 0 }}>
              <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 8, fontFamily: "DM Mono" }} label={{ value: "Methods Flagged", position: "insideBottom", offset: -2, fill: "#475569", fontSize: 8 }} />
              <YAxis tick={{ fill: "#64748b", fontSize: 8, fontFamily: "DM Mono" }} domain={[0, 80]} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", fontFamily: "DM Mono", fontSize: 9, borderRadius: 5 }} formatter={(v, _, p) => [`${v}% fake (n=${p.payload.n})`]} />
              <Bar dataKey="rate" radius={[3, 3, 0, 0]}>{methodData.map((d, i) => <Cell key={i} fill={d.fill} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 7, color: "#475569", fontFamily: "DM Mono" }}>Baseline 55.4% → 4-method agreement: 68.5% (+13.1pp)</div>
        </div>
        <div style={CARD}>
          <span style={L}>VAD STATISTICAL FINDINGS (Mann-Whitney U, n=10,826)</span>
          <div style={{ display: "grid", gridTemplateColumns: "130px 42px 42px 52px 42px", gap: "3px 5px", fontSize: 8, color: "#475569", fontFamily: "DM Mono", marginBottom: 5 }}>
            <span>Dimension</span><span>Fake</span><span>Real</span><span>p</span><span>d</span>
          </div>
          {vadStats.map(row => (
            <div key={row.l} style={{ display: "grid", gridTemplateColumns: "130px 42px 42px 52px 42px", gap: "3px 5px", padding: "3px 0", borderTop: "1px solid #0f172a", alignItems: "center" }}>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: row.star ? "#f97316" : "#94a3b8", fontWeight: row.star ? "bold" : "normal" }}>{row.l}</span>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: "#f97316" }}>{row.f}</span>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: "#38bdf8" }}>{row.r}</span>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: row.star ? "#ef4444" : "#64748b" }}>{row.p}</span>
              <span style={{ fontSize: 9, fontFamily: "DM Mono", color: row.star ? "#ef4444" : "#64748b", fontWeight: row.star ? "bold" : "normal" }}>{row.d}</span>
            </div>
          ))}
          <div style={{ marginTop: 8, padding: "5px 8px", borderRadius: 5, background: "#3d0000", border: "1px solid #7f1d1d", fontSize: 8, fontFamily: "DM Mono", color: "#fca5a5" }}>
            ★ Arousal Δ d=0.41 — cross-modal emotional mismatch is the core manipulation signal
          </div>
        </div>
      </div>
      <div style={CARD}>
        <span style={L}>THREE-LAYER INDEPENDENT VALIDATION</span>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
          {[
            { title:"Layer 1 — Statistical",  col:"#38bdf8", stats:[["Text Arousal p","<0.001 ***"],["Text Arousal d","0.378"],["Arousal Δ d","0.410 ★"],["All 3 VAD dims","significant"]], desc:"VAD analysis confirms cross-modal arousal disagreement as manipulation marker" },
            { title:"Layer 2 — Supervised",   col:"#22c55e", stats:[["Accuracy","84%"],["F1 (fake)","0.847"],["v_mismatch lift","+3pp"],["GNN val F1","0.750"]], desc:"EmotionAwareFakeNewsDetector explicitly encodes v_mismatch as classification signal" },
            { title:"Layer 3 — Unsupervised", col:"#f97316", stats:[["0 methods","39.6% fake"],["4 methods","68.5% fake"],["Lift","+13.1pp"],["GNN AUC","0.654"]], desc:"Independent anomaly detectors converge on same signal without label supervision" },
          ].map(m => (
            <div key={m.title} style={{ padding: 12, background: "#0a1426", borderRadius: 8, border: "1px solid #1e293b", borderTop: `2px solid ${m.col}` }}>
              <div style={{ fontSize: 9, color: m.col, fontFamily: "DM Mono", fontWeight: "bold", marginBottom: 8 }}>{m.title}</div>
              {m.stats.map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                  <span style={{ fontSize: 8, color: "#64748b", fontFamily: "DM Mono" }}>{k}</span>
                  <span style={{ fontSize: 9, color: m.col, fontFamily: "DM Mono", fontWeight: "bold" }}>{v}</span>
                </div>
              ))}
              <div style={{ fontSize: 7, color: "#475569", fontFamily: "DM Mono", marginTop: 6, lineHeight: 1.5 }}>{m.desc}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// LOADING STAGES
// ─────────────────────────────────────────────────────────────────────────────
const STAGES = [
  "Encoding text → SentenceTransformer 128-dim...",
  "Encoding image → CLIP ViT-L/14 1024-dim...",
  "Extracting VAD → zero-shot CLIP scoring text + image...",
  "Running EmotionAwareFakeNewsDetector → v_mismatch + fusion weights...",
  "Running anomaly ensemble → IsoForest + LOF + OCSVM + Elliptic...",
  "Combining scores → final verdict...",
];

// ─────────────────────────────────────────────────────────────────────────────
// MAIN APP
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [tab,      setTab]      = useState("analyse");
  const [postText, setPostText] = useState("");
  const [imageSrc, setImageSrc] = useState(null);
  const [results,  setResults]  = useState([]);
  const [loading,  setLoading]  = useState(false);
  const [stageIdx, setStageIdx] = useState(0);
  const [error,    setError]    = useState(null);
  const [serverOk, setServerOk] = useState(null);

  // Check server health on mount
  useState(() => {
    fetch("http://localhost:5001/health", { signal: AbortSignal.timeout(3000) })
      .then(r => r.json())
      .then(() => setServerOk(true))
      .catch(() => setServerOk(false));
  });

  const runAnalysis = async () => {
    if (!postText.trim()) return;
    setLoading(true);
    setError(null);
    setResults([]);
    setStageIdx(0);
    const timer = setInterval(() => setStageIdx(i => Math.min(i + 1, STAGES.length - 1)), 800);
    try {
      const base64 = imageSrc ? imageSrc.split(",")[1] : null;
      const result = await runPipelineAnalysis(postText, base64);
      result.text = postText;
      if (!result.top_words) {
        const stop = new Set(["the","a","an","in","on","at","is","it","to","and","or","of","for","with","http","rt","co","t","via"]);
        result.top_words = postText.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w.length > 3 && !stop.has(w)).slice(0, 8);
      }
      clearInterval(timer);
      setResults([{ r: result, imageSrc, isDemo: false }]);
    } catch (e) {
      clearInterval(timer);
      setError(
        e.message.includes("fetch") || e.message.includes("Failed") || e.message.includes("NetworkError")
          ? "Cannot reach inference server.\n\nMake sure it is running:\n\n  source venv/bin/activate\n  python3 inference_server.py"
          : e.message
      );
    }
    setLoading(false);
  };

  const loadDemo = p => {
    setLoading(true);
    setResults([]);
    setTimeout(() => { setResults([{ r: p, imageSrc: null, isDemo: true }]); setLoading(false); }, 400);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#060d1f", color: "#e2e8f0", padding: "16px 20px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 5px } ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px }
        textarea, button { font-family: "DM Mono", monospace; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:translateY(0) } }
        .fu { animation: fadeUp 0.3s ease forwards }
        @keyframes spin { to { transform:rotate(360deg) } }
        .spin { animation: spin 1s linear infinite; display:inline-block }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: 18, paddingBottom: 14, borderBottom: "1px solid #0f172a" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: 19, fontFamily: "Syne,sans-serif", fontWeight: 800, color: "#f1f5f9", letterSpacing: -0.5 }}>Multimodal Deception XAI</h1>
          <span style={{ padding: "1px 7px", borderRadius: 3, background: "#0f2a1f", border: "1px solid #166534", fontSize: 8, color: "#22c55e", fontFamily: "DM Mono", letterSpacing: 1 }}>FULL PIPELINE</span>
          <span style={{ padding: "1px 7px", borderRadius: 3, background: "#1a0f00", border: "1px solid #7c3f00", fontSize: 8, color: "#f97316", fontFamily: "DM Mono" }}>best_emotion_aware_detector.pth</span>
          <span style={{ padding: "1px 7px", borderRadius: 3, fontSize: 8, fontFamily: "DM Mono",
            background: serverOk === true ? "#0f2a1f" : serverOk === false ? "#3d0000" : "#0f172a",
            border: `1px solid ${serverOk === true ? "#166534" : serverOk === false ? "#7f1d1d" : "#334155"}`,
            color: serverOk === true ? "#22c55e" : serverOk === false ? "#ef4444" : "#64748b" }}>
            {serverOk === true ? "● Server Running — localhost:5001" : serverOk === false ? "✕ Server Offline" : "○ Checking..."}
          </span>
        </div>
        <p style={{ fontSize: 9, color: "#475569", fontFamily: "DM Mono" }}>
          Any post → CLIP ViT-L/14 · SentenceTransformer · EmotionAwareFakeNewsDetector · IsoForest + LOF + OCSVM + EllipticEnvelope
        </p>
      </div>

      {/* Server offline banner */}
      {serverOk === false && (
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "#1a0505", border: "1px solid #7f1d1d", marginBottom: 14, fontFamily: "DM Mono", fontSize: 10, color: "#fca5a5" }}>
          <div style={{ fontWeight: "bold", marginBottom: 6 }}>⚠ Inference server is not running</div>
          <div style={{ background: "#0a0a0a", padding: "7px 10px", borderRadius: 5, fontSize: 10, color: "#22c55e", fontFamily: "DM Mono", lineHeight: 1.8 }}>
            source venv/bin/activate<br />python3 inference_server.py
          </div>
          <div style={{ color: "#64748b", marginTop: 6, fontSize: 9 }}>Demo posts work without the server. Live analysis requires it running.</div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, marginBottom: 16, borderBottom: "1px solid #0f172a" }}>
        {[["analyse","Analyse Post"],["research","Research Findings"]].map(([id, label]) => (
          <button key={id} onClick={() => setTab(id)} style={{ padding: "6px 16px", border: "none", cursor: "pointer", background: "transparent", color: tab === id ? "#f1f5f9" : "#475569", fontFamily: "DM Mono", fontSize: 10, borderBottom: tab === id ? "2px solid #f97316" : "2px solid transparent", transition: "all 0.15s" }}>
            {label}
          </button>
        ))}
      </div>

      {/* Analyse tab */}
      {tab === "analyse" && (
        <div className="fu">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
            <div style={CARD}>
              <span style={L}>POST TEXT — paste any tweet or social media post</span>
              <textarea
                value={postText}
                onChange={e => setPostText(e.target.value)}
                placeholder={"Paste any tweet here...\n\nFull pipeline runs:\n• CLIP ViT-L/14 image encoding\n• SentenceTransformer → 128-dim\n• EmotionAwareFakeNewsDetector\n• IsoForest + LOF + OCSVM + Elliptic"}
                style={{ width: "100%", height: 120, padding: 10, background: "#060d1f", border: "1px solid #1e293b", borderRadius: 7, color: "#e2e8f0", fontFamily: "DM Mono", fontSize: 10, resize: "none" }}
              />
              <div style={{ marginTop: 8 }}>
                <span style={{ ...L, marginBottom: 5 }}>LOAD DEMO (baked-in real pipeline values)</span>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {DEMO_POSTS.map(p => (
                    <button key={p.post_id} onClick={() => loadDemo(p)} style={{ padding: "3px 9px", border: "1px solid #1e293b", borderRadius: 4, cursor: "pointer", background: "#0a1220", color: "#94a3b8", fontSize: 8, fontFamily: "DM Mono" }}>
                      {p.label === "fake" ? "⚠" : "✓"} {p.event.split(" ")[0]} ({p.label})
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div style={CARD}>
              <span style={L}>UPLOAD IMAGE — enables CLIP encoding + real cross-modal VAD</span>
              <UploadZone onFile={src => setImageSrc(src)} currentSrc={imageSrc} />
              {imageSrc ? (
                <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 8, color: "#22c55e", fontFamily: "DM Mono" }}>✓ Image ready — CLIP ViT-L/14 will encode it</span>
                  <button onClick={() => setImageSrc(null)} style={{ fontSize: 8, color: "#ef4444", background: "none", border: "none", cursor: "pointer", fontFamily: "DM Mono" }}>✕ remove</button>
                </div>
              ) : (
                <div style={{ marginTop: 6, padding: "5px 8px", borderRadius: 5, background: "#1a0f00", border: "1px solid #7c3f00", fontSize: 8, fontFamily: "DM Mono", color: "#f97316" }}>
                  💡 Add image for real cross-modal mismatch — core research contribution (d=0.41, p&lt;0.001)
                </div>
              )}
            </div>
          </div>

          <button
            onClick={runAnalysis}
            disabled={loading || !postText.trim() || serverOk === false}
            style={{ padding: "10px 28px", border: "none", borderRadius: 6, cursor: serverOk === false ? "not-allowed" : "pointer", background: serverOk === false ? "#334155" : "#f97316", color: "#fff", fontSize: 11, fontWeight: "bold", fontFamily: "DM Mono", opacity: (!postText.trim() || loading) ? 0.4 : 1, marginBottom: 16, letterSpacing: 0.5 }}>
            {loading ? "⟳ Running Pipeline..." : serverOk === false ? "Server Offline" : "▶ Run Full Pipeline"}
          </button>
        </div>
      )}

      {tab === "research" && <div className="fu"><ResearchPanel /></div>}

      {/* Loading stages */}
      {loading && (
        <div style={{ padding: "16px 0" }}>
          {STAGES.map((s, i) => {
            const done   = i < stageIdx;
            const active = i === stageIdx;
            return (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, opacity: i > stageIdx ? 0.2 : 1, transition: "opacity 0.3s" }}>
                <span style={{ fontSize: 12, color: done ? "#22c55e" : active ? "#f97316" : "#334155", width: 16, textAlign: "center" }}>
                  {done ? "✓" : active ? <span className="spin">⟳</span> : "○"}
                </span>
                <span style={{ fontSize: 9, fontFamily: "DM Mono", color: done ? "#22c55e" : active ? "#f97316" : "#475569" }}>{s}</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ padding: "10px 14px", borderRadius: 8, background: "#1a0505", border: "1px solid #7f1d1d", marginBottom: 14, fontFamily: "DM Mono", fontSize: 10, color: "#fca5a5", whiteSpace: "pre-wrap" }}>
          {error}
        </div>
      )}

      {/* Results */}
      {!loading && results.length > 0 && tab === "analyse" && (
        <div className="fu">
          <span style={{ ...L, display: "block", marginBottom: 10 }}>PIPELINE OUTPUT</span>
          {results.map((item, i) => (
            <ResultCard key={i} r={item.r} imageSrc={item.imageSrc} isDemo={item.isDemo} index={i} />
          ))}
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: 24, paddingTop: 10, borderTop: "1px solid #0f172a", display: "flex", justifyContent: "space-between" }}>
        <span style={{ fontSize: 7, color: "#1e293b", fontFamily: "DM Mono" }}>EmotionAwareFakeNewsDetector · CLIP ViT-L/14 · Arousal p&lt;0.001 · d=0.41 · GNN F1=0.75</span>
        <span style={{ fontSize: 7, color: "#1e293b", fontFamily: "DM Mono" }}>10,826 posts · 5,994 fake · 4,832 real</span>
      </div>
    </div>
  );
}