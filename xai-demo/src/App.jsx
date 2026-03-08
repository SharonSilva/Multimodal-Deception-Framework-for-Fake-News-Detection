import { useState, useRef, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, Legend
} from "recharts";

// ─── PIPELINE FUNCTIONS ───────────────────────────────────────────────────────
async function runPipelineAnalysis(text, imageBase64) {
  const body = { text };
  if (imageBase64) body.image_base64 = imageBase64;
  const [res, imgDesc] = await Promise.all([
    fetch("http://localhost:5001/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(120000) }),
    imageBase64 ? describeImageWithGemini(imageBase64) : Promise.resolve(null),
  ]);
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  const vt = data.vad_text, vi = data.vad_image;
  const dA = Math.abs(vt.A - vi.A), dV = Math.abs(vt.V - vi.V);
  const fw = data.fusion_weights, n = data.n_methods_flagged;
  const finalIsFake = data.label === "fake";
  const [aiTextDesc, aiMismatch] = await Promise.all([
    describeTextWithAI(text, vt),
    imgDesc ? explainMismatchWithAI(text, imgDesc, vt, vi, dA, dV, finalIsFake) : Promise.resolve(null),
  ]);
  const textDesc = aiTextDesc || `the text encodes ${vt.A > 0.60 ? "elevated" : "moderate"}-arousal semantics (A=${vt.A.toFixed(2)}, V=${vt.V.toFixed(2)})`;
  let pipeline_narrative = "";
  if (data.image_analysed && imgDesc) {
    const mismatchReason = aiMismatch || (dA > 0.50 ? `Arousal gap Δ=${dA.toFixed(3)} is extreme.` : `Emotional profiles misaligned (Δ=${dA.toFixed(3)}).`);
    pipeline_narrative = finalIsFake
      ? `${textDesc} (A=${vt.A.toFixed(2)}, V=${vt.V.toFixed(2)}) while the image shows ${imgDesc.toLowerCase().replace(/^image depicts /, "").replace(/\.$/, "")} (A=${vi.A.toFixed(2)}, V=${vi.V.toFixed(2)}). ${mismatchReason} Emotion gate weights text at ${Math.round(fw.text * 100)}%.`
      : `Emotionally consistent. ${textDesc} (A=${vt.A.toFixed(2)}, V=${vt.V.toFixed(2)}), image corroborates (A=${vi.A.toFixed(2)}, V=${vi.V.toFixed(2)}). Arousal Δ=${dA.toFixed(3)} within threshold. ${n}/4 anomaly detectors triggered.`;
  } else {
    pipeline_narrative = `Text-only: ${textDesc}. A=${vt.A.toFixed(3)}, V=${vt.V.toFixed(3)}. ${n}/4 detectors flagged. Fake prob ${data.fake_prob.toFixed(3)}.`;
  }
  const manipulation_signals = [];
  if (dA > 0.20) manipulation_signals.push(`Arousal mismatch Δ=${dA.toFixed(3)}`);
  if (dV > 0.15) manipulation_signals.push(`Valence inversion Δ=${dV.toFixed(3)}`);
  if (fw.text > 0.75) manipulation_signals.push(`Caption-driven fusion ${Math.round(fw.text * 100)}%`);
  if (n >= 3) manipulation_signals.push(`${n}/4 anomaly detectors`);
  const stop = new Set(["the","a","an","in","on","at","is","it","to","and","or","of","for","with","http","rt","co","t","via","this","that","was","are"]);
  const top_words = text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(w => w.length > 3 && !stop.has(w)).slice(0, 6);
  return { ...data, label: data.label, top_words, manipulation_signals, pipeline_narrative, image_description: imgDesc || (data.image_analysed ? `Image VAD: V=${vi.V.toFixed(3)}, A=${vi.A.toFixed(3)}` : null) };
}
async function runBatchAnalysis(posts, onProgress) {
  const CHUNK = 10, allResults = [];
  for (let i = 0; i < posts.length; i += CHUNK) {
    const chunk = posts.slice(i, i + CHUNK);
    const res = await fetch("http://localhost:5001/batch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ posts: chunk }), signal: AbortSignal.timeout(300000) });
    if (!res.ok) throw new Error(`Server error: ${res.status}`);
    const data = await res.json(); if (data.error) throw new Error(data.error);
    allResults.push(...data.results); onProgress(Math.min(i + CHUNK, posts.length), posts.length);
  }
  const fake = allResults.filter(r => r.label === "fake").length, real = allResults.filter(r => r.label === "real").length;
  const total = allResults.filter(r => r.label !== "error").length;
  return { results: allResults, summary: { total, fake, real, fake_rate: total > 0 ? fake / total : 0, avg_fake_prob: total > 0 ? allResults.filter(r => r.label !== "error").reduce((s, r) => s + (r.fake_prob || 0), 0) / total : 0, errors: allResults.filter(r => r.label === "error").length } };
}
async function describeTextWithAI(text, vt) {
  try { const res = await fetch("http://localhost:5001/ai_describe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ max_tokens: 120, prompt: `One sentence: what event this claims, emotional tone, sensational framing. Post: "${text}" A=${vt.A.toFixed(2)},V=${vt.V.toFixed(2)}` }), signal: AbortSignal.timeout(15000) }); const d = await res.json(); return d?.text || null; } catch (e) { return null; }
}
async function explainMismatchWithAI(text, imgDesc, vt, vi, dA, dV, isFake) {
  try { const res = await fetch("http://localhost:5001/ai_describe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ max_tokens: 150, prompt: `2 sentences: why text "${text}" and image "${imgDesc}" ${isFake ? "don't match" : "are consistent"}. Text A=${vt.A.toFixed(2)}, Image A=${vi.A.toFixed(2)}, Δ=${dA.toFixed(3)}` }), signal: AbortSignal.timeout(15000) }); const d = await res.json(); return d?.text || null; } catch (e) { return null; }
}
async function describeImageWithGemini(imageBase64) {
  if (!imageBase64) return null;
  try { const res = await fetch("http://localhost:5001/describe", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_base64: imageBase64.includes(",") ? imageBase64.split(",")[1] : imageBase64 }), signal: AbortSignal.timeout(30000) }); const d = await res.json(); return d?.description || null; } catch (e) { return null; }
}

// ─── DEMO DATA ────────────────────────────────────────────────────────────────
const DEMOS = [
  { post_id: "d1", label: "fake", anomaly_level: "critical", anomaly_score: 0.5191, fake_prob: 0.6795, contradiction_score: 0.386, n_methods_flagged: 3, text: "RT @Franke609: Shark in the street in #Brigantine, New Jersey during #HurricaneSandy", event: "Hurricane Sandy", vad_text: { V: 0.327, A: 0.695, D: 0.531 }, vad_image: { V: 0.321, A: 0.342, D: 0.301 }, fusion_weights: { text: 0.892, image: 0.054, meta: 0.054 }, method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: false }, top_words: ["shark","street","brigantine","hurricane","sandy"], image_description: "Digitally composited shark in flooded suburban street — dramatic, surreal.", pipeline_narrative: "Classic manipulation: elevated-arousal disaster text (A=0.70) vs calm composited image (A=0.34). Arousal Δ=0.353 exceeds threshold. 89% text-driven fusion — caption-led deception.", manipulation_signals: ["Arousal mismatch Δ=0.353","Caption-driven fusion 89%","3/4 anomaly detectors"] },
  { post_id: "d2", label: "fake", anomaly_level: "critical", anomaly_score: 0.8116, fake_prob: 0.9576, contradiction_score: 0.4247, n_methods_flagged: 4, text: "ISS wins. The Solar eclipse as seen from the International Space Station. #SolarEclipse #Space", event: "Solar Eclipse", vad_text: { V: 0.318, A: 0.636, D: 0.516 }, vad_image: { V: 0.326, A: 0.356, D: 0.317 }, fusion_weights: { text: 0.912, image: 0.044, meta: 0.044 }, method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: true }, top_words: ["solar","eclipse","space","station","iss"], image_description: "CGI-rendered solar eclipse from orbit — fabricated provenance.", pipeline_narrative: "All 4 detectors fired. CGI render falsely attributed to ISS. Arousal Δ=0.280, 91% text-driven. Fake prob 0.9576 — highest in dataset.", manipulation_signals: ["Arousal mismatch Δ=0.280","4/4 anomaly detectors","Fabricated ISS provenance"] },
  { post_id: "d3", label: "real", anomaly_level: "normal", anomaly_score: 0.0421, fake_prob: 0.3218, contradiction_score: 0.471, n_methods_flagged: 0, text: "Nepal's historic Dharahara Tower collapses in massive earthquake", event: "Nepal Earthquake", vad_text: { V: 0.420, A: 0.580, D: 0.610 }, vad_image: { V: 0.390, A: 0.560, D: 0.580 }, fusion_weights: { text: 0.741, image: 0.142, meta: 0.117 }, method_flags: { iso_forest: false, lof: false, ocsvm: false, elliptic: false }, top_words: ["dharahara","tower","collapses","earthquake","nepal"], image_description: "Collapsed rubble and structural damage.", pipeline_narrative: "Emotionally consistent. Text and image both show disaster semantics. Arousal Δ=0.020, well within threshold. Zero detectors triggered.", manipulation_signals: [] },
  { post_id: "d4", label: "real", anomaly_level: "normal", anomaly_score: 0.0438, fake_prob: 0.3041, contradiction_score: 0.406, n_methods_flagged: 0, text: "#LowerManhattan #nyc #hurricaneSandy — real footage of flooding at the seaport", event: "Hurricane Sandy (real)", vad_text: { V: 0.360, A: 0.600, D: 0.550 }, vad_image: { V: 0.340, A: 0.570, D: 0.520 }, fusion_weights: { text: 0.712, image: 0.168, meta: 0.120 }, method_flags: { iso_forest: false, lof: false, ocsvm: false, elliptic: false }, top_words: ["manhattan","nyc","hurricanesandy","flooding","seaport"], image_description: "Flooded urban streets, dark rising water, alarming.", pipeline_narrative: "Consistent profiles. Arousal Δ=0.030 within normal range. Zero anomaly detectors triggered.", manipulation_signals: [] },
];

const STAGES = ["Encoding text → SentenceTransformer 128-dim...","Encoding image → CLIP ViT-L/14 1024-dim...","Extracting VAD → zero-shot CLIP scoring...","Running EmotionAwareFakeNewsDetector...","Running anomaly ensemble (IsoForest + LOF + OCSVM + Elliptic)...","LLM entity consistency check...","Generating XAI narrative...","Combining scores → final verdict..."];

// ─── SUB-COMPONENTS ───────────────────────────────────────────────────────────
function VADRadar({ vt, vi }) {
  const scale = v => Math.min(v * 1.85, 1);
  const data = [
    { dim: "Valence",   text: scale(vt.V), image: scale(vi.V), rawText: vt.V, rawImage: vi.V },
    { dim: "Arousal",   text: scale(vt.A), image: scale(vi.A), rawText: vt.A, rawImage: vi.A },
    { dim: "Dominance", text: scale(vt.D), image: scale(vi.D), rawText: vt.D, rawImage: vi.D },
  ];
  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data} margin={{ top: 20, right: 40, bottom: 20, left: 40 }} outerRadius="72%">
        <PolarGrid stroke="#E8ECF4" strokeWidth={1.5} />
        <PolarAngleAxis dataKey="dim" tick={{ fill: "#3D4A6B", fontSize: 11, fontWeight: 700 }} />
        <Radar name="Text"  dataKey="text"  stroke="#4F6EF7" fill="#4F6EF7" fillOpacity={0.55} strokeWidth={3.5} dot={{ fill: "#4F6EF7", r: 6, strokeWidth: 2, stroke: "#fff" }} />
        <Radar name="Image" dataKey="image" stroke="#14B8A6" fill="#14B8A6" fillOpacity={0.45} strokeWidth={3.5} dot={{ fill: "#14B8A6", r: 6, strokeWidth: 2, stroke: "#fff" }} />
        <Legend iconSize={9} wrapperStyle={{ fontSize: 10, color: "#5A6480", fontWeight: 600 }} />
        <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E8ECF4", borderRadius: 8, fontSize: 10, boxShadow: "0 4px 16px rgba(0,0,0,0.08)" }} formatter={(v, name, props) => { const raw = name === "Text" ? props.payload.rawText : props.payload.rawImage; return [raw?.toFixed(3), name]; }} />
      </RadarChart>
    </ResponsiveContainer>
  );
}

function Gauge({ label, val, color }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: "#6B7A99", fontWeight: 600 }}>{label}</span>
        <span style={{ fontSize: 11, color, fontWeight: 800 }}>{typeof val === "number" ? val.toFixed(3) : val}</span>
      </div>
      <div style={{ height: 6, background: "#F0F2F8", borderRadius: 99, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${Math.min((val || 0) * 100, 100)}%`, background: color, borderRadius: 99, transition: "width 0.8s cubic-bezier(.4,0,.2,1)" }} />
      </div>
    </div>
  );
}

function MismatchBars({ vt, vi }) {
  const dims = [{ name: "Valence", tv: vt.V, iv: vi.V, thresh: 0.15 }, { name: "Arousal", tv: vt.A, iv: vi.A, thresh: 0.20 }, { name: "Dominance", tv: vt.D, iv: vi.D, thresh: 0.15 }];
  const dA = Math.abs(vt.A - vi.A);
  return (
    <div>
      <SectionLabel>Text–Image Mismatch</SectionLabel>
      {dims.map(d => {
        const delta = Math.abs(d.tv - d.iv), hot = delta > d.thresh;
        return (
          <div key={d.name} style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 10, color: hot ? "#DC2626" : "#6B7A99", fontWeight: hot ? 700 : 600 }}>{hot ? "⚠ " : ""}{d.name}</span>
              <span style={{ fontSize: 10, color: hot ? "#DC2626" : "#4F6EF7", fontWeight: 800 }}>Δ {delta.toFixed(3)}</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 8px 1fr", gap: 4, alignItems: "center" }}>
              {[{ val: d.tv, color: "#4F6EF7", label: "T" }, { val: d.iv, color: "#14B8A6", label: "I" }].map((bar, i) => (
                <div key={i} style={{ height: 16, background: "#F0F2F8", borderRadius: 4, overflow: "hidden", position: "relative" }}>
                  <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${bar.val * 100}%`, background: bar.color, borderRadius: 4, opacity: 0.8 }} />
                  <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: bar.val > 0.5 ? "#fff" : "#3D4A6B", fontWeight: 700 }}>{bar.label} {bar.val.toFixed(3)}</span>
                </div>
              )).reduce((acc, el, i) => i === 1 ? [...acc, <div key="sep" style={{ height: 1, background: hot ? "#DC2626" : "#CBD2E8" }} />, el] : [...acc, el], [])}
            </div>
          </div>
        );
      })}
      <div style={{ marginTop: 8, padding: "6px 10px", borderRadius: 8, background: dA > 0.20 ? "#FEF2F2" : "#F0FDFA", border: `1px solid ${dA > 0.20 ? "#FECACA" : "#99F6E4"}`, fontSize: 9, color: dA > 0.20 ? "#DC2626" : "#0F766E", fontWeight: 700 }}>
        {dA > 0.20 ? `⚠ Arousal Δ=${dA.toFixed(3)} — Manipulation Signal (d=0.41, p<0.001)` : `✓ Arousal Δ=${dA.toFixed(3)} — Modalities Consistent`}
      </div>
    </div>
  );
}

function AnomalyFlags({ flags, n }) {
  const methods = ["iso_forest", "lof", "ocsvm", "elliptic"];
  const labels = { iso_forest: "IsoForest", lof: "LOF", ocsvm: "OCSVM", elliptic: "Elliptic" };
  const weights = { iso_forest: "0.35", lof: "0.30", ocsvm: "0.20", elliptic: "0.15" };
  return (
    <div>
      <SectionLabel>Anomaly Ensemble ({n}/4)</SectionLabel>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6 }}>
        {methods.map(m => (
          <div key={m} style={{ padding: "8px 4px", borderRadius: 10, textAlign: "center", background: flags?.[m] ? "#FEF2F2" : "#F0FDFA", border: `1px solid ${flags?.[m] ? "#FECACA" : "#99F6E4"}` }}>
            <div style={{ fontSize: 14, marginBottom: 2 }}>{flags?.[m] ? "⚠" : "✓"}</div>
            <div style={{ fontSize: 8, color: "#6B7A99", fontWeight: 700 }}>{labels[m]}</div>
            <div style={{ fontSize: 7, color: "#9BA3BF" }}>w={weights[m]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function WordHighlight({ text, topWords }) {
  const flagged = new Set((topWords || []).map(w => w.toLowerCase()));
  return (
    <div>
      <SectionLabel>Word Attribution</SectionLabel>
      <div style={{ padding: "10px 12px", background: "#F7F8FC", borderRadius: 10, border: "1px solid #E8ECF4", lineHeight: 2.6 }}>
        {text.split(" ").map((word, i) => {
          const clean = word.toLowerCase().replace(/[^a-z0-9]/g, ""), hit = flagged.has(clean);
          return <span key={i} style={{ display: "inline-block", margin: "1px 2px", padding: "1px 7px", borderRadius: 5, background: hit ? "#FEF2F2" : "transparent", border: hit ? "1px solid #FECACA" : "1px solid transparent", color: hit ? "#DC2626" : "#5A6480", fontSize: 10.5, fontWeight: hit ? 700 : 400 }}>{word}</span>;
        })}
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return <div style={{ fontSize: 10, fontWeight: 800, color: "#3D4A6B", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 10 }}>{children}</div>;
}

function ResultCard({ r, imageSrc, isDemo, index }) {
  const [open, setOpen] = useState(index === 0);
  const isFake = r.label === "fake";
  const fw = r.fusion_weights || { text: 0.9, image: 0.05, meta: 0.05 };
  const vt = r.vad_text || { V: 0.5, A: 0.5, D: 0.5 };
  const vi = r.vad_image || { V: 0.5, A: 0.5, D: 0.5 };
  const dA = Math.abs(vt.A - vi.A);
  const accentColor = isFake ? "#DC2626" : "#0F766E";
  const accentBg = isFake ? "#FEF2F2" : "#F0FDFA";
  const accentBorder = isFake ? "#FECACA" : "#99F6E4";

  return (
    <div style={{ background: "#fff", borderRadius: 18, border: `1px solid ${accentBorder}`, borderTop: `4px solid ${accentColor}`, boxShadow: "0 4px 24px rgba(60,80,140,0.08)", marginBottom: 16, overflow: "hidden" }}>
      <div onClick={() => setOpen(!open)} style={{ padding: "16px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", background: open ? "#FAFBFE" : "#fff", borderBottom: open ? "1px solid #F0F2F8" : "none" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ padding: "4px 14px", borderRadius: 99, background: accentBg, color: accentColor, fontSize: 11, fontWeight: 800, letterSpacing: "0.06em", border: `1px solid ${accentBorder}` }}>
            {isFake ? "⚠ FAKE" : "✓ AUTHENTIC"}
          </span>
          <span style={{ fontSize: 9, padding: "3px 10px", borderRadius: 99, background: isDemo ? "#F7F8FC" : "#EEF2FF", color: isDemo ? "#9BA3BF" : "#4F6EF7", border: `1px solid ${isDemo ? "#E8ECF4" : "#C7CEFF"}`, fontWeight: 700 }}>
            {isDemo ? "Demo" : "🔬 Live Pipeline"}
          </span>
        </div>
        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          {[["Anomaly", (r.anomaly_score || 0).toFixed(3), isFake ? "#DC2626" : "#0F766E"], ["Arousal Δ", dA.toFixed(3), dA > 0.2 ? "#DC2626" : "#0F766E"], ["Fake Prob", (r.fake_prob || 0).toFixed(3), "#7C3AED"]].map(([l, v, c]) => (
            <div key={l} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 9, color: "#9BA3BF", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 2 }}>{l}</div>
              <div style={{ fontSize: 18, fontWeight: 900, color: c, lineHeight: 1 }}>{v}</div>
            </div>
          ))}
          <span style={{ color: "#C7CEDD", fontSize: 12 }}>{open ? "▲" : "▼"}</span>
        </div>
      </div>

      {open && (
        <div style={{ padding: "22px 24px" }}>
          {imageSrc && (
            <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 18, marginBottom: 20 }}>
              <div>
                <SectionLabel>Uploaded Image</SectionLabel>
                <img src={imageSrc} alt="uploaded" style={{ width: "100%", borderRadius: 10, border: "1px solid #E8ECF4", maxHeight: 160, objectFit: "cover" }} />
                {r.image_description && <div style={{ marginTop: 8, padding: "8px 10px", background: "#F0FDFA", borderRadius: 8, border: "1px solid #99F6E4", fontSize: 9, color: "#0F766E", lineHeight: 1.6, fontStyle: "italic" }}><span style={{ fontStyle: "normal", fontWeight: 700 }}>▸ </span>{r.image_description}</div>}
              </div>
              <div style={{ padding: "12px 16px", background: "#F7F8FC", borderRadius: 10, border: "1px solid #E8ECF4", fontSize: 12, color: "#3D4A6B", lineHeight: 1.7, fontStyle: "italic", alignSelf: "start" }}>
                <SectionLabel>Post Text</SectionLabel>"{r.text?.length > 240 ? r.text.slice(0, 240) + "…" : r.text}"
              </div>
            </div>
          )}
          {!imageSrc && (
            <div style={{ padding: "12px 16px", background: "#F7F8FC", borderRadius: 10, border: "1px solid #E8ECF4", fontSize: 12, color: "#3D4A6B", lineHeight: 1.7, fontStyle: "italic", marginBottom: 20 }}>
              <SectionLabel>Post Text</SectionLabel>"{r.text?.length > 280 ? r.text.slice(0, 280) + "…" : r.text}"
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "200px 1fr 1fr", gap: 20, marginBottom: 20 }}>
            <div>
              <SectionLabel>Detection Scores</SectionLabel>
              <Gauge label="Anomaly Score" val={r.anomaly_score} color={isFake ? "#DC2626" : "#0F766E"} />
              <Gauge label="Fake Probability" val={r.fake_prob} color="#7C3AED" />
              <Gauge label="Contradiction" val={r.contradiction_score} color="#F59E0B" />
              <Gauge label="Arousal Mismatch" val={dA} color={dA > 0.2 ? "#DC2626" : "#0F766E"} />
              <div style={{ marginTop: 16 }}>
                <SectionLabel>Emotion Gate</SectionLabel>
                {(() => {
                  const total = Math.abs(fw.text) + Math.abs(fw.image) + Math.abs(fw.meta) || 1;
                  const items = [["text", fw.text, "#4F6EF7", "TXT"], ["image", fw.image, "#14B8A6", "IMG"], ["meta", fw.meta, "#A78BFA", "META"]];
                  return (
                    <>
                      <div style={{ display: "flex", height: 20, borderRadius: 6, overflow: "hidden", border: "1px solid #E8ECF4" }}>
                        {items.map(([k, v, c, lbl]) => {
                          const pct = (Math.abs(v) / total) * 100;
                          return (
                            <div key={k} style={{ width: `${pct}%`, background: c, display: "flex", alignItems: "center", justifyContent: "center", minWidth: pct > 0 ? 2 : 0 }}>

                            </div>
                          );
                        })}
                      </div>
                      <div style={{ display: "flex", gap: 8, marginTop: 5 }}>
                        {items.map(([k, v, c, lbl]) => {
                          const pct = (Math.abs(v) / total) * 100;
                          return (
                            <div key={k} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                              <div style={{ width: 8, height: 8, borderRadius: 2, background: c, flexShrink: 0 }} />
                              <span style={{ fontSize: 8, color: "#6B7A99", fontWeight: 700 }}>{lbl} {pct.toFixed(1)}%</span>
                            </div>
                          );
                        })}
                      </div>
                    </>
                  );
                })()}
              </div>
              <div style={{ marginTop: 16 }}><AnomalyFlags flags={r.method_flags} n={r.n_methods_flagged} /></div>
            </div>
            <div>
              <SectionLabel>VAD Emotion Radar</SectionLabel>
              <VADRadar vt={vt} vi={vi} />
            </div>
            <MismatchBars vt={vt} vi={vi} />
          </div>

          <div style={{ marginBottom: 16 }}><WordHighlight text={r.text || ""} topWords={r.top_words || []} /></div>

          <div style={{ padding: "14px 18px", borderRadius: 12, background: accentBg, border: `1px solid ${accentBorder}`, fontSize: 11, color: "#3D4A6B", lineHeight: 1.85 }}>
            <div style={{ fontSize: 10, fontWeight: 800, color: accentColor, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.08em" }}>{isFake ? "⚠ Pipeline Verdict — Fake" : "✓ Pipeline Verdict — Authentic"}</div>
            <p style={{ margin: 0 }}>{r.pipeline_narrative}</p>
            {r.manipulation_signals?.length > 0 && (
              <div style={{ marginTop: 10, display: "flex", gap: 6, flexWrap: "wrap" }}>
                {r.manipulation_signals.map((s, i) => <span key={i} style={{ fontSize: 9, padding: "2px 9px", borderRadius: 99, background: "#fff", border: "1px solid #FECACA", color: "#DC2626", fontWeight: 700 }}>{s}</span>)}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function UploadZone({ onFile, currentSrc }) {
  const ref = useRef();
  const [drag, setDrag] = useState(false);
  const handle = useCallback(file => { if (!file || !file.type.startsWith("image/")) return; const reader = new FileReader(); reader.onload = e => onFile(e.target.result); reader.readAsDataURL(file); }, [onFile]);
  return (
    <div onClick={() => ref.current.click()} onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }} onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
      style={{ cursor: "pointer", borderRadius: 12, overflow: "hidden", border: `2px dashed ${drag ? "#4F6EF7" : "#CBD2E8"}`, background: drag ? "#EEF2FF" : "#F7F8FC", minHeight: 120, display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
      <input ref={ref} type="file" accept="image/*" style={{ display: "none" }} onChange={e => handle(e.target.files[0])} />
      {currentSrc ? <img src={currentSrc} alt="preview" style={{ width: "100%", maxHeight: 150, objectFit: "cover" }} /> : (
        <div style={{ textAlign: "center", padding: 20 }}>
          <div style={{ fontSize: 28, marginBottom: 8 }}>🖼</div>
          <div style={{ fontSize: 11, color: "#6B7A99", fontWeight: 700, marginBottom: 4 }}>Drop image or click to upload</div>
          <div style={{ fontSize: 9, color: "#9BA3BF" }}>CLIP ViT-L/14 · VAD zero-shot · LLM entity check</div>
        </div>
      )}
    </div>
  );
}

function ResearchPanel() {
  const barData = [{ name: "0", rate: 39.6, n: 3876 }, { name: "1", rate: 59.8, n: 2829 }, { name: "2", rate: 67.1, n: 1890 }, { name: "3", rate: 66.5, n: 1412 }, { name: "4", rate: 68.5, n: 819 }];
  const barColors = ["#CBD2E8","#93C5FD","#FCA5A5","#F87171","#DC2626"];
  const vadStats = [{ l: "Text Arousal", f: 0.637, r: 0.610, p: "<0.001", d: "0.378", star: true }, { l: "Text Valence", f: 0.505, r: 0.493, p: "0.001", d: "0.061", star: false }, { l: "Text Dominance", f: 0.579, r: 0.573, p: "0.003", d: "0.057", star: false }, { l: "Arousal Δ", f: 0.586, r: 0.542, p: "<0.001", d: "0.410", star: true }, { l: "Valence Δ", f: 0.421, r: 0.379, p: "<0.001", d: "0.187", star: false }];
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 14, marginBottom: 22 }}>
        {[["10,826","Posts Analysed","#4F6EF7"],["84%","Classifier Accuracy","#16A34A"],["p<0.001","Arousal Significance","#DC2626"],["d=0.41","Effect Size","#F59E0B"]].map(([v,l,c]) => (
          <div key={l} style={{ background: "#fff", borderRadius: 14, padding: "22px 18px", textAlign: "center", border: "1px solid #E8ECF4", boxShadow: "0 2px 12px rgba(60,80,140,0.06)", borderTop: `4px solid ${c}` }}>
            <div style={{ fontSize: 30, fontWeight: 900, color: c, marginBottom: 4, fontFamily: "'Sora', sans-serif" }}>{v}</div>
            <div style={{ fontSize: 10, color: "#9BA3BF", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em" }}>{l}</div>
          </div>
        ))}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={{ background: "#fff", borderRadius: 14, padding: 22, border: "1px solid #E8ECF4", boxShadow: "0 2px 12px rgba(60,80,140,0.06)" }}>
          <SectionLabel>Ensemble Agreement → Fake Rate (n=10,826)</SectionLabel>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={barData} margin={{ top: 4, right: 8, left: -24, bottom: 14 }}>
              <XAxis dataKey="name" tick={{ fill: "#9BA3BF", fontSize: 10 }} label={{ value: "Methods Flagged", position: "insideBottom", offset: -8, fill: "#9BA3BF", fontSize: 9 }} />
              <YAxis tick={{ fill: "#9BA3BF", fontSize: 10 }} domain={[0, 80]} />
              <Tooltip contentStyle={{ background: "#fff", border: "1px solid #E8ECF4", borderRadius: 8, fontSize: 10, boxShadow: "0 4px 16px rgba(0,0,0,0.08)" }} formatter={(v, _, p) => [`${v}% fake (n=${p.payload.n})`]} />
              <Bar dataKey="rate" radius={[4,4,0,0]}>{barData.map((d,i) => <Cell key={i} fill={barColors[i]} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 9, color: "#9BA3BF" }}>Baseline 55.4% → 4-method agreement: 68.5% (+13.1pp)</div>
        </div>
        <div style={{ background: "#fff", borderRadius: 14, padding: 22, border: "1px solid #E8ECF4", boxShadow: "0 2px 12px rgba(60,80,140,0.06)" }}>
          <SectionLabel>VAD Statistical Findings (Mann-Whitney U, n=10,826)</SectionLabel>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 44px 44px 56px 44px", gap: "3px 6px", fontSize: 9, color: "#9BA3BF", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8, borderBottom: "1px solid #F0F2F8", paddingBottom: 6 }}>
            <span>Dimension</span><span>Fake</span><span>Real</span><span>p</span><span>d</span>
          </div>
          {vadStats.map(row => (
            <div key={row.l} style={{ display: "grid", gridTemplateColumns: "1fr 44px 44px 56px 44px", gap: "3px 6px", padding: "5px 0", borderBottom: "1px solid #F7F8FC", alignItems: "center" }}>
              <span style={{ fontSize: 11, color: row.star ? "#4F6EF7" : "#5A6480", fontWeight: row.star ? 700 : 500 }}>{row.l}</span>
              <span style={{ fontSize: 11, color: "#4F6EF7", fontWeight: 700 }}>{row.f}</span>
              <span style={{ fontSize: 11, color: "#14B8A6", fontWeight: 700 }}>{row.r}</span>
              <span style={{ fontSize: 11, color: row.star ? "#DC2626" : "#9BA3BF", fontWeight: row.star ? 700 : 500 }}>{row.p}</span>
              <span style={{ fontSize: 11, color: row.star ? "#DC2626" : "#9BA3BF", fontWeight: row.star ? 700 : 500 }}>{row.d}</span>
            </div>
          ))}
          <div style={{ marginTop: 12, padding: "7px 12px", borderRadius: 8, background: "#FEF2F2", border: "1px solid #FECACA", fontSize: 10, color: "#DC2626", fontWeight: 700 }}>★ Arousal Δ d=0.41 — cross-modal emotional mismatch is the core manipulation signal</div>
        </div>
      </div>
    </div>
  );
}

function BatchPanel() {
  const [posts, setPosts] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState([0, 0]);
  const [error, setError] = useState(null);
  const [sortKey, setSortKey] = useState("fake_prob");
  const [filterFake, setFilterFake] = useState("all");
  const fileRef = useRef();
  const parseCSV = text => { const lines = text.trim().split("\n").filter(Boolean); const header = lines[0].toLowerCase().split(",").map(h => h.trim().replace(/"/g,"")); const ti = header.findIndex(h => h.includes("text")||h.includes("tweet")); if (ti===-1) throw new Error("CSV needs 'text' column"); return lines.slice(1).map(l => { const c = l.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/).map(x => x.trim().replace(/^"|"$/g,"")); return { text: c[ti]||"" }; }).filter(p => p.text.length>3); };
  const handleFile = async e => { const file = e.target.files[0]; if (!file) return; setError(null); setResults(null); try { const text = await file.text(); setPosts(file.name.endsWith(".csv") ? parseCSV(text) : text.split("\n").filter(l=>l.trim()).map(l=>({text:l.trim()}))); } catch (err) { setError(err.message); } };
  const runBatch = async () => { if (!posts.length) return; setLoading(true); setError(null); setResults(null); setProgress([0, posts.length]); try { const data = await runBatchAnalysis(posts, (d,t)=>setProgress([d,t])); setResults(data); } catch(e) { setError(e.message); } setLoading(false); };
  const sorted = results ? [...results.results].filter(r=>filterFake==="all"||r.label===filterFake).sort((a,b)=>(b[sortKey]||0)-(a[sortKey]||0)) : [];
  const pct = p => `${(p*100).toFixed(1)}%`;
  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <div style={{ background: "#fff", borderRadius: 14, padding: 22, border: "1px solid #E8ECF4", boxShadow: "0 2px 12px rgba(60,80,140,0.06)" }}>
          <SectionLabel>Upload CSV or TXT</SectionLabel>
          <div onClick={()=>fileRef.current.click()} style={{ cursor:"pointer", border:"2px dashed #CBD2E8", borderRadius:12, padding:24, textAlign:"center", background:"#F7F8FC", marginBottom:12 }}>
            <input ref={fileRef} type="file" accept=".csv,.txt" style={{display:"none"}} onChange={handleFile} />
            <div style={{ fontSize:26, marginBottom:8 }}>📂</div>
            <div style={{ fontSize:11, color:"#6B7A99", fontWeight:700 }}>Click to upload CSV or TXT</div>
            <div style={{ fontSize:9, color:"#9BA3BF", marginTop:4 }}>CSV needs "text" column · max 100 posts</div>
          </div>
          {posts.length>0&&<div style={{padding:"8px 12px",background:"#F0FDFA",border:"1px solid #99F6E4",borderRadius:8,fontSize:10,color:"#0F766E",fontWeight:700}}>✓ {posts.length} posts loaded</div>}
          {error&&<div style={{marginTop:8,padding:"8px 12px",background:"#FEF2F2",border:"1px solid #FECACA",borderRadius:8,fontSize:10,color:"#DC2626",fontWeight:600}}>⚠ {error}</div>}
        </div>
        <div style={{ background: "#fff", borderRadius: 14, padding: 22, border: "1px solid #E8ECF4", boxShadow: "0 2px 12px rgba(60,80,140,0.06)" }}>
          <SectionLabel>Sample Format</SectionLabel>
          <div style={{ background:"#F7F8FC", border:"1px solid #E8ECF4", borderRadius:10, padding:12, fontSize:10, color:"#6B7A99", lineHeight:1.9 }}>
            <div style={{color:"#4F6EF7",fontWeight:700,marginBottom:4}}>CSV:</div>
            <div>text,source</div><div>"Shark in the street during Sandy",twitter</div>
            <div style={{color:"#4F6EF7",fontWeight:700,margin:"8px 0 4px"}}>TXT (one per line):</div>
            <div>Shark in the street during Sandy</div>
          </div>
          <button onClick={runBatch} disabled={loading||posts.length===0} style={{ marginTop:12, width:"100%", padding:"12px 0", border:"none", borderRadius:10, cursor:"pointer", background:"linear-gradient(135deg,#4F6EF7,#6D8BFF)", color:"#fff", fontSize:12, fontWeight:800, opacity:(loading||!posts.length)?0.4:1, boxShadow:"0 4px 16px rgba(79,110,247,0.3)" }}>
            {loading ? `⟳ Analysing ${progress[0]}/${progress[1]}...` : `▶ Run Batch Analysis (${posts.length} posts)`}
          </button>
          {loading && <div style={{marginTop:8,height:4,background:"#F0F2F8",borderRadius:99,overflow:"hidden"}}><div style={{height:"100%",width:`${progress[1]>0?(progress[0]/progress[1])*100:0}%`,background:"#4F6EF7",borderRadius:99,transition:"width 0.3s"}}/></div>}
        </div>
      </div>
      {results && (
        <>
          <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:12,marginBottom:16}}>
            {[["TOTAL",results.summary.total,"#4F6EF7"],["FAKE",results.summary.fake,"#DC2626"],["REAL",results.summary.real,"#16A34A"],["FAKE RATE",pct(results.summary.fake_rate),"#F59E0B"],["AVG PROB",results.summary.avg_fake_prob.toFixed(3),"#7C3AED"]].map(([l,v,c])=>(
              <div key={l} style={{background:"#fff",borderRadius:12,padding:"14px 12px",textAlign:"center",border:"1px solid #E8ECF4",borderTop:`3px solid ${c}`}}>
                <div style={{fontSize:22,fontWeight:900,color:c}}>{v}</div>
                <div style={{fontSize:9,color:"#9BA3BF",fontWeight:700,textTransform:"uppercase",letterSpacing:"0.06em",marginTop:3}}>{l}</div>
              </div>
            ))}
          </div>
          <div style={{display:"flex",gap:8,marginBottom:12,alignItems:"center",flexWrap:"wrap"}}>
            {["fake_prob","anomaly_score"].map(k=><button key={k} onClick={()=>setSortKey(k)} style={{padding:"4px 12px",border:`1px solid ${sortKey===k?"#4F6EF7":"#E8ECF4"}`,borderRadius:99,cursor:"pointer",background:sortKey===k?"#EEF2FF":"#fff",color:sortKey===k?"#4F6EF7":"#9BA3BF",fontSize:10,fontWeight:700}}>{k.replace(/_/g," ")}</button>)}
            {["all","fake","real"].map(f=><button key={f} onClick={()=>setFilterFake(f)} style={{padding:"4px 12px",border:`1px solid ${filterFake===f?"#4F6EF7":"#E8ECF4"}`,borderRadius:99,cursor:"pointer",background:filterFake===f?"#EEF2FF":"#fff",color:filterFake===f?"#4F6EF7":"#9BA3BF",fontSize:10,fontWeight:700}}>{f}</button>)}
          </div>
          <div style={{background:"#fff",borderRadius:14,border:"1px solid #E8ECF4",overflow:"hidden"}}>
            <div style={{display:"grid",gridTemplateColumns:"28px 1fr 88px 88px 70px",padding:"8px 16px",background:"#F7F8FC",borderBottom:"1px solid #E8ECF4"}}>
              {["#","TEXT","LABEL","FAKE PROB","DETECTORS"].map(h=><span key={h} style={{fontSize:9,color:"#9BA3BF",fontWeight:800,textTransform:"uppercase",letterSpacing:"0.08em"}}>{h}</span>)}
            </div>
            <div style={{maxHeight:380,overflowY:"auto"}}>
              {sorted.map((r,i)=>{const isFake=r.label==="fake";return(<div key={i} style={{display:"grid",gridTemplateColumns:"28px 1fr 88px 88px 70px",padding:"8px 16px",borderBottom:"1px solid #F7F8FC",background:i%2===0?"#fff":"#FAFBFE",alignItems:"center"}}><span style={{fontSize:9,color:"#C7CEDD"}}>{(r.index||i)+1}</span><span style={{fontSize:10,color:"#5A6480",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",paddingRight:8}} title={r.text}>{r.text}</span><span style={{fontSize:9,padding:"2px 8px",borderRadius:99,background:isFake?"#FEF2F2":"#F0FDFA",border:`1px solid ${isFake?"#FECACA":"#99F6E4"}`,color:isFake?"#DC2626":"#0F766E",fontWeight:700,display:"inline-block"}}>{isFake?"⚠ FAKE":"✓ REAL"}</span><span style={{fontSize:11,color:"#7C3AED",fontWeight:700}}>{(r.fake_prob||0).toFixed(3)}</span><span style={{fontSize:11,color:r.n_methods_flagged>1?"#DC2626":"#0F766E",fontWeight:700}}>{r.n_methods_flagged}/4</span></div>);})}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("analyse");
  const [mode, setMode] = useState("full");
  const [postText, setPostText] = useState("");
  const [imageSrc, setImageSrc] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stageIdx, setStageIdx] = useState(0);
  const [error, setError] = useState(null);

  const runAnalysis = async () => {
    if (!postText.trim()) return;
    setLoading(true); setError(null); setResults([]); setStageIdx(0);
    const timer = setInterval(() => setStageIdx(i => Math.min(i + 1, STAGES.length - 1)), 800);
    try {
      const base64 = imageSrc ? imageSrc.split(",")[1] : null;
      const result = await runPipelineAnalysis(postText, base64);
      result.text = postText;
      if (!result.top_words) { const stop = new Set(["the","a","an","in","on","at","is","it","to","and","or","of","for","with","http","rt","co","t","via"]); result.top_words = postText.toLowerCase().replace(/[^a-z0-9\s]/g," ").split(/\s+/).filter(w=>w.length>3&&!stop.has(w)).slice(0,8); }
      clearInterval(timer);
      setResults([{ r: result, imageSrc, isDemo: false }]);
    } catch (e) { clearInterval(timer); setError(e.message || "Pipeline analysis failed."); }
    setLoading(false);
  };

  const loadDemo = p => { setLoading(true); setResults([]); setTimeout(() => { setResults([{ r: p, imageSrc: null, isDemo: true }]); setLoading(false); }, 300); };

  return (
    <div style={{ minHeight: "100vh", background: "#F7F8FC", fontFamily: "'Nunito', 'Segoe UI', sans-serif", color: "#1E2A4A" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&family=Sora:wght@700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px } ::-webkit-scrollbar-thumb { background: #CBD2E8; border-radius: 3px }
        @keyframes fadeUp { from { opacity:0; transform:translateY(10px) } to { opacity:1; transform:translateY(0) } }
        .fu { animation: fadeUp 0.35s ease forwards }
        @keyframes spin { to { transform:rotate(360deg) } }
        .spin { animation: spin 1s linear infinite; display:inline-block }
        button { font-family: 'Nunito', sans-serif; transition: all 0.15s; }
        textarea { font-family: 'Nunito', sans-serif; }
        .mode-card:hover { transform: translateY(-2px) !important; }
        .tab-btn:hover { color: #4F6EF7 !important; }
        .demo-btn:hover { background: #EEF2FF !important; border-color: #C7CEFF !important; color: #4F6EF7 !important; }
        .run-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(79,110,247,0.45) !important; }
        .run-btn:active:not(:disabled) { transform: translateY(0); }
      `}</style>

      {/* NAV */}
      <nav style={{ background: "rgba(255,255,255,0.95)", backdropFilter: "blur(16px)", borderBottom: "1px solid #E8ECF4", position: "sticky", top: 0, zIndex: 100, boxShadow: "0 1px 8px rgba(60,80,140,0.06)" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", padding: "0 28px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 56 }}>
          <div style={{ fontSize: 17, fontWeight: 900, color: "#1E2A4A", fontFamily: "'Sora', sans-serif", letterSpacing: "-0.5px" }}>
            Deception<span style={{ color: "#4F6EF7" }}>XAI</span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            {[["analyse","Analyser"],["batch","Batch"],["research","Research"]].map(([id,label]) => (
              <button key={id} onClick={() => setTab(id)} className="tab-btn" style={{ padding: "6px 18px", border: "none", background: tab===id ? "#EEF2FF" : "transparent", color: tab===id ? "#4F6EF7" : "#6B7A99", fontSize: 12, fontWeight: tab===id ? 800 : 600, borderRadius: 8, cursor: "pointer" }}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* HERO */}
      <div style={{ background: "linear-gradient(140deg, #E0F2FE 0%, #EEF2FF 45%, #ECFDF5 100%)", padding: "52px 28px 44px", textAlign: "center", borderBottom: "1px solid #E0EAF8" }}>
        <div style={{ maxWidth: 680, margin: "0 auto" }}>
          <div style={{ display: "inline-block", padding: "3px 14px", borderRadius: 99, background: "rgba(79,110,247,0.1)", border: "1px solid rgba(79,110,247,0.2)", fontSize: 10, fontWeight: 800, color: "#4F6EF7", textTransform: "uppercase", letterSpacing: "0.14em", marginBottom: 16 }}>
            EmotionAwareFakeNewsDetector
          </div>
          <h1 style={{ fontSize: 44, fontWeight: 900, color: "#1E2A4A", fontFamily: "'Sora', sans-serif", lineHeight: 1.12, marginBottom: 14, letterSpacing: "-1.5px" }}>
            Multimodal<br />
            <span style={{ background: "linear-gradient(125deg, #4F6EF7 0%, #14B8A6 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", backgroundClip: "text" }}>
              Deception XAI
            </span>
          </h1>
          <p style={{ fontSize: 14, color: "#6B7A99", lineHeight: 1.75, marginBottom: 32, maxWidth: 500, margin: "0 auto 32px" }}>
            Detect misinformation using cross-modal emotional analysis with CLIP, SentenceTransformer, and anomaly ensemble scoring.
          </p>

          {tab === "analyse" && (
            <div style={{ maxWidth: 320, margin: "0 auto" }}>
              <div className="mode-card" style={{ padding: "18px 20px", borderRadius: 16, background: "#F0FDFA", border: "2px solid #14B8A6", textAlign: "left", boxShadow: "0 6px 24px rgba(20,184,166,0.18)" }}>
                <div style={{ fontSize: 12, fontWeight: 800, color: "#134E4A", marginBottom: 6 }}>Full Multimodal Analysis</div>
                <div style={{ fontSize: 10, color: "#4B5563", lineHeight: 1.65 }}>Combines <strong>CLIP ViT-L/14</strong> image encoding with cross-modal VAD mismatch for full multimodal detection (d=0.41, p&lt;0.001).</div>
                <div style={{ marginTop: 10, fontSize: 9, fontWeight: 800, color: "#14B8A6", textTransform: "uppercase", letterSpacing: "0.1em" }}>● Selected</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* CONTENT */}
      <div style={{ maxWidth: 1080, margin: "0 auto", padding: "30px 28px" }}>
        {tab === "analyse" && (
          <div className="fu">
            {/* Input card */}
            <div style={{ background: "#fff", borderRadius: 20, border: "1px solid #E8ECF4", boxShadow: "0 4px 28px rgba(60,80,140,0.08)", padding: "28px 30px", marginBottom: 24 }}>
              <div style={{ display: "grid", gridTemplateColumns: mode==="full" ? "1fr 280px" : "1fr", gap: 22 }}>
                <div>
                  <div style={{ fontSize: 11, fontWeight: 800, color: "#3D4A6B", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 10 }}>Paste a Social Media Post</div>
                  <textarea value={postText} onChange={e => setPostText(e.target.value)} placeholder="Paste any tweet or social media post here for full pipeline analysis..." style={{ width: "100%", height: 115, padding: "13px 15px", background: "#F7F8FC", border: "1.5px solid #E8ECF4", borderRadius: 12, color: "#1E2A4A", fontSize: 12.5, resize: "none", lineHeight: 1.7, outline: "none", transition: "border-color 0.2s" }} onFocus={e=>e.target.style.borderColor="#4F6EF7"} onBlur={e=>e.target.style.borderColor="#E8ECF4"} />
                  <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <span style={{ fontSize: 10, color: "#9BA3BF", fontWeight: 700 }}>Sample Posts:</span>
                    {DEMOS.map(p => (
                      <button key={p.post_id} onClick={() => loadDemo(p)} className="demo-btn" style={{ padding: "4px 12px", border: "1.5px solid #E8ECF4", borderRadius: 99, cursor: "pointer", background: "#F7F8FC", color: "#6B7A99", fontSize: 9, fontWeight: 700 }}>
                        {p.label==="fake" ? "⚠" : "✓"} {p.event.split(" ").slice(0,2).join(" ")} ({p.label})
                      </button>
                    ))}
                  </div>
                </div>
                {mode==="full" && (
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 800, color: "#3D4A6B", textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 10 }}>
                      Upload Image <span style={{ fontSize: 9, color: "#9BA3BF", textTransform: "none", letterSpacing: 0, fontWeight: 600 }}>(optional)</span>
                    </div>
                    <UploadZone onFile={src => setImageSrc(src)} currentSrc={imageSrc} />
                    {imageSrc && <div style={{ marginTop: 8, display: "flex", justifyContent: "space-between", alignItems: "center" }}><span style={{ fontSize: 9, color: "#0F766E", fontWeight: 700 }}>✓ CLIP + LLM entity check ready</span><button onClick={() => setImageSrc(null)} style={{ fontSize: 9, color: "#DC2626", background: "none", border: "none", cursor: "pointer", fontWeight: 700 }}>✕ remove</button></div>}
                  </div>
                )}
              </div>

              {/* Analyse button */}
              <div style={{ marginTop: 22, display: "flex", alignItems: "center", gap: 14 }}>
                <button onClick={runAnalysis} disabled={loading||!postText.trim()} className="run-btn" style={{ padding: "13px 40px", border: "none", borderRadius: 12, cursor: "pointer", background: "linear-gradient(135deg,#4F6EF7,#6D8BFF)", color: "#fff", fontSize: 13, fontWeight: 800, opacity: (!postText.trim()||loading) ? 0.4 : 1, letterSpacing: "0.03em", boxShadow: "0 4px 16px rgba(79,110,247,0.35)" }}>
                  {loading ? "⟳ Running Pipeline..." : "Analyse Post"}
                </button>
                {!loading && <span style={{ fontSize: 10, color: "#9BA3BF" }}>{mode==="full" ? "CLIP ViT-L/14 · SentenceTransformer · Anomaly Ensemble · LLM Entity Check" : "SentenceTransformer · Anomaly Ensemble"}</span>}
              </div>
            </div>

            {/* Loading */}
            {loading && (
              <div style={{ background: "#fff", borderRadius: 16, border: "1px solid #E8ECF4", padding: "22px 26px", marginBottom: 22, boxShadow: "0 2px 12px rgba(60,80,140,0.06)" }}>
                <SectionLabel>Pipeline Execution</SectionLabel>
                {STAGES.map((s, i) => {
                  const done=i<stageIdx, active=i===stageIdx;
                  return (
                    <div key={i} style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8, opacity: i>stageIdx?0.22:1, transition: "opacity 0.3s" }}>
                      <div style={{ width: 22, height: 22, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", background: done?"#F0FDFA":active?"#EEF2FF":"#F7F8FC", border: `1.5px solid ${done?"#99F6E4":active?"#C7CEFF":"#E8ECF4"}`, flexShrink:0, fontSize: 11 }}>
                        {done?<span style={{color:"#0F766E"}}>✓</span>:active?<span className="spin" style={{color:"#4F6EF7"}}>⟳</span>:<span style={{color:"#CBD2E8"}}>○</span>}
                      </div>
                      <span style={{ fontSize: 11, color: done?"#0F766E":active?"#4F6EF7":"#9BA3BF", fontWeight: active?700:500 }}>{s}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {tab==="batch" && <div className="fu"><BatchPanel /></div>}
        {tab==="research" && <div className="fu"><ResearchPanel /></div>}

        {error && <div style={{ padding:"12px 18px", borderRadius:12, background:"#FEF2F2", border:"1px solid #FECACA", marginBottom:18, fontSize:11, color:"#DC2626", fontWeight:600 }}>⚠ {error}</div>}

        {!loading && results.length>0 && tab==="analyse" && (
          <div className="fu">
            <SectionLabel>Analysis Results</SectionLabel>
            {results.map((item,i) => <ResultCard key={i} r={item.r} imageSrc={item.imageSrc} isDemo={item.isDemo} index={i} />)}
          </div>
        )}

        <div style={{ marginTop: 36, paddingTop: 16, borderTop: "1px solid #E8ECF4", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 6 }}>
          <span style={{ fontSize: 9, color: "#C7CEDD" }}>EmotionAwareFakeNewsDetector · CLIP ViT-L/14 · LLM Entity Check · Arousal p&lt;0.001 · d=0.41</span>
          <span style={{ fontSize: 9, color: "#C7CEDD" }}>10,826 posts · 5,994 fake · 4,832 real</span>
        </div>
      </div>
    </div>
  );
}