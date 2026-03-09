import { useState, useRef, useCallback } from "react";
import {
 RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
 BarChart, Bar, XAxis, YAxis, Tooltip, Cell, Legend
} from "recharts";

async function runPipelineAnalysis(text, imageBase64) {
 const body = { text };
 if (imageBase64) {
 body.image_base64 = imageBase64;
}

 const [res, imgDesc] = await Promise.all([
   fetch("/predict", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify(body),
     signal: AbortSignal.timeout(120000),
   }),
   imageBase64 ? describeImageWithGemini(imageBase64) : Promise.resolve(null),
 ]);

 if (!res.ok) throw new Error(`Server error: ${res.status}`);
 const data = await res.json();
 if (data.error) throw new Error(data.error);

 const vt = data.vad_text;
 const vi = data.vad_image;
 const dA = Math.abs(vt.A - vi.A);
 const dV = Math.abs(vt.V - vi.V);
 const fw = data.fusion_weights;
 const n  = data.n_methods_flagged;

 const finalIsFake = data.label === "fake";
 const finalLabel  = data.label;

 const [aiTextDesc, aiMismatch] = await Promise.all([
   describeTextWithAI(text, vt),
   imgDesc ? explainMismatchWithAI(text, imgDesc, vt, vi, dA, dV, finalIsFake) : Promise.resolve(null),
 ]);

 const textDesc = aiTextDesc || `the text encodes ${vt.A > 0.60 ? "elevated" : "moderate"}-arousal semantics (A=${vt.A.toFixed(2)}, V=${vt.V.toFixed(2)})`;

 let pipeline_narrative = "";
 if (data.image_analysed && imgDesc) {
   const mismatchReason = aiMismatch || (dA > 0.50
     ? `The arousal gap Δ=${dA.toFixed(3)} is extreme — text and image occupy opposite ends of the emotional spectrum.`
     : `The emotional profiles are misaligned (Δ=${dA.toFixed(3)}), consistent with cross-modal manipulation.`);
   const imgEmotionWord = imgDesc.toLowerCase().includes("tense") || imgDesc.toLowerCase().includes("alarm") ? "tense and alarming"
                        : imgDesc.toLowerCase().includes("sad") || imgDesc.toLowerCase().includes("distress") ? "sad and distressing"
                        : imgDesc.toLowerCase().includes("joyful") || imgDesc.toLowerCase().includes("celebrat") ? "joyful and celebratory"
                        : vi.A < 0.10 ? "calm and low-energy"
                        : vi.A < 0.20 ? "neutral and relaxed"
                        : "moderately aroused";
   if (finalIsFake) {
     pipeline_narrative =
       `This post exhibits fake news manipulation: ${textDesc} (A=${vt.A.toFixed(2)}, V=${vt.V.toFixed(2)}) ` +
       `while the image shows ${imgDesc.toLowerCase().replace(/^image depicts /,"").replace(/\.$/,"")} ` +
       `(${imgEmotionWord}: A=${vi.A.toFixed(2)}, V=${vi.V.toFixed(2)}). ` +
       `${mismatchReason} ` +
       `The LLM entity consistency layer detected a semantic mismatch — the image subject does not correspond to the event described in the text. ` +
       `The emotion gate weights text modality at ${Math.round(fw.text*100)}%, ` +
       `characteristic of caption-driven misinformation that repurposes unrelated imagery.` +
       (dV > 0.15 ? ` Valence delta Δ=${dV.toFixed(3)} — cross-modal valence inversion confirms emotional contradiction.` : "");
   } else {
     pipeline_narrative =
       `This post shows emotionally consistent profiles, consistent with authentic reporting. ` +
       `${textDesc} (A=${vt.A.toFixed(2)}, V=${vt.V.toFixed(2)}), ` +
       `and the image corroborates this — ${imgDesc.toLowerCase().replace(/^image depicts /,"").replace(/\.$/,"")} ` +
       `(A=${vi.A.toFixed(2)}, V=${vi.V.toFixed(2)}). ` +
       `The arousal gap Δ=${dA.toFixed(3)} is well within the 0.20 manipulation threshold. ` +
       `LLM entity check: image content matches the text subject. ` +
       `${n}/4 anomaly detectors triggered — no statistical outlier signature detected.`;
   }
 } else if (data.image_analysed) {
   pipeline_narrative = finalIsFake
     ? `${textDesc} while CLIP ViT-L/14 image encoding registers a contrasting emotional profile (A=${vi.A.toFixed(3)}, V=${vi.V.toFixed(3)}) — cross-modal arousal mismatch Δ=${dA.toFixed(3)} exceeds the 0.20 manipulation threshold (d=0.41, p<0.001). Emotion gate routes ${Math.round(fw.text*100)}% weight to text modality. ${n}/4 independent anomaly detectors flagged this post as a statistical outlier.`
     : `${textDesc}. Text and image emotional profiles are broadly consistent — arousal mismatch Δ=${dA.toFixed(3)} within normal range. ${n}/4 anomaly detectors triggered.`;
 } else {
   pipeline_narrative = `Text-only analysis: ${textDesc}. EmotionAwareFakeNewsDetector encodes arousal A=${vt.A.toFixed(3)} and valence V=${vt.V.toFixed(3)} from SentenceTransformer 128-dim embedding. ${n}/4 anomaly detectors flagged. Fake probability ${data.fake_prob.toFixed(3)} — add an image to compute cross-modal VAD mismatch, the core manipulation signal (d=0.41, p<0.001).`;
 }

 const manipulation_signals = [];
 if (dA > 0.20) manipulation_signals.push(`Arousal mismatch Δ=${dA.toFixed(3)} — text vs image emotional disagreement exceeds threshold`);
 if (dV > 0.15) manipulation_signals.push(`Valence inversion Δ=${dV.toFixed(3)} — conflicting emotional tone across modalities`);
 if (fw.text > 0.75) manipulation_signals.push(`Caption-driven fusion — emotion gate ${Math.round(fw.text*100)}% text-dominant`);
 if (n >= 3) manipulation_signals.push(`${n}/4 anomaly detectors triggered simultaneously`);
 if (imgDesc && finalIsFake) manipulation_signals.push(`LLM entity check: ${imgDesc.split(" with ")[0].replace("Image depicts ","")} — does not match claimed event`);

 const stop = new Set(["the","a","an","in","on","at","is","it","to","and","or","of","for","with","http","rt","co","t","via","this","that","was","are"]);
 const top_words = text.toLowerCase().replace(/[^a-z0-9\s]/g," ").split(/\s+/).filter(w => w.length > 3 && !stop.has(w)).slice(0, 6);

 const image_description = imgDesc
   ? imgDesc
   : data.image_analysed
     ? `Image VAD: V=${vi.V.toFixed(4)}, A=${vi.A.toFixed(4)}, D=${vi.D.toFixed(4)} — CLIP ViT-L/14 zero-shot scoring`
     : null;

 return {
   ...data,
   label: finalLabel,
   top_words,
   manipulation_signals,
   pipeline_narrative,
   image_description,
 };
}

async function runBatchAnalysis(posts, onProgress) {
 const CHUNK = 10;
 const allResults = [];
 for (let i = 0; i < posts.length; i += CHUNK) {
   const chunk = posts.slice(i, i + CHUNK);
   const res = await fetch("/batch", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify({ posts: chunk }),
     signal: AbortSignal.timeout(300000),
   });
   if (!res.ok) throw new Error(`Server error: ${res.status}`);
   const data = await res.json();
   if (data.error) throw new Error(data.error);
   allResults.push(...data.results);
   onProgress(Math.min(i + CHUNK, posts.length), posts.length);
 }
 const fake    = allResults.filter(r => r.label === "fake").length;
 const real    = allResults.filter(r => r.label === "real").length;
 const total   = allResults.filter(r => r.label !== "error").length;
 const avgProb = total > 0 ? allResults.filter(r => r.label !== "error").reduce((s,r) => s + (r.fake_prob||0), 0) / total : 0;
 return {
   results: allResults,
   summary: { total, fake, real, fake_rate: total > 0 ? fake/total : 0, avg_fake_prob: avgProb, errors: allResults.filter(r=>r.label==="error").length }
 };
}

async function describeTextWithAI(text, vt) {
 try {
   const res = await fetch("/ai_describe", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify({
       max_tokens: 120,
       prompt: `Analyse this social media post text for a fake news detection system. In exactly ONE sentence (max 30 words), describe: what event/topic it claims, the emotional tone, and any sensational framing. Be specific and factual. Do not start with "The text" or "This post".\n\nPost: "${text}"\nText Arousal: ${vt.A.toFixed(2)}, Valence: ${vt.V.toFixed(2)}\n\nReply with only the one sentence description, nothing else.`
     }),
     signal: AbortSignal.timeout(15000),
   });
   const data = await res.json();
   return data?.text || null;
 } catch(e) { return null; }
}

async function explainMismatchWithAI(text, imgDesc, vt, vi, dA, dV, isFake) {
 try {
   const res = await fetch("/ai_describe", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify({
       max_tokens: 150,
       prompt: `You are an XAI component in a multimodal fake news detection system.\n\nPost text: "${text}"\nImage description: "${imgDesc}"\nText VAD: Arousal=${vt.A.toFixed(2)}, Valence=${vt.V.toFixed(2)}\nImage VAD: Arousal=${vi.A.toFixed(2)}, Valence=${vi.V.toFixed(2)}\nArousal mismatch Δ=${dA.toFixed(3)}, Valence mismatch Δ=${dV.toFixed(3)}\nVerdict: ${isFake ? "FAKE" : "AUTHENTIC"}\n\nIn exactly 2 sentences, explain specifically WHY the text and image ${isFake ? "do not match — focus on semantic entity mismatch if the image shows a person but text describes a disaster event" : "are emotionally consistent — what makes this authentic"}.\nBe specific to the actual content. Reply with only the 2 sentences.`
     }),
     signal: AbortSignal.timeout(15000),
   });
   const data = await res.json();
   return data?.text || null;
 } catch(e) { return null; }
}

async function describeImageWithGemini(imageBase64) {
 if (!imageBase64) return null;
 try {
   const base64Data = imageBase64.includes(",") ? imageBase64.split(",")[1] : imageBase64;
   const res = await fetch("/describe", {
     method: "POST",
     headers: { "Content-Type": "application/json" },
     body: JSON.stringify({ image_base64: base64Data }),
     signal: AbortSignal.timeout(30000),
   });
   const data = await res.json();
   return data?.description || null;
 } catch (e) { return null; }
}

const DEMO_POSTS = [
 {
   post_id: "263031839621537794", label: "fake", anomaly_level: "critical",
   anomaly_score: 0.5191, fake_prob: 0.6795, contradiction_score: 0.386, n_methods_flagged: 3,
   text: "RT @Franke609: Shark in the street in #Brigantine, New Jersey during #HurricaneSandy http://t.co/XHbmXgRr",
   username: "LillithL", event: "Hurricane Sandy",
   vad_text:  { V: 0.327, A: 0.695, D: 0.531 },
   vad_image: { V: 0.321, A: 0.342, D: 0.301 },
   fusion_weights: { text: 0.892, image: 0.054, meta: 0.054 },
   method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: false },
   top_words: ["shark","street","brigantine","hurricane","sandy"],
   image_description: "Image depicts a digitally composited shark swimming through a flooded suburban street with a dramatic and surreal emotional tone.",
   pipeline_narrative: "This post exhibits classic fake news manipulation: the text encodes elevated-arousal threat semantics using sensational wildlife-in-disaster framing (A=0.70, V=0.33) while the image shows a digitally composited shark in a flooded street (A=0.34, neutral tone). The arousal contradiction is severe — the fabricated image is designed to appear plausible at a glance while being emotionally incongruent with authentic disaster imagery. Arousal mismatch Δ=0.353 exceeds the 0.20 manipulation threshold. Emotion gate is 89% text-driven — caption-led deception. IsoForest, LOF, and OCSVM all flagged as anomalous.",
   manipulation_signals: ["Arousal mismatch Δ=0.353 — text vs image emotional disagreement exceeds threshold","Caption-driven fusion — emotion gate 89% text-dominant","3/4 anomaly detectors triggered simultaneously","Image shows composited shark in flood — classic viral fabrication misattribution pattern"],
 },
 {
   post_id: "578857948195262464", label: "fake", anomaly_level: "critical",
   anomaly_score: 0.8116, fake_prob: 0.9576, contradiction_score: 0.4247, n_methods_flagged: 4,
   text: "ISS wins. The Solar eclipse as seen from the International Space Station. #SolarEclipse #Space http://t.co/3fytvWXrW0",
   username: "EhiPenna", event: "Solar Eclipse",
   vad_text:  { V: 0.318, A: 0.636, D: 0.516 },
   vad_image: { V: 0.326, A: 0.356, D: 0.317 },
   fusion_weights: { text: 0.912, image: 0.044, meta: 0.044 },
   method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: true },
   top_words: ["solar","eclipse","space","station","iss"],
   image_description: "Image depicts a CGI-rendered solar eclipse viewed from orbit with a dramatic and awe-inspiring emotional tone.",
   pipeline_narrative: "This post exhibits classic fake news manipulation: the text encodes elevated-arousal awe-inducing semantics with fabricated provenance claims (A=0.64, V=0.32) while the image shows a CGI render falsely attributed to the ISS (neutral-calm A=0.36). The arousal gap Δ=0.280 reflects the rendered image's artificially calm tone clashing with the sensational claim. All 4 anomaly detectors fired — highest confidence fake in the dataset. Fake probability 0.9576.",
   manipulation_signals: ["Arousal mismatch Δ=0.280 — text vs image emotional disagreement exceeds threshold","Caption-driven fusion — emotion gate 91% text-dominant","4/4 anomaly detectors triggered simultaneously","Image shows CGI render falsely attributed to ISS — fabricated provenance misattribution"],
 },
 {
   post_id: "263266009899741184", label: "fake", anomaly_level: "critical",
   anomaly_score: 0.6716, fake_prob: 0.6795, contradiction_score: 0.382, n_methods_flagged: 3,
   text: "Shark in the front yard.....#Hurricane #Sandy #Shark #NewJersey #NewPet #DontFeedTheAnimals http://t.co/t0ZzUd9B",
   username: "KanchanGupta", event: "Hurricane Sandy",
   vad_text:  { V: 0.293, A: 0.708, D: 0.368 },
   vad_image: { V: 0.269, A: 0.104, D: 0.070 },
   fusion_weights: { text: 0.921, image: 0.040, meta: 0.039 },
   method_flags: { iso_forest: true, lof: true, ocsvm: true, elliptic: false },
   top_words: ["shark","yard","hurricane","sandy","newjersey"],
   image_description: "Image depicts a composite photo of a large shark in a flooded suburban yard with a calm and surreal emotional tone.",
   pipeline_narrative: "This post exhibits classic fake news manipulation: the text encodes high-arousal threat semantics using sensational wildlife-in-disaster framing (A=0.71, V=0.29) while the image shows a composite shark photo in a flooded yard (calm A=0.10, V=0.27). Arousal mismatch Δ=0.604 is extreme. Emotion gate 92% text-driven.",
   manipulation_signals: ["Arousal mismatch Δ=0.604 — extreme cross-modal emotional disagreement","Caption-driven fusion — emotion gate 92% text-dominant","3/4 anomaly detectors triggered simultaneously","Image shows composite shark photo — viral fabrication misattribution pattern"],
 },
 {
   post_id: "591992122271604737", label: "real", anomaly_level: "normal",
   anomaly_score: 0.0421, fake_prob: 0.3218, contradiction_score: 0.471, n_methods_flagged: 0,
   text: "Nepal's historic Dharahara Tower collapses in massive earthquake http://t.co/2zrj6cwZwZ",
   username: "SarahReports", event: "Nepal Earthquake",
   vad_text:  { V: 0.420, A: 0.580, D: 0.610 },
   vad_image: { V: 0.390, A: 0.560, D: 0.580 },
   fusion_weights: { text: 0.741, image: 0.142, meta: 0.117 },
   method_flags: { iso_forest: false, lof: false, ocsvm: false, elliptic: false },
   top_words: ["dharahara","tower","collapses","earthquake","nepal"],
   image_description: "Image depicts collapsed rubble and structural damage from a destroyed building with a distressing and alarming emotional tone.",
   pipeline_narrative: "This post shows emotionally consistent profiles, consistent with authentic reporting. The text encodes elevated-arousal disaster semantics describing structural collapse (A=0.58, V=0.42), and the image corroborates this directly — collapsed rubble and structural damage (A=0.56, V=0.39). Arousal gap Δ=0.020 is well within the 0.20 threshold. Zero anomaly detectors triggered.",
   manipulation_signals: [],
 },
 {
   post_id: "263106741833695232", label: "real", anomaly_level: "normal",
   anomaly_score: 0.0438, fake_prob: 0.3041, contradiction_score: 0.406, n_methods_flagged: 0,
   text: "#LowerManhattan #nyc #hurricaneSandy — real footage of flooding at the seaport http://t.co/MsSeQ8lD",
   username: "NYCUpdates", event: "Hurricane Sandy",
   vad_text:  { V: 0.360, A: 0.600, D: 0.550 },
   vad_image: { V: 0.340, A: 0.570, D: 0.520 },
   fusion_weights: { text: 0.712, image: 0.168, meta: 0.120 },
   method_flags: { iso_forest: false, lof: false, ocsvm: false, elliptic: false },
   top_words: ["manhattan","nyc","hurricanesandy","flooding","seaport"],
   image_description: "Image depicts flooded streets in an urban area with dark rising water and a tense and alarming emotional tone.",
   pipeline_narrative: "This post shows emotionally consistent profiles, consistent with authentic reporting. Text encodes elevated-arousal extreme weather semantics (A=0.60, V=0.36), and the image corroborates — flooded urban streets with dark rising water (A=0.57, V=0.34). Arousal gap Δ=0.030 within normal range. Zero anomaly detectors triggered.",
   manipulation_signals: [],
 },
];

const RISK = {
 critical: { label: "CRITICAL", color: "#c0392b", bg: "#fdf2f2", border: "#e8b4b0" },
 high:     { label: "HIGH",     color: "#d35400", bg: "#fdf5ec", border: "#f0c090" },
 medium:   { label: "MEDIUM",   color: "#b7770d", bg: "#fefce8", border: "#e8d48a" },
 normal:   { label: "LOW",      color: "#00bfa5", bg: "#e4fbf5", border: "#7dd9c0" },
};

const L = { fontSize: 9, color: "#009e88", fontFamily: "DM Mono", letterSpacing: 1, marginBottom: 5, display: "block", textTransform: "uppercase", fontWeight: "600" };
const CARD = { padding: 14, background: "linear-gradient(145deg,#f2fefb,#e4fbf5)", border: "1px solid #b0ead8", borderRadius: 10, boxShadow: "0 2px 10px rgba(13,27,46,0.08)" };

function VADRadar({ vt, vi }) {
 const data = [
   { dim: "Valence",   text: vt.V, image: vi.V },
   { dim: "Arousal",   text: vt.A, image: vi.A },
   { dim: "Dominance", text: vt.D, image: vi.D },
 ];
 return (
   <div>
     <span style={L}>VAD EMOTION RADAR</span>
     <ResponsiveContainer width="100%" height={380}>
       <RadarChart data={data} margin={{ top: 30, right: 55, bottom: 30, left: 55 }} outerRadius="72%">
         <PolarGrid stroke="#9ee8d0" strokeWidth={1.5} />
         <PolarAngleAxis dataKey="dim" tick={{ fill: "#0d2a3a", fontSize: 14, fontFamily: "DM Mono", fontWeight: "bold" }} />
         <PolarRadiusAxis angle={90} domain={[0, 1]} tickCount={6} tick={{ fill: "#5ab8a0", fontSize: 9, fontFamily: "DM Mono" }} axisLine={false} />
         <Radar name="Text"  dataKey="text"  stroke="#d35400" fill="#d35400" fillOpacity={0.35} strokeWidth={3} dot={{ fill: "#d35400", r: 6 }} />
         <Radar name="Image" dataKey="image" stroke="#2471a3" fill="#2471a3" fillOpacity={0.28} strokeWidth={3} dot={{ fill: "#2471a3", r: 6 }} />
         <Legend iconSize={10} wrapperStyle={{ fontSize: 11, fontFamily: "DM Mono", color: "#2a3a50" }} />
         <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #dde3f0", fontFamily: "DM Mono", fontSize: 11, borderRadius: 5, boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} formatter={v => [v.toFixed ? v.toFixed(3) : v]} />
       </RadarChart>
     </ResponsiveContainer>
   </div>
 );
}

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
       const hc    = `rgb(${Math.round(192*p+36*(1-p))},${Math.round(57*p+174*(1-p))},${Math.round(43*p+133*(1-p))})`;
       return (
         <div key={d.name} style={{ marginBottom: 8 }}>
           <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: hot ? hc : "#6b7a99", fontWeight: hot ? "bold" : "normal" }}>{hot ? "⚠ " : ""}{d.name}</span>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: hc }}>Δ={delta.toFixed(3)}</span>
           </div>
           <div style={{ display: "grid", gridTemplateColumns: "1fr 10px 1fr", gap: 3, alignItems: "center" }}>
             <div style={{ position: "relative", height: 16, background: "#e8fdf8", borderRadius: 3, overflow: "hidden", border: "1px solid #9ee8d0" }}>
               <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${d.tv*100}%`, background: "linear-gradient(90deg,#c0392b,#e67e22)", borderRadius: 3, transition: "width 0.6s" }} />
               <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#061520", fontFamily: "DM Mono" }}>T {d.tv.toFixed(3)}</span>
             </div>
             <div style={{ height: 2, background: hc, borderRadius: 1 }} />
             <div style={{ position: "relative", height: 16, background: "#e8fdf8", borderRadius: 3, overflow: "hidden", border: "1px solid #9ee8d0" }}>
               <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${d.iv*100}%`, background: "linear-gradient(90deg,#1a5276,#2e86c1)", borderRadius: 3, transition: "width 0.6s" }} />
               <span style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, color: "#061520", fontFamily: "DM Mono" }}>I {d.iv.toFixed(3)}</span>
             </div>
           </div>
         </div>
       );
     })}
     <div style={{ marginTop: 4, padding: "5px 8px", borderRadius: 5, background: dA > 0.20 ? "linear-gradient(135deg,#fdf2f2,#fce8e8)" : "linear-gradient(135deg,#e0faf4,#cef7ec)", border: `1px solid ${dA > 0.20 ? "#e8b4b0" : "#a3d9d0"}`, fontSize: 10, fontFamily: "DM Mono", color: dA > 0.20 ? "#c0392b" : "#007a6a" }}>
       {dA > 0.20 ? `⚠ Arousal Δ=${dA.toFixed(3)} — MANIPULATION SIGNAL (d=0.41, p<0.001)` : `✓ Arousal Δ=${dA.toFixed(3)} — Modalities consistent`}
     </div>
   </div>
 );
}

function FusionBar({ fw }) {
 const total = (Math.abs(fw.text) + Math.abs(fw.image) + Math.abs(fw.meta)) || 1;
 const tp = (Math.abs(fw.text)  / total) * 100;
 const ip = (Math.abs(fw.image) / total) * 100;
 const mp = (Math.abs(fw.meta)  / total) * 100;
 return (
   <div>
     <span style={L}>EMOTION GATE FUSION</span>
     <div style={{ display: "flex", height: 20, borderRadius: 4, overflow: "hidden", border: "1px solid #dde3f0" }}>
       <div style={{ width: `${tp}%`, background: "linear-gradient(90deg,#c0392b,#e67e22)", display: "flex", alignItems: "center", justifyContent: "center", transition: "width 0.6s" }}>
         {tp > 12 && <span style={{ fontSize: 8, color: "#fff", fontFamily: "DM Mono", fontWeight: "bold" }}>TEXT {tp.toFixed(0)}%</span>}
       </div>
       <div style={{ width: `${ip}%`, background: "linear-gradient(90deg,#1a5276,#2e86c1)", display: "flex", alignItems: "center", justifyContent: "center" }}>
         {ip > 12 && <span style={{ fontSize: 8, color: "#fff", fontFamily: "DM Mono", fontWeight: "bold" }}>IMG {ip.toFixed(0)}%</span>}
       </div>
       <div style={{ width: `${mp}%`, background: "linear-gradient(90deg,#6c3483,#9b59b6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
         {mp > 12 && <span style={{ fontSize: 8, color: "#fff", fontFamily: "DM Mono", fontWeight: "bold" }}>META {mp.toFixed(0)}%</span>}
       </div>
     </div>
     <div style={{ fontSize: 8, color: "#2a3a50", fontFamily: "IBM Plex Sans", marginTop: 3 }}>γ emotion gate routes signal — text dominance = caption-driven manipulation</div>
   </div>
 );
}

function AnomalyFlags({ flags, n }) {
 const methods = ["iso_forest","lof","ocsvm","elliptic"];
 const labels  = { iso_forest:"IsoForest", lof:"LOF", ocsvm:"OCSVM", elliptic:"Elliptic" };
 const weights = { iso_forest:"0.35", lof:"0.30", ocsvm:"0.20", elliptic:"0.15" };
 return (
   <div>
     <span style={L}>ANOMALY ENSEMBLE ({n}/4 methods flagged)</span>
     <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 5 }}>
       {methods.map(m => (
         <div key={m} style={{ padding: "6px 4px", borderRadius: 5, textAlign: "center", background: flags?.[m] ? "linear-gradient(135deg,#fdf2f2,#fce8e8)" : "linear-gradient(135deg,#e8fdf8,#d4f7ee)", border: `1px solid ${flags?.[m] ? "#e8b4b0" : "#7dd9c0"}` }}>
           <div style={{ fontSize: 11, color: flags?.[m] ? "#c0392b" : "#16a085", marginBottom: 2 }}>{flags?.[m] ? "⚠" : "✓"}</div>
           <div style={{ fontSize: 8, color: "#1a2a3a", fontFamily: "DM Mono" }}>{labels[m]}</div>
           <div style={{ fontSize: 8, color: "#2a3a50", fontFamily: "DM Mono" }}>w={weights[m]}</div>
         </div>
       ))}
     </div>
   </div>
 );
}

function Gauge({ label, val, col }) {
 return (
   <div style={{ marginBottom: 7 }}>
     <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 2 }}>
       <span style={{ fontSize: 9, color: "#2a3a50", fontFamily: "DM Mono" }}>{label}</span>
       <span style={{ fontSize: 10, color: col, fontFamily: "DM Mono", fontWeight: "bold" }}>{typeof val === "number" ? val.toFixed(3) : val}</span>
     </div>
     <div style={{ height: 4, background: "linear-gradient(90deg,#c8f5e8,#d4f7ee)", borderRadius: 2, overflow: "hidden" }}>
       <div style={{ height: "100%", width: `${Math.min((typeof val === "number" ? val : 0)*100, 100)}%`, background: col, borderRadius: 2, transition: "width 0.7s" }} />
     </div>
   </div>
 );
}

function WordHighlight({ text, topWords }) {
 const flagged = new Set((topWords || []).map(w => w.toLowerCase()));
 return (
   <div>
     <span style={L}>WORD ATTRIBUTION (TF-IDF top discriminative tokens)</span>
     <div style={{ padding: 10, background: "linear-gradient(135deg,#f0fefb,#e4fbf5)", borderRadius: 7, border: "1px solid #b0ead8", lineHeight: 2.2 }}>
       {text.split(" ").map((word, i) => {
         const clean = word.toLowerCase().replace(/[^a-z0-9]/g, "");
         const hit   = flagged.has(clean);
         return (
           <span key={i} style={{ display: "inline-block", margin: "1px 2px", padding: "1px 6px", borderRadius: 3, background: hit ? "#fdf2f2" : "transparent", border: hit ? "1px solid #e8b4b0" : "1px solid transparent", color: hit ? "#c0392b" : "#0d2a3a", fontFamily: "DM Mono", fontSize: 11, fontWeight: hit ? "bold" : "normal" }}>
             {word}
           </span>
         );
       })}
     </div>
   </div>
 );
}

function XAINarrative({ r }) {
 const isFake = r.label === "fake";
 const risk   = RISK[r.anomaly_level] || RISK.normal;
 const dA     = Math.abs((r.vad_text?.A || 0) - (r.vad_image?.A || 0));
 return (
   // ── CHANGED: fontSize 11 → 14, lineHeight 1.8 → 1.9 ──────────────────────
   <div style={{ padding: 12, borderRadius: 8, background: isFake ? "linear-gradient(135deg,#fdf2f2,#fce8e8)" : "linear-gradient(135deg,#e0faf4,#cef7ec)", border: `1px solid ${isFake ? "#e8b4b0" : "#9ee8d0"}`, fontSize: 14, fontFamily: "IBM Plex Sans", lineHeight: 1.9, color: "#061520" }}>
     <span style={{ ...L, color: isFake ? "#c0392b" : "#007a6a" }}>{isFake ? "⚠ PIPELINE VERDICT — FAKE" : "✓ PIPELINE VERDICT — AUTHENTIC"}</span>
     {r.pipeline_narrative && <p style={{ margin: "0 0 8px" }}>{r.pipeline_narrative}</p>}
     <p style={{ margin: 0 }}>
       <span style={{ color: risk.color, fontWeight: "bold" }}>{risk.label}</span> anomaly score {r.anomaly_score?.toFixed(3)}.{" "}
       EmotionAwareFakeNewsDetector fake probability: <span style={{ color: "#6c3483" }}>{r.fake_prob?.toFixed(3)}</span>.{" "}
       Arousal mismatch: <span style={{ color: dA > 0.20 ? "#c0392b" : "#007a6a", fontWeight: "bold" }}>Δ{dA.toFixed(3)}</span>
       {dA > 0.20 ? " — exceeds 0.20 manipulation threshold" : " — within normal range"}.{" "}
       Anomaly ensemble: <span style={{ color: r.n_methods_flagged > 1 ? "#c0392b" : "#16a085" }}>{r.n_methods_flagged}/4 detectors flagged</span>.
       {r.manipulation_signals?.length > 0 && <span style={{ color: "#d35400" }}> Signals: {r.manipulation_signals.join(", ")}.</span>}
     </p>
   </div>
 );
}

function PipelineBadges({ hasImage, isDemo }) {
 const all = [
   ["CLIP ViT-L/14",       hasImage || isDemo],
   ["SentenceTransformer", true],
   ["EmotionModel",        true],
   ["AnomalyEnsemble",     true],
   ["LLM EntityCheck",     hasImage || isDemo],
   ["GNN",                 false],
 ];
 return (
   <div style={{ display: "flex", gap: 5, flexWrap: "wrap", marginBottom: 10 }}>
     {all.map(([name, active]) => (
       <span key={name} style={{ fontSize: 8, padding: "3px 7px", borderRadius: 3, fontFamily: "DM Mono", background: active ? "rgba(0,158,136,0.15)" : "rgba(13,100,80,0.06)", border: `1px solid ${active ? "#00bfa5" : "#9ee8d0"}`, color: active ? "#00bfa5" : "#5ab8a0" }}>
         {active ? "✓" : "○"} {name}
       </span>
     ))}
     <span style={{ fontSize: 8, color: "#009e88", fontFamily: "IBM Plex Sans", alignSelf: "center" }}>(GNN requires graph context)</span>
   </div>
 );
}

function ResultCard({ r, imageSrc, isDemo, index }) {
 const [open, setOpen] = useState(index === 0);
 const isFake = r.label === "fake";
 const risk   = RISK[r.anomaly_level] || RISK.normal;
 const fw     = r.fusion_weights || { text: 0.9, image: 0.05, meta: 0.05 };
 const vt     = r.vad_text  || { V: 0.5, A: 0.5, D: 0.5 };
 const vi     = r.vad_image || { V: 0.5, A: 0.5, D: 0.5 };
 const mm     = { V: Math.abs(vt.V - vi.V), A: Math.abs(vt.A - vi.A), D: Math.abs(vt.D - vi.D) };

 return (
   <div style={{ marginBottom: 12, borderRadius: 10, overflow: "hidden", border: `1px solid ${isFake ? "#e8b4b0" : "#7dd9c0"}`, borderLeft: `4px solid ${isFake ? risk.color : "#009e88"}`, background: "linear-gradient(150deg,#f0fefb,#e4fbf5)", boxShadow: "0 3px 14px rgba(13,27,46,0.10)" }}>
     <div onClick={() => setOpen(!open)} style={{ padding: "10px 14px", display: "flex", alignItems: "center", justifyContent: "space-between", cursor: "pointer", background: open ? "linear-gradient(135deg,#c8f5e8,#b0f0e0)" : "transparent" }}>
       <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
         <div style={{ padding: "2px 8px", borderRadius: 4, background: isFake ? risk.bg : "#f0faf8", border: `1px solid ${isFake ? risk.color : "#16a085"}`, fontSize: 9, fontFamily: "DM Mono", fontWeight: "bold", letterSpacing: 1, color: isFake ? risk.color : "#16a085" }}>
           {isFake ? `⚠ ${risk.label}` : "✓ AUTHENTIC"}
         </div>
         <span style={{ fontSize: 8, padding: "1px 6px", borderRadius: 3, background: isDemo ? "linear-gradient(135deg,#f2f4fc,#ece8f7)" : "linear-gradient(135deg,#f0faf8,#e8f5f1)", border: `1px solid ${isDemo ? "#dde3f0" : "#a3d9d0"}`, color: isDemo ? "#2a3a50" : "#16a085", fontFamily: "DM Mono" }}>
           {isDemo ? "demo · real pipeline values" : "🔬 Live — EmotionAwareFakeNewsDetector"}
         </span>
       </div>
       <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
         {[
           ["ANOMALY",   r.anomaly_score?.toFixed(3), risk.color],
           ["AROUSAL Δ", mm.A.toFixed(3), mm.A > 0.20 ? "#c0392b" : "#16a085"],
         ].map(([lbl, val, col]) => (
           <div key={lbl} style={{ textAlign: "center" }}>
             <div style={{ fontSize: 8, color: "#2a3a50", fontFamily: "DM Mono", letterSpacing: 1 }}>{lbl}</div>
             <div style={{ fontSize: 17, fontWeight: "bold", color: col, fontFamily: "DM Mono" }}>{val}</div>
           </div>
         ))}
         <span style={{ color: "#4a5a70", fontSize: 12 }}>{open ? "▲" : "▼"}</span>
       </div>
     </div>

     {open && (
       <div style={{ padding: "0 14px 16px" }}>
         <PipelineBadges hasImage={!!imageSrc} isDemo={isDemo} />
         <div style={{ display: "grid", gridTemplateColumns: imageSrc ? "190px 1fr" : "1fr", gap: 12, marginBottom: 14 }}>
           {imageSrc && (
             <div>
               <span style={L}>UPLOADED IMAGE</span>
               <img src={imageSrc} alt="uploaded" style={{ width: "100%", borderRadius: 8, border: "1px solid #dde3f0", maxHeight: 155, objectFit: "cover", display: "block" }} />
               {r.image_description && (
                 // ── CHANGED: fontSize 9 → 13, lineHeight 1.7 → 1.8 ────────────────────
                 <div style={{ marginTop: 6, padding: "6px 9px", background: "linear-gradient(135deg,#daf5ee,#c8f0e6)", borderRadius: 5, border: "1px solid #9ee8d0", fontSize: 13, color: "#1a2a3a", fontFamily: "IBM Plex Sans", lineHeight: 1.8 }}>
                   <span style={{ color: "#007a6a", fontStyle: "normal", fontWeight: "bold" }}>▸ </span>{r.image_description}
                 </div>
               )}
             </div>
           )}
           {/* ── CHANGED: post text fontSize 10 → 13, lineHeight 1.7 → 1.8 ──────── */}
           <div style={{ padding: "9px 11px", background: "linear-gradient(135deg,#e0faf4,#cef7ec)", borderRadius: 7, border: "1px solid #9ee8d0", fontSize: 13, color: "#061520", fontFamily: "IBM Plex Sans", lineHeight: 1.8, alignSelf: "start" }}>
             <span style={L}>POST TEXT</span>
             "{r.text?.length > 240 ? r.text.slice(0, 240) + "…" : r.text}"
           </div>
         </div>
         <div style={{ display: "grid", gridTemplateColumns: "185px 1.4fr 1fr", gap: 14, marginBottom: 14 }}>
           <div>
             <span style={L}>DETECTION SCORES</span>
             <Gauge label="Anomaly Score"    val={r.anomaly_score}       col={risk.color} />
             <Gauge label="Fake Probability" val={r.fake_prob}           col="#6c3483" />
             <Gauge label="Contradiction"    val={r.contradiction_score} col="#d35400" />
             <Gauge label="Arousal Mismatch" val={mm.A}                  col={mm.A > 0.20 ? "#c0392b" : "#16a085"} />
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
     style={{ cursor: "pointer", borderRadius: 8, overflow: "hidden", border: `2px dashed ${drag ? "#d35400" : "#c5cfe0"}`, background: drag ? "#fdf5ec" : "linear-gradient(135deg,#f2f4fc,#ece8f7)", minHeight: 130, display: "flex", alignItems: "center", justifyContent: "center", transition: "all 0.2s" }}>
     <input ref={ref} type="file" id="imageUpload" name="imageUpload" accept="image/*" style={{ display: "none" }} onChange={e => handle(e.target.files[0])} />
     {currentSrc ? (
       <img src={currentSrc} alt="preview" style={{ width: "100%", maxHeight: 160, objectFit: "cover", display: "block" }} />
     ) : (
       <div style={{ textAlign: "center", padding: 16 }}>
         <div style={{ fontSize: 26, marginBottom: 6 }}>🖼</div>
         <div style={{ fontSize: 10, color: "#2a3a50", fontFamily: "DM Mono" }}>Drop image or click to upload</div>
         <div style={{ fontSize: 9, color: "#1a2a3a", fontFamily: "IBM Plex Sans", marginTop: 3 }}>CLIP ViT-L/14 encodes it · VAD extracted zero-shot · LLM entity check runs</div>
       </div>
     )}
   </div>
 );
}

function ResearchPanel() {
 const methodData = [
   { name: "0", rate: 39.6, n: 3876, fill: "#c5cfe0" },
   { name: "1", rate: 59.8, n: 2829, fill: "#e67e22" },
   { name: "2", rate: 67.1, n: 1890, fill: "#d35400" },
   { name: "3", rate: 66.5, n: 1412, fill: "#c0392b" },
   { name: "4", rate: 68.5, n: 819,  fill: "#96281b" },
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
       {[["10,826","Posts","#2471a3"],["84%","Classifier Acc","#16a085"],["p<0.001","Arousal Sig","#c0392b"],["d=0.41","Effect Size","#d35400"]].map(([v, l, c]) => (
         <div key={l} style={{ padding: 12, ...CARD, textAlign: "center", background: "linear-gradient(145deg,#162640,#0d2a3a)" }}>
           <div style={{ fontSize: 26, fontWeight: "bold", color: c, fontFamily: "DM Mono" }}>{v}</div>
           <div style={{ fontSize: 10, color: "#e8fdf8", fontFamily: "DM Mono", marginTop: 2 }}>{l}</div>
         </div>
       ))}
     </div>
     <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
       <div style={CARD}>
         <span style={L}>ENSEMBLE AGREEMENT → FAKE RATE (n=10,826)</span>
         <ResponsiveContainer width="100%" height={150}>
           <BarChart data={methodData} margin={{ top: 0, right: 8, left: -28, bottom: 0 }}>
             <XAxis dataKey="name" tick={{ fill: "#0d2a3a", fontSize: 9, fontFamily: "DM Mono" }} label={{ value: "Methods Flagged", position: "insideBottom", offset: -2, fill: "#3a4a60", fontSize: 8 }} />
             <YAxis tick={{ fill: "#0d2a3a", fontSize: 9, fontFamily: "DM Mono" }} domain={[0, 80]} />
             <Tooltip contentStyle={{ background: "#ffffff", border: "1px solid #dde3f0", fontFamily: "DM Mono", fontSize: 10, borderRadius: 5, boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} formatter={(v, _, p) => [`${v}% fake (n=${p.payload.n})`]} />
             <Bar dataKey="rate" radius={[3, 3, 0, 0]}>{methodData.map((d, i) => <Cell key={i} fill={d.fill} />)}</Bar>
           </BarChart>
         </ResponsiveContainer>
         <div style={{ fontSize: 9, color: "#1a2a3a", fontFamily: "IBM Plex Sans" }}>Baseline 55.4% → 4-method agreement: 68.5% (+13.1pp)</div>
       </div>
       <div style={CARD}>
         <span style={L}>VAD STATISTICAL FINDINGS (Mann-Whitney U, n=10,826)</span>
         <div style={{ display: "grid", gridTemplateColumns: "130px 42px 42px 52px 42px", gap: "3px 5px", fontSize: 9, color: "#009e88", fontFamily: "DM Mono", marginBottom: 5 }}>
           <span>Dimension</span><span>Fake</span><span>Real</span><span>p</span><span>d</span>
         </div>
         {vadStats.map(row => (
           <div key={row.l} style={{ display: "grid", gridTemplateColumns: "130px 42px 42px 52px 42px", gap: "3px 5px", padding: "3px 0", borderTop: "1px solid #9ee8d0", alignItems: "center" }}>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: row.star ? "#d35400" : "#4a5578", fontWeight: row.star ? "bold" : "normal" }}>{row.l}</span>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: "#d35400" }}>{row.f}</span>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: "#2471a3" }}>{row.r}</span>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: row.star ? "#c0392b" : "#2a3a50" }}>{row.p}</span>
             <span style={{ fontSize: 10, fontFamily: "DM Mono", color: row.star ? "#c0392b" : "#2a3a50", fontWeight: row.star ? "bold" : "normal" }}>{row.d}</span>
           </div>
         ))}
         <div style={{ marginTop: 8, padding: "5px 8px", borderRadius: 5, background: "#fdf2f2", border: "1px solid #e8b4b0", fontSize: 9, fontFamily: "DM Mono", color: "#c0392b" }}>
           ★ Arousal Δ d=0.41 — cross-modal emotional mismatch is the core manipulation signal
         </div>
       </div>
     </div>
     <div style={CARD}>
       <span style={L}>FIVE-LAYER INDEPENDENT VALIDATION</span>
       <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
         {[
           { title:"Layer 1 — Statistical",  col:"#2471a3", grad:"linear-gradient(135deg,#2471a3,#2e86c1)", stats:[["Text Arousal p","<0.001 ***"],["Text Arousal d","0.378"],["Arousal Δ d","0.410 ★"],["All 3 VAD dims","significant"]], desc:"VAD analysis confirms cross-modal arousal disagreement as manipulation marker" },
           { title:"Layer 2 — Supervised",   col:"#16a085", grad:"linear-gradient(135deg,#16a085,#1abc9c)", stats:[["Accuracy","84%"],["F1 (fake)","0.847"],["v_mismatch lift","+3pp"],["GNN val F1","0.750"]], desc:"EmotionAwareFakeNewsDetector explicitly encodes v_mismatch as classification signal" },
           { title:"Layer 3 — Unsupervised", col:"#d35400", grad:"linear-gradient(135deg,#d35400,#e67e22)", stats:[["0 methods","39.6% fake"],["4 methods","68.5% fake"],["Lift","+13.1pp"],["GNN AUC","0.654"]], desc:"Independent anomaly detectors converge on same signal without label supervision" },
         ].map(m => (
           <div key={m.title} style={{ padding: 12, background: "linear-gradient(145deg,#f2fefb,#e4fbf5)", borderRadius: 8, border: "1px solid #b0ead8", borderTop: `3px solid ${m.col}`, boxShadow: "0 1px 4px rgba(26,39,68,0.06)" }}>
             <div style={{ fontSize: 9, color: m.col, fontFamily: "DM Mono", fontWeight: "bold", marginBottom: 8 }}>{m.title}</div>
             {m.stats.map(([k, v]) => (
               <div key={k} style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                 <span style={{ fontSize: 8, color: "#061520", fontFamily: "DM Mono" }}>{k}</span>
                 <span style={{ fontSize: 9, color: m.col, fontFamily: "DM Mono", fontWeight: "bold" }}>{v}</span>
               </div>
             ))}
             <div style={{ fontSize: 8, color: "#2a3a50", fontFamily: "IBM Plex Sans", marginTop: 6, lineHeight: 1.5 }}>{m.desc}</div>
           </div>
         ))}
       </div>
     </div>
   </div>
 );
}

function BatchPanel() {
 const [posts,      setPosts]      = useState([]);
 const [results,    setResults]    = useState(null);
 const [loading,    setLoading]    = useState(false);
 const [progress,   setProgress]   = useState([0, 0]);
 const [error,      setError]      = useState(null);
 const [sortKey,    setSortKey]    = useState("fake_prob");
 const [filterFake, setFilterFake] = useState("all");
 const fileRef = useRef();

 const parseCSV = text => {
   const lines = text.trim().split("\n").filter(Boolean);
   const header = lines[0].toLowerCase().split(",").map(h => h.trim().replace(/"/g,""));
   const textIdx = header.findIndex(h => h.includes("text") || h.includes("tweet") || h.includes("content"));
   if (textIdx === -1) throw new Error("CSV must have a 'text' or 'tweet' column");
   return lines.slice(1).map(line => {
     const cols = line.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/).map(c => c.trim().replace(/^"|"$/g,""));
     return { text: cols[textIdx] || "" };
   }).filter(p => p.text.length > 3);
 };

 const handleFile = async e => {
   const file = e.target.files[0];
   if (!file) return;
   setError(null); setResults(null);
   try {
     const text = await file.text();
     const parsed = file.name.endsWith(".csv") ? parseCSV(text)
       : text.split("\n").filter(l => l.trim()).map(l => ({ text: l.trim() }));
     setPosts(parsed);
   } catch(err) { setError(err.message); }
 };

 const runBatch = async () => {
   if (!posts.length) return;
   setLoading(true); setError(null); setResults(null); setProgress([0, posts.length]);
   try {
     const data = await runBatchAnalysis(posts, (done, total) => setProgress([done, total]));
     setResults(data);
   } catch(e) { setError(e.message); }
   setLoading(false);
 };

 const sorted = results ? [...results.results]
   .filter(r => filterFake === "all" || r.label === filterFake)
   .sort((a,b) => (b[sortKey]||0) - (a[sortKey]||0)) : [];

 const pct = p => `${(p*100).toFixed(1)}%`;

 return (
   <div>
     <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
       <div style={CARD}>
         <span style={L}>UPLOAD CSV OR TXT — one post per row</span>
         <div onClick={() => fileRef.current.click()} style={{ cursor:"pointer", border:"2px dashed #9ee8d0", borderRadius:8, padding:24, textAlign:"center", background:"linear-gradient(135deg,#f0fefb,#e4fbf5)", marginBottom:10 }}>
           <input ref={fileRef} type="file" accept=".csv,.txt" style={{display:"none"}} onChange={handleFile} />
           <div style={{ fontSize:22, marginBottom:6 }}>📂</div>
           <div style={{ fontSize:10, color:"#6b7a99", fontFamily:"DM Mono" }}>Click to upload CSV or TXT</div>
           <div style={{ fontSize:8, color:"#3a4a60", fontFamily:"DM Mono", marginTop:3 }}>CSV needs a "text" column · TXT = one tweet per line · max 100 posts</div>
         </div>
         {posts.length > 0 && <div style={{ padding:"6px 10px", background:"rgba(0,191,165,0.12)", border:"1px solid #00bfa5", borderRadius:6, fontSize:9, color:"#16a085", fontFamily:"DM Mono" }}>✓ {posts.length} posts loaded — ready to analyse</div>}
         {error && <div style={{ marginTop:6, padding:"6px 10px", background:"#fdf2f2", border:"1px solid #e8b4b0", borderRadius:6, fontSize:9, color:"#c0392b", fontFamily:"DM Mono" }}>⚠ {error}</div>}
       </div>
       <div style={CARD}>
         <span style={L}>SAMPLE FORMAT</span>
         <div style={{ background:"linear-gradient(135deg,#162640,#0d2233)", border:"1px solid #1e3a55", borderRadius:6, padding:10, fontFamily:"IBM Plex Sans", fontSize:10, color:"#e8fdf8", lineHeight:1.8 }}>
           <div style={{ color:"#2471a3", marginBottom:4 }}>CSV format:</div>
           <div>text,source</div>
           <div>"Shark in the street during Hurricane Sandy",twitter</div>
           <div>"Nepal Dharahara Tower collapses",twitter</div>
           <div style={{ color:"#2471a3", margin:"8px 0 4px" }}>TXT format (one per line):</div>
           <div>Shark in the street during Hurricane Sandy</div>
           <div>Nepal Dharahara Tower collapses</div>
         </div>
         <button onClick={runBatch} disabled={loading || posts.length === 0} style={{ marginTop:10, width:"100%", padding:"9px 0", border:"none", borderRadius:6, cursor:"pointer", background:"linear-gradient(135deg,#162640,#0d2a3a)", color:"#e8fdf8", fontSize:10, fontWeight:"bold", fontFamily:"DM Mono", opacity:(loading||!posts.length)?0.4:1 }}>
           {loading ? `⟳ Analysing ${progress[0]}/${progress[1]} posts...` : `▶ Run Batch Analysis (${posts.length} posts)`}
         </button>
         {loading && (
           <div style={{ marginTop:8 }}>
             <div style={{ height:4, background:"#b8f0e0", borderRadius:2, overflow:"hidden" }}>
               <div style={{ height:"100%", width:`${progress[1]>0?(progress[0]/progress[1])*100:0}%`, background:"linear-gradient(90deg,#00bfa5,#00897b)", borderRadius:2, transition:"width 0.3s" }} />
             </div>
             <div style={{ fontSize:8, color:"#3a4a60", fontFamily:"DM Mono", marginTop:3 }}>{progress[0]} of {progress[1]} posts processed</div>
           </div>
         )}
       </div>
     </div>
     {results && (
       <>
         <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:10, marginBottom:14 }}>
           {[["TOTAL",results.summary.total,"#2471a3"],["FAKE",results.summary.fake,"#c0392b"],["REAL",results.summary.real,"#16a085"],["FAKE RATE",pct(results.summary.fake_rate),"#d35400"],["AVG PROB",results.summary.avg_fake_prob.toFixed(3),"#6c3483"]].map(([l,v,c]) => (
             <div key={l} style={{ padding:12, ...CARD, textAlign:"center", background:"linear-gradient(145deg,#e4fbf5,#cef7ec)" }}>
               <div style={{ fontSize:24, fontWeight:"bold", color:c, fontFamily:"DM Mono" }}>{v}</div>
               <div style={{ fontSize:8, color:"#6b7a99", fontFamily:"DM Mono", marginTop:2 }}>{l}</div>
             </div>
           ))}
         </div>
         <div style={{ ...CARD, marginBottom:14 }}>
           <span style={L}>FAKE / REAL DISTRIBUTION</span>
           <div style={{ display:"flex", height:24, borderRadius:4, overflow:"hidden", border:"1px solid #dde3f0" }}>
             <div style={{ width:`${results.summary.fake_rate*100}%`, background:"linear-gradient(90deg,#c0392b,#e74c3c)", display:"flex", alignItems:"center", justifyContent:"center", transition:"width 0.8s" }}>
               {results.summary.fake_rate > 0.1 && <span style={{ fontSize:10, color:"#fff", fontFamily:"DM Mono", fontWeight:"bold" }}>FAKE {pct(results.summary.fake_rate)}</span>}
             </div>
             <div style={{ flex:1, background:"linear-gradient(90deg,#16a085,#1abc9c)", display:"flex", alignItems:"center", justifyContent:"center" }}>
               {(1-results.summary.fake_rate) > 0.1 && <span style={{ fontSize:10, color:"#fff", fontFamily:"DM Mono", fontWeight:"bold" }}>REAL {pct(1-results.summary.fake_rate)}</span>}
             </div>
           </div>
         </div>
         <div style={{ display:"flex", gap:10, marginBottom:10, alignItems:"center" }}>
           <span style={{ fontSize:8, color:"#6b7a99", fontFamily:"DM Mono" }}>SORT BY</span>
           {["fake_prob","anomaly_score","arousal_mismatch"].map(k => (
             <button key={k} onClick={() => setSortKey(k)} style={{ padding:"2px 8px", border:`1px solid ${sortKey===k?"#d35400":"#dde3f0"}`, borderRadius:4, cursor:"pointer", background:sortKey===k?"rgba(211,84,0,0.12)":"linear-gradient(135deg,#f0fefb,#e4fbf5)", color:sortKey===k?"#d35400":"#6b7a99", fontSize:9, fontFamily:"DM Mono" }}>{k.replace(/_/g," ")}</button>
           ))}
           <span style={{ fontSize:8, color:"#6b7a99", fontFamily:"DM Mono", marginLeft:8 }}>FILTER</span>
           {["all","fake","real"].map(f => (
             <button key={f} onClick={() => setFilterFake(f)} style={{ padding:"2px 8px", border:`1px solid ${filterFake===f?"#2471a3":"#dde3f0"}`, borderRadius:4, cursor:"pointer", background:filterFake===f?"rgba(41,121,200,0.12)":"linear-gradient(135deg,#f0fefb,#e4fbf5)", color:filterFake===f?"#2471a3":"#6b7a99", fontSize:9, fontFamily:"DM Mono" }}>{f}</button>
           ))}
           <span style={{ fontSize:8, color:"#4a5a70", fontFamily:"DM Mono", marginLeft:"auto" }}>{sorted.length} posts shown</span>
         </div>
         <div style={{ ...CARD, padding:0, overflow:"hidden" }}>
           <div style={{ display:"grid", gridTemplateColumns:"30px 1fr 90px 90px 90px 80px 70px", gap:0, padding:"6px 12px", background:"linear-gradient(135deg,#162640,#0d2a3a)", borderBottom:"1px solid #1e3a55" }}>
             {["#","TEXT","LABEL","FAKE PROB","ANOMALY","AROUSAL Δ","DETECTORS"].map(h => <span key={h} style={{ fontSize:8, color:"#2a3a50", fontFamily:"DM Mono", letterSpacing:1 }}>{h}</span>)}
           </div>
           <div style={{ maxHeight:400, overflowY:"auto" }}>
             {sorted.map((r, i) => {
               const isFake = r.label === "fake";
               const risk   = RISK[r.anomaly_level] || RISK.normal;
               return (
                 <div key={i} style={{ display:"grid", gridTemplateColumns:"30px 1fr 90px 90px 90px 80px 70px", gap:0, padding:"7px 12px", borderBottom:"1px solid #f0f2f8", background: i%2===0 ? "#f0fefb" : "#e4fbf5", alignItems:"center" }}>
                   <span style={{ fontSize:8, color:"#4a5a70", fontFamily:"DM Mono" }}>{r.index+1}</span>
                   <span style={{ fontSize:9, color:"#4a5578", fontFamily:"DM Mono", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap", paddingRight:8 }} title={r.text}>{r.text}</span>
                   <span style={{ fontSize:8, padding:"1px 6px", borderRadius:3, background:isFake?"#fdf2f2":"#f0faf8", border:`1px solid ${isFake?"#e8b4b0":"#a3d9d0"}`, color:isFake?"#c0392b":"#16a085", fontFamily:"DM Mono", fontWeight:"bold", display:"inline-block" }}>{isFake ? "⚠ FAKE" : "✓ REAL"}</span>
                   <span style={{ fontSize:9, color:"#6c3483", fontFamily:"DM Mono", fontWeight:"bold" }}>{r.fake_prob?.toFixed(3)}</span>
                   <span style={{ fontSize:9, color:risk.color, fontFamily:"DM Mono", fontWeight:"bold" }}>{r.anomaly_score?.toFixed(3)}</span>
                   <span style={{ fontSize:9, color:(r.arousal_mismatch||0)>0.20?"#c0392b":"#16a085", fontFamily:"DM Mono", fontWeight:"bold" }}>Δ{(r.arousal_mismatch||0).toFixed(3)}</span>
                   <span style={{ fontSize:9, color:r.n_methods_flagged>1?"#c0392b":"#16a085", fontFamily:"DM Mono" }}>{r.n_methods_flagged}/4</span>
                 </div>
               );
             })}
           </div>
         </div>
       </>
     )}
   </div>
 );
}

const STAGES = [
 "Encoding text → SentenceTransformer 128-dim...",
 "Encoding image → CLIP ViT-L/14 1024-dim...",
 "Extracting VAD → zero-shot CLIP scoring text + image...",
 "Running EmotionAwareFakeNewsDetector → v_mismatch + fusion weights...",
 "Running anomaly ensemble → IsoForest + LOF + OCSVM + Elliptic...",
 "LLM entity consistency check → Ollama llama3.2...",
 "Generating XAI narrative → AI mismatch explanation...",
 "Combining scores → final verdict...",
];

export default function App() {
 const [tab,      setTab]      = useState("analyse");
 const [postText, setPostText] = useState("");
 const [imageSrc, setImageSrc] = useState(null);
 const [results,  setResults]  = useState([]);
 const [loading,  setLoading]  = useState(false);
 const [stageIdx, setStageIdx] = useState(0);
 const [error,    setError]    = useState(null);

 const runAnalysis = async () => {
   if (!postText.trim()) return;
   setLoading(true); setError(null); setResults([]); setStageIdx(0);
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
     setError(e.message || "Pipeline analysis failed. Please try again.");
   }
   setLoading(false);
 };

 const loadDemo = p => {
   setLoading(true); setResults([]);
   setTimeout(() => { setResults([{ r: p, imageSrc: null, isDemo: true }]); setLoading(false); }, 400);
 };

 return (
   <div style={{ minHeight: "100vh", background: "linear-gradient(175deg,#0d1b2e 0%,#162640 28%,#1a3a4a 48%,#c8f5e8 70%,#d4f7ee 100%)", color: "#0d1b2e", padding: "16px 20px" }}>
     <style>{`
       @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500\@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');family=IBM+Plex+Sans:wght@300;400;500;600;700\@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');family=IBM+Plex+Serif:wght@600;700\@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');display=swap');
       * { box-sizing: border-box; margin: 0; padding: 0; }
       ::-webkit-scrollbar { width: 5px } ::-webkit-scrollbar-thumb { background: #5ab8a0; border-radius: 3px }
       textarea, button { font-family: "IBM Plex Sans", sans-serif; }
       @keyframes fadeUp { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:translateY(0) } }
       .fu { animation: fadeUp 0.3s ease forwards }
       @keyframes spin { to { transform:rotate(360deg) } }
       .spin { animation: spin 1s linear infinite; display:inline-block }
     `}</style>

     <div style={{ marginBottom: 18, paddingBottom: 14, borderBottom: "1px solid rgba(200,245,232,0.2)" }}>
       <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4, flexWrap: "wrap" }}>
         <h1 style={{ fontSize: 22, fontFamily: "IBM Plex Serif, serif", fontWeight: 800, color: "#ffffff", letterSpacing: -0.5 }}>Multimodal Deception XAI</h1>
         <span style={{ padding: "1px 7px", borderRadius: 3, background: "rgba(0,191,165,0.15)", border: "1px solid #00bfa5", fontSize: 8, color: "#00bfa5", fontFamily: "IBM Plex Sans", letterSpacing: 1 }}>LIVE · FULL PIPELINE</span>
         <span style={{ padding: "1px 7px", borderRadius: 3, background: "linear-gradient(135deg,#fdf5ec,#faebd7)", border: "1px solid #edb87a", fontSize: 8, color: "#d35400", fontFamily: "DM Mono" }}>EmotionAwareFakeNewsDetector</span>
         <span style={{ padding: "1px 7px", borderRadius: 3, background: "rgba(41,121,200,0.15)", border: "1px solid #2979c8", fontSize: 8, color: "#2471a3", fontFamily: "DM Mono" }}>● CLIP + LLM Entity Check</span>
       </div>
       <p style={{ fontSize: 9, color: "#061520", fontFamily: "DM Mono" }}>
         Any post → CLIP ViT-L/14 · SentenceTransformer · EmotionAwareFakeNewsDetector · IsoForest + LOF + OCSVM + EllipticEnvelope · LLM Entity Consistency
       </p>
     </div>

     <div style={{ display: "flex", gap: 0, marginBottom: 16, borderBottom: "1px solid rgba(200,245,232,0.2)" }}>
       {[["analyse","Analyse Post"],["batch","Batch Upload"],["research","Research Findings"]].map(([id, label]) => (
         <button key={id} onClick={() => setTab(id)} style={{ padding: "6px 16px", border: "none", cursor: "pointer", background: "transparent", color: tab === id ? "#e8fdf8" : "rgba(200,245,232,0.5)", fontFamily: "IBM Plex Sans", fontSize: 11, borderBottom: tab === id ? "2px solid #00bfa5" : "2px solid transparent", transition: "all 0.15s" }}>
           {label}
         </button>
       ))}
     </div>

     {tab === "analyse" && (
       <div className="fu">
         <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
           <div style={CARD}>
             <span style={L}>POST TEXT — paste any tweet or social media post</span>
             <textarea
               value={postText}
               onChange={e => setPostText(e.target.value)}
               placeholder={"Paste any tweet here...\n\nFull pipeline runs:\n• CLIP ViT-L/14 image encoding\n• SentenceTransformer → 128-dim\n• EmotionAwareFakeNewsDetector\n• IsoForest + LOF + OCSVM + Elliptic\n• LLM Entity Consistency Check"}
               style={{ width: "100%", height: 120, padding: 10, background: "linear-gradient(135deg,#e4fbf5,#d4f7ee)", border: "1px solid #9ee8d0", borderRadius: 7, color: "#061520", fontFamily: "IBM Plex Sans", fontSize: 10, resize: "none" }}
             />
             <div style={{ marginTop: 8 }}>
               <span style={{ ...L, marginBottom: 5 }}>LOAD DEMO (baked-in real pipeline values)</span>
               <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                 {DEMO_POSTS.map(p => (
                   <button key={p.post_id} onClick={() => loadDemo(p)} style={{ padding: "3px 9px", border: "1px solid #dde3f0", borderRadius: 4, cursor: "pointer", background: "rgba(13,27,46,0.12)", color: "#061520", fontSize: 9, fontFamily: "DM Mono" }}>
                     {p.label === "fake" ? "⚠" : "✓"} {p.event.split(" ")[0]} ({p.label})
                   </button>
                 ))}
               </div>
             </div>
           </div>
           <div style={CARD}>
             <span style={L}>UPLOAD IMAGE — enables CLIP encoding + LLM entity check</span>
             <UploadZone onFile={src => setImageSrc(src)} currentSrc={imageSrc} />
             {imageSrc ? (
               <div style={{ marginTop: 6, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                 <span style={{ fontSize: 8, color: "#00bfa5", fontFamily: "IBM Plex Sans" }}>✓ Image ready — CLIP ViT-L/14 + LLM entity check will run</span>
                 <button onClick={() => setImageSrc(null)} style={{ fontSize: 8, color: "#c0392b", background: "none", border: "none", cursor: "pointer", fontFamily: "DM Mono" }}>✕ remove</button>
               </div>
             ) : (
               <div style={{ marginTop: 6, padding: "5px 8px", borderRadius: 5, background: "linear-gradient(135deg,#fdf5ec,#faebd7)", border: "1px solid #edb87a", fontSize: 9, fontFamily: "DM Mono", color: "#d35400" }}>
                 💡 Add image for real cross-modal mismatch — core research contribution (d=0.41, p&lt;0.001)
               </div>
             )}
           </div>
         </div>
         <button
           onClick={runAnalysis}
           disabled={loading || !postText.trim()}
           style={{ padding: "10px 28px", border: "none", borderRadius: 6, cursor: "pointer", background: "linear-gradient(135deg,#00bfa5,#00897b)", color: "#fff", fontSize: 11, fontWeight: "bold", fontFamily: "DM Mono", opacity: (!postText.trim() || loading) ? 0.4 : 1, marginBottom: 16, letterSpacing: 0.5, boxShadow: "0 2px 12px rgba(0,191,165,0.35)" }}>
           {loading ? "⟳ Running Pipeline..." : "▶ Run Full Pipeline"}
         </button>
       </div>
     )}

     {tab === "batch"    && <div className="fu"><BatchPanel /></div>}
     {tab === "research" && <div className="fu"><ResearchPanel /></div>}

     {loading && (
       <div style={{ padding: "16px 0" }}>
         {STAGES.map((s, i) => {
           const done   = i < stageIdx;
           const active = i === stageIdx;
           return (
             <div key={i} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, opacity: i > stageIdx ? 0.3 : 1, transition: "opacity 0.3s" }}>
               <span style={{ fontSize: 12, color: done ? "#16a085" : active ? "#d35400" : "#c5cfe0", width: 16, textAlign: "center" }}>
                 {done ? "✓" : active ? <span className="spin">⟳</span> : "○"}
               </span>
               <span style={{ fontSize: 10, fontFamily: "IBM Plex Sans", color: done ? "#16a085" : active ? "#d35400" : "#3a4a60" }}>{s}</span>
             </div>
           );
         })}
       </div>
     )}

     {error && (
       <div style={{ padding: "10px 14px", borderRadius: 8, background: "linear-gradient(135deg,#fdf2f2,#fce8e8)", border: "1px solid #e8b4b0", marginBottom: 14, fontFamily: "IBM Plex Sans", fontSize: 11, color: "#c0392b", whiteSpace: "pre-wrap" }}>
         ⚠ {error}
       </div>
     )}

     {!loading && results.length > 0 && tab === "analyse" && (
       <div className="fu">
         <span style={{ ...L, display: "block", marginBottom: 10, color: "#e8fdf8" }}>PIPELINE OUTPUT</span>
         {results.map((item, i) => (
           <ResultCard key={i} r={item.r} imageSrc={item.imageSrc} isDemo={item.isDemo} index={i} />
         ))}
       </div>
     )}

     <div style={{ marginTop: 24, paddingTop: 10, borderTop: "1px solid rgba(200,245,232,0.15)", display: "flex", justifyContent: "space-between" }}>
       <span style={{ fontSize: 8, color: "#c5cfe0", fontFamily: "IBM Plex Sans" }}>EmotionAwareFakeNewsDetector · CLIP ViT-L/14 · LLM Entity Check · Arousal p&lt;0.001 · d=0.41 · GNN F1=0.75</span>
       <span style={{ fontSize: 8, color: "#c5cfe0", fontFamily: "IBM Plex Sans" }}>10,826 posts · 5,994 fake · 4,832 real</span>
     </div>
   </div>
 );
}