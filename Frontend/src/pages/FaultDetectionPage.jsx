import { useEffect, useMemo, useRef, useState } from "react";
import jsPDF from "jspdf";
import html2canvas from "html2canvas";
import {
  FiUploadCloud,
  FiDownload,
  FiRefreshCcw,
  FiMaximize2,
  FiCheckCircle,
  FiClock,
  FiImage,
} from "react-icons/fi";
import "./FaultDetectionPage.css";

const API_BASE =
  import.meta.env?.VITE_API_BASE?.replace(/\/$/, "") || "http://localhost:8000";
const resolveImgUrl = (p) => (p?.startsWith("http") ? p : `${API_BASE}${p}`);
const fmtTime = (d) =>
  new Date(d).toLocaleString([], {
    hour12: false,
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });

function FaultDetectionPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [imgCacheBust, setImgCacheBust] = useState(0);
  const [viewMode, setViewMode] = useState("heatmap"); // original | heatmap | overlay | side
  const [blend, setBlend] = useState(1);
  const [queue, setQueue] = useState([]);
  const [history, setHistory] = useState([]);
  const [toast, setToast] = useState(null);
  const [confettiOn, setConfettiOn] = useState(false);

  const reportRef = useRef(null);
  const inputRef = useRef(null);
  const confettiCanvasRef = useRef(null);
  const isProcessingBatch = useRef(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("fd_history");
      if (raw) setHistory(JSON.parse(raw));
    } catch { /* empty */ }
  }, []);

  const pushHistory = async (h) => {
    const item = { id: Date.now() + Math.random(), ...h, createdAt: Date.now() };
    const next = [item, ...history].slice(0, 30);
    setHistory(next);
    try {
      localStorage.setItem("fd_history", JSON.stringify(next));
    } catch { /* empty */ }
    showToast("Saved to History");
  };

  const burstConfetti = () => {
    setConfettiOn(true);
    const canvas = confettiCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { innerWidth: w, innerHeight: h } = window;
    canvas.width = w;
    canvas.height = h;
    const N = 90;
    const parts = new Array(N).fill(0).map(() => ({
      x: Math.random() * w,
      y: -10,
      r: 2 + Math.random() * 4,
      vy: 2 + Math.random() * 3,
      vx: (Math.random() - 0.5) * 2,
      a: Math.random() * Math.PI,
    }));
    let t = 0;
    const tick = () => {
      t += 1;
      ctx.clearRect(0, 0, w, h);
      parts.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;
        p.a += 0.1;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(p.a);
        ctx.fillStyle = ["#22d3ee", "#38bdf8", "#22d3ee", "#38bdf8"][p.r | 0 % 4];
        ctx.fillRect(-p.r, -p.r, p.r * 2, p.r * 2);
        ctx.restore();
      });
      if (t < 80) requestAnimationFrame(tick);
      else setConfettiOn(false);
    };
    requestAnimationFrame(tick);
  };

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(null), 1800);
  };

  const resetAll = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setIsModalOpen(false);
    setDragActive(false);
    setUploading(false);
    setProgress(0);
    setError(null);
    setImgCacheBust(0);
    setViewMode("heatmap");
    setBlend(1);
    setQueue([]);
    if (inputRef.current) inputRef.current.value = "";
  };

  const validImage = (f) => f && f.type.startsWith("image/") && f.size <= 10 * 1024 * 1024;

  const addFilesToQueue = (files) => {
    const accepted = [];
    [...files].forEach((f) => {
      if (!validImage(f)) return;
      accepted.push({
        id: `${f.name}-${Date.now()}-${Math.random()}`,
        file: f,
        name: f.name,
        preview: URL.createObjectURL(f),
        status: "queued",
        pct: 0,
        res: null,
        err: null,
      });
    });
    if (!accepted.length) return;
    setQueue((q) => {
      const next = [...q, ...accepted];
      if (!file) {
        setFile(accepted[0].file);
        setPreview(accepted[0].preview);
      }
      return next;
    });
  };

  const handleFileChange = (e) => addFilesToQueue(e.target.files || []);
  const onDragOver = (e) => { e.preventDefault(); setDragActive(true); };
  const onDragLeave = (e) => { e.preventDefault(); setDragActive(false); };
  const onDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer?.files?.length) addFilesToQueue(e.dataTransfer.files);
  };

  const uploadOne = (f, onPct) =>
    new Promise((resolve, reject) => {
      const formData = new FormData();
      formData.append("file", f);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/fault-detection/predict-fault`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onPct(Math.round((e.loaded / e.total) * 100));
        else onPct(60);
      };
      xhr.onload = () => {
        try {
          if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
          else reject(new Error(`HTTP ${xhr.status}`));
        } catch (err) { reject(err); }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(formData);
    });

  const handleUpload = async () => {
    if (uploading) return;
    setError(null);

    // single
    if (!queue.length && file) {
      setUploading(true);
      setProgress(0);
      try {
        const data = await uploadOne(file, setProgress);
        setResult(data);
        setImgCacheBust(Date.now());
        showToast("Analysis complete");
        burstConfetti();
        await pushHistory({
          name: file.name,
          predicted: data.predicted_class,
          confidence: Math.round((data.confidence ?? 0) * 100),
          gradcam: resolveImgUrl(data.gradcam_image),
          originalPreview: preview,
        });
      } catch (e) {
        setError(String(e.message || e));
      } finally {
        setUploading(false);
      }
      return;
    }

    // batch
    if (!queue.length) return;
    isProcessingBatch.current = true;
    setUploading(true);
    for (let i = 0; i < queue.length; i++) {
      if (!isProcessingBatch.current) break;
      if (queue[i].status === "done") continue;

      setFile(queue[i].file);
      setPreview(queue[i].preview);
      setResult(null);
      setProgress(0);
      setQueue((q) => {
        const c = [...q];
        c[i] = { ...c[i], status: "running", pct: 0 };
        return c;
      });

      try {
        const data = await uploadOne(queue[i].file, (p) => {
          setProgress(p);
          setQueue((q) => {
            const c = [...q];
            c[i] = { ...c[i], pct: p };
            return c;
          });
        });
        setResult(data);
        setImgCacheBust(Date.now());
        setQueue((q) => {
          const c = [...q];
          c[i] = { ...c[i], status: "done", res: data, pct: 100 };
          return c;
        });
        showToast(`Analysis complete: ${queue[i].name}`);
        burstConfetti();
        await pushHistory({
          name: queue[i].name,
          predicted: data.predicted_class,
          confidence: Math.round((data.confidence ?? 0) * 100),
          gradcam: resolveImgUrl(data.gradcam_image),
          originalPreview: queue[i].preview,
        });
      } catch (err) {
        setQueue((q) => {
          const c = [...q];
          c[i] = { ...c[i], status: "error", err: String(err), pct: 0 };
          return c;
        });
        setError(`Failed: ${queue[i].name} — ${String(err)}`);
      }
    }
    setUploading(false);
    isProcessingBatch.current = false;
  };

  const handleDownloadPDF = async () => {
    if (!reportRef.current) return;
    const canvas = await html2canvas(reportRef.current, { scale: 2, useCORS: true });
    const imgData = canvas.toDataURL("image/png");
    const pdf = new jsPDF("p", "mm", "a4");
    const imgWidth = 190;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    pdf.setFontSize(18);
    pdf.text("Fault Detection Report", 15, 20);
    pdf.addImage(imgData, "PNG", 10, 30, imgWidth, imgHeight);
    pdf.save("fault_detection_report.pdf");
  };

  const step1 = true;
  const step2 = uploading || !!result;
  const step3 = !!result;

  const gradUrl = useMemo(
    () => (result?.gradcam_image ? `${resolveImgUrl(result.gradcam_image)}?t=${imgCacheBust}` : null),
    [result, imgCacheBust]
  );

  return (
    <div className="fd-page">
      {confettiOn && <canvas className="fd-confetti" ref={confettiCanvasRef} />}
      <div className="fd-container">
        <div className="fd-header">
          <h1 className="fd-title">🔍 Fault Detection</h1>
          <p className="fd-subtitle">
            Upload a machine image → run detection → review the Grad-CAM heatmap and export a PDF report.
          </p>
        </div>

        <div className="fd-steps">
          <div className={`fd-step ${step1 ? "is-active" : ""}`}>
            <div className="fd-step-title">1 • Upload</div>
            <div className="fd-step-desc">Choose or drag files</div>
          </div>
          <div className={`fd-step ${step2 ? "is-active" : ""}`}>
            <div className="fd-step-title">2 • Analyze</div>
            <div className="fd-step-desc">Send to AI model</div>
          </div>
          <div className={`fd-step ${step3 ? "is-active" : ""}`}>
            <div className="fd-step-title">3 • Review & Export</div>
            <div className="fd-step-desc">Compare, blend & PDF</div>
          </div>
        </div>

        {error && <div className="fd-error">{error}</div>}

        <div className="fd-grid2">
          <div className="fd-card">
            <div
              className={`fd-dropzone ${dragActive ? "is-active" : ""}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              role="button"
              tabIndex={0}
            >
              <div className="fd-uploadIcon">
                <FiUploadCloud size={42} color="#06131d" />
              </div>
              <div className="fd-droptext">
                Drag & drop one or more images
                <br />
                <button className="fd-browse" onClick={() => inputRef.current?.click()}>
                  or browse files
                </button>
              </div>
              <div className="fd-hint">PNG/JPG • Max 10MB each</div>
              <input
                ref={inputRef}
                id="fileInput"
                type="file"
                accept="image/*"
                multiple
                className="fd-hidden"
                onChange={handleFileChange}
              />

              {queue.length > 0 && (
                <div className="fd-queue">
                  {queue.map((it, idx) => (
                    <div key={it.id} className={`fd-queue-item is-${it.status}`}>
                      <span className="fd-queue-name">
                        {idx + 1}. {it.name}
                      </span>
                      <div className="fd-queue-bar">
                        <div style={{ width: `${it.pct || 0}%` }} />
                      </div>
                      <span className="fd-queue-status">
                        {it.status === "queued" && "Queued"}
                        {it.status === "running" && `${it.pct}%`}
                        {it.status === "done" && (<><FiCheckCircle /> Done</>)}
                        {it.status === "error" && "Error"}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {uploading && (
                <>
                  <div className="fd-progress">
                    <div className="fd-progress-fill" style={{ width: `${progress}%` }} />
                  </div>
                  <div className="fd-progress-meta">
                    <span>Uploading…</span>
                    <span>{progress}%</span>
                  </div>
                </>
              )}
            </div>

            <div className="fd-buttons">
              <button
                className="fd-btn fd-btn-primary"
                onClick={handleUpload}
                disabled={uploading || (!file && queue.length === 0)}
              >
                🚀 Upload & Detect {queue.length > 1 ? `(${queue.length})` : ""}
              </button>
              <button className="fd-btn fd-btn-ghost" onClick={resetAll}>
                <FiRefreshCcw /> Reset
              </button>
              {result && (
                <button className="fd-btn fd-btn-export" onClick={handleDownloadPDF}>
                  <FiDownload /> Export PDF
                </button>
              )}
            </div>

            {preview && (
              <div className="fd-preview">
                <div className="fd-sectionTitle">
                  <FiImage /> Preview
                </div>
                <img src={preview} alt="Selected" className="fd-preview-img" />
              </div>
            )}
          </div>

          <div ref={reportRef} className="fd-card fd-resultcard">
            <div className="fd-result-header">
              <h2 className="fd-sectionTitle">📊 Detection Results</h2>
              {!result && <span className="fd-muted">(will appear after analysis)</span>}
            </div>

            {!result && (
              <div className="fd-placeholder">
                Tip: center the subject for clearer Grad-CAM highlights. After analysis, prediction, confidence and heatmap will appear here.
              </div>
            )}

            {result && (
              <>
                <div className="fd-stats">
                  <div className="fd-stat">
                    <div className="fd-stat-label">Prediction</div>
                    <div className="fd-pill">{result.predicted_class}</div>
                  </div>
                  <div className="fd-stat">
                    <div className="fd-stat-label">Confidence</div>
                    <div className="fd-confbar">
                      <div
                        className="fd-confbar-fill"
                        style={{ width: `${result.confidence ? Math.round(result.confidence * 100) : 0}%` }}
                      />
                    </div>
                    <div className="fd-confval">
                      {result.confidence ? `${Math.round(result.confidence * 100)}%` : "N/A"}
                    </div>
                  </div>
                </div>

                <div className="fd-toolbar">
                  <div className="fd-seg">
                    <button className={`fd-seg-btn ${viewMode === "original" ? "is-active" : ""}`} onClick={() => setViewMode("original")}>Original</button>
                    <button className={`fd-seg-btn ${viewMode === "heatmap" ? "is-active" : ""}`} onClick={() => setViewMode("heatmap")}>Heatmap</button>
                    <button className={`fd-seg-btn ${viewMode === "overlay" ? "is-active" : ""}`} onClick={() => setViewMode("overlay")}>Overlay</button>
                    <button className={`fd-seg-btn ${viewMode === "side" ? "is-active" : ""}`} onClick={() => setViewMode("side")}>Side-by-side</button>
                  </div>

                  {viewMode === "overlay" && (
                    <div className="fd-blend">
                      <label>Blend</label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={blend}
                        onChange={(e) => setBlend(parseFloat(e.target.value))}
                      />
                    </div>
                  )}

                  <button className="fd-btn fd-btn-mini" onClick={() => setIsModalOpen(true)} disabled={!gradUrl}>
                    <FiMaximize2 /> View fullscreen
                  </button>
                </div>

                <div className="fd-view">
                  {viewMode === "original" && preview && (
                    <img src={preview} alt="original" className="fd-gradcam-img" />
                  )}

                  {viewMode === "heatmap" && gradUrl && (
                    <img src={gradUrl} alt="GradCAM heatmap" className="fd-gradcam-img" crossOrigin="anonymous" />
                  )}

                  {viewMode === "overlay" && gradUrl && (
                    <div className="fd-stack">
                      {preview && <img src={preview} alt="original" className="fd-stack-img" crossOrigin="anonymous" />}
                      <img src={gradUrl} alt="GradCAM heatmap" className="fd-stack-img" style={{ opacity: blend }} crossOrigin="anonymous" />
                    </div>
                  )}

                  {viewMode === "side" && (
                    <div className="fd-side">
                      {preview && <img src={preview} alt="original" className="fd-side-img" />}
                      {gradUrl && <img src={gradUrl} alt="GradCAM heatmap" className="fd-side-img" crossOrigin="anonymous" />}
                    </div>
                  )}
                </div>

                <div className="fd-legend">
                  <div className="fd-legend-bar" />
                  <span>Low → High importance</span>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="fd-history">
          <div className="fd-history-head">
            <h3 className="fd-sectionTitle">🗂️ History</h3>
            <span className="fd-muted">{history.length ? `${history.length} item(s)` : "Empty"}</span>
          </div>
          {!!history.length && (
            <div className="fd-history-grid">
              {history.map((h) => (
                <div key={h.id} className="fd-history-card">
                  <div className="fd-history-thumb">
                    <img src={h.gradcam || h.originalPreview} alt={h.name} crossOrigin="anonymous" />
                  </div>
                  <div className="fd-history-meta">
                    <div className="fd-history-name" title={h.name}>{h.name}</div>
                    <div className="fd-history-sub">
                      <span className="fd-pill-sm">{h.predicted}</span>
                      <span className="fd-dot" />
                      <span className="fd-muted"><FiClock /> {fmtTime(h.createdAt)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {isModalOpen && gradUrl && (
        <div className="fd-modal" onClick={() => setIsModalOpen(false)}>
          <img src={gradUrl} alt="GradCAM fullscreen" className="fd-modal-img" crossOrigin="anonymous" />
        </div>
      )}

      {toast && <div className="fd-toast">{toast}</div>}
    </div>
  );
}

export default FaultDetectionPage;
