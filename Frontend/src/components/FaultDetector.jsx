import { useState } from "react";

function FaultDetector() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/predict-fault", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="p-4 bg-gray-900 rounded-lg shadow-lg text-white">
      <h2 className="text-xl mb-4 font-bold">📸 Fault Detector</h2>

      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button
        onClick={handleUpload}
        className="mt-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded"
      >
        Upload & Predict
      </button>

      {result && (
        <div className="mt-4">
          <p>
            🔍 <b>Prediction:</b> {result.predicted_class}
          </p>
          <p>
            📊 <b>Confidence:</b> {Math.round(result.confidence * 100)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default FaultDetector;
