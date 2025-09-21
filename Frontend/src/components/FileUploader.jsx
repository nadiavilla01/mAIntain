import { useState } from "react";
import { FiUploadCloud } from "react-icons/fi";

function FileUploader({ onUpload }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileSelect = (f) => {
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  };

  const handleFileChange = (e) => {
    handleFileSelect(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    handleFileSelect(e.dataTransfer.files[0]);
  };

  const handleUpload = () => {
    if (file && onUpload) {
      onUpload(file);
    }
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <div
        className={`border-2 border-dashed rounded-2xl p-10 text-center transition relative 
        ${dragActive ? "border-cyan-400 bg-gray-800" : "border-gray-600 bg-gray-900"}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="hidden"
          id="fileInput"
        />

        <label htmlFor="fileInput" className="cursor-pointer">
          <div className="relative flex items-center justify-center w-32 h-32 mx-auto rounded-full 
            bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-400 
            shadow-xl hover:scale-105 transition transform">
            <FiUploadCloud className="text-white" size={48} />
            <span className="absolute inline-flex h-full w-full rounded-full bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-400 opacity-30 animate-ping"></span>
          </div>
          <p className="mt-6 text-gray-300">
            {file ? (
              <span className="text-cyan-400 font-medium">✅ {file.name}</span>
            ) : (
              "Drag & drop or click to upload"
            )}
          </p>
        </label>

        {preview && (
          <div className="mt-6">
            <img
              src={preview}
              alt="Preview"
              className="rounded-lg border border-gray-700 max-h-48 mx-auto"
            />
          </div>
        )}
      </div>

      <button
        onClick={handleUpload}
        disabled={!file}
        className="mt-6 w-full py-3 rounded-lg 
        bg-gradient-to-r from-blue-600 to-purple-600 hover:from-cyan-500 hover:to-blue-700 
        font-semibold text-white shadow-lg transition disabled:opacity-50"
      >
        🚀 Upload & Detect
      </button>
    </div>
  );
}

export default FileUploader;
