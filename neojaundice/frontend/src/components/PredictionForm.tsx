import {
  useEffect,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";
import {
  MAX_UPLOAD_BYTES,
  predictFromImage,
} from "../services/predictionApi";
import type { PredictionResponse } from "../types/prediction";

const ACCEPTED_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
];

export default function PredictionForm() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;

    setFile(null);
    setResult(null);
    setError(null);

    if (!selected) return;

    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      event.target.value = "";
      return;
    }

    if (selected.size === 0) {
      setError("The selected file is empty.");
      event.target.value = "";
      return;
    }

    if (selected.size > MAX_UPLOAD_BYTES) {
      setError("The image must be 10 MB or smaller.");
      event.target.value = "";
      return;
    }

    setFile(selected);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!file || loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();

    try {
      const prediction = await predictFromImage(
        file,
        controller.signal,
      );
      setResult(prediction);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred.",
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setFile(null);
    setResult(null);
    setError(null);
    setLoading(false);
  }

  return (
    <section className="card">
      <div className="mb-6">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-cyan-700">
          Image analysis
        </p>

        <h2 className="mt-2 text-2xl font-bold text-slate-900">
          Upload a skin image
        </h2>

        <p className="mt-2 text-sm leading-6 text-slate-600">
          Select a supported image to request a prediction from
          the research model.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <label
          htmlFor="skin-image"
          className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center transition hover:border-cyan-500 hover:bg-cyan-50"
        >
          <span className="text-4xl" aria-hidden="true">
            ↑
          </span>

          <span className="mt-3 font-semibold text-slate-800">
            Choose an image
          </span>

          <span className="mt-1 text-sm text-slate-500">
            JPEG, PNG, or WebP · Maximum 10 MB
          </span>

          <input
            id="skin-image"
            name="skin-image"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            disabled={loading}
            onChange={handleFileChange}
            className="sr-only"
          />
        </label>

        {file && (
          <div className="rounded-xl border border-slate-200 p-4">
            <p className="break-all text-sm font-medium text-slate-700">
              Selected: {file.name}
            </p>

            <p className="mt-1 text-xs text-slate-500">
              {(file.size / 1024).toFixed(1)} KB
            </p>

            {previewUrl && (
              <img
                src={previewUrl}
                alt="Preview of the selected skin image"
                className="mt-4 max-h-72 w-full rounded-lg object-contain"
              />
            )}
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-800"
          >
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={!file || loading}
          className="w-full rounded-xl bg-slate-900 px-5 py-3 font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Analyzing image…" : "Run research prediction"}
        </button>
      </form>

      {result && (
        <section
          aria-labelledby="results-heading"
          className="mt-8 border-t border-slate-200 pt-6"
        >
          <h3
            id="results-heading"
            className="text-xl font-bold text-slate-900"
          >
            Model output
          </h3>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="rounded-xl bg-cyan-50 p-4">
              <p className="text-sm text-cyan-900">
                Regression output
              </p>
              <p className="mt-2 text-3xl font-bold text-cyan-950">
                {result.regression_prediction.toFixed(2)}
              </p>
              <p className="mt-1 text-xs text-cyan-900">
                Model estimate in mg/dL; not clinically validated.
              </p>
            </div>

            <div className="rounded-xl bg-indigo-50 p-4">
              <p className="text-sm text-indigo-900">
                Classification output
              </p>
              <p className="mt-2 text-3xl font-bold text-indigo-950">
                {result.classification_prediction}
              </p>
              <p className="mt-1 text-sm text-indigo-900">
                Returned probability:{" "}
                {(result.classification_probability * 100).toFixed(1)}%
              </p>
              <p className="mt-2 text-xs text-indigo-900">
                Class meaning and probability calibration must be
                verified before clinical interpretation.
              </p>
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-amber-300 bg-amber-50 p-4 text-sm leading-6 text-amber-950">
            <strong>Research use only.</strong>{" "}
            {result.message} Do not use these outputs to diagnose
            jaundice, delay care, or make treatment decisions.
          </div>

          <button
            type="button"
            onClick={resetForm}
            className="mt-5 rounded-xl border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
          >
            Analyze another image
          </button>
        </section>
      )}
    </section>
  );
}
