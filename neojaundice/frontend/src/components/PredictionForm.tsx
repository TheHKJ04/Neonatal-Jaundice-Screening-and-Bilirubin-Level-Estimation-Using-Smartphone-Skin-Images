import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
} from "react";
import {
  MAX_UPLOAD_BYTES,
  predictFromImage,
} from "../services/predictionApi";
import type { PredictionResponse } from "../types/prediction";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"] as const;


/**
 * UI-only wording for the binary classifier output.
 *
 * IMPORTANT: the current training target is `treatment_label`, whose clinical
 * semantics have not yet been verified. These labels are provisional display
 * text only; they are not a validated clinical diagnosis.
 */
function getClassificationLabel(value: number | string): string {
  const label = Number(value);

  if (label === 1) return "Jaundice signal indicated (provisional)";
  if (label === 0) return "No jaundice signal indicated (provisional)";
  return `Unrecognized model class (${String(value)})`;
}

function getClassificationReason(value: number | string): string {
  const label = Number(value);

  if (label === 1) {
    return (
      "The classifier returned class 1. For readability, this interface provisionally " +
      "shows it as a jaundice signal. The results are not " +
      "clinically verified, so this must not be treated as a diagnosis."
    );
  }

  if (label === 0) {
    return (
      "The classifier returned class 0. For readability, this interface provisionally " +
      "shows it as no jaundice signal indicated.This output does not rule out jaundice."
    );
  }

  return (
    "The returned class is not one of the expected binary values (0 or 1). " +
    "Do not interpret this output clinically; verify the backend response and label mapping."
  );
}

type RecentAnalysis = {
  id: string;
  estimate: number;
  classification: string;
  probability: number;
  createdAt: string;
};

function formatFileSize(bytes: number): string {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(2)} MB`
    : `${(bytes / 1024).toFixed(1)} KB`;
}

export default function PredictionForm() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [recentAnalyses, setRecentAnalyses] = useState<RecentAnalysis[]>([]);

  const inputRef = useRef<HTMLInputElement>(null);
  const previewUrlRef = useRef<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      controllerRef.current?.abort();
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  function releasePreview() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  }

  function selectFile(selected: File | null) {
    releasePreview();
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);

    if (!selected) return;

    if (!(ACCEPTED_TYPES as readonly string[]).includes(selected.type)) {
      setError("Choose a JPEG, PNG, or WebP image.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    if (selected.size === 0) {
      setError("The selected file is empty. Choose another image.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    if (selected.size > MAX_UPLOAD_BYTES) {
      setError("The image must be 10 MB or smaller.");
      if (inputRef.current) inputRef.current.value = "";
      return;
    }

    const objectUrl = URL.createObjectURL(selected);
    previewUrlRef.current = objectUrl;
    setFile(selected);
    setPreviewUrl(objectUrl);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] ?? null);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    if (loading) return;
    selectFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || loading) return;

    const controller = new AbortController();
    controllerRef.current = controller;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const prediction = await predictFromImage(file, controller.signal);
      setResult(prediction);
      setRecentAnalyses((previous) => [
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          estimate: prediction.regression_prediction,
          classification: getClassificationLabel(prediction.classification_prediction),
          probability: prediction.classification_probability,
          createdAt: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        },
        ...previous,
      ].slice(0, 5));
    } catch (err) {
      // Ignore stale responses after cancellation or if a newer request has started.
      if (controllerRef.current !== controller) return;

      if (err instanceof Error && err.name === "AbortError") {
        setError("Analysis cancelled. You can choose an image and try again.");
      } else {
        setError(
          err instanceof Error
            ? err.message
            : "An unexpected error occurred. Please try again.",
        );
      }
    } finally {
      // Do not let an older, cancelled request change the state of a newer request.
      if (controllerRef.current === controller) {
        controllerRef.current = null;
        setLoading(false);
      }
    }
  }

  function cancelAnalysis() {
    const controller = controllerRef.current;
    if (!controller) return;

    controllerRef.current = null;
    controller.abort();
    setLoading(false);
    setError("Analysis cancelled. You can choose an image and try again.");
  }

  function resetForm() {
    controllerRef.current?.abort();
    controllerRef.current = null;
    releasePreview();
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setLoading(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-700 dark:bg-slate-900 sm:p-7">
      <div className="mb-6">
        <p className="text-sm font-semibold uppercase tracking-[0.18em] text-cyan-700 dark:text-cyan-300">
          Image analysis
        </p>
        <h2 className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
          Upload a skin image
        </h2>
        <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">
          Choose a supported image or drag it into the upload area. This research
          tool returns model outputs, not a medical diagnosis.
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div
          onDragOver={(event) => {
            event.preventDefault();
            if (!loading) setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`rounded-2xl border-2 border-dashed px-6 py-8 text-center transition-colors sm:py-10 ${
            isDragging
              ? "border-cyan-500 bg-cyan-50 dark:bg-cyan-950/40"
              : "border-slate-300 bg-slate-50 hover:border-cyan-500 hover:bg-cyan-50 dark:border-slate-600 dark:bg-slate-800 dark:hover:bg-slate-700"
          }`}
        >
          <p className="text-4xl" aria-hidden="true">↑</p>
          <p className="mt-3 font-semibold text-slate-900 dark:text-white">
            Drag and drop an image here
          </p>
          <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
            JPEG, PNG, or WebP · Maximum 10 MB
          </p>
          <button
            type="button"
            disabled={loading}
            onClick={() => inputRef.current?.click()}
            className="mt-4 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 hover:bg-slate-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-700"
          >
            Browse files
          </button>
          <input
            ref={inputRef}
            id="skin-image"
            name="skin-image"
            aria-label="Choose an image"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            disabled={loading}
            onChange={handleFileChange}
            className="sr-only"
          />
        </div>

        {file && (
          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="break-all text-sm font-semibold text-slate-900 dark:text-white">
                  {file.name}
                </p>
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">
                  {formatFileSize(file.size)}
                </p>
              </div>
              {!loading && (
                <button
                  type="button"
                  onClick={resetForm}
                  className="text-sm font-semibold text-slate-600 underline hover:text-slate-900 dark:text-slate-300 dark:hover:text-white"
                >
                  Remove image
                </button>
              )}
            </div>
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Preview of the selected skin image"
                className="mt-4 max-h-72 w-full rounded-xl bg-white object-contain dark:bg-slate-950"
              />
            )}
          </div>
        )}

        {error && (
          <div
            role="alert"
            className="rounded-xl border border-red-300 bg-red-50 p-4 text-sm leading-6 text-red-900 dark:border-red-800 dark:bg-red-950/50 dark:text-red-200"
          >
            {error}
          </div>
        )}

        <div className="flex flex-col gap-3 sm:flex-row">
          <button
            type="submit"
            disabled={!file || loading}
            className="flex-1 rounded-xl bg-cyan-700 px-5 py-3 font-semibold text-white transition hover:bg-cyan-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-cyan-600 dark:hover:bg-cyan-500"
          >
            {loading ? "Analyzing image…" : "Run research prediction"}
          </button>
          {loading && (
            <button
              type="button"
              onClick={cancelAnalysis}
              className="rounded-xl border border-slate-300 px-5 py-3 font-semibold text-slate-800 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800"
            >
              Cancel request
            </button>
          )}
        </div>
      </form>

      {result && (
        <section aria-labelledby="results-heading" className="mt-8 border-t border-slate-200 pt-6 dark:border-slate-700">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.16em] text-cyan-700 dark:text-cyan-300">
              Output from the model
            </p>
            <h3 id="results-heading" className="mt-2 text-2xl font-bold text-slate-900 dark:text-white">
              Research prediction results
            </h3>
            <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">
              These numbers describe what the current model returned for this image. They are not a laboratory test result or a clinical risk assessment.
            </p>
          </div>
          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <article className="rounded-2xl border border-cyan-200 bg-cyan-50 p-5 dark:border-cyan-800 dark:bg-slate-800">
              <p className="text-sm font-semibold text-cyan-900 dark:text-cyan-200">
                Model-estimated bilirubin
              </p>
              <p className="mt-3 break-words text-4xl font-extrabold tabular-nums text-slate-950 dark:text-white">
                {Number(result.regression_prediction).toFixed(2)}
                <span className="ml-2 text-base font-semibold text-slate-700 dark:text-slate-200">mg/dL</span>
              </p>
              <p className="mt-3 text-sm leading-6 text-slate-700 dark:text-slate-200">
                The regression model estimates a bilirubin-related value from image features learned during training. It does not directly measure bilirubin in the blood.
              </p>
            </article>

            <article className="rounded-2xl border border-indigo-200 bg-indigo-50 p-5 dark:border-indigo-800 dark:bg-slate-800">
              <p className="text-sm font-semibold text-indigo-900 dark:text-indigo-200">
                Model classification output
              </p>
              <p className="mt-3 break-words text-2xl font-extrabold leading-tight text-slate-950 dark:text-white">
                {getClassificationLabel(result.classification_prediction)}
              </p>
              <p className="mt-3 text-sm leading-6 text-slate-700 dark:text-slate-200">
                {/* <strong className="text-slate-950 dark:text-white">Why this is shown:</strong>{" "} */}
                {getClassificationReason(result.classification_prediction)}
              </p>
              <p className="mt-3 text-sm leading-6 text-slate-700 dark:text-slate-200">
                Classifier-reported probability:{" "}
                <strong className="text-slate-950 dark:text-white">
                  {(Number(result.classification_probability) * 100).toFixed(1)}%
                </strong>
              </p>
              <p className="mt-2 text-xs leading-5 text-slate-600 dark:text-slate-300">
              It must not be interpreted as a baby's clinical probability of having jaundice.
              </p>
            </article>
          </div>

          <div className="mt-5 rounded-2xl border border-amber-300 bg-amber-50 p-5 text-amber-950 dark:border-amber-700 dark:bg-amber-950/50 dark:text-amber-100">
            <h4 className="font-bold">Does this mean the baby is OK or needs a doctor?</h4>
            <p className="mt-2 text-sm leading-6">
              This app cannot safely answer “OK” or “no doctor needed” from the image estimate. If you suspect jaundice or are worried about the baby, contact a paediatrician or neonatal care professional promptly and follow their advice. Jaundice appearing in the first 24 hours needs urgent medical assessment.
            </p>
            <p className="mt-2 text-sm leading-6">
              Seek urgent medical care if the baby is difficult to wake, is not feeding well, has unusual limpness or stiffness, has a high-pitched cry, breathing difficulty, or seems seriously unwell. Do not wait for another app prediction.
            </p>
          </div>

          <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-700 dark:bg-slate-900">
            <h4 className="font-bold text-slate-900 dark:text-white">What does bilirubin mean?</h4>
            <p className="mt-2 text-sm leading-6 text-slate-700 dark:text-slate-300">
              Bilirubin is a yellow substance produced when the body breaks down old red blood cells. Newborns can develop jaundice because bilirubin builds up while their liver is still maturing. A clinician interprets a measured bilirubin result using the baby's age in hours, gestational age, medical condition, and other risk factors.
            </p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800">
                <p className="font-semibold text-slate-900 dark:text-white">What a higher result may suggest</p>
                <p className="mt-1 text-sm leading-6 text-slate-700 dark:text-slate-300">
                  It may indicate a higher estimated bilirubin level, but this model output alone cannot determine severity or whether treatment is needed.
                </p>
              </div>
              <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800">
                <p className="font-semibold text-slate-900 dark:text-white">What a lower result means</p>
                <p className="mt-1 text-sm leading-6 text-slate-700 dark:text-slate-300">
                  It does not rule out jaundice or guarantee that the baby is safe. Image quality, lighting, skin tone, and model limitations can affect the estimate.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-slate-300 bg-slate-100 p-4 text-sm leading-6 text-slate-800 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200">
            <strong>Research use only.</strong> {result.message} Do not use this output to diagnose jaundice, decide whether treatment is needed, delay a clinical assessment, or replace a bilirubin blood test or clinician's advice.
          </div>

          <button
            type="button"
            onClick={resetForm}
            className="mt-5 rounded-xl border border-slate-300 px-4 py-2.5 text-sm font-semibold text-slate-800 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-100 dark:hover:bg-slate-800"
          >
            Analyze another image
          </button>
        </section>
      )}

      {recentAnalyses.length > 0 && (
        <section className="mt-8 border-t border-slate-200 pt-6 dark:border-slate-700" aria-labelledby="recent-heading">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 id="recent-heading" className="font-bold text-slate-900 dark:text-white">Recent results in this session</h3>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">Kept in component memory only; not saved to local storage or a database by this UI.</p>
            </div>
            <button
              type="button"
              onClick={() => setRecentAnalyses([])}
              className="text-sm font-semibold text-slate-600 underline hover:text-slate-900 dark:text-slate-300 dark:hover:text-white"
            >
              Clear history
            </button>
          </div>
          <ul className="mt-3 space-y-2">
            {recentAnalyses.map((analysis) => (
              <li key={analysis.id} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm dark:border-slate-700 dark:bg-slate-900">
                <span className="text-slate-600 dark:text-slate-300">{analysis.createdAt}</span>
                <span className="font-semibold text-slate-900 dark:text-white">Estimate: {analysis.estimate.toFixed(2)} mg/dL</span>
                <span className="text-slate-700 dark:text-slate-200">Class: {analysis.classification}</span>
                <span className="text-slate-700 dark:text-slate-200">Probability: {(analysis.probability * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-xs leading-5 text-slate-500 dark:text-slate-400">Recent model outputs are for technical review only. They do not indicate whether a newborn is clinically safe.</p>
        </section>
      )}
    </section>
  );
}
