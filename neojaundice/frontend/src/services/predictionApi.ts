import type { PredictionResponse } from "../types/prediction";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? ""
).replace(/\/+$/, "");

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

const ALLOWED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

export class PredictionApiError extends Error {
  public readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "PredictionApiError";
    this.status = status;
  }
}

function validateImage(file: File): void {
  if (file.size === 0) {
    throw new PredictionApiError("The selected file is empty.");
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    throw new PredictionApiError(
      "The image must be 10 MB or smaller.",
    );
  }

  if (!ALLOWED_IMAGE_TYPES.has(file.type)) {
    throw new PredictionApiError(
      "Choose a JPEG, PNG, or WebP image.",
    );
  }
}

export async function predictFromImage(
  file: File,
  signal?: AbortSignal,
): Promise<PredictionResponse> {
  validateImage(file);

  const formData = new FormData();
  formData.append("file", file);

  let response: Response;

  try {
    response = await fetch(
      `${API_BASE_URL}/api/v1/predict`,
      {
        method: "POST",
        body: formData,
        signal,
      },
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }

    throw new PredictionApiError(
      "Unable to reach the prediction service. Check that the backend is running.",
    );
  }

  if (!response.ok) {
    let message = `Prediction request failed (HTTP ${response.status}).`;

    try {
      const body: unknown = await response.json();

      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body
      ) {
        const detail = (body as { detail: unknown }).detail;

        if (typeof detail === "string") {
          message = detail;
        } else if (Array.isArray(detail)) {
          message = detail
            .map((item) =>
              typeof item === "object" &&
              item !== null &&
              "msg" in item
                ? String(item.msg)
                : "",
            )
            .filter(Boolean)
            .join("; ") || message;
        }
      }
    } catch {
      // Retain the HTTP error message if the response isn't JSON.
    }

    throw new PredictionApiError(message, response.status);
  }

  let result: PredictionResponse;

  try {
    result = (await response.json()) as PredictionResponse;
  } catch {
    throw new PredictionApiError(
      "The prediction service returned an invalid response.",
      response.status,
    );
  }

  if (
    typeof result.regression_prediction !== "number" ||
    !Number.isFinite(result.regression_prediction) ||
    typeof result.classification_probability !== "number" ||
    !Number.isFinite(result.classification_probability) ||
    result.classification_probability < 0 ||
    result.classification_probability > 1 ||
    typeof result.classification_prediction !== "number" ||
    !Number.isFinite(result.classification_prediction) ||
    typeof result.research_only !== "boolean" ||
    typeof result.message !== "string"
  ) {
    throw new PredictionApiError(
      "The prediction service returned an unexpected response format.",
    );
  }

  return result;
}
