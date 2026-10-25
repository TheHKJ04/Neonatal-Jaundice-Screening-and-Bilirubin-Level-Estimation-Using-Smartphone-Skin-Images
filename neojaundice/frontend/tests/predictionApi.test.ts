import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  predictFromImage,
  PredictionApiError,
} from "../src/services/predictionApi";

import type { PredictionResponse } from "../src/types/prediction";

const validPrediction: PredictionResponse = {
  regression_prediction: 12.4,
  classification_probability: 0.35,
  classification_prediction: 1,
  research_only: true,
  message: "Research use only.",
};

function mockResponse(
  body: unknown,
  status = 200,
): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  } as unknown as Response;
}

function createTestImage(): File {
  return new File(
    ["fake image contents"],
    "newborn.jpg",
    { type: "image/jpeg" },
  );
}

describe("prediction API service", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("uploads the image and returns a valid prediction", async () => {
    vi.mocked(fetch).mockResolvedValue(
      mockResponse(validPrediction),
    );

    const image = createTestImage();
    const result = await predictFromImage(image);

    expect(result).toEqual(validPrediction);
    expect(fetch).toHaveBeenCalledTimes(1);

    const [requestUrl, requestOptions] =
      vi.mocked(fetch).mock.calls[0];

    expect(String(requestUrl)).toContain("/api/v1/predict");
    expect(requestOptions?.method).toBe("POST");
    expect(requestOptions?.body).toBeInstanceOf(FormData);

    const body = requestOptions?.body as FormData;

    expect(body.get("file")).toBe(image);
  });

  it("reports the backend status when the request fails", async () => {
    vi.mocked(fetch).mockResolvedValue(
      mockResponse(
        { detail: "Model is temporarily unavailable." },
        503,
      ),
    );

    const image = createTestImage();

    const error = await predictFromImage(image).catch(
      (caught: unknown) => caught,
    );

    expect(error).toBeInstanceOf(PredictionApiError);
    expect(error).toMatchObject({
      status: 503,
    });
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("rejects unsupported file types without calling the backend", async () => {
    const unsupportedImage = new File(
      ["not an image"],
      "notes.txt",
      { type: "text/plain" },
    );

    await expect(
      predictFromImage(unsupportedImage),
    ).rejects.toBeInstanceOf(PredictionApiError);

    expect(fetch).not.toHaveBeenCalled();
  });
});
