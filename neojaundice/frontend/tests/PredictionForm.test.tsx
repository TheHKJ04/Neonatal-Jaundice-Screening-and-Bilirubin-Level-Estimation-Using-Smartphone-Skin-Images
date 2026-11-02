import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

import PredictionForm from "../src/components/PredictionForm";
import { predictFromImage } from "../src/services/predictionApi";
import type { PredictionResponse } from "../src/types/prediction";

vi.mock("../src/services/predictionApi", () => ({
  MAX_UPLOAD_BYTES: 10 * 1024 * 1024,
  predictFromImage: vi.fn(),
}));

const successfulResponse: PredictionResponse = {
  regression_prediction: 10.25,
  classification_prediction: 1,
  classification_probability: 0.82,
  research_only: true,
  message: "Research output only.",
};

function chooseImage(
  type = "image/png",
  name = "newborn.png",
): HTMLInputElement {
  const input = screen.getByLabelText(
    /Choose an image/i,
  ) as HTMLInputElement;

  const file = new File(["image bytes"], name, { type });

  fireEvent.change(input, {
    target: { files: [file] },
  });

  return input;
}

describe("PredictionForm", () => {
  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn(() => "blob:test-image"),
    });

    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      writable: true,
      value: vi.fn(),
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("disables image selection while a prediction is running", async () => {
    let resolvePrediction!: (value: PredictionResponse) => void;

    vi.mocked(predictFromImage).mockReturnValue(
      new Promise<PredictionResponse>((resolve) => {
        resolvePrediction = resolve;
      }),
    );

    render(<PredictionForm />);

    const input = chooseImage();

    fireEvent.click(
      screen.getByRole("button", {
        name: /Run research prediction/i,
      }),
    );

    await waitFor(() => {
      expect(input).toBeDisabled();
    });

    resolvePrediction(successfulResponse);

    await screen.findByRole("heading", {
      name: /Research prediction results/i,
    });
  });

  it("disables submission until a valid image is selected", () => {
    render(<PredictionForm />);

    expect(
      screen.getByRole("button", {
        name: /Run research prediction/i,
      }),
    ).toBeDisabled();
  });

  it("rejects an unsupported file type", async () => {
    render(<PredictionForm />);

    chooseImage("text/plain", "notes.txt");

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Choose a JPEG, PNG, or WebP image/i,
    );

    expect(predictFromImage).not.toHaveBeenCalled();
  });

  it("submits a valid image and displays readable model outputs", async () => {
    vi.mocked(predictFromImage).mockResolvedValue(successfulResponse);

    render(<PredictionForm />);

    chooseImage();

    fireEvent.click(
      screen.getByRole("button", {
        name: /Run research prediction/i,
      }),
    );

    expect(
      await screen.findByText("10.25", {
        selector: "p",
      }),
    ).toBeInTheDocument();

    expect(
      screen.getByText("Jaundice signal indicated (provisional)", {
        selector: "p",
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Research use only/i),
    ).toBeInTheDocument();
  });

  it("displays an API error when prediction fails", async () => {
    vi.mocked(predictFromImage).mockRejectedValue(
      new Error("API unavailable"),
    );

    render(<PredictionForm />);

    chooseImage();

    fireEvent.click(
      screen.getByRole("button", {
        name: /Run research prediction/i,
      }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /API unavailable/i,
    );
  });
});
