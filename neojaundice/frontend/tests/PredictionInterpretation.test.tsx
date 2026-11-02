import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import PredictionForm from "../src/components/PredictionForm";
import { predictFromImage } from "../src/services/predictionApi";
import type { PredictionResponse } from "../src/types/prediction";

vi.mock("../src/services/predictionApi", () => ({
  MAX_UPLOAD_BYTES: 10 * 1024 * 1024,
  predictFromImage: vi.fn(),
}));

const mockPrediction = {
  regression_prediction: 10.25,
  classification_prediction: 1,
  classification_probability: 0.87,
  message: "Research model response.",
} as PredictionResponse;

beforeEach(() => {
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    value: vi.fn(() => "blob:mock-image"),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    value: vi.fn(),
  });
  vi.mocked(predictFromImage).mockResolvedValue(mockPrediction);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PredictionForm result interpretation", () => {
  it("shows the estimate, meaning, and clinician-safety guidance after prediction", async () => {
    const { container } = render(<PredictionForm />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["test image"], "newborn.png", { type: "image/png" });

    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /run research prediction/i }));

    await waitFor(() => expect(screen.getAllByText(/10\.25/).length).toBeGreaterThan(0));
    expect(screen.getByText(/What does bilirubin mean\?/i)).toBeTruthy();
    expect(screen.getByText(/cannot safely answer “OK” or “no doctor needed”/i)).toBeTruthy();
  });

  it("uses explicit dark-mode foreground colors on prediction scores", async () => {
    const { container } = render(<PredictionForm />);
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    const file = new File(["test image"], "newborn.png", { type: "image/png" });

    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByRole("button", { name: /run research prediction/i }));

    await waitFor(() => expect(screen.getAllByText(/10\.25/).length).toBeGreaterThan(0));
    const score = screen.getAllByText(/10\.25/)[0];
    expect(score.className).toContain("dark:text-white");
  });
});
