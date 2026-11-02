import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import PredictionForm from "../src/components/PredictionForm";
import { predictFromImage } from "../src/services/predictionApi";

vi.mock("../src/services/predictionApi", () => ({
  MAX_UPLOAD_BYTES: 10 * 1024 * 1024,
  predictFromImage: vi.fn(),
}));

const mockPrediction = (classification_prediction: number) => ({
  regression_prediction: 10.25,
  classification_prediction,
  classification_probability:
    classification_prediction === 1 ? 0.82 : 0.18,
  research_only: true,
  message: "Research output only.",
});

function chooseImage() {
  const input = screen.getByLabelText(
    /Choose an image/i,
  ) as HTMLInputElement;

  const file = new File(["image bytes"], "newborn.png", {
    type: "image/png",
  });

  fireEvent.change(input, {
    target: { files: [file] },
  });

  return input;
}

describe("provisional classification wording", () => {
  it("shows a human-readable provisional jaundice signal for class 1", async () => {
    vi.mocked(predictFromImage).mockResolvedValue(mockPrediction(1));

    render(<PredictionForm />);

    chooseImage();

    fireEvent.click(
      screen.getByRole("button", {
        name: /Run research prediction/i,
      }),
    );

    expect(
      await screen.findByText(
        "Jaundice signal indicated (provisional)",
        { selector: "p" },
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(/classifier returned class 1/i),
    ).toBeInTheDocument();
  });

  it("shows a human-readable provisional no-signal label for class 0", async () => {
    vi.mocked(predictFromImage).mockResolvedValue(mockPrediction(0));

    render(<PredictionForm />);

    chooseImage();

    fireEvent.click(
      screen.getByRole("button", {
        name: /Run research prediction/i,
      }),
    );

    expect(
      await screen.findByText(
        "No jaundice signal indicated (provisional)",
        { selector: "p" },
      ),
    ).toBeInTheDocument();

    expect(
      screen.getByText(
        /It does not rule out jaundice or guarantee that the baby is safe/i,
      ),
    ).toBeInTheDocument();
  });
});
