import { afterEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";

import PredictionForm from "../src/components/PredictionForm";
import { predictFromImage } from "../src/services/predictionApi";

vi.mock("../src/services/predictionApi", () => ({
  MAX_UPLOAD_BYTES: 10 * 1024 * 1024,
  predictFromImage: vi.fn(),
}));

const mockPrediction = {
  regression_prediction: 9.75,
  classification_probability: 0.35,
  classification_prediction: 0,
  research_only: true,
  message: "Research prototype only.",
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PredictionForm", () => {

  it("disables image selection while a prediction is running", async () => {
  vi.mocked(predictFromImage).mockImplementation(
    () => new Promise(() => {}),
  );

  render(<PredictionForm />);

  const input = screen.getByLabelText(
    /Choose an image/i,
  ) as HTMLInputElement;

  const file = new File(
    ["image bytes"],
    "skin.jpg",
    { type: "image/jpeg" },
  );

  fireEvent.change(input, {
    target: { files: [file] },
  });

  fireEvent.click(
    screen.getByRole("button", {
      name: "Run research prediction",
    }),
  );

  await waitFor(() => {
    expect(input).toBeDisabled();
  });

  expect(
    screen.getByRole("button", {
      name: "Analyzing image…",
    }),
  ).toBeDisabled();
});

  it("disables submission until a valid image is selected", () => {
    render(<PredictionForm />);

    expect(
      screen.getByRole("button", {
        name: "Run research prediction",
      }),
    ).toBeDisabled();
  });

  it("rejects an unsupported file type", () => {
    render(<PredictionForm />);

    const file = new File(["document"], "notes.txt", {
      type: "text/plain",
    });

    fireEvent.change(screen.getByLabelText(/Choose an image/i), {
      target: { files: [file] },
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Choose a JPEG, PNG, or WebP image.",
    );
  });

  it("submits a valid image and displays the returned model outputs", async () => {
    vi.mocked(predictFromImage).mockResolvedValue(mockPrediction);

    render(<PredictionForm />);

    const file = new File(["image bytes"], "skin.jpg", {
      type: "image/jpeg",
    });

    fireEvent.change(screen.getByLabelText(/Choose an image/i), {
      target: { files: [file] },
    });

    expect(
      screen.getByRole("button", {
        name: "Run research prediction",
      }),
    ).toBeEnabled();

    fireEvent.click(
      screen.getByRole("button", {
        name: "Run research prediction",
      }),
    );

    expect(
      await screen.findByRole("heading", { name: "Model output" }),
    ).toBeInTheDocument();

    expect(screen.getByText("9.75")).toBeInTheDocument();
    expect(screen.getByText(/Returned probability:\s*35\.0\s*%/)).toBeInTheDocument();
    expect(predictFromImage).toHaveBeenCalledWith(
      file,
      expect.any(AbortSignal),
    );
  });

  it("displays an API error when prediction fails", async () => {
    vi.mocked(predictFromImage).mockRejectedValue(
      new Error("Prediction service unavailable."),
    );

    render(<PredictionForm />);

    const file = new File(["image bytes"], "skin.jpg", {
      type: "image/jpeg",
    });

    fireEvent.change(screen.getByLabelText(/Choose an image/i), {
      target: { files: [file] },
    });

    fireEvent.click(
      screen.getByRole("button", {
        name: "Run research prediction",
      }),
    );

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Prediction service unavailable.",
      );
    });
  });
});
