import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import App from "../src/App";

describe("NeoJaundice dashboard", () => {
  beforeEach(() => {
    localStorage.clear();
    localStorage.setItem("neojaundice-theme", "light");

    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
      }),
    );
  });

  afterEach(() => {
    cleanup();
    document.documentElement.classList.remove("dark");
    document.documentElement.style.colorScheme = "";
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    localStorage.clear();
  });

  it("renders the dashboard and research safety notice", () => {
    render(<App />);

    expect(
      screen.getByRole("heading", {
        name: /exploring bilirubin estimation/i,
      }),
    ).toBeTruthy();

    expect(
      screen.getByRole("heading", {
        name: /image analysis/i,
      }),
    ).toBeTruthy();

    expect(
      screen.getByRole("heading", {
        name: /important safety notice/i,
      }),
    ).toBeTruthy();
  });

  it("switches from light mode to dark mode", () => {
    render(<App />);

    const toggle = screen.getByRole("button", {
      name: /switch to dark mode/i,
    });

    fireEvent.click(toggle);

    expect(document.documentElement.classList.contains("dark")).toBe(true);
    expect(document.documentElement.style.colorScheme).toBe("dark");
    expect(localStorage.getItem("neojaundice-theme")).toBe("dark");
  });

  it("switches back to light mode", () => {
    localStorage.setItem("neojaundice-theme", "dark");

    render(<App />);

    const toggle = screen.getByRole("button", {
      name: /switch to light mode/i,
    });

    fireEvent.click(toggle);

    expect(document.documentElement.classList.contains("dark")).toBe(false);
    expect(document.documentElement.style.colorScheme).toBe("light");
    expect(localStorage.getItem("neojaundice-theme")).toBe("light");
  });

  it("checks backend health when the dashboard loads", () => {
    render(<App />);

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/health",
      expect.objectContaining({
        method: "GET",
        cache: "no-store",
      }),
    );
  });
});
