import { useEffect, useState } from "react";
import PredictionForm from "./components/PredictionForm";

type ThemePreference = "system" | "light" | "dark";
type Theme = "light" | "dark";
type ServiceStatus = "checking" | "online" | "offline";

const THEME_STORAGE_KEY = "neojaundice-theme";

function getInitialThemePreference(): ThemePreference {
  try {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);

    if (
      savedTheme === "light" ||
      savedTheme === "dark"
    ) {
      return savedTheme;
    }
  } catch {
    // Continue with the operating system preference.
  }

  return "system";
}

function getSystemTheme(): Theme {
  if (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  ) {
    return "dark";
  }

  return "light";
}

function useTheme() {
  const [preference, setPreference] =
    useState<ThemePreference>(getInitialThemePreference);

  const [systemTheme, setSystemTheme] =
    useState<Theme>(getSystemTheme);

  const theme: Theme =
    preference === "system" ? systemTheme : preference;

 useEffect(() => {
  if (typeof window.matchMedia !== "function") {
    return;
  }

  const mediaQuery = window.matchMedia(
    "(prefers-color-scheme: dark)",
  );

  const updateSystemTheme = (event: MediaQueryListEvent) => {
    setSystemTheme(event.matches ? "dark" : "light");
  };

  mediaQuery.addEventListener("change", updateSystemTheme);

  return () => {
    mediaQuery.removeEventListener("change", updateSystemTheme);
  };
}, []);

  useEffect(() => {
    document.documentElement.classList.toggle(
      "dark",
      theme === "dark",
    );

    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  function toggleTheme() {
    const nextTheme: Theme =
      theme === "dark" ? "light" : "dark";

    setPreference(nextTheme);

    try {
      localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch {
      // The theme still works if storage is unavailable.
    }
  }

  return { theme, toggleTheme };
}

function useBackendStatus(): ServiceStatus {
  const [status, setStatus] =
    useState<ServiceStatus>("checking");

  useEffect(() => {
    let active = true;
    let controller: AbortController | undefined;

    async function checkBackend() {
      controller?.abort();
      controller = new AbortController();

      try {
        const response = await fetch("/api/v1/health", {
          method: "GET",
          cache: "no-store",
          signal: controller.signal,
        });

        if (active) {
          setStatus(response.ok ? "online" : "offline");
        }
      } catch {
        if (active && !controller.signal.aborted) {
          setStatus("offline");
        }
      }
    }

    void checkBackend();

    const intervalId = window.setInterval(() => {
      void checkBackend();
    }, 30_000);

    return () => {
      active = false;
      window.clearInterval(intervalId);
      controller?.abort();
    };
  }, []);

  return status;
}

function ServiceIndicator({
  status,
}: {
  status: ServiceStatus;
}) {
  const labels: Record<ServiceStatus, string> = {
    checking: "Checking API",
    online: "API reachable",
    offline: "API unavailable",
  };

  const indicatorStyles: Record<ServiceStatus, string> = {
    checking: "bg-amber-500",
    online: "bg-emerald-500",
    offline: "bg-red-500",
  };

  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
      role="status"
      aria-live="polite"
    >
      <span
        aria-hidden="true"
        className={`h-2 w-2 rounded-full ${indicatorStyles[status]}`}
      />
      {labels[status]}
    </div>
  );
}

const workflowSteps = [
  {
    number: "01",
    title: "Select an image",
    description:
      "Choose a supported smartphone image from your device.",
  },
  {
    number: "02",
    title: "Run the model",
    description:
      "Submit the image to the configured research API.",
  },
  {
    number: "03",
    title: "Review the output",
    description:
      "Inspect the model's outputs and understand their limitations.",
  },
];

const researchFeatures = [
  {
    icon: "01",
    title: "Image-based analysis",
    description:
      "Explore computer-vision approaches to smartphone skin-image analysis.",
  },
  {
    icon: "02",
    title: "Model evaluation",
    description:
      "Support experimentation with regression and classification outputs.",
  },
  {
    icon: "03",
    title: "Responsible research",
    description:
      "Keep research limitations visible and separate model output from clinical decisions.",
  },
];

export default function App() {
  const { theme, toggleTheme } = useTheme();
  const backendStatus = useBackendStatus();

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 transition-colors duration-200 dark:bg-slate-950 dark:text-slate-100">
      <a
        href="#main-content"
        className="sr-only z-50 rounded-lg bg-white p-3 text-cyan-800 focus:not-sr-only focus:fixed focus:left-4 focus:top-4"
      >
        Skip to main content
      </a>

      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-950/90">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6 lg:px-8">
          <a
            href="#home"
            className="rounded-lg"
            aria-label="NeoJaundice home"
          >
            <span className="text-2xl font-extrabold tracking-tight">
              <span className="text-cyan-600 dark:text-cyan-400">
                Neo
              </span>
              Jaundice
            </span>

            <span className="mt-1 block text-xs text-slate-500 dark:text-slate-400">
              Neonatal jaundice research platform
            </span>
          </a>

          <div className="flex flex-wrap items-center gap-3">
            <ServiceIndicator status={backendStatus} />

            <button
              type="button"
              onClick={toggleTheme}
              className="inline-flex min-h-10 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 focus-visible:outline-cyan-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              title={`Current theme: ${theme}. Click to switch.`}
            >
              <span aria-hidden="true">
                {theme === "dark" ? "☀️" : "🌙"}
              </span>

              <span>
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </span>
            </button>
          </div>
        </div>

        <nav
          aria-label="Main navigation"
          className="border-t border-slate-100 dark:border-slate-800"
        >
          <div className="mx-auto flex max-w-7xl gap-6 overflow-x-auto px-4 py-3 text-sm font-medium sm:px-6 lg:px-8">
            <a
              href="#home"
              className="whitespace-nowrap text-cyan-700 hover:text-cyan-900 dark:text-cyan-400 dark:hover:text-cyan-300"
            >
              Overview
            </a>

            <a
              href="#analysis"
              className="whitespace-nowrap text-slate-600 hover:text-cyan-700 dark:text-slate-300 dark:hover:text-cyan-400"
            >
              Image analysis
            </a>

            <a
              href="#workflow"
              className="whitespace-nowrap text-slate-600 hover:text-cyan-700 dark:text-slate-300 dark:hover:text-cyan-400"
            >
              Workflow
            </a>

            <a
              href="#research"
              className="whitespace-nowrap text-slate-600 hover:text-cyan-700 dark:text-slate-300 dark:hover:text-cyan-400"
            >
              Research
            </a>
          </div>
        </nav>
      </header>

      <main
        id="main-content"
        className="mx-auto max-w-7xl space-y-10 px-4 py-8 sm:px-6 sm:py-10 lg:px-8"
      >
        <section id="home" className="scroll-mt-40">
          <div className="overflow-hidden rounded-3xl border border-cyan-100 bg-gradient-to-br from-cyan-50 via-white to-sky-50 p-6 sm:p-10 dark:border-slate-800 dark:from-slate-900 dark:via-slate-900 dark:to-cyan-950/30">
            <div className="max-w-3xl">
              <span className="inline-flex items-center gap-2 rounded-full border border-cyan-200 bg-white/80 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-cyan-800 dark:border-cyan-900 dark:bg-slate-900/80 dark:text-cyan-300">
                <span aria-hidden="true">✦</span>
                Computer vision research
              </span>

              <h1 className="mt-6 text-3xl font-extrabold leading-tight tracking-tight sm:text-5xl">
                Exploring bilirubin estimation through{" "}
                <span className="text-cyan-700 dark:text-cyan-400">
                  smartphone skin images.
                </span>
              </h1>

              <p className="mt-5 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-300 sm:text-lg">
                NeoJaundice is a research prototype for investigating
                computer-vision models that estimate bilirubin-related
                outputs from images. This dashboard supports research
                experimentation, not clinical diagnosis.
              </p>

              <div className="mt-7 flex flex-wrap gap-3">
                <a
                  href="#analysis"
                  className="inline-flex min-h-11 items-center justify-center rounded-xl bg-cyan-700 px-5 py-3 text-sm font-bold text-white shadow-sm transition hover:bg-cyan-800 dark:bg-cyan-600 dark:hover:bg-cyan-500"
                >
                  Open image analysis
                  <span className="ml-2" aria-hidden="true">
                    →
                  </span>
                </a>

                <a
                  href="#workflow"
                  className="inline-flex min-h-11 items-center justify-center rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-bold text-slate-700 transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800"
                >
                  Explore workflow
                </a>
              </div>
            </div>
          </div>
        </section>

        <section
          id="analysis"
          className="scroll-mt-40"
          aria-labelledby="analysis-heading"
        >
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-sm font-bold uppercase tracking-widest text-cyan-700 dark:text-cyan-400">
                Workspace
              </p>

              <h2
                id="analysis-heading"
                className="mt-2 text-2xl font-extrabold tracking-tight sm:text-3xl"
              >
                Image analysis
              </h2>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600 dark:text-slate-400">
                Upload an image to explore the outputs returned by the
                configured research model.
              </p>
            </div>

            <span className="rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-bold text-amber-900 dark:border-amber-900 dark:bg-amber-950/50 dark:text-amber-200">
              Research prototype
            </span>
          </div>

          <PredictionForm />
        </section>

        <section
          id="workflow"
          className="scroll-mt-40"
          aria-labelledby="workflow-heading"
        >
          <div className="mb-5">
            <p className="text-sm font-bold uppercase tracking-widest text-cyan-700 dark:text-cyan-400">
              Getting started
            </p>

            <h2
              id="workflow-heading"
              className="mt-2 text-2xl font-extrabold tracking-tight sm:text-3xl"
            >
              How the workflow works
            </h2>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {workflowSteps.map((step) => (
              <article
                key={step.number}
                className="rounded-2xl border border-slate-200 bg-white p-5 transition hover:-translate-y-0.5 hover:shadow-md dark:border-slate-800 dark:bg-slate-900"
              >
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-50 text-sm font-extrabold text-cyan-800 dark:bg-cyan-950 dark:text-cyan-300">
                  {step.number}
                </span>

                <h3 className="mt-4 font-bold">{step.title}</h3>

                <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-400">
                  {step.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section
          id="research"
          className="scroll-mt-40"
          aria-labelledby="research-heading"
        >
          <div className="mb-5">
            <p className="text-sm font-bold uppercase tracking-widest text-cyan-700 dark:text-cyan-400">
              Platform capabilities
            </p>

            <h2
              id="research-heading"
              className="mt-2 text-2xl font-extrabold tracking-tight sm:text-3xl"
            >
              Built for responsible experimentation
            </h2>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            {researchFeatures.map((feature) => (
              <article
                key={feature.icon}
                className="rounded-2xl border border-slate-200 bg-white p-5 dark:border-slate-800 dark:bg-slate-900"
              >
                <p className="text-sm font-extrabold text-cyan-700 dark:text-cyan-400">
                  {feature.icon}
                </p>

                <h3 className="mt-3 font-bold">{feature.title}</h3>

                <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-400">
                  {feature.description}
                </p>
              </article>
            ))}
          </div>
        </section>

        <section
          aria-labelledby="safety-heading"
          className="rounded-2xl border border-amber-200 bg-amber-50 p-5 sm:p-6 dark:border-amber-900 dark:bg-amber-950/30"
        >
          <div className="flex items-start gap-3">
            <span className="text-xl" aria-hidden="true">
              ⚠️
            </span>

            <div>
              <h2
                id="safety-heading"
                className="font-extrabold text-amber-950 dark:text-amber-200"
              >
                Important safety notice
              </h2>

              <p className="mt-2 text-sm leading-6 text-amber-900 dark:text-amber-100">
                NeoJaundice is a research and educational prototype. Its
                outputs have not been established as a safe substitute for
                laboratory bilirubin measurements or professional medical
                assessment. Do not use predictions to make treatment
                decisions or delay medical care for a newborn.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-950">
        <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-xs leading-5 text-slate-500 dark:text-slate-400 sm:px-6 md:flex-row md:items-center md:justify-between lg:px-8">
          <p>
            <span className="font-bold text-slate-700 dark:text-slate-200">
              NeoJaundice
            </span>{" "}
            · Research and educational use only
          </p>

          <p>
            No clinical decision-making · API status checks every 30 seconds
          </p>
        </div>
      </footer>
    </div>
  );
}
