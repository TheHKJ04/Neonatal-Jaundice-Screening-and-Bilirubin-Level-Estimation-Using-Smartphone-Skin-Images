import PredictionForm from "./components/PredictionForm";

export default function App() {
  return (
    <main className="min-h-screen bg-slate-100 text-slate-900">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-5 sm:px-8">
          <div>
            <p className="text-xl font-extrabold tracking-tight">
              <span className="text-cyan-700">Neo</span>Jaundice
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Computer vision research platform
            </p>
          </div>

          <span className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-900">
            Research prototype
          </span>
        </div>
      </header>

      <div className="mx-auto grid max-w-6xl gap-10 px-5 py-10 sm:px-8 lg:grid-cols-[0.9fr_1.1fr] lg:py-16">
        <section className="pt-2">
          <p className="text-sm font-semibold uppercase tracking-[0.2em] text-cyan-700">
            Neonatal jaundice research
          </p>

          <h1 className="mt-4 text-4xl font-extrabold leading-tight tracking-tight sm:text-5xl">
            Exploring skin-image-based bilirubin estimation.
          </h1>

          <p className="mt-5 max-w-xl text-base leading-7 text-slate-600">
            NeoJaundice is a research prototype investigating
            computer vision models for estimating bilirubin-related
            outputs from smartphone skin images.
          </p>

          <div className="mt-8 space-y-4">
            {[
              [
                "01",
                "Select an image",
                "Choose a supported image from your device.",
              ],
              [
                "02",
                "Run the model",
                "Send the image to the local research API.",
              ],
              [
                "03",
                "Review the output",
                "Inspect the model outputs and their limitations.",
              ],
            ].map(([number, title, description]) => (
              <div key={number} className="flex gap-4">
                <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white text-sm font-bold text-cyan-800 shadow-sm">
                  {number}
                </span>

                <div>
                  <h2 className="font-semibold">{title}</h2>
                  <p className="mt-1 text-sm leading-6 text-slate-600">
                    {description}
                  </p>
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 p-5">
            <h2 className="font-bold text-amber-950">
              Important safety notice
            </h2>
            <p className="mt-2 text-sm leading-6 text-amber-900">
              This software has not been established as a safe
              substitute for laboratory bilirubin measurements or
              professional medical assessment. Do not use its
              predictions to guide treatment or delay medical care.
            </p>
          </div>
        </section>

        <div>
          <PredictionForm />
        </div>
      </div>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-5 py-5 text-xs leading-5 text-slate-500 sm:px-8">
          NeoJaundice · Research and educational use only · No
          clinical decision-making
        </div>
      </footer>
    </main>
  );
}
