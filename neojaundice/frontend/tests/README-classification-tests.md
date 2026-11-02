# Classification UI test notes

`PredictionClassification.test.tsx` checks that class 1 and class 0 are rendered as readable provisional labels and that the unverified `treatment_label` caveat is visible.

Existing `PredictionForm.test.tsx` already uses `getByLabelText(/Choose an image/i)`. The file input now has that `aria-label`, fixing the current selector failure.

If the existing form test asserts the raw text `0` or `1`, update that assertion to expect:
- `Jaundice signal indicated (provisional)` for class 1; or
- `No jaundice signal indicated (provisional)` for class 0.

Do not remove assertions for regression output, API error behavior, disabled controls, or research-only messaging.
