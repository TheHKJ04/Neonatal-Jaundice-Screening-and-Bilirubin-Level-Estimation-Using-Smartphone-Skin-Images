# Research protocol and evidence gaps

## Working research questions

1. Can smartphone images support an estimate correlated with measured neonatal bilirubin?
2. How robust are estimates across image lighting, device, skin tone, gestational age, and postnatal age?
3. Can model explanations help identify model failure modes without being mistaken for causal explanations?

## Experimental design requirements

- Define the intended population, image capture protocol, inclusion/exclusion criteria, and reference bilirubin measurement.
- Ensure patient-level train/validation/test separation and document how repeated images are handled.
- Predefine primary metrics, model selection rules, and test-set freeze.
- Compare against simple baselines and report uncertainty intervals.
- Report subgroup performance, missingness, image-quality exclusions, and failure cases.
- Preserve an audit trail of data versions, model hashes, parameters, and code commit.

## Critical limitations to resolve

- `treatment_label` clinical semantics and the binary class mapping are not verified in the available training/evaluation notes.
- Smartphone color varies with illumination, white balance, device, skin tone, camera processing, and region of interest.
- A skin-image estimate is not equivalent to serum bilirubin or a validated transcutaneous bilirubin measurement.
- A classifier probability is not a calibrated clinical risk unless this has been evaluated against appropriate reference labels.
- This prototype has no authority to advise treatment, rule out jaundice, or tell caregivers that consultation is unnecessary.

## Current safe UI language

Use “model-estimated bilirubin” and “provisional classifier signal.” Avoid “confirmed bilirubin,” “healthy,” “safe,” “no jaundice” without a provisional qualifier, and any treatment recommendation. The app must advise clinical assessment when jaundice is suspected and urgent assessment for jaundice in the first 24 hours or a seriously unwell baby.
