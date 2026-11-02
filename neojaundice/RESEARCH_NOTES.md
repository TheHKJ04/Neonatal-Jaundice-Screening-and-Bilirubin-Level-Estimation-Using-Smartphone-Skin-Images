# Research notes (index)

The current research protocol, questions, limitations, and evaluation requirements live in [`docs/research.md`](docs/research.md). The ML pipeline and metrics are described in [`docs/ml_pipeline.md`](docs/ml_pipeline.md).

Before making clinical claims, resolve these evidence gaps:
- source semantics of `treatment_label`;
- mapping between numeric classifier classes and source labels;
- definition/calibration of `classification_probability`;
- target units and provenance for bilirubin regression;
- patient-level held-out evaluation and subgroup limitations.

Reference repositories are architectural inspiration only. Record paper-level bibliographic details and actual methods/results in a proper literature-review table before citing them in a report.
