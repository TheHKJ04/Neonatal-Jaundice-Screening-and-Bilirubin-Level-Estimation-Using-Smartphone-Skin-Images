NeoJaundice Phase 7.6 patch

1. Extract this ZIP into /content/neojaundice/neojaundice (preserve relative paths).
2. Run unit tests:
   python -m pytest tests/unit/test_evaluate_classification_test.py -v
3. Run docs/ml/phase76_classification_evaluation.md's Colab command.

If the test manifest is not automatically detected, pass --test-manifest with its exact path. Ensure the classification checkpoint was saved by Phase 7.5 and that the dataset/model APIs match the existing project. The target semantics of treatment_label are unverified; this is not clinical validation.
