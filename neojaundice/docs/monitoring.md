# NeoJaundice Monitoring and Observability

## Purpose

NeoJaundice exposes operational metrics for debugging and performance
analysis. Monitoring results do not establish clinical validity.

## Metrics endpoint

The FastAPI application exposes Prometheus metrics at:

    GET /metrics

## Available metrics

- `neojaundice_http_requests_total`
  Completed HTTP requests grouped by method, route, and status code.

- `neojaundice_http_request_duration_seconds`
  HTTP request duration histogram.

- `neojaundice_model_inference_duration_seconds`
  Model-registry inference duration histogram, grouped by outcome.

- `neojaundice_model_inference_errors_total`
  Number of inference calls that raised exceptions.

## Logs

Application logs are emitted as JSON records with timestamps, severity,
logger name, message, and selected request metadata.

Uploaded image contents, request bodies, and patient identifiers must not
be included in ordinary application logs.

## Local validation

Run unit tests:

    python -m pytest tests/unit/test_metrics.py -v

Run the full unit suite:

    python -m pytest tests/unit -q

Start the API using the project's existing development command, then visit:

    http://127.0.0.1:8000/metrics

Use the actual host and port configured for your environment.

## Interpretation

HTTP duration includes the complete request processing time. Model inference
duration measures only the model-registry prediction call.

The two values should not be treated as interchangeable.

Metrics describe software operation and performance. They do not prove
bilirubin estimation accuracy, calibration, clinical safety, or suitability
for treatment decisions.

## Future work

- Configure Prometheus scraping.
- Build Grafana dashboards.
- Add deployment health checks and alert rules.
- Evaluate latency under concurrent requests.
- Restrict access to monitoring endpoints in deployed environments.
