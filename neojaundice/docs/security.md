# NeoJaundice Security and Privacy

## 1. Scope

NeoJaundice is a healthcare-oriented research project.

Medical images and associated metadata may contain
sensitive personal information.

## 2. Current Security Controls

The repository currently uses:

- Git ignore rules
- Local environment variables
- Secret-exclusion conventions
- Docker build exclusions
- Basic pre-commit checks

These are foundational controls, not a complete
healthcare security program.

## 3. Data Handling Principles

- Collect only necessary data.
- Use appropriate authorization and consent.
- Prefer pseudonymized identifiers.
- Restrict dataset access.
- Avoid public storage of identifiable images.
- Document data retention requirements.
- Maintain secure backups where appropriate.

## 4. Source Control

Do not commit:

- Patient-identifiable images
- Medical records
- Passwords
- API keys
- Private certificates
- Local database files
- Unapproved dataset exports

## 5. Environment Variables

Use `.env.example` as a template.

Store local secrets in `.env`.

Never commit `.env`.

## 6. API Security

Planned controls include:

- Request validation
- Authentication
- Authorization
- File upload restrictions
- Rate limiting
- Secure error handling
- Audit logging

## 7. Database Security

Planned controls include:

- Least-privilege database accounts
- Restricted network access
- Secure credentials
- Backup protection
- Migration controls

## 8. ML Security

Potential threats include:

- Malicious image uploads
- Data poisoning
- Model extraction
- Sensitive information leakage
- Dependency vulnerabilities
- Supply-chain compromise

## 9. Incident Response

If sensitive data is accidentally committed:

1. Stop further distribution.
2. Assess exposure.
3. Revoke affected credentials.
4. Follow applicable incident procedures.
5. Review Git history.
6. Document corrective actions.

## 10. Clinical Safety

A model prediction must not be presented as a
clinically validated bilirubin measurement without
appropriate evidence and authorization.

## 11. Compliance

Applicable privacy, medical-device, and research
requirements depend on the jurisdiction, intended
use, and deployment context.

A formal compliance assessment has not yet been
completed.
