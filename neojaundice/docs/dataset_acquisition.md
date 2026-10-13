# NeoJaundice Dataset Acquisition Protocol

## Status

Preliminary research acquisition controls.

No clinical dataset is approved or downloaded
by this document.

## 1. Purpose

Establish a controlled process for acquiring,
verifying, and storing neonatal jaundice datasets.

## 2. Required Reviews

Before acquisition:

1. Confirm the original dataset source.
2. Identify the exact dataset version.
3. Review the applicable license.
4. Verify permission for the intended research use.
5. Review privacy and consent conditions.
6. Record the reviewer and approval reference.
7. Confirm suitable storage and access controls.

## 3. Approval Status

Possible statuses:

- pending
- approved
- rejected
- suspended

Only approved datasets may enter the
controlled acquisition workflow.

## 4. Download Policy

Datasets must be acquired from the approved source.

Do not download patient images from unknown mirrors.

Do not upload clinical archives to public repositories.

## 5. Integrity Verification

Calculate SHA-256 for downloaded archives.

Compare against a trusted provider checksum
when available.

Record the checksum and source version.

A self-computed checksum alone does not prove
the authenticity of the source.

## 6. Archive Inspection

Before extraction:

- Reject unsafe paths.
- Reject unsupported file types.
- Check archive size.
- Check member count.
- Check compression ratios.
- Inspect archive contents.

## 7. Storage

Raw datasets belong in protected storage.

The repository's data/raw directory is intended
for local development and must not be treated
as secure clinical storage by default.

## 8. Audit Logging

Record:

- Dataset identifier
- Action
- Timestamp
- Outcome

Do not log patient identifiers, images,
credentials, or sensitive metadata.

## 9. Reproducibility

Record:

- Dataset source
- Dataset version
- License
- Approval reference
- Archive checksum
- Extraction method
- Code version

## 10. Current Limitations

The current utilities do not provide:

- Automated permission verification
- Enterprise-grade access control
- Encryption key management
- Malware scanning
- Full archive format support
- Clinical compliance certification

These requirements must be assessed before
handling sensitive real-world clinical data.
