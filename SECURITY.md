# Security Policy

## Supported versions

Until the project reaches a stable release, security fixes are made on the
latest revision of the default branch only.

## Reporting a vulnerability

Do not open a public issue for suspected vulnerabilities. Use the repository's
private vulnerability-reporting feature on GitHub. If that feature is not
available, contact the maintainers privately through the owning organization's
GitHub profile.

Include the affected revision, impact, reproduction steps, and any suggested
mitigation. Do not include live wallet seed phrases, private keys, access
tokens, or private evaluation data. Use disposable test credentials when a
proof of concept requires authentication.

## Operational safety

- Use scoped, revocable Hugging Face and W&B tokens.
- Keep Bittensor coldkey material off validator hosts where possible.
- Review downloaded adapters and model artifacts as untrusted input.
- Keep `.thinker/`, wallet directories, local logs, and environment files out
  of source control.
- Rotate a credential immediately if it appears in a command, log, issue, or
  commit that could be exposed.
