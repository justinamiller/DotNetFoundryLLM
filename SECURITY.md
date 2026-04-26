# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

Security fixes will be backported to 1.0.x releases until version 1.1.0 is released.

## Reporting a Vulnerability

**Do NOT open a public issue for security vulnerabilities.**

To report a security vulnerability:

1. **Use GitHub Security Advisories**: Go to the [Security tab](https://github.com/justinamiller/dotnetfoundryllm/security/advisories) and click "Report a vulnerability"
2. Provide detailed information:
   - Type of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if known)
3. You will receive an acknowledgment within **7 business days** (best effort)
4. We will work with you to understand and address the issue
5. A fix will be developed and released as soon as possible
6. You will be credited in the security advisory (unless you prefer to remain anonymous)

## Security Best Practices

When using DotNetFoundryLLM:

- **Model Files**: Only load GGUF model files from trusted sources. Maliciously crafted model files could potentially exploit parsing vulnerabilities.
- **API Exposure**: If exposing the REST API publicly, implement appropriate authentication, rate limiting, and input validation.
- **Resource Limits**: Configure appropriate memory and compute limits to prevent resource exhaustion attacks.
- **Dependencies**: Keep your .NET runtime updated to receive security patches.

## Disclosure Policy

We follow a **coordinated disclosure** policy:

1. Vulnerability is reported privately
2. Fix is developed and tested
3. Security advisory is published
4. Patch release is published
5. Public disclosure is made after users have had time to update

Thank you for helping keep DotNetFoundryLLM and its users safe!
