FROM python:3.11-slim

WORKDIR /workspace

# Install test dependencies
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy repo + generated tests at runtime via volume mount
CMD ["pytest", "tests/", "-v", "--tb=short", "--cov=repo", "--cov-branch", "--cov-report=json:/workspace/coverage.json", "--cov-report=term-missing"]
