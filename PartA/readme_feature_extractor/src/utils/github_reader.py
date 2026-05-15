import requests
import base64
import os
from urllib.parse import urlparse


def get_github_file_content(repo_url: str, file_path: str) -> str | None:
    """Fetch and decode a single file from a GitHub repository via the REST API."""
    try:
        parsed_url = urlparse(repo_url)
        path_parts = parsed_url.path.strip("/").split("/")

        # BUG FIX: original code used path_parts (the full list) instead of path_parts[0]
        if len(path_parts) >= 2:
            owner = path_parts[0]   # was: path_parts  ← bug
            repo  = path_parts[1]
        else:
            return None

        api_url = (
            f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
        )
        headers = {"Accept": "application/vnd.github.v3+json"}

        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"

        response = requests.get(api_url, headers=headers, timeout=30)

        if response.status_code == 200:
            content_b64 = response.json().get("content", "")
            # GitHub pads the base64 with newlines – strip them first
            return base64.b64decode(content_b64.replace("\n", "")).decode("utf-8")
        return None
    except Exception as e:
        print(f"[github_reader] Error fetching '{file_path}': {e}")
        return None


def fetch_readme(repo_name: str) -> str | None:
    """
    Fetch the README for a repo given as 'owner/repo'.
    Tries README.md, readme.md, and README.rst in order.
    """
    repo_url = f"https://github.com/{repo_name}"
    for candidate in ("README.md", "readme.md", "README.rst", "Readme.md"):
        content = get_github_file_content(repo_url, candidate)
        if content:
            return content
    return None


def fetch_repository_data(repo_url: str) -> dict:
    """
    Fetch README, requirements.txt, and package.json from a full GitHub URL.
    Returns a dict with keys: 'readme', 'requirements', 'package_json'.
    """
    data: dict = {}

    # README
    for candidate in ("README.md", "readme.md", "README.rst"):
        content = get_github_file_content(repo_url, candidate)
        if content:
            data["readme"] = content
            break

    # Python requirements
    req_content = get_github_file_content(repo_url, "requirements.txt")
    if req_content:
        data["requirements"] = req_content

    # Node / JS projects
    pkg_content = get_github_file_content(repo_url, "package.json")
    if pkg_content:
        data["package_json"] = pkg_content

    return data