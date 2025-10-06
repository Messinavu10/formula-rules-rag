#!/usr/bin/env python3
"""
Local CI/CD Testing Script

This script runs the same checks that the GitHub Actions CI/CD pipeline runs,
allowing you to test locally before pushing changes.
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def run_command(command, description):
    """Run a command and return success status."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    print("-" * 50)

    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print("✅ Success")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed")
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def check_environment():
    """Check if required environment variables are set."""
    print("🔍 Checking environment...")

    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing_vars = []
    found_vars = []

    for var in required_vars:
        if os.getenv(var):
            found_vars.append(var)
        else:
            missing_vars.append(var)

    if found_vars:
        print(f"✅ Found variables: {', '.join(found_vars)}")

    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file or system environment.")
        print("Example .env file:")
        print("OPENAI_API_KEY=your_openai_api_key")
        print("PINECONE_API_KEY=your_pinecone_api_key")
        return False

    print("✅ All required environment variables set")
    return True


def main():
    """Run local CI/CD tests."""
    print("🚀 Local CI/CD Testing")
    print("=" * 50)

    # Load environment variables from .env file
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        print(f"📁 Loading environment from {env_file}")
        load_dotenv(env_file)
    else:
        print("⚠️  No .env file found, using system environment variables")
        print("💡 Create a .env file with your API keys:")
        print("   OPENAI_API_KEY=your_openai_api_key")
        print("   PINECONE_API_KEY=your_pinecone_api_key")

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Change to project root
    os.chdir(project_root)

    # Test results
    results = []

    # 1. Code Quality Checks
    print("\n📋 CODE QUALITY CHECKS")
    print("=" * 50)

    results.append(
        (
            "Black formatting",
            run_command(
                "black --check --diff src/ scripts/ tests/",
                "Checking code formatting with Black",
            ),
        )
    )

    results.append(
        (
            "Import sorting",
            run_command(
                "isort --check-only --diff src/ scripts/ tests/",
                "Checking import sorting with isort",
            ),
        )
    )

    results.append(
        (
            "Linting",
            run_command("flake8 src/ scripts/ tests/", "Running linting with flake8"),
        )
    )

    results.append(
        (
            "Security linting",
            run_command(
                "bandit -r src/ scripts/", "Running security linting with bandit"
            ),
        )
    )

    results.append(
        (
            "Dependency security",
            run_command(
                "safety check", "Checking dependency vulnerabilities with safety"
            ),
        )
    )

    # 2. Testing
    print("\n🧪 TESTING")
    print("=" * 50)

    results.append(
        (
            "Unit tests",
            run_command(
                "pytest tests/ --cov=src --cov-report=term-missing --cov-report=xml --junitxml=test-results.xml",
                "Running unit tests with coverage",
            ),
        )
    )

    # 3. Performance Tests
    print("\n⚡ PERFORMANCE TESTING")
    print("=" * 50)

    results.append(
        (
            "Performance tests",
            run_command(
                "pytest tests/test_performance.py -v -m performance",
                "Running performance tests",
            ),
        )
    )

    # 4. Documentation Check
    print("\n📚 DOCUMENTATION CHECK")
    print("=" * 50)

    results.append(
        ("README exists", run_command("test -f README.md", "Checking README.md exists"))
    )

    # 5. UI Testing
    print("\n🖥️ UI TESTING")
    print("=" * 50)

    results.append(
        (
            "Streamlit app syntax",
            run_command(
                "python -c \"import streamlit_app; print('Streamlit app syntax OK')\"",
                "Testing Streamlit app syntax",
            ),
        )
    )

    results.append(
        (
            "UI launcher",
            run_command(
                "python -c \"import run_ui; print('UI launcher OK')\"",
                "Testing UI launcher",
            ),
        )
    )

    # 6. Evaluation (Optional)
    print("\n📊 EVALUATION (Optional)")
    print("=" * 50)

    try:
        run_evaluation = input("Run agent evaluation? (y/N): ").lower().strip() == "y"
    except EOFError:
        # Handle non-interactive environments
        print("Skipping evaluation (non-interactive environment)")
        run_evaluation = False

    if run_evaluation:
        results.append(
            (
                "Agent evaluation",
                run_command(
                    "python scripts/evaluate_agent.py --verbose",
                    "Running agent evaluation",
                ),
            )
        )

    # Summary
    print("\n📊 SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Ready to push.")
        return 0
    else:
        print("❌ Some tests failed. Please fix issues before pushing.")
        print("\nFailed tests:")
        for test_name, success in results:
            if not success:
                print(f"  - {test_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
