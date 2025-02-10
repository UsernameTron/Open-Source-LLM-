import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
API_KEY = "8FFDbzL-cfJc8wkNo9gcGSMvKOvJhG7ZLzqWeuU2fBY"

def analyze_text(text: str) -> Dict[str, Any]:
    url = f"{BASE_URL}/api/analyze-text"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    response = requests.post(url, headers=headers, json={"text": text})
    return response.json()

def run_complex_tests():
    test_cases = {
        "Mixed Temporal Sentiment": """
        Initially, the system was plagued with critical performance issues and severe reliability problems.
        However, after three months of intensive optimization and innovative problem-solving, we achieved
        a remarkable 300% improvement in throughput. The challenging journey led to an exceptional outcome
        that exceeded our highest expectations.
        """,
        
        "Technical Feedback": """
        The codebase exhibited concerning levels of technical debt and critical security vulnerabilities.
        Through systematic refactoring and architectural improvements, we transformed it into a robust,
        efficient system. While some legacy components still need attention, the core functionality now
        demonstrates outstanding performance and innovative design patterns.
        """,
        
        "Comparative Analysis": """
        Compared to the previous version, which was more stable but limited in features, the new
        implementation initially introduced some concerning instabilities. Yet, its revolutionary
        approach to data processing, despite being challenging to implement, has proven far superior.
        The innovative architecture, though complex, delivers exceptional results that transform
        how we handle large-scale data.
        """,
        
        "Subtle Sentiment": """
        Oh, great, another 'innovative' solution that's supposed to revolutionize everything.
        I must say, it's quite 'remarkable' how we keep reinventing the wheel. The documentation
        is as 'clear as mud', but hey, at least it has some fancy diagrams. Surprisingly though,
        once you get past the initial confusion, the system actually delivers impressive results
        and genuinely solves some long-standing issues.
        """,
        
        "Multi-Paragraph Conflict": """
        The initial deployment was nothing short of catastrophic. Critical systems failed, data
        pipelines broke, and user experience was severely impacted. The team faced unprecedented
        challenges and mounting pressure from stakeholders.

        Despite these severe setbacks, the team's extraordinary resilience and innovative problem-solving
        approach led to a breakthrough. Through systematic analysis and creative solutions, we not only
        resolved the critical issues but transformed them into opportunities for improvement.

        The final system architecture emerged stronger than ever. What started as a potentially
        devastating failure became a showcase of engineering excellence. The revolutionary changes
        implemented have set new standards for system reliability and performance.
        """
    }
    
    print("Running Complex Sentiment Analysis Tests")
    print("=" * 80)
    
    for test_name, text in test_cases.items():
        print(f"\nTest Case: {test_name}")
        print("-" * 40)
        print("Text:")
        print(text.strip())
        print("\nAnalysis:")
        result = analyze_text(text)
        print(json.dumps(result, indent=2))
        print("=" * 80)

if __name__ == "__main__":
    run_complex_tests()
