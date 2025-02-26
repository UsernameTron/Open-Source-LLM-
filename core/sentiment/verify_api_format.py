#!/usr/bin/env python
"""
verify_api_format.py

Script to verify that the sentiment API response format follows the improved design.
This script validates all API endpoints to ensure they return data in the expected format.
"""

import requests
import json
import logging
import argparse
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

# Test sentences
TEST_SENTENCES = [
    "This product exceeded all my expectations! Absolutely fantastic.",
    "Terrible experience, would not recommend to anyone.",
    "The product works as expected, neither good nor bad."
]

def check_api_health(api_url):
    """Check the health endpoint format."""
    console.print("\n[bold cyan]Testing /health endpoint...[/bold cyan]")
    
    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        data = response.json()
        
        # Verify expected fields
        expected_fields = ["status", "api_version", "model"]
        missing_fields = [field for field in expected_fields if field not in data]
        
        if missing_fields:
            console.print(f"[red]❌ Health endpoint missing fields: {', '.join(missing_fields)}[/red]")
        else:
            console.print("[green]✓ Health endpoint contains all required fields[/green]")
        
        # Verify model info
        if "model" in data:
            model_info = data["model"]
            expected_model_fields = ["type", "version"]
            missing_model_fields = [field for field in expected_model_fields if field not in model_info]
            
            if missing_model_fields:
                console.print(f"[red]❌ Model info missing fields: {', '.join(missing_model_fields)}[/red]")
            else:
                console.print("[green]✓ Model info contains all required fields[/green]")
        
        # Display the response
        console.print("Health endpoint response:")
        console.print(Panel(json.dumps(data, indent=2), border_style="cyan"))
        
        return True
    except Exception as e:
        console.print(f"[red]❌ Health endpoint test failed: {str(e)}[/red]")
        return False

def check_predict_endpoint(api_url):
    """Check the predict endpoint format."""
    console.print("\n[bold cyan]Testing /predict endpoint...[/bold cyan]")
    
    try:
        # Prepare request
        request_data = {"texts": TEST_SENTENCES}
        
        # Call API
        response = requests.post(
            f"{api_url}/predict",
            json=request_data
        )
        response.raise_for_status()
        data = response.json()
        
        # Verify expected fields
        expected_fields = ["results", "processing_time", "model_version"]
        missing_fields = [field for field in expected_fields if field not in data]
        
        if missing_fields:
            console.print(f"[red]❌ Predict endpoint missing fields: {', '.join(missing_fields)}[/red]")
        else:
            console.print("[green]✓ Predict endpoint contains all required fields[/green]")
        
        # Verify results array
        if "results" in data:
            results = data["results"]
            
            if not isinstance(results, list):
                console.print(f"[red]❌ Results is not an array (type: {type(results)})[/red]")
            elif len(results) != len(TEST_SENTENCES):
                console.print(f"[red]❌ Results count mismatch: got {len(results)}, expected {len(TEST_SENTENCES)}[/red]")
            else:
                console.print(f"[green]✓ Results contains {len(results)} items as expected[/green]")
            
            # Check first result format
            if results and isinstance(results, list):
                first_result = results[0]
                expected_result_fields = ["text", "sentiment", "confidence", "scores"]
                missing_result_fields = [field for field in expected_result_fields if field not in first_result]
                
                if missing_result_fields:
                    console.print(f"[red]❌ Result item missing fields: {', '.join(missing_result_fields)}[/red]")
                else:
                    console.print("[green]✓ Result item contains all required fields[/green]")
        
        # Display the response
        console.print("Predict endpoint response:")
        console.print(Panel(json.dumps(data, indent=2), border_style="cyan"))
        
        return True
    except Exception as e:
        console.print(f"[red]❌ Predict endpoint test failed: {str(e)}[/red]")
        return False

def check_batch_endpoints(api_url):
    """Check the batch endpoints format."""
    console.print("\n[bold cyan]Testing /batch endpoints...[/bold cyan]")
    
    try:
        # 1. Submit batch job
        request_data = {"texts": TEST_SENTENCES}
        
        submit_response = requests.post(
            f"{api_url}/batch",
            json=request_data
        )
        submit_response.raise_for_status()
        submit_data = submit_response.json()
        
        # Verify submit response format
        expected_submit_fields = ["job_id", "status", "model_version"]
        missing_submit_fields = [field for field in expected_submit_fields if field not in submit_data]
        
        if missing_submit_fields:
            console.print(f"[red]❌ Batch submit response missing fields: {', '.join(missing_submit_fields)}[/red]")
        else:
            console.print("[green]✓ Batch submit response contains all required fields[/green]")
        
        # Display the submit response
        console.print("Batch submit response:")
        console.print(Panel(json.dumps(submit_data, indent=2), border_style="cyan"))
        
        # 2. Check batch status
        job_id = submit_data.get("job_id")
        if not job_id:
            console.print("[red]❌ Batch job ID not found in submit response[/red]")
            return False
        
        # Wait for job to complete
        console.print(f"Waiting for job {job_id} to complete...")
        status_data = None
        
        for _ in range(5):  # Try for 5 seconds
            status_response = requests.get(f"{api_url}/batch/{job_id}")
            status_response.raise_for_status()
            status_data = status_response.json()
            
            status = status_data.get("status")
            if status in ["complete", "completed", "failed"]:
                break
            
            time.sleep(1)
        
        # Verify status response format
        expected_status_fields = ["job_id", "status", "model_version"]
        missing_status_fields = [field for field in expected_status_fields if field not in status_data]
        
        if missing_status_fields:
            console.print(f"[red]❌ Batch status response missing fields: {', '.join(missing_status_fields)}[/red]")
        else:
            console.print("[green]✓ Batch status response contains all required fields[/green]")
        
        # Verify results format for completed jobs
        if status_data.get("status") in ["complete", "completed"]:
            results = status_data.get("results") or status_data.get("predictions") or []
            
            if not results:
                console.print("[yellow]⚠️ Batch results not found in status response (may use a different key)[/yellow]")
            elif not isinstance(results, list):
                console.print(f"[red]❌ Batch results is not an array (type: {type(results)})[/red]")
            else:
                console.print(f"[green]✓ Batch results contains {len(results)} items[/green]")
                
                # Check first result format
                if results:
                    first_result = results[0]
                    expected_result_fields = ["text", "sentiment", "confidence", "scores"]
                    missing_result_fields = [field for field in expected_result_fields if field not in first_result]
                    
                    if missing_result_fields:
                        console.print(f"[red]❌ Batch result item missing fields: {', '.join(missing_result_fields)}[/red]")
                    else:
                        console.print("[green]✓ Batch result item contains all required fields[/green]")
        
        # Display the status response
        console.print("Batch status response:")
        console.print(Panel(json.dumps(status_data, indent=2), border_style="cyan"))
        
        return True
    except Exception as e:
        console.print(f"[red]❌ Batch endpoints test failed: {str(e)}[/red]")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify sentiment API format")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of sentiment API")
    args = parser.parse_args()
    
    api_url = args.api_url
    
    console.print("[bold]Verifying Sentiment API Response Format[/bold]")
    console.print(f"API URL: {api_url}")
    
    # Check all endpoints
    health_ok = check_api_health(api_url)
    predict_ok = check_predict_endpoint(api_url)
    batch_ok = check_batch_endpoints(api_url)
    
    # Summary
    console.print("\n[bold]API Format Verification Summary:[/bold]")
    console.print(f"Health Endpoint: {'[green]✓ OK[/green]' if health_ok else '[red]❌ FAILED[/red]'}")
    console.print(f"Predict Endpoint: {'[green]✓ OK[/green]' if predict_ok else '[red]❌ FAILED[/red]'}")
    console.print(f"Batch Endpoints: {'[green]✓ OK[/green]' if batch_ok else '[red]❌ FAILED[/red]'}")
    
    if health_ok and predict_ok and batch_ok:
        console.print("\n[bold green]✅ API FORMAT VERIFICATION PASSED[/bold green]")
        console.print("[green]All endpoints return data in the expected format[/green]")
    else:
        console.print("\n[bold red]❌ API FORMAT VERIFICATION FAILED[/bold red]")
        console.print("[red]Some endpoints do not conform to the expected format[/red]")

if __name__ == "__main__":
    main()
