#!/usr/bin/env python
"""
sample_batch_client.py

A demonstration client for the batch processing feature of the improved Sentiment Analysis API.
Shows how to submit a batch job, check its status, and retrieve results.
"""

import requests
import json
import logging
import argparse
import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

def submit_batch_job(api_url, texts):
    """Submit a batch job to the API."""
    try:
        # Prepare request
        request_data = {"texts": texts}
        
        # Call API
        with Progress() as progress:
            task = progress.add_task("[cyan]Submitting batch job...", total=1)
            response = requests.post(
                f"{api_url}/batch",
                json=request_data
            )
            progress.update(task, advance=1)
            
        response.raise_for_status()
        
        # Return the parsed JSON response
        return response.json()
    except Exception as e:
        logger.error(f"Batch job submission failed: {str(e)}")
        raise

def check_batch_status(api_url, job_id):
    """Check the status of a batch job."""
    try:
        response = requests.get(f"{api_url}/batch/{job_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to check batch status: {str(e)}")
        raise

def display_batch_results(result):
    """Display the batch processing results."""
    console.print("\n[bold green]Batch Processing Results:[/bold green]")
    console.print(f"Job ID: [cyan]{result.get('job_id', 'unknown')}[/cyan]")
    console.print(f"Status: [cyan]{result.get('status', 'unknown')}[/cyan]")
    console.print(f"Model Version: [cyan]{result.get('model_version', 'unknown')}[/cyan]")
    
    # Check for results - they might be under a different key based on the status response
    results = result.get("results") or result.get("predictions") or []
    
    if (result.get("status") == "completed" or result.get("status") == "complete") and results:
        table = Table(title="Sentiment Predictions")
        table.add_column("Text", style="cyan", no_wrap=True)
        table.add_column("Sentiment", style="magenta")
        table.add_column("Confidence", justify="right")
        table.add_column("Negative", justify="right")
        table.add_column("Neutral", justify="right")
        table.add_column("Positive", justify="right")
        
        for prediction in results:
            text = prediction["text"]
            if len(text) > 50:
                text = text[:47] + "..."
                
            sentiment = prediction["sentiment"]
            sentiment_style = {
                "Positive": "[green]Positive[/green]",
                "Neutral": "[yellow]Neutral[/yellow]",
                "Negative": "[red]Negative[/red]"
            }.get(sentiment, sentiment)
            
            confidence = f"{prediction['confidence']:.2f}"
            scores = prediction["scores"]
            
            table.add_row(
                text,
                sentiment_style,
                confidence,
                f"{scores.get('Negative', 0):.2f}",
                f"{scores.get('Neutral', 0):.2f}",
                f"{scores.get('Positive', 0):.2f}"
            )
        
        console.print(table)
    elif result.get("status") == "complete" and not results:
        console.print("[yellow]Job is complete but no results were found in the response.[/yellow]")
        console.print("[yellow]To view results, try making another request to the API.[/yellow]")
    elif result.get("status") == "processing":
        console.print("[yellow]Job is still processing...[/yellow]")
    elif result.get("status") == "failed":
        console.print(f"[bold red]Job failed:[/bold red] {result.get('error', 'Unknown error')}")

def main():
    parser = argparse.ArgumentParser(description="Sample batch client for the Sentiment Analysis API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of sentiment API")
    parser.add_argument("--input", "-i", help="Input file with texts, one per line")
    parser.add_argument("--job-id", "-j", help="Existing job ID to check status")
    args = parser.parse_args()
    
    api_url = args.api_url
    
    # If job ID is provided, check status
    if args.job_id:
        console.print(f"[bold]Checking status of job {args.job_id}...[/bold]")
        try:
            result = check_batch_status(api_url, args.job_id)
            display_batch_results(result)
            return
        except Exception as e:
            console.print(f"[bold red]Failed to check job status:[/bold red] {str(e)}")
            return
    
    # Example texts to analyze
    example_texts = [
        "This product exceeded all my expectations! Absolutely fantastic.",
        "I love how well this works, best purchase I've made all year.",
        "Terrible experience, would not recommend to anyone.",
        "This is the worst product I've ever used, complete waste of money.",
        "The product works as expected, neither good nor bad.",
        "It arrived on time and functions adequately.",
        "While the design is nice, the functionality is lacking.",
        "Great features but the customer service was disappointing."
    ]
    
    # If input file is provided, read texts from it
    if args.input:
        try:
            with open(args.input, 'r') as f:
                example_texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to read input file: {str(e)}")
            console.print(f"[bold red]Failed to read input file:[/bold red] {str(e)}")
            return
    
    # Submit batch job
    console.print("[bold]Submitting batch job...[/bold]")
    try:
        result = submit_batch_job(api_url, example_texts)
        job_id = result.get("job_id")
        console.print(f"[green]Batch job submitted successfully.[/green] Job ID: [cyan]{job_id}[/cyan]")
        
        # Poll for job completion
        max_retries = 5
        retry_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Waiting for batch processing to complete...")
        ) as progress:
            task = progress.add_task("Waiting", total=None)
            try:
                for _ in range(10):  # Maximum 10 seconds wait time
                    time.sleep(1)  # Wait for 1 second before checking again
                    try:
                        status_result = check_batch_status(api_url, job_id)
                        retry_count = 0  # Reset retry counter on successful request
                        if status_result.get("status") in ["completed", "complete", "failed"]:
                            break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            console.print(f"[bold red]Failed to check status after {max_retries} retries.[/bold red]")
                            raise
                        console.print(f"[yellow]Retry {retry_count}/{max_retries}: Failed to check status. Retrying...[/yellow]")
                else:
                    console.print("[yellow]Batch processing taking longer than expected. Check status manually with:[/yellow]")
                    console.print(f"[cyan]python -m core.sentiment.sample_batch_client --job-id {job_id}[/cyan]")
                    return
            except Exception as e:
                console.print(f"[bold red]Error during batch processing:[/bold red] {str(e)}")
                return
        
        # Display final results
        display_batch_results(status_result)
        
    except Exception as e:
        console.print(f"[bold red]Batch processing failed:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
