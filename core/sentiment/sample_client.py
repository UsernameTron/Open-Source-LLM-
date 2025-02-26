#!/usr/bin/env python
"""
sample_client.py

A demonstration client for the improved Sentiment Analysis API.
Shows how to call the API and process the results.
"""

import requests
import json
import logging
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich import print as rprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
console = Console()

def predict_sentiment(api_url, texts):
    """Make a sentiment prediction request to the API."""
    try:
        # Prepare request
        request_data = {"texts": texts}
        
        # Call API
        with Progress() as progress:
            task = progress.add_task("[cyan]Making API request...", total=1)
            response = requests.post(
                f"{api_url}/predict",
                json=request_data
            )
            progress.update(task, advance=1)
            
        response.raise_for_status()
        
        # Return the parsed JSON response
        return response.json()
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        raise

def display_results(result):
    """Display the results in a nicely formatted way."""
    console.print("\n[bold green]API Response Summary:[/bold green]")
    console.print(f"Processing Time: [cyan]{result.get('processing_time', 0):.6f}[/cyan] seconds")
    console.print(f"Model Version: [cyan]{result.get('model_version', 'unknown')}[/cyan]")
    
    table = Table(title="Sentiment Predictions")
    table.add_column("Text", style="cyan", no_wrap=True)
    table.add_column("Sentiment", style="magenta")
    table.add_column("Confidence", justify="right")
    table.add_column("Negative", justify="right")
    table.add_column("Neutral", justify="right")
    table.add_column("Positive", justify="right")
    
    for prediction in result["results"]:
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

def check_api_health(api_url):
    """Check the health of the API."""
    try:
        response = requests.get(f"{api_url}/health")
        response.raise_for_status()
        health_data = response.json()
        
        panel = Panel(
            f"Status: [green]{health_data.get('status', 'unknown')}[/green]\n"
            f"API Version: [cyan]{health_data.get('api_version', 'unknown')}[/cyan]\n"
            f"Model Type: [cyan]{health_data.get('model', {}).get('type', 'unknown')}[/cyan]\n"
            f"Model Version: [cyan]{health_data.get('model', {}).get('version', 'unknown')}[/cyan]",
            title="API Health Check",
            border_style="green"
        )
        console.print(panel)
        return True
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        console.print(f"[bold red]API health check failed:[/bold red] {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Sample client for the Sentiment Analysis API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of sentiment API")
    parser.add_argument("--input", "-i", help="Input file with texts, one per line")
    args = parser.parse_args()
    
    api_url = args.api_url
    
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
    
    # First check API health
    console.print("[bold]Checking API health...[/bold]")
    if not check_api_health(api_url):
        return
    
    # Now make prediction request
    console.print("[bold]Making sentiment prediction request...[/bold]")
    try:
        result = predict_sentiment(api_url, example_texts)
        display_results(result)
    except Exception as e:
        console.print(f"[bold red]Failed to get predictions:[/bold red] {str(e)}")

if __name__ == "__main__":
    main()
