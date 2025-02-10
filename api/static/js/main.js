// Function to create the chart
function createProbabilityChart(ctx, probabilities, labels) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability',
                data: probabilities,
                backgroundColor: probabilities.map((_, index) => {
                    // Red for Negative, Blue for Neutral, Green for Positive
                    const colors = [
                        'rgba(255, 99, 132, 0.7)',  // Negative - Red
                        'rgba(54, 162, 235, 0.7)',  // Neutral - Blue
                        'rgba(75, 192, 192, 0.7)'   // Positive - Green
                    ];
                    return colors[index];
                }),
                borderColor: probabilities.map((_, index) => {
                    const colors = [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(75, 192, 192, 1)'
                    ];
                    return colors[index];
                }),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Probability: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            }
        }
    });
}

// Function to update results
function updateResults(data) {
    const resultsDiv = document.getElementById('results');
    
    if (!data || !data.output) {
        resultsDiv.innerHTML = '<div class="alert alert-warning">No results available</div>';
        return;
    }

    const { prediction, confidence, probabilities, explanation } = data.output;
    
    // Create results HTML
    let resultsHTML = `
        <div class="sentiment-result">
            <h4>Prediction: ${prediction}</h4>
            <div class="confidence-section mb-3">
                <p>Confidence: ${(confidence * 100).toFixed(2)}%</p>
                <div class="confidence-meter">
                    <div class="confidence-bar" style="width: ${confidence * 100}%"></div>
                </div>
            </div>
            
            <div class="explanation-section mb-3">
                <h6>Analysis Explanation:</h6>
                ${Array.isArray(explanation) && explanation.length > 0 ? 
                    explanation.map(exp => `<p class="mb-1">${exp}</p>`).join('') :
                    '<p class="text-muted">Analyzing sentiment distribution...</p>'
                }
            </div>
            
            <div class="chart-section">
                <h6>Probability Distribution:</h6>
                <canvas id="probabilityChart" width="400" height="200"></canvas>
            </div>
        </div>
    `;

    resultsDiv.innerHTML = resultsHTML;

    // Create chart
    const ctx = document.getElementById('probabilityChart').getContext('2d');
    createProbabilityChart(ctx, probabilities, ['Negative', 'Neutral', 'Positive']);
}

// Function to update metrics
function updateMetrics(metrics) {
    if (!metrics) {
        document.getElementById('latency').textContent = 'N/A';
        document.getElementById('throughput').textContent = 'N/A';
        document.getElementById('resourceUsage').textContent = 'N/A';
        return;
    }

    const metricsConfig = [
        {
            id: 'latency',
            value: metrics.latency_ms,
            format: (v) => `${v.toFixed(2)} ms`
        },
        {
            id: 'throughput',
            value: metrics.throughput,
            format: (v) => `${v.toFixed(2)} req/s`
        },
        {
            id: 'resourceUsage',
            value: metrics.gpu_utilization,
            format: (v) => `GPU: ${v.toFixed(1)}% | Memory: ${metrics.memory_utilization.toFixed(1)}%`
        }
    ];

    metricsConfig.forEach(metric => {
        const el = document.getElementById(metric.id);
        if (metric.value > 0) {
            el.textContent = metric.format(metric.value);
            el.parentElement.classList.add('metrics-card');
        } else {
            el.textContent = 'Calculating...';
        }
    });
}

// Handle form submission
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('inferenceForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Get text input and file
        const text = document.getElementById('textInput').value;
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        
        if (!text && !file) {
            document.getElementById('results').innerHTML = 
                '<div class="alert alert-warning">Please enter text or upload a file</div>';
            return;
        }
        
        // Show loading
        document.getElementById('results').innerHTML = 
            '<div class="alert alert-info">Processing...</div>';
        
        try {
            let response;
            let data;
            
            if (text) {
                // Send text analysis request
                response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        text: text,
                        explain: true
                    })
                });
            } else if (file) {
                // Send file upload request
                const formData = new FormData();
                formData.append('files', file);
                formData.append('options', JSON.stringify({
                    csv_options: {
                        text_column: 'Review Text',
                        batch_size: 10
                    }
                }));
                
                response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
            }
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            
            data = await response.json();
            
            // Handle file upload response differently
            if (file) {
                const resultDiv = document.getElementById('results');
                let resultsHTML = '';
                
                if (data.errors) {
                    resultsHTML += `
                        <div class="alert alert-warning">
                            <h4>Some files had errors:</h4>
                            <ul>
                                ${data.errors.map(err => `
                                    <li>${err.filename}: ${err.error}</li>
                                `).join('')}
                            </ul>
                        </div>
                    `;
                }
                
                if (data.files && data.files.length > 0) {
                    resultsHTML += `
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">File Processing Results</h5>
                                <p>Successfully processed ${data.successful_files} of ${data.total_files} files</p>
                                <p>Total processing time: ${data.total_processing_time.toFixed(2)}s</p>
                                <p>Average time per file: ${data.average_time_per_file.toFixed(2)}s</p>
                            </div>
                        </div>
                    `;
                    
                    // Show results for each file
                    data.files.forEach(fileResult => {
                        resultsHTML += `
                            <div class="card mt-3">
                                <div class="card-body">
                                    <h6 class="card-subtitle mb-2 text-muted">File: ${fileResult.filename}</h6>
                                    <div class="results-container">
                                        ${fileResult.results.map(result => `
                                            <div class="sentiment-result">
                                                <div class="text-section mb-3">
                                                    <h6>Analyzed Text:</h6>
                                                    <p class="text-muted">${result.text}</p>
                                                </div>
                                                
                                                <h4>Prediction: ${result.output.prediction}</h4>
                                                <div class="confidence-section mb-3">
                                                    <p class="mb-1">Confidence: ${(result.output.confidence * 100).toFixed(2)}%</p>
                                                    <div class="confidence-meter">
                                                        <div class="confidence-bar" 
                                                             style="width: ${result.output.confidence * 100}%">
                                                        </div>
                                                    </div>
                                                </div>
                                                
                                                <div class="explanation-section mb-3">
                                                    <h6>Analysis Explanation:</h6>
                                                    ${Array.isArray(result.output.explanation) && result.output.explanation.length > 0 ? 
                                                        result.output.explanation.map(exp => `<p class="mb-1">${exp}</p>`).join('') :
                                                        '<p class="text-muted">Analyzing sentiment patterns...</p>'
                                                    }
                                                </div>
                                                
                                                <div class="chart-section mb-4">
                                                    <h6>Probability Distribution:</h6>
                                                    <div style="height: 200px;">
                                                        <canvas class="probability-chart"></canvas>
                                                    </div>
                                                </div>
                                            </div>
                                        `).join('<hr>')}
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                }
                
                resultDiv.innerHTML = resultsHTML;
                
                // Initialize charts for each result
                data.files.forEach(fileResult => {
                    fileResult.results.forEach((result, index) => {
                        const chartCanvases = document.querySelectorAll('.probability-chart');
                        const ctx = chartCanvases[index].getContext('2d');
                        createProbabilityChart(ctx, result.output.probabilities, ['Negative', 'Neutral', 'Positive']);
                    });
                });
            } else {
                // Update results for text input
                updateResults(data);
            }
            
            // Clear inputs after successful submission
            document.getElementById('textInput').value = '';
            fileInput.value = '';
            
            // Update metrics in both cases
            updateMetrics(data.metrics);
        } catch (error) {
            document.getElementById('results').innerHTML = 
                `<div class="alert alert-danger">Error: ${error.message}</div>`;
            updateMetrics(null);
        }
    });
});
