document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const textInput = document.getElementById('text-input');
    const classifyBtn = document.getElementById('classify-btn');
    const resultsSection = document.getElementById('results-section');
    const loadingIndicator = document.getElementById('loading-indicator');
    const classificationResult = document.getElementById('classification-result');
    const confidenceValue = document.getElementById('confidence-value');
    const batchInput = document.getElementById('batch-input');
    const batchClassifyBtn = document.getElementById('batch-classify-btn');
    const batchResults = document.getElementById('batch-results');
    const batchResultsBody = document.getElementById('batch-results-body');
    
    // Chart variables
    let resultChart = null;
    let batchChart = null;
    
    // Function to classify a single text
    classifyBtn.addEventListener('click', function() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter a comment to classify.');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        resultsSection.style.display = 'none';
        
        // Send classification request
        fetch('/api/a7/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Display results
            displayClassificationResult(data);
            resultsSection.style.display = 'block';
            
            // Update model name if provided
            if (data.model_name) {
                document.getElementById('model-name').textContent = data.model_name;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.style.display = 'none';
            alert('An error occurred while classifying the text. Please try again.');
        });
    });
    
    // Function to display classification result with visualization
    function displayClassificationResult(result) {
        // Update text elements
        classificationResult.textContent = result.label;
        confidenceValue.textContent = `${(result.confidence * 100).toFixed(2)}%`;
        
        // Set color based on classification
        if (result.label.toLowerCase().includes('toxic')) {
            classificationResult.className = 'toxic';
        } else {
            classificationResult.className = 'non-toxic';
        }
        
        // Create/update chart visualization
        const ctx = document.getElementById('result-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (resultChart) {
            resultChart.destroy();
        }
        
        // Get label names from the result
        const labels = Object.values(result.id2label);
        
        // Create new chart
        resultChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability',
                    data: result.probabilities,
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.7)',  // Green for non-toxic
                        'rgba(220, 53, 69, 0.7)'   // Red for toxic
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(220, 53, 69, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Function to handle batch classification
    batchClassifyBtn.addEventListener('click', function() {
        const texts = batchInput.value.trim().split('\n').filter(text => text.trim() !== '');
        
        if (texts.length === 0) {
            alert('Please enter at least one comment for batch classification.');
            return;
        }
        
        if (texts.length > 10) {
            alert('Please limit batch classification to 10 comments at a time.');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        batchResults.style.display = 'none';
        
        // Send batch classification request
        fetch('/api/a7/batch_classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ texts: texts })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            loadingIndicator.style.display = 'none';
            
            // Display batch results
            displayBatchResults(data.results);
            batchResults.style.display = 'block';
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.style.display = 'none';
            alert('An error occurred during batch classification. Please try again.');
        });
    });
    
    // Function to display batch classification results
    function displayBatchResults(results) {
        // Clear previous results
        batchResultsBody.innerHTML = '';
        
        // Add each result to the table
        results.forEach(result => {
            const row = document.createElement('tr');
            
            // Truncate text if too long
            const truncatedText = result.text.length > 50 
                ? result.text.substring(0, 50) + '...' 
                : result.text;
            
            // Determine CSS class based on classification
            const labelClass = result.label.toLowerCase().includes('toxic') 
                ? 'toxic' 
                : 'non-toxic';
            
            // Create table row with result data
            row.innerHTML = `
                <td>${truncatedText}</td>
                <td><span class="${labelClass}">${result.label}</span></td>
                <td>${(result.confidence * 100).toFixed(2)}%</td>
            `;
            
            batchResultsBody.appendChild(row);
        });
        
        // Create batch visualization chart
        createBatchChart(results);
    }
    
    // Function to create batch visualization chart
    function createBatchChart(results) {
        const ctx = document.getElementById('batch-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (batchChart) {
            batchChart.destroy();
        }
        
        // Prepare data for chart
        const labels = results.map(r => r.text.substring(0, 20) + (r.text.length > 20 ? '...' : ''));
        const toxicities = results.map(r => {
            // Get toxic probability (assuming binary classification with toxic as index 1)
            // For multi-class, adjust this logic accordingly
            return r.probabilities[1]; // Index 1 typically corresponds to "Toxic" class
        });
        
        // Create colors based on toxicity level
        const backgroundColors = toxicities.map(t => 
            `rgba(${Math.round(255 * t)}, ${Math.round(255 * (1 - t))}, 0, 0.7)`
        );
        
        // Create border colors
        const borderColors = toxicities.map(t => 
            `rgba(${Math.round(255 * t)}, ${Math.round(255 * (1 - t))}, 0, 1)`
        );
        
        // Create new chart
        batchChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Toxicity Probability',
                    data: toxicities,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const result = results[context.dataIndex];
                                return [
                                    `Classification: ${result.label}`,
                                    `Probability: ${(context.raw * 100).toFixed(2)}%`
                                ];
                            }
                        }
                    }
                }
            }
        });
    }
});
