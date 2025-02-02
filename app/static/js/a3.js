document.addEventListener('DOMContentLoaded', function() {
    const translateBtn = document.getElementById('translateBtn');
    const clearBtn = document.getElementById('clearBtn');
    const sourceText = document.getElementById('sourceText');
    const targetText = document.getElementById('targetText');
    const modelSelect = document.getElementById('modelSelect');
    const attentionMap = document.getElementById('attentionMap');
    const attentionText = document.getElementById('attentionText');

    // Add loading state handlers
    function setLoading(isLoading) {
        const elements = [sourceText, modelSelect, translateBtn];
        elements.forEach(el => {
            el.disabled = isLoading;
        });
        translateBtn.innerHTML = isLoading ? 
            '<span class="spinner-border spinner-border-sm me-2"></span>Translating...' : 
            '<i class="fas fa-language me-2"></i>Translate';
        
        if (isLoading) {
            document.getElementById('translationOutput').classList.add('loading');
            document.getElementById('attentionMap').classList.add('loading');
        } else {
            document.getElementById('translationOutput').classList.remove('loading');
            document.getElementById('attentionMap').classList.remove('loading');
        }
    }

    // Create text-based attention weights display
    function displayAttentionText(data) {
        let text = '';
        const maxTokenLength = Math.max(...data.target_tokens.map(t => t.length));
        
        data.target_tokens.forEach((target, i) => {
            // Pad target token for alignment
            const paddedTarget = target.padEnd(maxTokenLength + 1);
            
            // Create source token weights string
            const weights = data.source_tokens.map((source, j) => {
                const weight = data.attention_map[i][j].toFixed(2);
                return `${source}(${weight})`;
            }).join(' ');
            
            text += `${paddedTarget}: ${weights}\n`;
        });
        
        attentionText.textContent = text;
    }

    // Create attention heatmap
    function createAttentionMap(data) {
        const trace = {
            z: data.attention_map,
            x: data.source_tokens,
            y: data.target_tokens,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true,
            hoverongaps: false,
            hovertemplate: 
                'Source: %{x}<br>' +
                'Target: %{y}<br>' +
                'Attention: %{z:.3f}<extra></extra>'
        };

        const layout = {
            title: {
                text: 'Attention Weights Visualization',
                font: { size: 16 }
            },
            xaxis: {
                title: 'Source Text (English)',
                tickangle: 45,
                tickfont: { size: 12 }
            },
            yaxis: {
                title: 'Target Text (Gujarati)',
                tickfont: { size: 12 }
            },
            margin: {
                l: 150,
                r: 50,
                t: 50,
                b: 100
            }
        };

        Plotly.newPlot('attentionPlot', [trace], layout);
    }

    // Handle translation
    translateBtn.addEventListener('click', async function() {
        const text = sourceText.value.trim();
        const model = modelSelect.value;
        
        if (!text) {
            alert('Please enter some text to translate.');
            return;
        }
        
        setLoading(true);
        
        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    model: model
                })
            });
            
            const data = await response.json();
            
            if (data.error) {
                alert('Translation error: ' + data.error);
                return;
            }
            
            targetText.textContent = data.translation;
            
            if (data.attention_map && data.source_tokens && data.target_tokens) {
                createAttentionMap(data);
                displayAttentionText(data);
            }
            
        } catch (error) {
            console.error('Translation error:', error);
            alert('An error occurred during translation.');
        } finally {
            setLoading(false);
        }
    });

    // Handle clear button
    clearBtn.addEventListener('click', function() {
        sourceText.value = '';
        targetText.textContent = '';
        attentionText.textContent = '';
        if (document.getElementById('attentionPlot').data) {
            Plotly.purge('attentionPlot');
        }
    });

    // Add keyboard shortcut (Ctrl/Cmd + Enter) for translation
    sourceText.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            translateBtn.click();
        }
    });

    // Initial setup
    sourceText.focus();
});
