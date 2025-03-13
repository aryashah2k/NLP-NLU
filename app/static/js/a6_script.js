document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const sourcesContent = document.getElementById('sources-content');
    const sourcesPanel = document.getElementById('sources-panel');
    
    // Analysis Elements
    const analysisInput = document.getElementById('analysis-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analysisLoading = document.getElementById('analysis-loading');
    const analysisContent = document.getElementById('analysis-content');
    
    // Initialize chat history
    loadChatHistory();
    
    // Send message when button is clicked
    sendBtn.addEventListener('click', sendMessage);
    
    // Send message when Enter key is pressed
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Analyze button functionality
    analyzeBtn.addEventListener('click', analyzeQuestion);
    
    // Analyze when Enter key is pressed in analysis input
    analysisInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeQuestion();
        }
    });
    
    // Function to send message
    function sendMessage() {
        const message = userInput.value.trim();
        if (message === '') return;
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        userInput.value = '';
        
        // Show loading indicator
        addMessageToChat('system', 'Thinking...', 'loading-message');
        
        // Send message to server
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            const loadingMessage = document.querySelector('.loading-message');
            if (loadingMessage) {
                chatMessages.removeChild(loadingMessage.parentNode);
            }
            
            // Add bot response to chat
            addMessageToChat('system', data.answer);
            
            // Update sources
            updateSources(data.sources);
            
            // Scroll to bottom of chat
            chatMessages.scrollTop = chatMessages.scrollHeight;
        })
        .catch(error => {
            console.error('Error:', error);
            // Remove loading message
            const loadingMessage = document.querySelector('.loading-message');
            if (loadingMessage) {
                chatMessages.removeChild(loadingMessage.parentNode);
            }
            
            // Add error message
            addMessageToChat('system', 'Sorry, there was an error processing your request.');
        });
    }
    
    // Function to add message to chat
    function addMessageToChat(sender, content, className = '') {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = `message-content ${className}`;
        
        const paragraph = document.createElement('p');
        paragraph.textContent = content;
        
        contentDiv.appendChild(paragraph);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to update sources
    function updateSources(sources) {
        sourcesContent.innerHTML = '';
        
        if (!sources || sources.length === 0) {
            const noSourcesP = document.createElement('p');
            noSourcesP.className = 'no-sources';
            noSourcesP.textContent = 'No sources available for this response.';
            sourcesContent.appendChild(noSourcesP);
            return;
        }
        
        sources.forEach(source => {
            const sourceDiv = document.createElement('div');
            sourceDiv.className = 'source-item';
            
            const sourceTitle = document.createElement('h5');
            sourceTitle.textContent = source.source || 'Unknown Source';
            
            const sourceContent = document.createElement('p');
            sourceContent.textContent = source.content || 'No content available';
            
            sourceDiv.appendChild(sourceTitle);
            sourceDiv.appendChild(sourceContent);
            sourcesContent.appendChild(sourceDiv);
        });
    }
    
    // Function to load chat history
    function loadChatHistory() {
        fetch('/qa_history')
        .then(response => response.json())
        .then(data => {
            if (data && data.length > 0) {
                data.forEach(qa => {
                    addMessageToChat('user', qa.question);
                    addMessageToChat('system', qa.answer);
                });
                
                // Scroll to bottom of chat
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        })
        .catch(error => {
            console.error('Error loading chat history:', error);
        });
    }
    
    // Function to analyze a question
    function analyzeQuestion() {
        const question = analysisInput.value.trim();
        if (question === '') return;
        
        // Show loading indicator
        analysisLoading.style.display = 'block';
        analysisContent.innerHTML = '';
        
        // Send question to server for analysis
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading indicator
            analysisLoading.style.display = 'none';
            
            // Create analysis results HTML
            const resultsHTML = `
                <div class="analysis-result">
                    <h4>Analysis Results for: "${question}"</h4>
                    
                    <div class="result-section">
                        <h5>Retriever Model Results</h5>
                        <p><strong>Top Retrieved Documents:</strong></p>
                        <ul class="retrieved-docs">
                            ${data.retriever_results.map(doc => `
                                <li>
                                    <p><strong>Source:</strong> ${doc.source.split('\\').pop()}</p>
                                    <p><strong>Content:</strong> ${doc.content}</p>
                                    <p><strong>Relevance Score:</strong> ${(1 / (1 + doc.score)).toFixed(2)}</p>
                                </li>
                            `).join('')}
                        </ul>
                        <p><strong>Retrieval Quality:</strong> ${assessRetrievalQuality(data.retriever_results, question)}</p>
                    </div>
                    
                    <div class="result-section">
                        <h5>Generator Model Results</h5>
                        <p><strong>Generated Answer:</strong> ${data.generation_result.answer}</p>
                        <p><strong>Generation Quality:</strong> ${assessGenerationQuality(data.generation_result.answer, data.retriever_results)}</p>
                    </div>
                </div>
            `;
            
            // Update analysis content
            analysisContent.innerHTML = resultsHTML;
        })
        .catch(error => {
            console.error('Error:', error);
            // Hide loading indicator
            analysisLoading.style.display = 'none';
            
            // Show error message
            analysisContent.innerHTML = '<p class="error">Error analyzing question. Please try again.</p>';
        });
    }
    
    // Helper function to calculate relevance score (simple keyword matching)
    function calculateRelevanceScore(content, question) {
        if (!content || !question) return 0;
        
        const contentLower = content.toLowerCase();
        const questionWords = question.toLowerCase().split(/\s+/).filter(word => word.length > 3);
        
        let matchCount = 0;
        questionWords.forEach(word => {
            if (contentLower.includes(word)) {
                matchCount++;
            }
        });
        
        return questionWords.length > 0 ? matchCount / questionWords.length : 0;
    }
    
    // Helper function to assess retrieval quality
    function assessRetrievalQuality(results, question) {
        if (!results || results.length === 0) return 'Poor - No documents retrieved';
        
        // FAISS returns L2 distance, so lower is better
        // Convert to similarity score (1 / (1 + distance))
        const relevantDocs = results.filter(doc => {
            const similarityScore = 1 / (1 + doc.score);
            return similarityScore > 0.4; // Threshold for relevance
        });
        
        if (relevantDocs.length === 0) return 'Poor - No relevant documents found';
        if (relevantDocs.length < results.length / 2) return 'Fair - Some relevant documents found';
        return 'Good - Most retrieved documents are relevant';
    }
    
    // Helper function to assess generation quality
    function assessGenerationQuality(answer, retrievedDocs) {
        if (!answer || answer.trim() === '') return 'Poor - No answer generated';
        
        // Check if answer is too short
        if (answer.length < 20) return 'Poor - Answer is too short';
        
        // Check if answer is relevant to retrieved documents
        const relevantCount = retrievedDocs.filter(doc => 
            doc.content.toLowerCase().includes(answer.toLowerCase().substring(0, 15)) || 
            answer.toLowerCase().includes(doc.content.toLowerCase().substring(0, 15))
        ).length;
        
        if (relevantCount === 0) return 'Fair - Answer may not be based on retrieved documents';
        if (relevantCount < retrievedDocs.length / 2) return 'Good - Answer is somewhat based on retrieved documents';
        return 'Excellent - Answer is well-grounded in retrieved documents';
    }
});
