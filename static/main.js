const app = Vue.createApp({
    data() {
        return {
            inputText: '',
            result: null,
            explanation: null,
            loading: false,
            error: null,
            metrics: null,
            recentErrors: [],
            uploadedFiles: [],
            uploadStatus: null,
            selectedFiles: []
        }
    },
    methods: {
        async runInference() {
            if (!this.inputText.trim()) {
                this.error = 'Please enter some text first';
                return;
            }
            
            this.loading = true;
            this.error = null;
            this.result = null;
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: this.inputText
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }
                
                this.result = await response.json();
                
            } catch (err) {
                this.error = err.message;
                console.error('Analysis error:', err);
            } finally {
                this.loading = false;
            }
        },
        
        async getExplanation() {
            this.loading = true;
            this.error = null;
            try {
                const response = await fetch('/api/explain', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: this.inputText
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Explanation failed');
                }
                const data = await response.json();
                if (data.status === 'success') {
                    this.explanation = data.explanation;
                } else {
                    throw new Error('Explanation failed');
                }
            } catch (err) {
                this.error = err.message;
            } finally {
                this.loading = false;
            }
        },
        
        async uploadFiles(event) {
            const files = event.target.files;
            if (!files || files.length === 0) {
                this.error = 'Please select files first';
                return;
            }
            
            const formData = new FormData();
            formData.append('file', files[0]); // We only support single file upload for now
            
            this.loading = true;
            this.error = null;
            this.uploadStatus = 'Uploading...';
            
            try {
                const response = await fetch('/api/analyze-file', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Upload failed');
                }
                
                const result = await response.json();
                this.result = result;
                this.uploadStatus = 'File analyzed successfully';
                this.selectedFiles = Array.from(files);
                
            } catch (err) {
                this.error = err.message;
                this.uploadStatus = 'Analysis failed';
                console.error('Upload error:', err);
            } finally {
                this.loading = false;
                // Reset file input
                event.target.value = '';
            }
        },
        
        async fetchMetrics() {
            try {
                const response = await fetch('/api/metrics');
                if (!response.ok) throw new Error('Failed to fetch metrics');
                this.metrics = await response.json();
            } catch (err) {
                console.error('Error fetching metrics:', err);
            }
        },
        
        async fetchErrors() {
            try {
                const response = await fetch('/api/errors');
                if (!response.ok) throw new Error('Failed to fetch errors');
                this.recentErrors = await response.json();
            } catch (err) {
                console.error('Error fetching errors:', err);
            }
        }
    },
    mounted() {
        // Fetch initial data
        this.fetchMetrics();
        this.fetchErrors();
        
        // Set up polling for metrics and errors
        setInterval(() => {
            this.fetchMetrics();
            this.fetchErrors();
        }, 5000);
    }
}).mount('#app');
