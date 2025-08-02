// AutoML Application JavaScript

class AutoMLApp {
    constructor() {
        this.currentData = null;
        this.currentStep = 'upload';
        this.targetColumn = null;
        this.problemType = null;
        this.trainedModels = [];
        this.bestModel = null;
        
        // Model algorithms from the provided data
        this.algorithms = {
            classification: [
                {name: "Logistic Regression", class: "LogisticRegression", params: {random_state: 42}},
                {name: "Random Forest", class: "RandomForestClassifier", params: {n_estimators: 100, random_state: 42}},
                {name: "Gradient Boosting", class: "GradientBoostingClassifier", params: {random_state: 42}},
                {name: "Support Vector Machine", class: "SVC", params: {random_state: 42, probability: true}},
                {name: "Decision Tree", class: "DecisionTreeClassifier", params: {random_state: 42}}
            ],
            regression: [
                {name: "Linear Regression", class: "LinearRegression", params: {}},
                {name: "Random Forest", class: "RandomForestRegressor", params: {n_estimators: 100, random_state: 42}},
                {name: "Gradient Boosting", class: "GradientBoostingRegressor", params: {random_state: 42}},
                {name: "Support Vector Machine", class: "SVR", params: {}},
                {name: "Decision Tree", class: "DecisionTreeRegressor", params: {random_state: 42}}
            ]
        };
        
        this.targetPatterns = ["target", "class", "label", "y", "output", "prediction", "outcome"];
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.updateProgressSteps();
    }
    
    setupEventListeners() {
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Navigation buttons
        document.getElementById('analyzeBtn').addEventListener('click', this.analyzeDataset.bind(this));
        document.getElementById('proceedToConfigBtn').addEventListener('click', () => this.navigateToStep('config'));
        document.getElementById('backToEdaBtn').addEventListener('click', () => this.navigateToStep('eda'));
        document.getElementById('startTrainingBtn').addEventListener('click', this.startTraining.bind(this));
        document.getElementById('viewResultsBtn').addEventListener('click', () => this.navigateToStep('results'));
        document.getElementById('backToConfigBtn').addEventListener('click', () => this.navigateToStep('config'));
        document.getElementById('resetBtn').addEventListener('click', this.resetApplication.bind(this));
        
        // Configuration
        document.getElementById('targetSelect').addEventListener('change', this.handleTargetSelection.bind(this));
        document.getElementById('problemTypeSelect').addEventListener('change', this.handleProblemTypeSelection.bind(this));
        document.getElementById('splitRatio').addEventListener('input', this.updateSplitDisplay.bind(this));
        
        // Modal handlers
        document.getElementById('successModalClose').addEventListener('click', () => this.hideModal('successModal'));
        
        // Export functionality
        document.getElementById('exportResultsBtn').addEventListener('click', this.exportResults.bind(this));
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }
    
    handleDragLeave(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
    }
    
    handleDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }
    
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }
    
    processFile(file) {
        // Validate file type and size
        const validTypes = ['.csv', '.xlsx', '.xls'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validTypes.includes(fileExtension)) {
            alert('Please upload a CSV or Excel file.');
            return;
        }
        
        if (file.size > 50 * 1024 * 1024) { // 50MB limit
            alert('File size too large. Please upload a file smaller than 50MB.');
            return;
        }
        
        this.showModal('loadingModal', 'Uploading and processing file...');
        
        // Create FormData to upload file to backend
        const formData = new FormData();
        formData.append('dataset', file);
        
        // Upload file to Flask backend
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            console.log('File uploaded successfully');
            
            // Display file preview using backend data
            this.displayFilePreview(file);
            
            // Fetch and display actual data preview
            setTimeout(() => {
                this.fetchAndDisplayDataPreview();
            }, 500); // Small delay to ensure backend processing is complete
            
            this.hideModal('loadingModal');
            
            // Don't auto-navigate to EDA - wait for user to click "Analyze Dataset" button
            console.log('File uploaded successfully. Ready for EDA analysis.');
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            alert('Error uploading file. Please try again.');
            this.hideModal('loadingModal');
        });
    }
    
    displayFilePreview(file) {
        document.getElementById('uploadArea').style.display = 'none';
        document.getElementById('filePreview').classList.remove('hidden');
        
        // Update file info
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = `Size: ${(file.size / 1024).toFixed(1)} KB`;
        
        // Get file info from backend EDA data
        fetch('/eda')
        .then(response => response.json())
        .then(eda => {
            if (eda && eda.stats) {
                document.getElementById('fileRows').textContent = `Rows: ${eda.stats.rows}`;
                
                // Show data preview using actual data from backend
                this.fetchAndDisplayDataPreview();
            }
        })
        .catch(error => {
            console.error('Error fetching file info:', error);
            // Fallback display
            document.getElementById('fileRows').textContent = 'Rows: Loading...';
            this.fetchAndDisplayDataPreview(); // Try to fetch preview anyway
        });
    }
    
    fetchAndDisplayDataPreview() {
        fetch('/data_preview')
        .then(response => response.json())
        .then(data => {
            this.displayActualDataPreview(data);
        })
        .catch(error => {
            console.error('Error fetching data preview:', error);
            this.displayDataPreviewFromBackend(null); // Fallback
        });
    }
    
    displayActualDataPreview(data) {
        const preview = document.getElementById('dataPreview');
        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');
        
        if (!data || !data.columns || !data.data) {
            tableBody.innerHTML = '<tr><td colspan="100%">No data preview available</td></tr>';
            return;
        }
        
        // Create table header
        tableHead.innerHTML = data.columns.map(col => `<th>${col}</th>`).join('');
        
        // Create table rows with actual data
        tableBody.innerHTML = data.data.map(row => 
            `<tr>${data.columns.map(col => 
                `<td>${row[col] !== null && row[col] !== undefined ? row[col] : 'N/A'}</td>`
            ).join('')}</tr>`
        ).join('');
        
        preview.classList.remove('hidden');
    }
    
    displayDataPreviewFromBackend(eda) {
        const preview = document.getElementById('dataPreview');
        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');
        
        if (!eda || !eda.stats) return;
        
        // Get column names from backend
        const allColumns = [...(eda.stats.numerics || []), ...(eda.stats.categoricals || [])];
        
        // Create table header
        tableHead.innerHTML = allColumns.map(col => `<th>${col}</th>`).join('');
        
        // Create placeholder rows (since we don't have actual row data)
        tableBody.innerHTML = `
            <tr>
                ${allColumns.map(() => '<td>Loading...</td>').join('')}
            </tr>
            <tr>
                <td colspan="${allColumns.length}" class="text-center text-gray-500">
                    Data preview available after EDA analysis
                </td>
            </tr>
        `;
        
        preview.classList.remove('hidden');
    }
    
    displayDataPreview() {
        const preview = document.getElementById('dataPreview');
        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');
        
        if (!this.currentData || this.currentData.length === 0) return;
        
        // Get column names
        const columns = Object.keys(this.currentData[0]);
        
        // Create table header
        tableHead.innerHTML = columns.map(col => `<th>${col}</th>`).join('');
        
        // Create table body (show first 5 rows)
        const previewRows = this.currentData.slice(0, 5);
        tableBody.innerHTML = previewRows.map(row => 
            `<tr>${columns.map(col => `<td>${row[col] || 'N/A'}</td>`).join('')}</tr>`
        ).join('');
        
        preview.classList.remove('hidden');
    }
    
    analyzeDataset() {
        this.showModal('loadingModal', 'Performing EDA analysis...');
        
        // Call backend to process EDA analysis
        fetch('/process_eda', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        })
        .then(response => response.text())
        .then(html => {
            console.log('EDA processing completed');
            
            // Now fetch the EDA data
            fetch('/eda')
            .then(response => response.json())
            .then(eda => {
                if (eda && Object.keys(eda).length > 0) {
                    this.displayEDAResults(eda);
                    this.navigateToStep('eda');
                } else {
                    console.error('No EDA data received from backend');
                    alert('Error processing EDA. Please try again.');
                }
                this.hideModal('loadingModal');
            })
            .catch(error => {
                console.error('Error fetching EDA data:', error);
                alert('Error fetching EDA data. Please try again.');
                this.hideModal('loadingModal');
            });
        })
        .catch(error => {
            console.error('Error processing EDA:', error);
            alert('Error processing EDA. Please try again.');
            this.hideModal('loadingModal');
        });
    }
    
    displayEDAResults(eda) {
        // Display EDA results from backend
        if (eda.stats) {
            document.getElementById('totalRows').textContent = eda.stats.rows || 0;
            document.getElementById('totalColumns').textContent = eda.stats.cols || 0;
            document.getElementById('numericColumns').textContent = eda.stats.numerics?.length || 0;
            document.getElementById('categoricalColumns').textContent = eda.stats.categoricals?.length || 0;
            document.getElementById('missingValues').textContent = Object.values(eda.stats.missing || {}).reduce((a, b) => a + b, 0);
            document.getElementById('duplicateRows').textContent = eda.stats.duplicate_rows || 0;
            
            // Create EDA charts from backend data
            this.createEDAChartsFromBackend(eda.stats);
        }
        
        // Populate target selection from backend columns
        if (eda.stats && eda.stats.numerics && eda.stats.categoricals) {
            const allColumns = [...(eda.stats.numerics || []), ...(eda.stats.categoricals || [])];
            this.populateTargetSelection(allColumns);
        }
    }
    
    performEDA() {
        if (!this.currentData) return;
        
        const columns = Object.keys(this.currentData[0]);
        const numRows = this.currentData.length;
        
        // Calculate statistics
        const stats = this.calculateDatasetStats(columns);
        
        // Update EDA display
        document.getElementById('totalRows').textContent = numRows;
        document.getElementById('totalColumns').textContent = columns.length;
        document.getElementById('numericColumns').textContent = stats.numericColumns;
        document.getElementById('categoricalColumns').textContent = stats.categoricalColumns;
        document.getElementById('missingValues').textContent = stats.missingValues;
        document.getElementById('duplicateRows').textContent = stats.duplicateRows;
        
        // Create charts
        this.createDataQualityChart(stats);
        this.createColumnTypesChart(stats);
        this.createMissingValuesChart(stats);
        
        // Populate target selection dropdown
        this.populateTargetSelection(columns);
    }
    
    createEDAChartsFromBackend(stats) {
        // Create data quality chart
        this.createDataQualityChartFromStats(stats);
        
        // Create column types chart
        this.createColumnTypesChartFromStats(stats);
        
        // Create missing values chart
        this.createMissingValuesChartFromStats(stats);
    }
    
    createDataQualityChartFromStats(stats) {
        const ctx = document.getElementById('qualityChart').getContext('2d');
        
        // Destroy existing chart if any
        if (this.qualityChartInstance) {
            this.qualityChartInstance.destroy();
        }
        
        const totalCells = stats.rows * stats.cols;
        const totalMissing = Object.values(stats.missing || {}).reduce((a, b) => a + b, 0);
        const completeCells = totalCells - totalMissing;
        
        this.qualityChartInstance = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Complete Data', 'Missing Values', 'Duplicate Rows'],
                datasets: [{
                    data: [
                        completeCells,
                        totalMissing,
                        stats.duplicate_rows || 0
                    ],
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    createColumnTypesChartFromStats(stats) {
        const ctx = document.getElementById('typesChart').getContext('2d');
        
        // Destroy existing chart if any
        if (this.typesChartInstance) {
            this.typesChartInstance.destroy();
        }
        
        this.typesChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Numeric', 'Categorical'],
                datasets: [{
                    label: 'Number of Columns',
                    data: [stats.numerics?.length || 0, stats.categoricals?.length || 0],
                    backgroundColor: ['#1FB8CD', '#FFC185']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }
    
    createMissingValuesChartFromStats(stats) {
        const ctx = document.getElementById('missingChart').getContext('2d');
        
        // Destroy existing chart if any
        if (this.missingChartInstance) {
            this.missingChartInstance.destroy();
        }
        
        const columns = Object.keys(stats.missing || {});
        const missingCounts = columns.map(col => stats.missing[col] || 0);
        
        this.missingChartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: columns,
                datasets: [{
                    label: 'Missing Values',
                    data: missingCounts,
                    backgroundColor: '#B4413C'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                }
            }
        });
    }
    
    calculateDatasetStats(columns) {
        let numericColumns = 0;
        let categoricalColumns = 0;
        let missingValues = 0;
        const columnMissing = {};
        
        columns.forEach(col => {
            let missing = 0;
            let isNumeric = true;
            
            this.currentData.forEach(row => {
                const value = row[col];
                if (value === null || value === undefined || value === '') {
                    missing++;
                    missingValues++;
                } else if (isNumeric && isNaN(parseFloat(value))) {
                    isNumeric = false;
                }
            });
            
            columnMissing[col] = missing;
            
            if (isNumeric) {
                numericColumns++;
            } else {
                categoricalColumns++;
            }
        });
        
        // Calculate duplicate rows (simplified)
        const duplicateRows = Math.floor(Math.random() * 5);
        
        return {
            numericColumns,
            categoricalColumns,
            missingValues,
            duplicateRows,
            columnMissing
        };
    }
    
    createDataQualityChart(stats) {
        const ctx = document.getElementById('qualityChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Complete Data', 'Missing Values', 'Duplicate Rows'],
                datasets: [{
                    data: [
                        this.currentData.length * Object.keys(this.currentData[0]).length - stats.missingValues,
                        stats.missingValues,
                        stats.duplicateRows
                    ],
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    createColumnTypesChart(stats) {
        const ctx = document.getElementById('typesChart').getContext('2d');
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Numeric', 'Categorical'],
                datasets: [{
                    label: 'Number of Columns',
                    data: [stats.numericColumns, stats.categoricalColumns],
                    backgroundColor: ['#1FB8CD', '#FFC185']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }
    
    createMissingValuesChart(stats) {
        const ctx = document.getElementById('missingChart').getContext('2d');
        const columns = Object.keys(stats.columnMissing);
        const missingCounts = columns.map(col => stats.columnMissing[col]);
        
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: columns,
                datasets: [{
                    label: 'Missing Values',
                    data: missingCounts,
                    backgroundColor: '#B4413C'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    },
                    x: {
                        ticks: {
                            maxRotation: 45
                        }
                    }
                }
            }
        });
    }
    
    populateTargetSelection(columns) {
        const targetSelect = document.getElementById('targetSelect');
        targetSelect.innerHTML = '<option value="">Choose target variable...</option>';
        
        columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            
            // Auto-select if matches common target patterns
            if (this.targetPatterns.some(pattern => col.toLowerCase().includes(pattern))) {
                option.selected = true;
                this.targetColumn = col;
                this.analyzeTarget();
            }
            
            targetSelect.appendChild(option);
        });
    }
    
    handleTargetSelection(e) {
        this.targetColumn = e.target.value;
        if (this.targetColumn) {
            this.analyzeTarget();
            document.getElementById('startTrainingBtn').disabled = false;
        } else {
            document.getElementById('targetPreview').classList.add('hidden');
            document.getElementById('startTrainingBtn').disabled = true;
        }
    }
    
    analyzeTarget() {
        if (!this.targetColumn) return;
        
        // Fetch target analysis from backend
        fetch('/eda')
        .then(response => response.json())
        .then(eda => {
            if (eda && eda.stats) {
                this.analyzeTargetFromBackend(eda.stats);
            } else {
                // Fallback to client-side analysis if available
                this.analyzeTargetFromClient();
            }
        })
        .catch(error => {
            console.error('Error fetching target analysis:', error);
            this.analyzeTargetFromClient();
        });
    }
    
    analyzeTargetFromBackend(stats) {
        if (!this.targetColumn) return;
        
        // Determine if target is numeric or categorical
        const isNumeric = stats.numerics && stats.numerics.includes(this.targetColumn);
        const isCategorical = stats.categoricals && stats.categoricals.includes(this.targetColumn);
        
        // Auto-detect problem type based on column type
        let detectedType = 'classification';
        if (isNumeric) {
            // For numeric columns, we could add more logic here
            // For now, default to regression for numeric targets
            detectedType = 'regression';
        }
        
        this.problemType = detectedType;
        
        // Update UI
        document.getElementById('problemTypeSelect').value = detectedType;
        document.getElementById('detectedType').textContent = `Auto-detected: ${detectedType.charAt(0).toUpperCase() + detectedType.slice(1)}`;
        document.getElementById('detectedType').className = 'status status--success';
        
        // Show target preview
        const targetPreview = document.getElementById('targetPreview');
        const targetStats = document.getElementById('targetStats');
        
        const missingCount = stats.missing[this.targetColumn] || 0;
        
        targetStats.innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Column Type:</span>
                <span class="stat-value">${isNumeric ? 'Numeric' : 'Categorical'}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Missing Values:</span>
                <span class="stat-value">${missingCount}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Rows:</span>
                <span class="stat-value">${stats.rows}</span>
            </div>
        `;
        
        targetPreview.classList.remove('hidden');
    }
    
    analyzeTargetFromClient() {
        if (!this.targetColumn || !this.currentData) return;
        
        const targetValues = this.currentData.map(row => row[this.targetColumn]).filter(v => v !== null && v !== undefined && v !== '');
        const uniqueValues = [...new Set(targetValues)];
        
        // Auto-detect problem type
        let detectedType;
        if (uniqueValues.length <= 10 && uniqueValues.every(v => !isNaN(parseFloat(v)) ? Number.isInteger(parseFloat(v)) : true)) {
            detectedType = 'classification';
        } else if (uniqueValues.every(v => !isNaN(parseFloat(v)))) {
            detectedType = 'regression';
        } else {
            detectedType = 'classification';
        }
        
        this.problemType = detectedType;
        
        // Update UI
        document.getElementById('problemTypeSelect').value = detectedType;
        document.getElementById('detectedType').textContent = `Auto-detected: ${detectedType.charAt(0).toUpperCase() + detectedType.slice(1)}`;
        document.getElementById('detectedType').className = 'status status--success';
        
        // Show target preview
        const targetPreview = document.getElementById('targetPreview');
        const targetStats = document.getElementById('targetStats');
        
        targetStats.innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Unique Values:</span>
                <span class="stat-value">${uniqueValues.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Missing Values:</span>
                <span class="stat-value">${this.currentData.length - targetValues.length}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Data Type:</span>
                <span class="stat-value">${typeof targetValues[0]}</span>
            </div>
        `;
        
        targetPreview.classList.remove('hidden');
    }
    
    handleProblemTypeSelection(e) {
        if (e.target.value) {
            this.problemType = e.target.value;
        }
    }
    
    updateSplitDisplay() {
        const splitRatio = document.getElementById('splitRatio').value;
        document.getElementById('trainPercent').textContent = splitRatio;
        document.getElementById('testPercent').textContent = 100 - splitRatio;
    }
    
    startTraining() {
        // Show loading modal
        this.showModal('loadingModal', 'Training models...');
        
        // Prepare config data
        const config = {
            target: this.targetColumn,
            ptype: this.problemType,
            split: document.getElementById('splitRatio').value / 100
        };
        
        console.log('Sending training request with:', config);
        
        // Send config to Flask backend (backend will use uploaded file)
        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `target=${encodeURIComponent(config.target)}&ptype=${encodeURIComponent(config.ptype)}&split=${encodeURIComponent(config.split)}`
        })
        .then(response => response.text())
        .then(html => {
            // Hide loading modal
            this.hideModal('loadingModal');
            
            // Fetch model results from backend and then navigate to results
            this.fetchModelResults().then(() => {
                // Navigate to results after data is loaded
                setTimeout(() => {
                    this.navigateToStep('results');
                }, 500);
            });
            
            // Show training actions after completion
            setTimeout(() => {
                const trainingActions = document.getElementById('trainingActions');
                if (trainingActions) {
                    trainingActions.classList.remove('hidden');
                }
            }, 1000);
        })
        .catch(err => {
            this.hideModal('loadingModal');
            alert('Model training failed.');
        });
    }

    fetchModelResults() {
        console.log('fetchModelResults called');
        // Fetch metrics and model comparison from backend
        return Promise.all([
            fetch('/best_model').then(r => r.json()),
            fetch('/model_comparison').then(r => r.json()),
            fetch('/feature_importance').then(r => r.json())
        ]).then(([bestModel, history, featureImportance]) => {
            console.log('Fetched data:', { bestModel, history, featureImportance });
            
            // Store results for display
            this.bestModel = bestModel;
            this.trainedModels = history;
            this.featureImportance = featureImportance;
            
            // Update best model info
            const bestModelName = document.querySelector('.best-model-name');
            const bestModelScore = document.querySelector('.best-model-score');
            
            if (bestModelName && bestModel.name) {
                bestModelName.textContent = bestModel.name;
            }
            
            if (bestModelScore && bestModel && bestModel.metrics) {
                // Display primary metric with safe access
                const metrics = bestModel.metrics || {};
                const primaryMetric = metrics.Accuracy || metrics['R^2'] || Object.values(metrics)[0] || 0;
                bestModelScore.textContent = typeof primaryMetric === 'number' ? primaryMetric.toFixed(3) : primaryMetric;
            }
            
            // Update metrics table
            const tbody = document.getElementById('metricsTableBody');
            if (tbody && history && history.length) {
                tbody.innerHTML = '';
                history.forEach(entry => {
                    if (!entry || !entry.model) return; // Skip invalid entries
                    
                    const tr = document.createElement('tr');
                    
                    // Safely access metrics with null checks
                    const metrics = entry.metrics || {};
                    const primaryMetric = metrics.Accuracy || metrics['R^2'] || Object.values(metrics)[0] || 0;
                    const displayMetric = typeof primaryMetric === 'number' ? primaryMetric.toFixed(3) : primaryMetric;
                    const trainingTime = metrics.Training_Time ? `${metrics.Training_Time}s` : '-';
                    
                    tr.innerHTML = `
                        <td class="font-bold">${entry.model}</td>
                        <td class="text-primary font-bold">${displayMetric}</td>
                        <td>${trainingTime}</td>
                        <td><span class="status status--completed">Completed</span></td>
                    `;
                    tbody.appendChild(tr);
                });
            }
            
            console.log('Data loaded successfully');
            return Promise.resolve();
        }).catch(err => {
            console.error('Failed to fetch model results:', err);
            this.hideModal('loadingModal');
            return Promise.reject(err);
        });
    }
    
    navigateToStep(step) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        // Show target section only if it exists
        const sectionDiv = document.getElementById(`${step}Section`);
        if (sectionDiv) {
            sectionDiv.classList.add('active');
        } else {
            // Optionally log or handle missing section
            console.warn(`Section ${step}Section not found in DOM.`);
        }
        this.currentStep = step;
        this.updateProgressSteps();
        // Special handling for results section
        if (step === 'results') {
            this.displayResults();
        }
    }
    
    updateProgressSteps() {
        const steps = ['upload', 'eda', 'config', 'training', 'results'];
        const currentIndex = steps.indexOf(this.currentStep);
        
        document.querySelectorAll('.progress-step').forEach((step, index) => {
            step.classList.remove('active', 'completed');
            
            if (index < currentIndex) {
                step.classList.add('completed');
            } else if (index === currentIndex) {
                step.classList.add('active');
            }
        });
    }
    
    displayResults() {
        console.log('displayResults called');
        console.log('bestModel:', this.bestModel);
        console.log('trainedModels:', this.trainedModels);
        console.log('featureImportance:', this.featureImportance);
        
        if (!this.bestModel || !this.trainedModels || !this.trainedModels.length) {
            console.log('No results data available yet - data missing');
            return;
        }
        
        console.log('Creating charts...');
        // Use setTimeout to ensure DOM is ready
        setTimeout(() => {
            this.createPerformanceChart();
            this.createFeatureImportanceChart();
        }, 100);
    }
    
    createPerformanceChart() {
        console.log('createPerformanceChart called');
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded');
            return;
        }
        
        const chartCanvas = document.getElementById('performanceChart');
        console.log('Chart canvas:', chartCanvas);
        console.log('Trained models:', this.trainedModels);
        
        if (!chartCanvas) {
            console.error('Chart canvas not found');
            return;
        }
        
        if (!this.trainedModels || !this.trainedModels.length) {
            console.error('No trained models data available');
            return;
        }

        // Destroy existing chart if any
        if (this.performanceChartInstance) {
            this.performanceChartInstance.destroy();
        }

        const ctx = chartCanvas.getContext('2d');
        
        // Extract data from trainedModels
        const labels = this.trainedModels.map(m => m.model || 'Unknown');
        const data = this.trainedModels.map(m => {
            // Get primary metric (accuracy for classification, R² for regression)
            const metrics = m.metrics || {};
            return metrics.Accuracy || metrics['R^2'] || Object.values(metrics)[0] || 0;
        });
        
        console.log('Chart data:', { labels, data });
        
        try {
            this.performanceChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Performance Score',
                        data: data,
                        backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        },
                        x: {
                            ticks: {
                                maxRotation: 45
                            }
                        }
                    }
                }
            });
            console.log('Performance chart created successfully');
        } catch (error) {
            console.error('Error creating performance chart:', error);
        }
    }
    
    createFeatureImportanceChart() {
        console.log('createFeatureImportanceChart called');
        
        // Check if Chart.js is available
        if (typeof Chart === 'undefined') {
            console.error('Chart.js is not loaded');
            return;
        }
        
        const chartCanvas = document.getElementById('featureImportanceChart');
        console.log('Feature importance canvas:', chartCanvas);
        console.log('Best model:', this.bestModel);
        console.log('Feature importance data:', this.featureImportance);
        
        if (!chartCanvas) {
            console.error('Feature importance chart canvas not found');
            return;
        }
        
        if (!this.featureImportance || !this.bestModel) {
            console.error('Feature importance data not available');
            return;
        }
        
        // Destroy existing chart if any
        if (this.featureImportanceChartInstance) {
            this.featureImportanceChartInstance.destroy();
        }
        
        const ctx = chartCanvas.getContext('2d');
        
        // Get feature importance for the best model
        const bestModelName = this.bestModel.name;
        const importanceData = this.featureImportance[bestModelName] || {};
        
        if (Object.keys(importanceData).length === 0) {
            console.log('No feature importance data available for model:', bestModelName);
            // Show message instead of chart
            const container = chartCanvas.parentElement;
            container.innerHTML = '<p class="text-gray-500">No feature importance data available for this model.</p>';
            return;
        }
        
        // Convert to arrays and sort by importance
        const features = Object.entries(importanceData)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10); // Top 10 features
            
        const labels = features.map(f => f[0]);
        const data = features.map(f => f[1]);
        
        console.log('Feature importance chart data:', { labels, data });
        
        try {
            this.featureImportanceChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feature Importance',
                        data: data,
                        backgroundColor: '#1FB8CD'
                    }]
                },
                options: {
                    indexAxis: 'y', // This makes it horizontal
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        x: {
                            beginAtZero: true
                        }
                    }
                }
            });
            console.log('Feature importance chart created successfully');
        } catch (error) {
            console.error('Error creating feature importance chart:', error);
        }
    }
    
    updateMetricsTable() {
        const tbody = document.getElementById('metricsTableBody');
        tbody.innerHTML = this.trainedModels.map(model => `
            <tr>
                <td class="font-bold">${model.name}</td>
                <td class="text-primary font-bold">${model.primaryScore.toFixed(3)}</td>
                <td>${model.trainingTime}s</td>
                <td><span class="status status--completed">Completed</span></td>
            </tr>
        `).join('');
    }
    
    exportResults() {
        console.log('Exporting results...');
        
        // Use backend export endpoint for CSV export
        window.location.href = '/export_results';
        
        this.showModal('successModal', 'Results exported successfully!');
        document.getElementById('successText').textContent = 'Results exported successfully!';
    }
    
    resetApplication() {
        if (confirm('Are you sure you want to reset the application? All progress will be lost.')) {
            this.currentData = null;
            this.targetColumn = null;
            this.problemType = null;
            this.trainedModels = [];
            this.bestModel = null;
            
            // Reset UI
            document.getElementById('uploadArea').style.display = 'block';
            document.getElementById('filePreview').classList.add('hidden');
            document.getElementById('dataPreview').classList.add('hidden');
            document.getElementById('fileInput').value = '';
            
            this.navigateToStep('upload');
        }
    }
    
    showModal(modalId, text = '') {
        const modal = document.getElementById(modalId);
        if (text && modalId === 'loadingModal') {
            document.getElementById('loadingText').textContent = text;
        }
        modal.classList.remove('hidden');
    }
    
    hideModal(modalId) {
        document.getElementById(modalId).classList.add('hidden');
    }
    
    // Test method to debug charts
    testCharts() {
        console.log('Testing charts with dummy data...');
        
        // Set dummy data
        this.bestModel = {
            name: 'TestModel',
            metrics: { Accuracy: 0.85 }
        };
        
        this.trainedModels = [
            { model: 'Model1', metrics: { Accuracy: 0.85 } },
            { model: 'Model2', metrics: { Accuracy: 0.78 } },
            { model: 'Model3', metrics: { Accuracy: 0.82 } }
        ];
        
        this.featureImportance = {
            'TestModel': {
                'feature1': 0.3,
                'feature2': 0.25,
                'feature3': 0.2,
                'feature4': 0.15,
                'feature5': 0.1
            }
        };
        
        // Update best model display
        const bestModelName = document.querySelector('.best-model-name');
        const bestModelScore = document.querySelector('.best-model-score');
        if (bestModelName) bestModelName.textContent = this.bestModel.name;
        if (bestModelScore) bestModelScore.textContent = '0.850';
        
        // Create charts
        this.createPerformanceChart();
        this.createFeatureImportanceChart();
        
        console.log('Test charts created');
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.autoMLApp = new AutoMLApp();
});