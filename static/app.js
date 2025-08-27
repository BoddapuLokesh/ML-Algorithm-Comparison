// AutoML Application UI logic
// - handles upload, EDA, configuration, training, and results rendering
// - communicates with Flask endpoints and draws charts/tables

class AutoMLApp {
    constructor() {
        this.currentData = null;
        this.currentStep = 'upload';
        this.targetColumn = null;
        this.problemType = null;
        this.trainedModels = [];
        this.bestModel = null;
        this.featureImportance = null;
        
        // Chart instances for cleanup
        this.qualityChartInstance = null;
        this.typesChartInstance = null;
        this.missingChartInstance = null;
        this.performanceChartInstance = null;
        this.featureImportanceChartInstance = null;
        
        // Model algorithms from the provided data
        this.algorithms = {
            classification: [
                {name: "Logistic Regression", class: "LogisticRegression", params: {random_state: 42}},
                {name: "Random Forest", class: "RandomForestClassifier", params: {n_estimators: 10, random_state: 42}},
                {name: "Gradient Boosting", class: "GradientBoostingClassifier", params: {random_state: 42}},
                {name: "Support Vector Machine", class: "SVC", params: {random_state: 42, probability: true}},
                {name: "Decision Tree", class: "DecisionTreeClassifier", params: {random_state: 42}}
            ],
            regression: [
                {name: "Linear Regression", class: "LinearRegression", params: {}},
                {name: "Random Forest", class: "RandomForestRegressor", params: {n_estimators: 10, random_state: 42}},
                {name: "Gradient Boosting", class: "GradientBoostingRegressor", params: {random_state: 42}},
                {name: "Support Vector Machine", class: "SVR", params: {}},
                {name: "Decision Tree", class: "DecisionTreeRegressor", params: {random_state: 42}}
            ]
        };
        
        this.targetPatterns = ["target", "class", "label", "y", "output", "prediction", "outcome"];
        
        this.init();
    }
    
    // Utility method to escape HTML to prevent XSS
    escapeHtml(text) {
        if (text === null || text === undefined) return 'N/A';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    init() {
        this.setupEventListeners();
        this.updateProgressSteps();
    }
    
    setupEventListeners() {
        // File upload - with null checks
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        if (!uploadArea || !fileInput) {
            console.error('Critical upload elements not found');
            return;
        }
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Navigation buttons - with null checks
        const navButtons = [
            { id: 'analyzeBtn', handler: this.analyzeDataset.bind(this) },
            { id: 'proceedToConfigBtn', handler: () => this.navigateToStep('config') },
            { id: 'backToEdaBtn', handler: () => this.navigateToStep('eda') },
            { id: 'startTrainingBtn', handler: this.startTraining.bind(this) },
            { id: 'viewResultsBtn', handler: () => this.navigateToStep('results') },
            { id: 'backToConfigBtn', handler: () => this.navigateToStep('config') },
            { id: 'resetBtn', handler: this.resetApplication.bind(this) },
            { id: 'downloadModelBtn', handler: this.downloadModel.bind(this) },
            { id: 'exportResultsBtn', handler: this.exportResults.bind(this) }
        ];
        
        navButtons.forEach(({ id, handler }) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('click', handler);
            } else {
                console.warn(`Navigation button ${id} not found`);
            }
        });
        
        // Configuration - with null checks
        const targetSelect = document.getElementById('targetSelect');
        const problemTypeSelect = document.getElementById('problemTypeSelect');
        const splitRatio = document.getElementById('splitRatio');
        const successModalClose = document.getElementById('successModalClose');
        
        if (targetSelect) targetSelect.addEventListener('change', this.handleTargetSelection.bind(this));
        if (problemTypeSelect) problemTypeSelect.addEventListener('change', this.handleProblemTypeSelection.bind(this));
        if (splitRatio) splitRatio.addEventListener('input', this.updateSplitDisplay.bind(this));
        if (successModalClose) successModalClose.addEventListener('click', () => this.hideModal('successModal'));
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
        // Basic validation - detailed validation is now handled by Python backend
        if (!file) {
            alert('Please select a file.');
            return;
        }
        
        this.showModal('loadingModal', 'Uploading and processing file...');
        
        // Create FormData to upload file to backend
        const formData = new FormData();
        formData.append('dataset', file);
        
        // Upload file to Flask backend - backend now handles all validation
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                console.log('File uploaded successfully');
                
                // Display file preview using backend data
                this.displayFilePreview(file, result);
                
                // Display data preview using server-generated HTML for better security
                this.displayDataPreviewFromBackend({
                    columns: result.columns,
                    data: result.preview_data
                });
                
                // Populate target selection dropdown
                this.populateTargetSelection(result.columns);
                
                this.hideModal('loadingModal');
                
                console.log('File uploaded successfully. Ready for EDA analysis.');
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            alert('Error uploading file: ' + error.message);
            this.hideModal('loadingModal');
        });
    }
    
    displayFilePreview(file, result) {
        const uploadArea = document.getElementById('uploadArea');
        const filePreview = document.getElementById('filePreview');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileRows = document.getElementById('fileRows');
        
        if (!uploadArea || !filePreview || !fileName || !fileSize || !fileRows) {
            console.error('File preview elements not found');
            return;
        }
        
        uploadArea.style.display = 'none';
        filePreview.classList.remove('hidden');
        
        // Update file info
        fileName.textContent = file.name;
        fileSize.textContent = `Size: ${(file.size / 1024).toFixed(1)} KB`;
        
        if (result && result.file_shape) {
            fileRows.textContent = `Rows: ${result.file_shape[0]}, Columns: ${result.file_shape[1]}`;
        } else {
            fileRows.textContent = 'Rows: Loading...';
        }
    }
    
    fetchAndDisplayDataPreview() {
        fetch('/data_preview')
        .then(response => response.json())
        .then(data => {
            this.displayActualDataPreview(data);
        })
        .catch(error => {
            console.error('Error fetching data preview:', error);
            // Show error message in preview
            const tableBody = document.getElementById('tableBody');
            tableBody.innerHTML = '<tr><td colspan="100%">Error loading data preview</td></tr>';
        });
    }
    
    displayActualDataPreview(data) {
        const preview = document.getElementById('dataPreview');
        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');
        
        if (!data || !data.columns || !data.data || data.data.length === 0) {
            tableHead.innerHTML = '<th>No Data</th>';
            tableBody.innerHTML = '<tr><td>No data preview available</td></tr>';
            preview.classList.remove('hidden');
            return;
        }
        
        // Store data for target analysis
        this.currentData = data.data;
        this.columns = data.columns;
        
        // Create table header - sanitize column names
        tableHead.innerHTML = data.columns.map(col => `<th>${this.escapeHtml(col)}</th>`).join('');
        
        // Create table rows - sanitize all data values
        tableBody.innerHTML = data.data.map(row => {
            return '<tr>' + data.columns.map(col => `<td>${this.escapeHtml(row[col])}</td>`).join('') + '</tr>';
        }).join('');
        
        // Show data preview
        preview.classList.remove('hidden');
    }
    
    displayDataPreviewFromBackend(data) {
        // Use server-generated HTML for better security
        fetch('/get_data_preview_html')
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                const preview = document.getElementById('dataPreview');
                const tableContainer = document.querySelector('#dataPreview .table-container');
                
                if (tableContainer) {
                    tableContainer.innerHTML = result.html;
                } else {
                    // Fallback: create table container if it doesn't exist
                    const previewTitle = document.querySelector('#dataPreview h3');
                    if (previewTitle) {
                        previewTitle.insertAdjacentHTML('afterend', `<div class="table-container">${result.html}</div>`);
                    }
                }
                
                // Store data for target analysis (fallback)
                this.currentData = data.data;
                this.columns = data.columns;
                
                preview.classList.remove('hidden');
                console.log(`Data preview loaded: ${result.columns_count} columns, ${result.rows_count} rows`);
            } else {
                console.error('Failed to load server-generated preview:', result.error);
                // Fallback to client-side generation
                this.displayActualDataPreview(data);
            }
        })
        .catch(error => {
            console.error('Error loading server-generated preview:', error);
            // Fallback to client-side generation
            this.displayActualDataPreview(data);
        });
    }
    
    displayDataPreview() {
        const preview = document.getElementById('dataPreview');
        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');
        
        if (!this.currentData || this.currentData.length === 0) return;
        
        // Check if first row exists before accessing it
        if (!this.currentData[0]) {
            console.error('First data row is missing');
            return;
        }
        
        // Get column names
        const columns = Object.keys(this.currentData[0]);
        
        // Create table header - sanitize column names
        tableHead.innerHTML = columns.map(col => `<th>${this.escapeHtml(col)}</th>`).join('');
        
        // Create table body (show first 5 rows) - sanitize data values
        const previewRows = this.currentData.slice(0, 5);
        tableBody.innerHTML = previewRows.map(row => 
            `<tr>${columns.map(col => `<td>${this.escapeHtml(row && row[col] !== undefined ? row[col] : 'N/A')}</td>`).join('')}</tr>`
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
        .then(response => response.json())
        .then(result => {
            this.hideModal('loadingModal');
            
            if (result.success) {
                console.log('EDA processing completed');
                
                // Display EDA results
                this.displayEDAResults(result.eda);
                
                // Populate target selection
                if (result.columns) {
                    this.populateTargetSelection(result.columns, result.target, result.ptype);
                }
                
                // Navigate to EDA section
                this.navigateToStep('eda');
                
                // Do not show success modal after EDA (removed per request)
            } else {
                console.error('EDA processing failed:', result.error);
                alert('Error processing EDA: ' + result.error);
            }
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
    
    // EDA methods moved to Python backend - see /process_eda endpoint
    // Client-side methods removed for efficiency and security
    
    createEDAChartsFromBackend(stats) {
        // Create all EDA charts using Chart.js (JavaScript frontend)
        this.createDataQualityChartFromStats(stats);
        this.createColumnTypesChartFromStats(stats);
        this.createMissingValuesChartFromStats(stats);
    }

    createDataQualityChartFromStats(stats) {
        const ctx = document.getElementById('qualityChart');
        if (!ctx) {
            console.error('Quality chart canvas not found');
            return;
        }
        
        // Destroy existing chart if any
        if (this.qualityChartInstance) {
            this.qualityChartInstance.destroy();
            this.qualityChartInstance = null;
        }
        
        // Calculate data quality metrics
        const totalRows = stats.rows || 0;
        const totalCols = stats.cols || 0;
        const totalCells = totalRows * totalCols;
        
        const missing = stats.missing || {};
        const totalMissing = Object.values(missing).reduce((a, b) => a + b, 0);
        const completeCells = totalCells - totalMissing;
        
        this.qualityChartInstance = new Chart(ctx.getContext('2d'), {
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
        const ctx = document.getElementById('typesChart');
        if (!ctx) {
            console.error('Types chart canvas not found');
            return;
        }
        
        // Destroy existing chart if any
        if (this.typesChartInstance) {
            this.typesChartInstance.destroy();
            this.typesChartInstance = null;
        }
        
        this.typesChartInstance = new Chart(ctx.getContext('2d'), {
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
        const ctx = document.getElementById('missingChart');
        if (!ctx) {
            console.error('Missing chart canvas not found');
            return;
        }
        
        // Destroy existing chart if any
        if (this.missingChartInstance) {
            this.missingChartInstance.destroy();
            this.missingChartInstance = null;
        }
        
        // Use the correct structure from backend: stats.missing
        const missing = stats.missing || {};
        const columns = Object.keys(missing);
        const missingCounts = columns.map(col => missing[col] || 0);
        
        this.missingChartInstance = new Chart(ctx.getContext('2d'), {
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

    // Chart creation methods using Chart.js for client-side visualization

    populateTargetSelection(columns, defaultTarget = null, defaultPtype = null) {
        const targetSelect = document.getElementById('targetSelect');
        const problemTypeSelect = document.getElementById('problemTypeSelect');
        
        targetSelect.innerHTML = '<option value="">Choose target variable...</option>';
        
        columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            
            // Auto-select if it's the default target or matches common patterns
            if (col === defaultTarget || (!defaultTarget && this.targetPatterns.some(pattern => col.toLowerCase().includes(pattern)))) {
                option.selected = true;
                this.targetColumn = col;
            }
            
            targetSelect.appendChild(option);
        });
        
        // Set default problem type if provided
        if (defaultPtype && problemTypeSelect) {
            problemTypeSelect.value = defaultPtype;
            this.problemType = defaultPtype;
        }
        
        // Analyze target if one is selected
        if (this.targetColumn) {
            this.analyzeTarget();
            const startBtn = document.getElementById('startTrainingBtn');
            if (startBtn) startBtn.disabled = false;
        }
    }
    
    handleTargetSelection(e) {
        this.targetColumn = e.target.value;
        const startTrainingBtn = document.getElementById('startTrainingBtn');
        const targetPreview = document.getElementById('targetPreview');
        
        if (this.targetColumn) {
            this.analyzeTarget();
            if (startTrainingBtn) startTrainingBtn.disabled = false;
        } else {
            if (targetPreview) targetPreview.classList.add('hidden');
            if (startTrainingBtn) startTrainingBtn.disabled = true;
        }
    }
    
    analyzeTarget() {
        if (!this.targetColumn) return;
        
        // Use Python backend for target analysis (moved from JavaScript)
        fetch('/analyze_target', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target: this.targetColumn
            })
        })
        .then(response => response.json())
        .then(result => {
            if (result.success) {
                // Use backend analysis results
                this.displayTargetAnalysisFromBackend(result);
            } else {
                console.error('Target analysis failed:', result.error);
                // Fallback to client-side analysis
                this.analyzeTargetFromClient();
            }
        })
        .catch(error => {
            console.error('Error in target analysis:', error);
            // Fallback to client-side analysis
            this.analyzeTargetFromClient();
        });
    }
    
    displayTargetAnalysisFromBackend(result) {
        // Update problem type from backend analysis
        this.problemType = result.detected_type;
        
        // Update UI elements
        const problemTypeSelect = document.getElementById('problemTypeSelect');
        const detectedTypeElement = document.getElementById('detectedType');
        const targetPreview = document.getElementById('targetPreview');
        const targetStats = document.getElementById('targetStats');
        
        if (problemTypeSelect) problemTypeSelect.value = result.detected_type;
        if (detectedTypeElement) {
            detectedTypeElement.textContent = `Auto-detected: ${result.detected_type.charAt(0).toUpperCase() + result.detected_type.slice(1)} (${result.type_confidence} confidence)`;
            detectedTypeElement.className = 'status status--success';
        }
        
        // Show target preview with backend data
        if (targetStats) {
            targetStats.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Unique Values:</span>
                    <span class="stat-value">${result.unique_count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Missing Values:</span>
                    <span class="stat-value">${result.missing_count} (${result.missing_percentage}%)</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Data Type:</span>
                    <span class="stat-value">${result.data_type}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Total Rows:</span>
                    <span class="stat-value">${result.total_rows}</span>
                </div>
            `;
        }
        
        if (targetPreview) targetPreview.classList.remove('hidden');
        
        // Enable training button
        const startBtn = document.getElementById('startTrainingBtn');
        if (startBtn) startBtn.disabled = false;
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
        
        // Update UI with null checks
        const problemTypeSelect = document.getElementById('problemTypeSelect');
        const detectedTypeElement = document.getElementById('detectedType');
        const targetPreview = document.getElementById('targetPreview');
        const targetStats = document.getElementById('targetStats');
        
        if (problemTypeSelect) problemTypeSelect.value = detectedType;
        if (detectedTypeElement) {
            detectedTypeElement.textContent = `Auto-detected: ${detectedType.charAt(0).toUpperCase() + detectedType.slice(1)}`;
            detectedTypeElement.className = 'status status--success';
        }
        
        // Show target preview
        if (targetStats) {
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
                    <span class="stat-value">${targetValues.length > 0 ? typeof targetValues[0] : 'unknown'}</span>
                </div>
            `;
        }
        
        if (targetPreview) targetPreview.classList.remove('hidden');
    }
    
    handleProblemTypeSelection(e) {
        if (e.target.value) {
            this.problemType = e.target.value;
        }
    }
    
    updateSplitDisplay() {
        const splitRatio = document.getElementById('splitRatio');
        const trainPercent = document.getElementById('trainPercent');
        const testPercent = document.getElementById('testPercent');
        
        if (!splitRatio || !trainPercent || !testPercent) {
            console.warn('Split display elements not found');
            return;
        }
        
        const ratio = splitRatio.value;
        trainPercent.textContent = ratio;
        testPercent.textContent = 100 - ratio;
    }
    
    startTraining() {
        // Show loading modal
        this.showModal('loadingModal', 'Validating configuration...');
        
        // Prepare config data
        const config = {
            target: this.targetColumn,
            ptype: this.problemType,
            split: document.getElementById('splitRatio').value / 100
        };
        
        console.log('Validating training configuration:', config);
        
        // First validate configuration with Python backend
        fetch('/validate_training_config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                target: config.target,
                ptype: config.ptype,
                split_ratio: config.split
            })
        })
        .then(response => response.json())
        .then(validationResult => {
            if (validationResult.success && validationResult.validation_result.valid) {
                // Configuration is valid, proceed with training
                this.showModal('loadingModal', 'Training models...');
                return this.performTraining(config);
            } else {
                // Configuration is invalid
                const errors = validationResult.validation_result.errors.join('; ');
                throw new Error(`Configuration validation failed: ${errors}`);
            }
        })
        .catch(error => {
            console.error('Training validation error:', error);
            this.hideModal('loadingModal');
            alert('Training validation failed: ' + error.message);
        });
    }
    
    performTraining(config) {
        console.log('Starting training with validated config:', config);
        
        // Send config to Flask backend for training
        fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: `target=${encodeURIComponent(config.target)}&ptype=${encodeURIComponent(config.ptype)}&split=${encodeURIComponent(config.split)}`
        })
        .then(response => response.json())
        .then(result => {
            // Hide loading modal
            this.hideModal('loadingModal');
            
            if (result.success) {
                console.log('Training completed successfully');
                
                // Store results
                this.bestModel = {
                    name: result.best_model,
                    metrics: result.metrics
                };
                this.trainedModels = result.all_results;
                this.featureImportance = result.feature_importance;
                
                // Display results
                this.displayTrainingResults(result);
                
                // Navigate to results
                this.navigateToStep('results');
            } else {
                console.error('Training failed:', result.error);
                alert('Training failed: ' + result.error);
            }
        })
        .catch(err => {
            console.error('Training error:', err);
            this.hideModal('loadingModal');
            alert('Model training failed: ' + err.message);
        });
    }
    
    displayTrainingResults(result) {
        // Update best model info
        const bestModelName = document.querySelector('.best-model-name');
        const bestModelScore = document.querySelector('.best-model-score');
        
        if (bestModelName && result.best_model) {
            bestModelName.textContent = result.best_model;
        }
        
        if (bestModelScore && result.metrics) {
            // Display primary metric based on problem type
            const primaryMetric = this.problemType === 'regression' ? 'R2' : 'Accuracy';
            const score = result.metrics[primaryMetric];
            if (score !== undefined) {
                bestModelScore.textContent = `${primaryMetric}: ${(score * 100).toFixed(2)}%`;
            }
        }
        
        // Update metrics table
        this.updateMetricsTable(result.all_results);
        
        // Create feature importance chart
        this.createFeatureImportanceChart(result.feature_importance, result.best_model);
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
                const primaryMetric = metrics.Accuracy || metrics.R2 || Object.values(metrics)[0] || 0;
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
                    const primaryMetric = metrics.Accuracy || metrics.R2 || Object.values(metrics)[0] || 0;
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
            // Create charts using Chart.js
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
            this.performanceChartInstance = null;
        }

        const ctx = chartCanvas.getContext('2d');
        
        // Extract data from trainedModels
        const labels = this.trainedModels.map(m => m.model || 'Unknown');
        const data = this.trainedModels.map(m => {
            // Get primary metric (accuracy for classification, R² for regression)
            const metrics = m.metrics || {};
            return metrics.Accuracy || metrics.R2 || Object.values(metrics)[0] || 0;
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
    
    // (Removed earlier duplicate createFeatureImportanceChart; unified optional-param version below)
    
    updateMetricsTable(allResults) {
        const tbody = document.getElementById('metricsTableBody');
        if (!tbody || !allResults) return;
        
        tbody.innerHTML = allResults.map(result => {
            const model = result.model;
            const metrics = result.metrics;
            
            // Get primary metric based on problem type
            let primaryScore = 'N/A';
            let primaryMetric = 'Score';
            
            if (this.problemType === 'regression') {
                primaryMetric = 'R²';
                primaryScore = metrics['R2'] !== undefined ? metrics['R2'].toFixed(3) : 'N/A';
            } else {
                primaryMetric = 'Accuracy';
                primaryScore = metrics['Accuracy'] !== undefined ? metrics['Accuracy'].toFixed(3) : 'N/A';
            }
            
            const trainingTime = metrics['Training_Time'] || 'N/A';
            
            return `
                <tr>
                    <td class="font-bold">${model}</td>
                    <td class="text-primary font-bold">${primaryScore}</td>
                    <td>${trainingTime}s</td>
                    <td><span class="status status--completed">Completed</span></td>
                </tr>
            `;
        }).join('');
    }
    
    createFeatureImportanceChart(featureImportance = null, bestModelName = null) {
        const fi = featureImportance || this.featureImportance;
        const model = bestModelName || (this.bestModel && this.bestModel.name);
        const canvas = document.getElementById('featureImportanceChart');
        
        if (!fi || !model || !fi[model] || !canvas) {
            console.warn('No feature importance data available');
            return;
        }
        
        try {
            // Destroy existing chart if any
            if (this.featureImportanceChartInstance) {
                this.featureImportanceChartInstance.destroy();
                this.featureImportanceChartInstance = null;
            }
            
            const importance = fi[model];
            const sorted = Object.entries(importance).sort((a,b)=>b[1]-a[1]).slice(0,10);
            
            this.featureImportanceChartInstance = new Chart(canvas.getContext('2d'), {
                type: 'bar',
                data: { 
                    labels: sorted.map(d=>d[0]), 
                    datasets: [{ 
                        label: 'Feature Importance', 
                        data: sorted.map(d=>d[1]), 
                        backgroundColor: '#1FB8CD', 
                        borderColor: '#1FB8CD', 
                        borderWidth: 1 
                    }] 
                },
                options: { 
                    indexAxis: 'y', 
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
        } catch (e) { 
            console.error('Error creating feature importance chart:', e); 
        }
    }
    
    downloadModel() {
        console.log('Downloading model package...');
        
        // Use backend download endpoint for complete model package
        window.location.href = '/download_model';
    }
    
    exportResults() {
        console.log('Exporting results...');
        
        // Use backend export endpoint for CSV export
        window.location.href = '/export_results';
    }
    
    resetApplication() {
        if (confirm('Are you sure you want to reset the application? All progress will be lost.')) {
            this.currentData = null;
            this.targetColumn = null;
            this.problemType = null;
            this.trainedModels = [];
            this.bestModel = null;
            
            // Reset UI with null checks
            const uploadArea = document.getElementById('uploadArea');
            const filePreview = document.getElementById('filePreview');
            const dataPreview = document.getElementById('dataPreview');
            const fileInput = document.getElementById('fileInput');
            
            if (uploadArea) uploadArea.style.display = 'block';
            if (filePreview) filePreview.classList.add('hidden');
            if (dataPreview) dataPreview.classList.add('hidden');
            if (fileInput) fileInput.value = '';
            
            this.navigateToStep('upload');
        }
    }
    
    showModal(modalId, text = '') {
        const modal = document.getElementById(modalId);
        if (!modal) {
            console.error(`Modal ${modalId} not found`);
            return;
        }
        
        if (text && modalId === 'loadingModal') {
            const loadingText = document.getElementById('loadingText');
            if (loadingText) loadingText.textContent = text;
        }
        modal.classList.remove('hidden');
    }
    
    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            modal.classList.add('hidden');
        } else {
            console.error(`Modal ${modalId} not found`);
        }
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
        // Load Python-generated performance chart
        this.loadPythonChart('/performance_chart', 'performanceChart');
        this.createFeatureImportanceChart();
        
        console.log('Test charts created');
    }
}

// ---------------------------------------------------------------------------
// Features implemented in this file
// - Upload handlers (drag/drop, select) and dataset preview rendering
// - EDA fetch/render: stats, charts (quality, types, missing, correlation)
// - Training workflow: config, validation, start, progress, results
// - Results UI: best model panel, metrics table, performance chart
// - Export/download actions and app reset utilities
// ---------------------------------------------------------------------------

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.autoMLApp = new AutoMLApp();
});