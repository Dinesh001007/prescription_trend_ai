document.addEventListener('DOMContentLoaded', () => {
    
    // --- MODE SWITCHING ---
    const modeBtns = document.querySelectorAll('.mode-btn');
    const sections = document.querySelectorAll('.app-section');
    
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // UI Toggle
            modeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const mode = btn.dataset.mode;
            sections.forEach(sec => {
                sec.classList.add('hidden');
            });
            document.getElementById(`section-${mode}`).classList.remove('hidden');
        });
    });

    // --- TEST CONNECTION ---
    document.getElementById('btn-test-connection').addEventListener('click', async (e) => {
        const btn = e.target;
        btn.innerText = "TESTING...";
        try {
            const res = await fetch('/api/drug_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt: "Say 'OK' only."})
            });
            if(res.ok) {
                btn.innerText = "CONNECTION OK";
                btn.style.color = "var(--gold-primary)";
                btn.style.borderColor = "var(--gold-primary)";
            } else {
                btn.innerText = "OLLAMA DOWN";
                btn.style.color = "var(--danger)";
                btn.style.borderColor = "var(--danger)";
            }
        } catch(err) {
            btn.innerText = "ERROR";
        }
        setTimeout(() => { btn.innerText = "TEST CONNECTION"; btn.style = ""; }, 3000);
    });

    // --- DRUG LOOKUP ---
    const btnAnalyzeDrug = document.getElementById('btn-analyze-drug');
    const drugInput = document.getElementById('drug-input');
    const drugResults = document.getElementById('drug-results');
    const drugChat = document.getElementById('drug-chat');

    btnAnalyzeDrug.addEventListener('click', async () => {
        const query = drugInput.value.trim();
        if(!query) return;
        
        btnAnalyzeDrug.innerText = "ANALYZING...";
        drugResults.classList.add('hidden');
        drugChat.classList.add('hidden');
        
        try {
            const res = await fetch('/api/drug_lookup', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query})
            });
            const data = await res.json();
            
            if(res.ok && data.sections) {
                let html = `<h3 class="section-heading mb-2">CLINICAL PROFILE: ${query.toUpperCase()}</h3>`;
                data.sections.forEach(sec => {
                    html += `
                        <div class="glass-card mb-1">
                            <h4 class="card-title mb-1" style="color:var(--gold-primary); font-size:12px;">${sec.header}</h4>
                            <div style="font-size:14px; line-height:1.6;">${marked(sec.body)}</div>
                        </div>
                    `;
                });
                drugResults.innerHTML = html;
                drugResults.classList.remove('hidden');
                drugChat.classList.remove('hidden');
            } else {
                alert(data.error || "Failed to analyze drug.");
            }
        } catch(err) {
            alert("Connection error.");
        }
        btnAnalyzeDrug.innerText = "ANALYZE";
    });

    // Helper: basic markdown to HTML for bold and lines
    function marked(text) {
        if(!text) return "";
        let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        html = html.replace(/\n/g, '<br>');
        return html;
    }

    // Drug Chat
    const btnDrugChat = document.getElementById('btn-drug-chat');
    const inputDrugChat = document.getElementById('drug-chat-input');
    const historyDrugChat = document.getElementById('drug-chat-history');

    btnDrugChat.addEventListener('click', async () => {
        const prompt = inputDrugChat.value.trim();
        if(!prompt) return;
        
        historyDrugChat.innerHTML += `<div class="chat-msg user">${prompt}</div>`;
        inputDrugChat.value = "";
        historyDrugChat.scrollTop = historyDrugChat.scrollHeight;
        
        try {
            const res = await fetch('/api/drug_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt})
            });
            const data = await res.json();
            if(data.response) {
                historyDrugChat.innerHTML += `<div class="chat-msg assistant">${marked(data.response)}</div>`;
            } else {
                historyDrugChat.innerHTML += `<div class="chat-msg assistant" style="border-color:var(--danger)">${data.error}</div>`;
            }
            historyDrugChat.scrollTop = historyDrugChat.scrollHeight;
        } catch(err) {}
    });

    // --- SCAN REPORT ---
    const scanFile = document.getElementById('scan-file');
    const btnAnalyzeScan = document.getElementById('btn-analyze-scan');
    const scanResults = document.getElementById('scan-results');
    
    scanFile.addEventListener('change', () => {
        if(scanFile.files.length > 0) {
            btnAnalyzeScan.classList.remove('hidden');
            document.querySelector('#scan-upload-zone p').innerText = "SELECTED: " + scanFile.files[0].name.toUpperCase();
        }
    });

    let currentScanAnalysis = "";

    btnAnalyzeScan.addEventListener('click', async () => {
        if(scanFile.files.length === 0) return;
        
        btnAnalyzeScan.innerText = "INTERPRETING...";
        scanResults.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('file', scanFile.files[0]);
        
        try {
            const res = await fetch('/api/scan_report', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if(res.ok) {
                document.getElementById('scan-modality').innerText = (data.modality || "O.C.R. TEXT EXTRACTION").toUpperCase();
                document.getElementById('scan-analysis').innerHTML = marked(data.analysis);
                currentScanAnalysis = data.analysis;
                scanResults.classList.remove('hidden');
            } else {
                alert(data.error);
            }
        } catch(err) {}
        btnAnalyzeScan.innerText = "INTERPRET REPORT";
    });

    // Scan Chat
    const btnScanChat = document.getElementById('btn-scan-chat');
    const inputScanChat = document.getElementById('scan-chat-input');
    const historyScanChat = document.getElementById('scan-chat-history');

    btnScanChat.addEventListener('click', async () => {
        const prompt = inputScanChat.value.trim();
        if(!prompt) return;
        
        historyScanChat.innerHTML += `<div class="chat-msg user">${prompt}</div>`;
        inputScanChat.value = "";
        
        try {
            const res = await fetch('/api/scan_chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, analysis: currentScanAnalysis})
            });
            const data = await res.json();
            if(data.response) {
                historyScanChat.innerHTML += `<div class="chat-msg assistant">${marked(data.response)}</div>`;
            }
        } catch(err) {}
    });

    // --- DATASET ANALYSIS ---
    const dsFile = document.getElementById('dataset-file');
    const dsWorkspace = document.getElementById('dataset-workspace');
    let currentDatasetId = null;
    let currentDatasetColMap = null;

    dsFile.addEventListener('change', async () => {
        if(dsFile.files.length === 0) return;
        
        const p = document.querySelector('#dataset-upload-zone p');
        p.innerText = "UPLOADING: " + dsFile.files[0].name.toUpperCase();
        
        const formData = new FormData();
        formData.append('file', dsFile.files[0]);
        
        try {
            const res = await fetch('/api/upload_dataset', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if(res.ok) {
                currentDatasetId = data.dataset_id;
                currentDatasetColMap = data.col_map;
                
                document.getElementById('ds-rows').innerText = data.rows.toLocaleString();
                document.getElementById('ds-cols').innerText = data.cols;
                document.getElementById('ds-missing').innerText = data.missing.toLocaleString();
                
                dsWorkspace.classList.remove('hidden');
                document.getElementById('dataset-upload-container').classList.add('hidden');
            } else {
                alert(data.error);
                p.innerText = "DROP DATASET HERE OR CLICK TO BROWSE";
            }
        } catch(err) {
            p.innerText = "DROP DATASET HERE OR CLICK TO BROWSE";
        }
    });

    // Run Agents
    const btnRunAgents = document.getElementById('btn-run-agents');
    const dsLoader = document.getElementById('dataset-loader');
    const dsResults = document.getElementById('dataset-results');

    btnRunAgents.addEventListener('click', async () => {
        if(!currentDatasetId) return;
        
        dsLoader.classList.remove('hidden');
        dsResults.classList.add('hidden');
        btnRunAgents.classList.add('hidden');
        
        const selections = {
            pattern: document.getElementById('agent-pattern').checked,
            risk: document.getElementById('agent-risk').checked,
            cohort: document.getElementById('agent-cohort').checked,
            anomaly: document.getElementById('agent-anomaly').checked,
            trend: document.getElementById('agent-trend').checked
        };
        
        try {
            const res = await fetch('/api/analyze_dataset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    dataset_id: currentDatasetId,
                    col_map: currentDatasetColMap,
                    selections: selections
                })
            });
            const data = await res.json();
            
            dsLoader.classList.add('hidden');
            
            if(res.ok) {
                // Populate LLM Insights
                document.getElementById('ds-llm-insights').innerHTML = marked(data.llm_insights);
                
                // Populate Alerts
                const alertsContainer = document.getElementById('ds-alerts');
                alertsContainer.innerHTML = '';
                
                if(data.results.risk && data.results.risk.high_risk_count > 0) {
                    alertsContainer.innerHTML += `
                        <div class="alert-box">
                            <div class="alert-title">⚠️ SYSTEM ALERT: HIGH RISK RECORDS DETECTED</div>
                            <div>${data.results.risk.high_risk_count} high-risk prescriptions identified. Immediate review advised.</div>
                        </div>
                    `;
                }
                if(data.results.anomaly && data.results.anomaly.anom_count > 0) {
                    alertsContainer.innerHTML += `
                        <div class="alert-box" style="border-left-color: #B28835; background: rgba(178,136,53,0.05);">
                            <div class="alert-title" style="color: #B28835;">🔴 ANOMALY DETECTION ALERT</div>
                            <div style="color:var(--text-secondary)">${data.results.anomaly.anom_count} statistical anomalies isolated by unsupervised isolation forest agent.</div>
                        </div>
                    `;
                }
                
                // Render Plotly Tabs
                const renderTab = (agentKey, divId) => {
                    const pane = document.getElementById(divId);
                    pane.innerHTML = '';
                    if(data.results[agentKey]) {
                        const resObj = data.results[agentKey];
                        if(resObj.summary) {
                            pane.innerHTML += `<div class="mb-2" style="font-size:14px; color:var(--text-secondary);">${resObj.summary}</div>`;
                        }
                        if(resObj.figures) {
                            resObj.figures.forEach((fig, i) => {
                                const plotterId = `${agentKey}-plot-${i}`;
                                pane.innerHTML += `
                                    <div class="glass-card mb-2">
                                        <h4 class="card-title mb-1" style="font-size:12px;">${fig.title.toUpperCase()}</h4>
                                        <div id="${plotterId}" style="width:100%; height:400px;"></div>
                                    </div>
                                `;
                            });
                        }
                    } else {
                        pane.innerHTML = '<p style="color:var(--text-secondary)">AGENT DID NOT RUN OR NO DATA AVAILABLE.</p>';
                    }
                };
                
                renderTab('pattern', 'tab-pattern');
                renderTab('risk', 'tab-risk');
                renderTab('cohort', 'tab-cohort');
                renderTab('anomaly', 'tab-anomaly');
                renderTab('trend', 'tab-trend');
                
                // Show Results
                dsResults.classList.remove('hidden');
                
                // Setup Plotly bindings after DOM update
                setTimeout(() => {
                    ['pattern', 'risk', 'cohort', 'anomaly', 'trend'].forEach(agentKey => {
                        if(data.results[agentKey] && data.results[agentKey].figures) {
                            data.results[agentKey].figures.forEach((fig, i) => {
                                const plotterId = `${agentKey}-plot-${i}`;
                                // Parse the JSON string figure into JS object
                                const figObj = JSON.parse(fig.plotly_json);
                                
                                // Override layout to match luxury theme
                                figObj.layout.paper_bgcolor = '#111111';
                                figObj.layout.plot_bgcolor = '#111111';
                                figObj.layout.font = { family: 'Montserrat, sans-serif', color: '#888888' };
                                figObj.layout.title = ''; // Managed by our HTML overlay
                                
                                Plotly.newPlot(plotterId, figObj.data, figObj.layout, {displayModeBar: false, responsive: true});
                            });
                        }
                    });
                }, 100);

            } else {
                alert(data.error);
                btnRunAgents.classList.remove('hidden');
            }
        } catch(err) {
            dsLoader.classList.add('hidden');
            btnRunAgents.classList.remove('hidden');
        }
    });

    // Tab Switching functionality for Dataset Results
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            tabPanes.forEach(p => p.classList.add('hidden'));
            const targetId = btn.getAttribute('data-tab');
            document.getElementById(targetId).classList.remove('hidden');
            
            // Trigger Plotly resize for charts hidden during render
            setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 100);
        });
    });

});
