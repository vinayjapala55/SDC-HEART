/**
 * BMD-HS Multi-Valvular PCG Diagnostic Workstation - Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    const patientPresetButtons = document.getElementById('patientPresetButtons');
    const btnRunAnalysis = document.getElementById('btnRunAnalysis');
    const loadingSection = document.getElementById('loadingSection');
    const resultsContainer = document.getElementById('resultsContainer');
    const progressTitle = document.getElementById('progressTitle');
    const progressSub = document.getElementById('progressSub');

    // 4 Valve file inputs & status indicators
    const valveInputs = {
        Aor: document.getElementById('inputAor'),
        Mit: document.getElementById('inputMit'),
        Pul: document.getElementById('inputPul'),
        Tri: document.getElementById('inputTri'),
    };

    const valveStatuses = {
        Aor: document.getElementById('statusAor'),
        Mit: document.getElementById('statusMit'),
        Pul: document.getElementById('statusPul'),
        Tri: document.getElementById('statusTri'),
    };

    let selectedFiles = { Aor: null, Mit: null, Pul: null, Tri: null };
    let activeSamplePatientId = null;
    let samplePatientsData = [];

    // 1. Fetch Authentic Clinical Patient Presets
    fetchSamplePatients();

    async function fetchSamplePatients() {
        try {
            const resp = await fetch('/api/sample_patients');
            const data = await resp.json();
            samplePatientsData = data.patients || [];
            renderPatientPresets(samplePatientsData);
        } catch (e) {
            console.error('Failed to load sample patients:', e);
            patientPresetButtons.innerHTML = `<div class="error-text">Failed to load presets from server.</div>`;
        }
    }

    function renderPatientPresets(patients) {
        if (!patients || patients.length === 0) {
            patientPresetButtons.innerHTML = `<div>No sample presets found.</div>`;
            return;
        }

        patientPresetButtons.innerHTML = '';
        patients.forEach(p => {
            const btn = document.createElement('button');
            btn.className = 'preset-chip';
            btn.innerHTML = `<strong>${p.category}</strong>: ${p.label}`;
            btn.addEventListener('click', () => selectPatientPreset(p));
            patientPresetButtons.appendChild(btn);
        });
    }

    function selectPatientPreset(p) {
        activeSamplePatientId = p.patient_id;
        selectedFiles = { Aor: null, Mit: null, Pul: null, Tri: null };

        // Highlight preset chips
        document.querySelectorAll('.preset-chip').forEach(b => b.classList.remove('active'));
        event.target.closest('.preset-chip')?.classList.add('active');

        // Update valve labels
        Object.keys(valveStatuses).forEach(v => {
            const fname = p.valves[v];
            valveStatuses[v].textContent = `Preset: ${fname}`;
            valveStatuses[v].classList.add('loaded');
        });

        btnRunAnalysis.disabled = false;
        btnRunAnalysis.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // 2. File Upload Listeners for all 4 valves
    Object.keys(valveInputs).forEach(valve => {
        valveInputs[valve].addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const file = e.target.files[0];
                selectedFiles[valve] = file;
                activeSamplePatientId = null;

                // Remove chip highlights
                document.querySelectorAll('.preset-chip').forEach(b => b.classList.remove('active'));

                valveStatuses[valve].textContent = `Selected: ${file.name}`;
                valveStatuses[valve].classList.add('loaded');

                checkReadyToAnalyze();
            }
        });
    });

    function checkReadyToAnalyze() {
        const allUploaded = Object.values(selectedFiles).every(f => f !== null);
        btnRunAnalysis.disabled = !allUploaded && !activeSamplePatientId;
    }

    // 3. Run Analysis Workflow
    btnRunAnalysis.addEventListener('click', async () => {
        resultsContainer.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        loadingSection.scrollIntoView({ behavior: 'smooth' });

        // Multi-Agent visual stepper
        progressTitle.textContent = "Executing Multi-Agent Workflow...";
        progressSub.textContent = "Agent 1: Inspecting signal quality and generating log-mel spectrograms...";

        const formData = new FormData();
        if (activeSamplePatientId) {
            formData.append('sample_patient_id', activeSamplePatientId);
        } else {
            formData.append('file_aor', selectedFiles.Aor);
            formData.append('file_mit', selectedFiles.Mit);
            formData.append('file_pul', selectedFiles.Pul);
            formData.append('file_tri', selectedFiles.Tri);
        }

        try {
            // Simulated agent handoff updates
            setTimeout(() => {
                progressSub.textContent = "Agent 2: Running RosaNet 4-valve inference & Grad-CAM attribution...";
            }, 1200);

            setTimeout(() => {
                progressSub.textContent = "Agent 3: Querying ChromaDB for ACC/AHA guidelines evidence...";
            }, 2400);

            const resp = await fetch('/api/predict_patient', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${resp.status}`);
            }

            const report = await resp.json();
            renderReportResults(report);

        } catch (err) {
            alert(`Analysis Failed: ${err.message}`);
            console.error(err);
        } finally {
            loadingSection.classList.add('hidden');
        }
    });

    // 4. Render Report Results
    function renderReportResults(report) {
        resultsContainer.classList.remove('hidden');
        resultsContainer.scrollIntoView({ behavior: 'smooth' });

        const quality = report.quality_summary;
        const pred = report.prediction_summary;
        const evidence = report.evidence_summary;

        document.getElementById('reportHeaderMeta').textContent = `Session: ${report.patient_session_id} | Model: ${report.model_version}`;

        // Risk Pill
        const riskPill = document.getElementById('riskPill');
        riskPill.textContent = pred.risk_level.toUpperCase();
        riskPill.className = `risk-pill ${pred.risk_level.toLowerCase().replace(' ', '-')}`;

        // Primary Banner
        const banner = document.getElementById('primaryBanner');
        const bTitle = document.getElementById('bannerTitle');
        const bSub = document.getElementById('bannerSubtitle');

        bTitle.textContent = pred.overall_status;
        bSub.textContent = pred.is_normal ? "All physical valve auscultation sites exhibit physiological sound patterns." : "Auscultation features exceed clinical decision thresholds for valvular pathology.";
        banner.className = `presence-banner ${pred.is_normal ? 'normal' : 'abnormal'}`;

        // Agent 1: Quality Summary
        const qContent = document.getElementById('qualityContent');
        let qHtml = `<p><strong>Inspection Status:</strong> <span class="badge ${quality.status.toLowerCase()}">${quality.status}</span> &bull; ${quality.summary}</p>`;
        qHtml += `<div class="valves-table-wrapper"><table class="classic-table"><thead><tr><th>Valve</th><th>File</th><th>Duration</th><th>Rate</th><th>Status</th></tr></thead><tbody>`;
        quality.valves_inspected.forEach(v => {
            qHtml += `<tr><td><strong>${v.valve_location}</strong></td><td><code>${v.filename}</code></td><td>${v.duration_seconds}s</td><td>${v.sample_rate} Hz</td><td>${v.is_valid ? 'Pass' : 'Fail'}</td></tr>`;
        });
        qHtml += `</tbody></table></div>`;
        qContent.innerHTML = qHtml;

        // Agent 2: Probabilities & XAI
        const pGrid = document.getElementById('probsGrid');
        pGrid.innerHTML = '';
        const dNames = { N: "Normal", AS: "Aortic Stenosis", AR: "Aortic Regurgitation", MR: "Mitral Regurgitation", MS: "Mitral Stenosis" };

        Object.keys(pred.raw_probabilities).forEach(code => {
            const prob = pred.raw_probabilities[code];
            const pct = Math.round(prob * 100);
            const thresh = (pred.decision_thresholds[code] || 0.45) * 100;
            const isHigh = pct >= thresh;

            const card = document.createElement('div');
            card.className = `prob-card ${isHigh ? 'elevated' : ''}`;
            card.innerHTML = `
                <div class="prob-header"><span class="code">${code}</span><span class="p-val">${pct}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill" style="width: ${pct}%"></div></div>
                <div class="prob-meta">${dNames[code] || code} (Thresh: ${thresh}%)</div>
            `;
            pGrid.appendChild(card);
        });

        // XAI Summary Box
        const xaiBox = document.getElementById('xaiSummaryBox');
        if (pred.valve_attributions && pred.valve_attributions.length > 0) {
            let xaiHtml = `<h5>Valve-Level Diagnostic Attribution (XAI)</h5><ul>`;
            pred.valve_attributions.forEach(item => {
                xaiHtml += `<li><strong>${item.disease_class} (${Math.round(item.probability*100)}%)</strong>: Primary convolutional activation concentrated at <strong>${item.top_contributing_valve}</strong>.</li>`;
            });
            xaiHtml += `</ul><p class="xai-note">Attribution heatmaps reflect learned convolutional filter activations across physical auscultation sites.</p>`;
            xaiBox.innerHTML = xaiHtml;
        } else {
            xaiBox.innerHTML = `<p class="xai-note">No elevated pathological classes requiring valve attribution analysis.</p>`;
        }

        // Agent 3: Guidelines Evidence
        const gContent = document.getElementById('guidelinesContent');
        if (evidence.retrieved_guidelines && evidence.retrieved_guidelines.length > 0) {
            let gHtml = `<div class="guideline-list">`;
            evidence.retrieved_guidelines.slice(0, 3).forEach(g => {
                gHtml += `
                    <div class="guideline-card">
                        <div class="g-header"><strong>${g.target_disease}</strong> &bull; ${g.section_name} (Page ${g.page_number})</div>
                        <div class="g-body">"${g.content_text}"</div>
                        <div class="g-source">Source: ${g.guideline_title}</div>
                    </div>
                `;
            });
            gHtml += `</div>`;
            gContent.innerHTML = gHtml;
        } else {
            gContent.innerHTML = `<p class="muted-text">No direct guideline excerpts retrieved for current findings.</p>`;
        }

        // Markdown Report View & Export Button
        const mdPreview = document.getElementById('markdownReportPreview');
        mdPreview.textContent = report.structured_markdown_report;

        document.getElementById('btnExportMarkdown').onclick = () => {
            navigator.clipboard.writeText(report.structured_markdown_report)
                .then(() => alert("Clinical Report copied to clipboard!"))
                .catch(() => alert("Failed to copy report to clipboard."));
        };
    }
});
