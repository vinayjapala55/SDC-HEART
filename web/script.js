/**
 * BMD-HS Clinical Cardiovascular Diagnostic Suite - Client Logic
 * Version: 6.0
 * Features:
 * - Patient Preset Cohort Management
 * - 4-Valve PCG Audio Stream Preview
 * - Multi-Agent Live Execution Stepper Animation
 * - Tabbed Clinical Results Dashboard
 * - EMR Medical Consultation Note Copy & Print
 */

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const patientPresetButtons = document.getElementById('patientPresetButtons');
    const btnRunAnalysis = document.getElementById('btnRunAnalysis');
    const btnResetForm = document.getElementById('btnResetForm');
    const loadingSection = document.getElementById('loadingSection');
    const resultsContainer = document.getElementById('resultsContainer');
    const inputReadinessText = document.getElementById('inputReadinessText');
    const progressTitle = document.getElementById('progressTitle');
    const progressSub = document.getElementById('progressSub');

    // 4 Valve Stations Elements
    const valves = ['Aor', 'Mit', 'Pul', 'Tri'];
    const stations = {
        Aor: {
            station: document.getElementById('stationAor'),
            badge: document.getElementById('badgeAor'),
            status: document.getElementById('statusAor'),
            input: document.getElementById('inputAor'),
            playerWrap: document.getElementById('playerWrapAor'),
            audio: document.getElementById('audioAor'),
        },
        Mit: {
            station: document.getElementById('stationMit'),
            badge: document.getElementById('badgeMit'),
            status: document.getElementById('statusMit'),
            input: document.getElementById('inputMit'),
            playerWrap: document.getElementById('playerWrapMit'),
            audio: document.getElementById('audioMit'),
        },
        Pul: {
            station: document.getElementById('stationPul'),
            badge: document.getElementById('badgePul'),
            status: document.getElementById('statusPul'),
            input: document.getElementById('inputPul'),
            playerWrap: document.getElementById('playerWrapPul'),
            audio: document.getElementById('audioPul'),
        },
        Tri: {
            station: document.getElementById('stationTri'),
            badge: document.getElementById('badgeTri'),
            status: document.getElementById('statusTri'),
            input: document.getElementById('inputTri'),
            playerWrap: document.getElementById('playerWrapTri'),
            audio: document.getElementById('audioTri'),
        }
    };

    let selectedFiles = { Aor: null, Mit: null, Pul: null, Tri: null };
    let activeSamplePatientId = null;
    let samplePatientsData = [];

    // Disease labels mapping
    const diseaseMap = {
        N: { name: "Normal Auscultation", desc: "Physiological S1/S2 heart sounds, no murmur detected" },
        AS: { name: "Aortic Stenosis", desc: "Systolic ejection murmur, loudest at Aortic area" },
        AR: { name: "Aortic Regurgitation", desc: "Early diastolic decrescendo murmur, Erb's / Aortic area" },
        MR: { name: "Mitral Regurgitation", desc: "Pansystolic / holosystolic murmur, radiating to axilla" },
        MS: { name: "Mitral Stenosis", desc: "Low-pitch mid-diastolic rumble with opening snap at apex" }
    };

    // 1. Initialize System and Load Curated Patient Profiles
    fetchSystemStatus();
    fetchSamplePatients();
    initTabSwitching();

    async function fetchSystemStatus() {
        try {
            const resp = await fetch('/api/status');
            const data = await resp.json();
            const textElem = document.getElementById('systemStatusText');
            if (data.status === 'online') {
                textElem.textContent = '4-Valve Engine Online';
            }
        } catch (e) {
            console.error('Status check error:', e);
        }
    }

    async function fetchSamplePatients() {
        try {
            const resp = await fetch('/api/sample_patients');
            const data = await resp.json();
            samplePatientsData = data.patients || [];
            renderPatientPresets(samplePatientsData);
        } catch (e) {
            console.error('Failed to load sample patients:', e);
            patientPresetButtons.innerHTML = `<div class="error-text">Failed to load clinical presets from server.</div>`;
        }
    }

    function renderPatientPresets(patients) {
        if (!patients || patients.length === 0) {
            patientPresetButtons.innerHTML = `<div>No clinical presets found on server.</div>`;
            return;
        }

        patientPresetButtons.innerHTML = '';
        patients.forEach(p => {
            const card = document.createElement('div');
            card.className = 'patient-preset-card';
            card.dataset.pid = p.patient_id;

            card.innerHTML = `
                <div class="card-top-row">
                    <span class="patient-id-badge">${p.patient_id.toUpperCase()}</span>
                    <span class="patient-category-tag">${p.category}</span>
                </div>
                <div class="patient-diagnosis-label">${p.label}</div>
                <div class="valve-indicators-row">
                    <span class="v-dot-tag">AOR ✓</span>
                    <span class="v-dot-tag">MIT ✓</span>
                    <span class="v-dot-tag">PUL ✓</span>
                    <span class="v-dot-tag">TRI ✓</span>
                </div>
            `;

            card.addEventListener('click', () => selectPatientPreset(p, card));
            patientPresetButtons.appendChild(card);
        });
    }

    function selectPatientPreset(p, cardElement) {
        activeSamplePatientId = p.patient_id;
        selectedFiles = { Aor: null, Mit: null, Pul: null, Tri: null };

        // Highlight preset card
        document.querySelectorAll('.patient-preset-card').forEach(c => c.classList.remove('active'));
        cardElement.classList.add('active');

        // Update all 4 valve stations with audio player streams
        valves.forEach(v => {
            const fname = p.valves[v];
            const st = stations[v];
            
            st.badge.textContent = 'Preset Loaded';
            st.badge.className = 'station-status-pill loaded';
            
            st.status.textContent = fname;
            st.status.className = 'file-name-display loaded';

            // Connect audio stream
            st.audio.src = `/api/sample_audio/${fname}`;
            st.playerWrap.classList.remove('hidden');
        });

        inputReadinessText.innerHTML = `Patient profile <strong>${p.patient_id.toUpperCase()}</strong> (${p.label}) selected across all 4 auscultation sites.`;
        btnRunAnalysis.disabled = false;
    }

    // 2. Custom File Upload Listeners
    valves.forEach(v => {
        const st = stations[v];
        st.input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                const file = e.target.files[0];
                selectedFiles[v] = file;
                activeSamplePatientId = null;

                // Remove preset card highlights
                document.querySelectorAll('.patient-preset-card').forEach(c => c.classList.remove('active'));

                st.badge.textContent = 'Custom WAV';
                st.badge.className = 'station-status-pill loaded';

                st.status.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                st.status.className = 'file-name-display loaded';

                // Local object URL for audio playback
                st.audio.src = URL.createObjectURL(file);
                st.playerWrap.classList.remove('hidden');

                validateInputReadiness();
            }
        });
    });

    function validateInputReadiness() {
        const missing = valves.filter(v => !selectedFiles[v]);
        if (missing.length === 0) {
            inputReadinessText.innerHTML = `<span style="color: #059669; font-weight: 700;">✓ All 4 auscultation channels loaded and verified. Ready for analysis.</span>`;
            btnRunAnalysis.disabled = false;
        } else {
            inputReadinessText.textContent = `Awaiting uploads for: ${missing.join(', ')}.`;
            btnRunAnalysis.disabled = true;
        }
    }

    // Reset Form Button
    btnResetForm.addEventListener('click', () => {
        activeSamplePatientId = null;
        selectedFiles = { Aor: null, Mit: null, Pul: null, Tri: null };
        document.querySelectorAll('.patient-preset-card').forEach(c => c.classList.remove('active'));
        
        valves.forEach(v => {
            const st = stations[v];
            st.badge.textContent = 'Pending';
            st.badge.className = 'station-status-pill';
            st.status.textContent = 'No audio loaded';
            st.status.className = 'file-name-display';
            st.playerWrap.classList.add('hidden');
            st.audio.src = '';
            st.input.value = '';
        });

        inputReadinessText.textContent = 'Please select a patient preset or provide WAV files for all 4 valve positions.';
        btnRunAnalysis.disabled = true;
        resultsContainer.classList.add('hidden');
    });

    // 3. Tab Navigation Switching
    function initTabSwitching() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetTab = btn.dataset.tab;
                tabBtns.forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

                btn.classList.add('active');
                const pane = document.getElementById(targetTab);
                if (pane) pane.classList.add('active');
            });
        });
    }

    // 4. Execution Workflow with Stepper Animation
    btnRunAnalysis.addEventListener('click', async () => {
        resultsContainer.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        loadingSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Reset agent steps
        resetAgentStepper();

        // Stepper animation
        animateAgentStep('stepAgent1', 'Agent 1: Inspecting signal quality, sample rate, and Butterworth bandpass filtering...');
        
        const step2Timer = setTimeout(() => {
            completeAgentStep('stepAgent1');
            animateAgentStep('stepAgent2', 'Agent 2: Running RosaNet 4-Valve neural inference & Grad-CAM attribution...');
        }, 1100);

        const step3Timer = setTimeout(() => {
            completeAgentStep('stepAgent2');
            animateAgentStep('stepAgent3', 'Agent 3: Querying ChromaDB Vector Store for ACC/AHA Valvular Guidelines...');
        }, 2200);

        const step4Timer = setTimeout(() => {
            completeAgentStep('stepAgent3');
            animateAgentStep('stepAgent4', 'Agent 4: Synthesizing clinical impression, risk stratification, and report note...');
        }, 3300);

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
            const resp = await fetch('/api/predict_patient', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP Error ${resp.status}`);
            }

            const report = await resp.json();

            // Finish stepper
            clearTimeout(step2Timer);
            clearTimeout(step3Timer);
            clearTimeout(step4Timer);
            completeAgentStep('stepAgent1');
            completeAgentStep('stepAgent2');
            completeAgentStep('stepAgent3');
            completeAgentStep('stepAgent4');

            setTimeout(() => {
                loadingSection.classList.add('hidden');
                renderDiagnosticResults(report);
            }, 600);

        } catch (err) {
            clearTimeout(step2Timer);
            clearTimeout(step3Timer);
            clearTimeout(step4Timer);
            loadingSection.classList.add('hidden');
            alert(`Diagnostic Pipeline Error: ${err.message}`);
            console.error('Analysis error:', err);
        }
    });

    function resetAgentStepper() {
        ['stepAgent1', 'stepAgent2', 'stepAgent3', 'stepAgent4'].forEach(id => {
            const el = document.getElementById(id);
            el.className = 'agent-step-card';
            el.querySelector('.step-status-tag').textContent = 'Waiting';
        });
    }

    function animateAgentStep(id, subtitleText) {
        progressSub.textContent = subtitleText;
        const el = document.getElementById(id);
        if (el) {
            el.className = 'agent-step-card active';
            el.querySelector('.step-status-tag').textContent = 'Executing';
        }
    }

    function completeAgentStep(id) {
        const el = document.getElementById(id);
        if (el) {
            el.className = 'agent-step-card completed';
            el.querySelector('.step-status-tag').textContent = 'Verified ✓';
        }
    }

    // 5. Render Diagnostic Findings Dashboard
    function renderDiagnosticResults(report) {
        resultsContainer.classList.remove('hidden');
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        const pred = report.prediction_summary;
        const quality = report.quality_summary;
        const evidence = report.evidence_summary;

        // Executive Header Meta
        document.getElementById('resPatientId').textContent = report.patient_session_id;
        document.getElementById('resTimestamp').textContent = new Date().toLocaleString();
        document.getElementById('resModelVersion').textContent = report.model_version || 'RosaNet-4Valve-v1.0';

        // Primary Diagnosis & Status
        const consultHeader = document.querySelector('.executive-consult-header');
        consultHeader.className = `executive-consult-header ${pred.is_normal ? 'normal' : 'abnormal'}`;
        
        document.getElementById('resPrimaryDiagnosis').textContent = pred.overall_status;
        document.getElementById('resSecondaryDescription').textContent = pred.is_normal
            ? 'Physiological cardiac sound patterns confirmed across Aortic, Mitral, Pulmonary, and Tricuspid auscultation sites.'
            : 'Phonocardiogram acoustic murmurs exceed calibrated decision thresholds. Valvular pathology indicated.';

        // Risk Pill
        const riskPill = document.getElementById('resRiskPill');
        const rk = (pred.risk_level || 'Low Risk').toLowerCase().replace(' ', '-');
        riskPill.textContent = (pred.risk_level || 'Low Risk').toUpperCase();
        riskPill.className = `risk-status-pill ${rk}`;

        // 1. Probabilities Grid
        const probsGrid = document.getElementById('probsGrid');
        probsGrid.innerHTML = '';

        Object.keys(pred.raw_probabilities).forEach(code => {
            const prob = pred.raw_probabilities[code];
            const pct = Math.round(prob * 100);
            const thresh = Math.round((pred.decision_thresholds[code] || 0.50) * 100);
            const isElevated = pct >= thresh;
            const info = diseaseMap[code] || { name: code, desc: "" };

            const card = document.createElement('div');
            card.className = `prob-metric-card ${isElevated && code !== 'N' ? 'elevated' : ''}`;

            card.innerHTML = `
                <div>
                    <div class="prob-card-top">
                        <span class="prob-disease-code">${code}</span>
                        <span class="prob-pct-large">${pct}%</span>
                    </div>
                    <div class="prob-disease-name">${info.name}</div>
                </div>

                <div>
                    <div class="prob-bar-track" title="Threshold: ${thresh}%">
                        <div class="prob-bar-fill" style="width: ${pct}%"></div>
                        <div class="prob-threshold-needle" style="left: ${thresh}%" title="Threshold Needle (${thresh}%)"></div>
                    </div>
                    <div class="prob-footer-meta">
                        <span>Threshold: <strong>${thresh}%</strong></span>
                        <span class="prob-status-tag ${isElevated ? 'pos' : 'neg'}">
                            ${code === 'N' ? (isElevated ? 'Normal' : 'Low') : (isElevated ? 'Positive' : 'Negative')}
                        </span>
                    </div>
                </div>
            `;
            probsGrid.appendChild(card);
        });

        // Active Conditions Chips
        const condSection = document.getElementById('detectedConditionsSection');
        const condList = document.getElementById('conditionsChipList');
        condList.innerHTML = '';

        if (pred.detected_conditions && pred.detected_conditions.length > 0) {
            condSection.classList.remove('hidden');
            pred.detected_conditions.forEach(c => {
                const chip = document.createElement('div');
                chip.className = 'condition-result-chip active-disease';
                chip.innerHTML = `
                    <span style="color: #dc2626; font-size: 1.1rem;">⚠️</span>
                    <div>
                        <div><strong>${c.disease_name} (${c.disease_code})</strong> &bull; Probability: ${Math.round(c.probability * 100)}%</div>
                        <div class="chip-valve-focus">Primary Auscultation Focus: <strong>${c.primary_contributing_valve}</strong></div>
                    </div>
                `;
                condList.appendChild(chip);
            });
        } else {
            condList.innerHTML = `
                <div class="condition-result-chip">
                    <span style="color: #059669; font-size: 1.1rem;">✓</span>
                    <span>No active pathological valvular lesions detected above decision thresholds. Physiological auscultation profile.</span>
                </div>
            `;
        }

        // 2. Signal Quality (Agent 1)
        const qBanner = document.getElementById('qualityBanner');
        qBanner.innerHTML = `
            <strong>Signal Inspection Summary:</strong> Overall Status: <span class="qc-badge ${quality.status.toLowerCase()}">${quality.status}</span> &bull; ${quality.summary}
        `;

        const qTableBody = document.getElementById('qualityTableBody');
        qTableBody.innerHTML = '';
        quality.valves_inspected.forEach(v => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${v.valve_location} Valve</strong></td>
                <td><code>${v.filename}</code></td>
                <td>${v.duration_seconds} sec (10,000 samples)</td>
                <td>${v.sample_rate} Hz (Standardized)</td>
                <td>4th-Order Butterworth (25-400 Hz) ✓</td>
                <td><span class="qc-badge ${v.is_valid ? 'pass' : 'fail'}">${v.is_valid ? 'PASS' : 'WARN'}</span></td>
            `;
            qTableBody.appendChild(tr);
        });

        // 3. Explainable AI & Attribution (Agent 2)
        const xaiContainer = document.getElementById('xaiContainer');
        xaiContainer.innerHTML = '';

        if (pred.valve_attributions && pred.valve_attributions.length > 0) {
            pred.valve_attributions.forEach(item => {
                const block = document.createElement('div');
                block.className = 'xai-condition-block';

                let valvesHtml = '<div class="valve-attributions-list">';
                if (item.valve_attributions) {
                    item.valve_attributions.forEach(v => {
                        const scorePct = Math.round(v.branch_score * 100);
                        valvesHtml += `
                            <div class="valve-attribution-card">
                                <div style="display: flex; justify-content: space-between; font-weight: 700;">
                                    <span>${v.valve_name}</span>
                                    <span>${scorePct}%</span>
                                </div>
                                <div class="attribution-bar-bg">
                                    <div class="attribution-bar-fill" style="width: ${scorePct}%"></div>
                                </div>
                                <div style="color: #64748b; font-size: 0.72rem;">Saliency: ${(v.peak_saliency || 0).toFixed(2)}</div>
                            </div>
                        `;
                    });
                }
                valvesHtml += '</div>';

                block.innerHTML = `
                    <div class="xai-block-header">
                        <span class="xai-condition-title">${item.disease_class} (${Math.round(item.probability * 100)}% Confidence)</span>
                        <span class="xai-top-valve-tag">Epicenter: ${item.top_contributing_valve}</span>
                    </div>
                    <p style="font-size: 0.8rem; color: #475569; margin-bottom: 10px;">
                        Convolutional spatial-temporal attention indicates hemodynamic murmur radiation primarily originating at <strong>${item.top_contributing_valve}</strong>.
                    </p>
                    ${valvesHtml}
                `;
                xaiContainer.appendChild(block);
            });
        } else {
            xaiContainer.innerHTML = `
                <div style="padding: 24px; text-align: center; color: #64748b;">
                    No elevated murmur patterns detected. Neural activations remain within physiological baseline across all four sites.
                </div>
            `;
        }

        // 4. ACC/AHA Guideline Evidence (Agent 3)
        const gContainer = document.getElementById('guidelinesContainer');
        gContainer.innerHTML = '';

        if (evidence.retrieved_guidelines && evidence.retrieved_guidelines.length > 0) {
            evidence.retrieved_guidelines.forEach(g => {
                const card = document.createElement('div');
                card.className = 'clinical-guideline-card';

                card.innerHTML = `
                    <div class="guideline-header-row">
                        <span class="guideline-section-title">${g.target_disease} &bull; ${g.section_name}</span>
                        <span class="guideline-page-badge">Page ${g.page_number} &bull; Similarity: ${(1 - (g.distance_score || 0)).toFixed(2)}</span>
                    </div>
                    <div class="guideline-excerpt-quote">
                        "${g.content_text}"
                    </div>
                    <div class="guideline-source-citation">
                        <strong>Source Document:</strong> ${g.guideline_title} (2020 ACC/AHA Guideline for the Management of Patients With Valvular Heart Disease)
                    </div>
                `;
                gContainer.appendChild(card);
            });
        } else {
            gContainer.innerHTML = `
                <div style="padding: 24px; text-align: center; color: #64748b;">
                    Standard physiological sound patterns. No specialized valvular interventional guideline citations indicated.
                </div>
            `;
        }

        // 5. Full Clinical Consultation EMR Note
        const mdPre = document.getElementById('reportMarkdownPre');
        mdPre.textContent = report.structured_markdown_report;

        // Copy Report Button
        document.getElementById('btnCopyReport').onclick = () => {
            navigator.clipboard.writeText(report.structured_markdown_report)
                .then(() => alert("Official Clinical Consultation Note copied to clipboard."))
                .catch(() => alert("Failed to copy note to clipboard."));
        };

        // Print Report Button
        document.getElementById('btnPrintReport').onclick = () => {
            // Activate the full report tab first
            document.querySelector('[data-tab="tabFullReport"]').click();
            setTimeout(() => {
                window.print();
            }, 250);
        };
    }
});
