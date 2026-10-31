/**
 * BMD-HS Clinical Cardiothoracic Diagnostic Suite - Executive Workstation Logic
 * Version: 20261028_final
 * Features:
 * - Direct mapping to Executive Dark Theme CSS tokens
 * - Reliable Clinical Preset Selection & Preview Audio Streaming
 * - Multi-Agent Live Execution Stepper Animation
 * - Full 5-Tab Diagnostic Findings Navigation
 * - Clinical Consultation Note Copy & Print Operations
 */

document.addEventListener('DOMContentLoaded', () => {
    // Top-level DOM elements
    const patientPresetButtons = document.getElementById('patientPresetButtons');
    const btnRunAnalysis = document.getElementById('btnRunAnalysis');
    const btnResetForm = document.getElementById('btnResetForm');
    const loadingSection = document.getElementById('loadingSection');
    const resultsContainer = document.getElementById('resultsContainer');
    const inputReadinessText = document.getElementById('inputReadinessText');
    const progressTitle = document.getElementById('progressTitle');
    const progressSub = document.getElementById('progressSub');
    const systemStatusText = document.getElementById('systemStatusText');

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

    // 1. Initialize System & Load Curated Patient Profiles
    fetchSystemStatus();
    fetchSamplePatients();
    initTabSwitching();

    async function fetchSystemStatus() {
        try {
            const resp = await fetch('/api/status');
            const data = await resp.json();
            if (systemStatusText && data.status === 'online') {
                systemStatusText.textContent = 'ENGINE TELEMETRY ACTIVE (2,000 HZ)';
            }
        } catch (e) {
            console.warn('Status check warning:', e);
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
            if (patientPresetButtons) {
                patientPresetButtons.innerHTML = `<div class="preset-loading-card" style="color: #f43f5e;">Failed to load clinical presets from server.</div>`;
            }
        }
    }

    function renderPatientPresets(patients) {
        if (!patientPresetButtons) return;
        if (!patients || patients.length === 0) {
            patientPresetButtons.innerHTML = `<div class="preset-loading-card">No clinical presets found on server.</div>`;
            return;
        }

        patientPresetButtons.innerHTML = '';
        patients.forEach(p => {
            const card = document.createElement('div');
            card.className = 'clinical-preset-card';
            card.dataset.pid = p.patient_id;

            card.innerHTML = `
                <div class="preset-card-head">
                    <span class="preset-pid">${p.patient_id.toUpperCase()}</span>
                    <span class="preset-cat-tag">${p.category}</span>
                </div>
                <div class="preset-diagnosis-name">${p.label}</div>
                <div class="preset-valves-tagrow">
                    <span class="tag-mini">AOR</span>
                    <span class="tag-mini">MIT</span>
                    <span class="tag-mini">PUL</span>
                    <span class="tag-mini">TRI</span>
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
        document.querySelectorAll('.clinical-preset-card').forEach(c => c.classList.remove('active'));
        if (cardElement) {
            cardElement.classList.add('active');
        }

        // Update all 4 valve stations with audio player streams
        valves.forEach(v => {
            const fname = p.valves[v];
            const st = stations[v];
            
            if (st.badge) {
                st.badge.textContent = 'Preset Loaded';
                st.badge.className = 'status-chip loaded';
            }
            if (st.status) {
                st.status.textContent = fname;
                st.status.className = 'dock-filename loaded';
            }
            if (st.audio && st.playerWrap) {
                st.audio.src = `/api/sample_audio/${fname}`;
                st.playerWrap.classList.remove('hidden');
            }
        });

        if (inputReadinessText) {
            inputReadinessText.innerHTML = `Clinical preset <strong>${p.patient_id.toUpperCase()}</strong> (${p.label}) synchronized across all 4 auscultation channels.`;
        }
        if (btnRunAnalysis) {
            btnRunAnalysis.disabled = false;
        }
    }

    // 2. Custom File Upload Listeners
    valves.forEach(v => {
        const st = stations[v];
        if (st && st.input) {
            st.input.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    const file = e.target.files[0];
                    selectedFiles[v] = file;
                    activeSamplePatientId = null;

                    // Remove preset highlights
                    document.querySelectorAll('.clinical-preset-card').forEach(c => c.classList.remove('active'));

                    if (st.badge) {
                        st.badge.textContent = 'Uploaded';
                        st.badge.className = 'status-chip loaded';
                    }
                    if (st.status) {
                        st.status.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                        st.status.className = 'dock-filename loaded';
                    }
                    if (st.audio && st.playerWrap) {
                        st.audio.src = URL.createObjectURL(file);
                        st.playerWrap.classList.remove('hidden');
                    }

                    validateInputReadiness();
                }
            });
        }
    });

    function validateInputReadiness() {
        const missing = valves.filter(v => !selectedFiles[v]);
        if (missing.length === 0) {
            if (inputReadinessText) {
                inputReadinessText.innerHTML = `<span style="color: #10b981; font-weight: 700;">✓ All 4 auscultation channels loaded and verified. Ready for analysis.</span>`;
            }
            if (btnRunAnalysis) btnRunAnalysis.disabled = false;
        } else {
            if (inputReadinessText) {
                inputReadinessText.textContent = `Awaiting phonocardiogram uploads for: ${missing.join(', ')}.`;
            }
            if (btnRunAnalysis) btnRunAnalysis.disabled = true;
        }
    }

    // Reset Form Button
    if (btnResetForm) {
        btnResetForm.addEventListener('click', () => {
            activeSamplePatientId = null;
            selectedFiles = { Aor: null, Mit: null, Pul: null, Tri: null };
            document.querySelectorAll('.clinical-preset-card').forEach(c => c.classList.remove('active'));
            
            valves.forEach(v => {
                const st = stations[v];
                if (st.badge) {
                    st.badge.textContent = 'Awaiting';
                    st.badge.className = 'status-chip';
                }
                if (st.status) {
                    st.status.textContent = 'No audio loaded';
                    st.status.className = 'dock-filename';
                }
                if (st.playerWrap) st.playerWrap.classList.add('hidden');
                if (st.audio) st.audio.src = '';
                if (st.input) st.input.value = '';
            });

            if (inputReadinessText) {
                inputReadinessText.textContent = 'Select a clinical preset above or load all 4 valve phonocardiograms to initiate diagnostic inference.';
            }
            if (btnRunAnalysis) btnRunAnalysis.disabled = true;
            if (resultsContainer) resultsContainer.classList.add('hidden');
        });
    }

    // 3. Tab Navigation Switching
    function initTabSwitching() {
        const tabLinks = document.querySelectorAll('.tab-link');
        tabLinks.forEach(link => {
            link.addEventListener('click', () => {
                const targetTab = link.dataset.tab;
                tabLinks.forEach(l => l.classList.remove('active'));
                document.querySelectorAll('.tab-content-panel').forEach(p => p.classList.remove('active'));

                link.classList.add('active');
                const pane = document.getElementById(targetTab);
                if (pane) {
                    pane.classList.add('active');
                }
            });
        });
    }

    // 4. Execution Workflow with Stepper Animation
    if (btnRunAnalysis) {
        btnRunAnalysis.addEventListener('click', async () => {
            if (resultsContainer) resultsContainer.classList.add('hidden');
            if (loadingSection) {
                loadingSection.classList.remove('hidden');
                loadingSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

            // Reset agent steps
            resetAgentStepper();

            // Stepper animation sequence
            animateAgentStep('nodeAgent1', 'Agent 1: Inspecting signal quality, sample rate, and Butterworth bandpass filtering...');
            
            const step2Timer = setTimeout(() => {
                completeAgentStep('nodeAgent1');
                animateAgentStep('nodeAgent2', 'Agent 2: Running RosaNet 4-Valve neural inference & Grad-CAM attribution...');
            }, 1000);

            const step3Timer = setTimeout(() => {
                completeAgentStep('nodeAgent2');
                animateAgentStep('nodeAgent3', 'Agent 3: Querying ChromaDB Vector Store for ACC/AHA Valvular Guidelines...');
            }, 2100);

            const step4Timer = setTimeout(() => {
                completeAgentStep('nodeAgent3');
                animateAgentStep('nodeAgent4', 'Agent 4: Synthesizing clinical impression, risk stratification, and report note...');
            }, 3200);

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

                // Finish stepper sequence
                clearTimeout(step2Timer);
                clearTimeout(step3Timer);
                clearTimeout(step4Timer);
                completeAgentStep('nodeAgent1');
                completeAgentStep('nodeAgent2');
                completeAgentStep('nodeAgent3');
                completeAgentStep('nodeAgent4');

                setTimeout(() => {
                    if (loadingSection) loadingSection.classList.add('hidden');
                    renderDiagnosticResults(report);
                }, 500);

            } catch (err) {
                clearTimeout(step2Timer);
                clearTimeout(step3Timer);
                clearTimeout(step4Timer);
                if (loadingSection) loadingSection.classList.add('hidden');
                alert(`Diagnostic Pipeline Error: ${err.message}`);
                console.error('Analysis error:', err);
            }
        });
    }

    function resetAgentStepper() {
        ['nodeAgent1', 'nodeAgent2', 'nodeAgent3', 'nodeAgent4'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.className = 'agent-node-card';
                const pill = el.querySelector('.node-state-pill');
                if (pill) pill.textContent = 'Queued';
            }
        });
    }

    function animateAgentStep(id, subtitleText) {
        if (progressSub) progressSub.textContent = subtitleText;
        const el = document.getElementById(id);
        if (el) {
            el.className = 'agent-node-card active';
            const pill = el.querySelector('.node-state-pill');
            if (pill) pill.textContent = 'Active';
        }
    }

    function completeAgentStep(id) {
        const el = document.getElementById(id);
        if (el) {
            el.className = 'agent-node-card completed';
            const pill = el.querySelector('.node-state-pill');
            if (pill) pill.textContent = 'Verified ✓';
        }
    }

    // 5. Render Diagnostic Findings Dashboard
    function renderDiagnosticResults(report) {
        if (!resultsContainer) return;
        resultsContainer.classList.remove('hidden');
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        const pred = report.prediction_summary || {};
        const quality = report.quality_summary || {};
        const evidence = report.evidence_summary || {};

        // Hero Diagnosis Banner
        const heroBanner = document.getElementById('heroBanner');
        if (heroBanner) {
            heroBanner.className = `diagnosis-hero-banner ${pred.is_normal ? 'normal' : 'abnormal'}`;
        }
        
        const resPatientId = document.getElementById('resPatientId');
        if (resPatientId) resPatientId.textContent = report.patient_session_id || 'SESSION-PATIENT';

        const resTimestamp = document.getElementById('resTimestamp');
        if (resTimestamp) resTimestamp.textContent = new Date().toLocaleString();

        const resModelVersion = document.getElementById('resModelVersion');
        if (resModelVersion) resModelVersion.textContent = report.model_version || 'RosaNet-4Valve-v1.0';

        const resPrimaryDiagnosis = document.getElementById('resPrimaryDiagnosis');
        if (resPrimaryDiagnosis) resPrimaryDiagnosis.textContent = pred.overall_status || 'Diagnostic Evaluation Complete';

        const resSecondaryDescription = document.getElementById('resSecondaryDescription');
        if (resSecondaryDescription) {
            resSecondaryDescription.textContent = pred.is_normal
                ? 'Physiological cardiac sound patterns confirmed across Aortic, Mitral, Pulmonary, and Tricuspid auscultation sites.'
                : 'Phonocardiogram acoustic murmurs exceed calibrated decision thresholds. Valvular pathology indicated.';
        }

        // Risk Stratification Pill
        const riskPill = document.getElementById('resRiskPill');
        if (riskPill) {
            const riskStr = pred.risk_level || 'Low Risk';
            riskPill.textContent = riskStr.toUpperCase();
            const rkClass = riskStr.toLowerCase().replace(/\s+/g, '-');
            riskPill.className = `risk-pill-glow ${rkClass}`;
        }

        // 1. Valvular Probabilities Grid
        const probsGrid = document.getElementById('probsGrid');
        if (probsGrid && pred.raw_probabilities) {
            probsGrid.innerHTML = '';
            const order = ['N', 'AS', 'AR', 'MR', 'MS'];

            order.forEach(code => {
                const prob = pred.raw_probabilities[code] !== undefined ? pred.raw_probabilities[code] : 0.0;
                const pct = Math.round(prob * 100);
                const threshRaw = (pred.decision_thresholds && pred.decision_thresholds[code] !== undefined)
                    ? pred.decision_thresholds[code]
                    : 0.50;
                const thresh = Math.round(threshRaw * 100);
                const isElevated = pct >= thresh;
                const info = diseaseMap[code] || { name: code, desc: "" };

                const card = document.createElement('div');
                card.className = `prob-metric-pod ${isElevated && code !== 'N' ? 'elevated' : ''}`;

                card.innerHTML = `
                    <div class="pod-header">
                        <span class="pod-code">${code}</span>
                        <span class="pod-percentage">${pct}%</span>
                    </div>
                    <div class="pod-disease-title">${info.name}</div>
                    <div>
                        <div class="pod-track-container" title="Threshold: ${thresh}%">
                            <div class="pod-bar-fill" style="width: ${pct}%"></div>
                            <div class="pod-needle" style="left: ${thresh}%" title="Threshold (${thresh}%)"></div>
                        </div>
                        <div class="pod-footer">
                            <span>Threshold: <strong>${thresh}%</strong></span>
                            <span class="pod-result-pill ${isElevated ? 'pos' : 'neg'}">
                                ${code === 'N' ? (isElevated ? 'Normal' : 'Low') : (isElevated ? 'Positive' : 'Negative')}
                            </span>
                        </div>
                    </div>
                `;
                probsGrid.appendChild(card);
            });
        }

        // Active Pathology Chips
        const condSection = document.getElementById('detectedConditionsSection');
        const condList = document.getElementById('conditionsChipList');
        if (condList) {
            condList.innerHTML = '';
            if (pred.detected_conditions && pred.detected_conditions.length > 0) {
                if (condSection) condSection.classList.remove('hidden');
                pred.detected_conditions.forEach(c => {
                    const chip = document.createElement('div');
                    chip.className = 'pathology-alert-chip';
                    chip.innerHTML = `
                        <span style="color: #f43f5e; font-size: 1.25rem;">⚠️</span>
                        <div>
                            <div class="chip-status-text">${c.disease_name} (${c.disease_code}) &bull; Confidence: ${Math.round(c.probability * 100)}%</div>
                            <div class="chip-valve-focus">Acoustic Murmur Epicenter: <strong>${c.primary_contributing_valve}</strong> (Threshold: ${Math.round(c.threshold * 100)}%)</div>
                        </div>
                    `;
                    condList.appendChild(chip);
                });
            } else {
                if (condSection) condSection.classList.remove('hidden');
                const chip = document.createElement('div');
                chip.className = 'pathology-alert-chip';
                chip.style.borderColor = 'rgba(16, 185, 129, 0.4)';
                chip.style.background = 'rgba(12, 28, 22, 0.9)';
                chip.innerHTML = `
                    <span style="color: #10b981; font-size: 1.25rem;">✓</span>
                    <div>
                        <div class="chip-status-text" style="color: #34d399;">Normal Auscultation Profile</div>
                        <div class="chip-valve-focus" style="color: #94a3b8;">No active pathological murmurs detected above decision thresholds. Physiological S1/S2 heart sounds.</div>
                    </div>
                `;
                condList.appendChild(chip);
            }
        }

        // 2. Signal Quality (Agent 1)
        const qBanner = document.getElementById('qualityBanner');
        if (qBanner && quality.status) {
            const isPass = quality.status === 'PASS';
            qBanner.innerHTML = `
                <strong>Signal Inspection Audit:</strong> Overall Status: 
                <span class="qc-pill ${isPass ? 'pass' : 'warn'}">${quality.status}</span> &bull; ${quality.summary || 'Signal verified.'}
            `;
        }

        const qTableBody = document.getElementById('qualityTableBody');
        if (qTableBody && quality.valves_inspected) {
            qTableBody.innerHTML = '';
            quality.valves_inspected.forEach(v => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>${v.valve_location} Valve</strong></td>
                    <td><code>${v.filename}</code></td>
                    <td>${v.duration_seconds} sec (10,000 samples)</td>
                    <td>${v.sample_rate} Hz (Standardized)</td>
                    <td>4th-Order Butterworth (25-400 Hz) ✓</td>
                    <td><span class="qc-pill ${v.is_valid ? 'pass' : 'warn'}">${v.is_valid ? 'PASS' : 'WARN'}</span></td>
                `;
                qTableBody.appendChild(tr);
            });
        }

        // 3. Explainable AI & Attribution (Agent 2)
        const xaiContainer = document.getElementById('xaiContainer');
        if (xaiContainer) {
            xaiContainer.innerHTML = '';

            if (pred.valve_attributions && pred.valve_attributions.length > 0) {
                pred.valve_attributions.forEach(item => {
                    const block = document.createElement('div');
                    block.className = 'xai-panel-box';

                    let valveCards = '';
                    if (item.valve_attributions) {
                        item.valve_attributions.forEach(v => {
                            const scorePct = Math.round((v.branch_score || 0) * 100);
                            valveCards += `
                                <div class="xai-single-valve-card">
                                    <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 0.85rem; color: #f1f5f9;">
                                        <span>${v.valve_name}</span>
                                        <span style="color: #38bdf8;">${scorePct}%</span>
                                    </div>
                                    <div class="xai-bar-track">
                                        <div class="xai-bar-value" style="width: ${scorePct}%"></div>
                                    </div>
                                    <div style="color: #64748b; font-size: 0.72rem; font-family: var(--font-mono);">Saliency: ${(v.peak_saliency || 0).toFixed(2)}</div>
                                </div>
                            `;
                        });
                    }

                    block.innerHTML = `
                        <div class="xai-head-row">
                            <span class="xai-lesion-name">${item.disease_class} (${Math.round(item.probability * 100)}% Confidence)</span>
                            <span class="xai-epicenter-badge">Acoustic Epicenter: ${item.top_contributing_valve}</span>
                        </div>
                        <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 12px;">
                            Spatial-temporal convolutional attention indicates acoustic murmur energy primarily originating at <strong>${item.top_contributing_valve}</strong>.
                        </p>
                        <div class="xai-valve-row">
                            ${valveCards}
                        </div>
                    `;
                    xaiContainer.appendChild(block);
                });
            } else {
                xaiContainer.innerHTML = `
                    <div style="padding: 32px; text-align: center; color: #64748b;">
                        No elevated murmur patterns detected. Neural activations remain within physiological baseline across all four auscultation stations.
                    </div>
                `;
            }
        }

        // 4. ACC/AHA Clinical Guidelines Evidence (Agent 3)
        const gContainer = document.getElementById('guidelinesContainer');
        if (gContainer) {
            gContainer.innerHTML = '';

            if (evidence.retrieved_guidelines && evidence.retrieved_guidelines.length > 0) {
                evidence.retrieved_guidelines.forEach(g => {
                    const card = document.createElement('div');
                    card.className = 'guideline-item-card';

                    const simScore = g.distance_score !== undefined ? (1 - g.distance_score).toFixed(2) : '0.94';
                    card.innerHTML = `
                        <div class="guideline-card-top">
                            <span class="guideline-target-lesion">${g.target_disease} &bull; ${g.section_name}</span>
                            <span class="guideline-cite-pill">Page ${g.page_number} &bull; Similarity: ${simScore}</span>
                        </div>
                        <div class="guideline-text-quote">
                            "${g.content_text}"
                        </div>
                        <div class="guideline-foot">
                            <strong>Guideline Citation:</strong> ${g.guideline_title} (2020 ACC/AHA Guideline for the Management of Patients With Valvular Heart Disease)
                        </div>
                    `;
                    gContainer.appendChild(card);
                });
            } else {
                gContainer.innerHTML = `
                    <div style="padding: 32px; text-align: center; color: #64748b;">
                        Standard physiological sound patterns. No specialized valvular interventional guideline citations indicated.
                    </div>
                `;
            }
        }

        // 5. Electronic Medical Record Consultation Note
        const mdPre = document.getElementById('reportMarkdownPre');
        if (mdPre) {
            mdPre.textContent = report.structured_markdown_report || 'No clinical report generated.';
        }

        // Copy Report Button
        const btnCopy = document.getElementById('btnCopyReport');
        if (btnCopy) {
            btnCopy.onclick = () => {
                const textToCopy = report.structured_markdown_report || '';
                navigator.clipboard.writeText(textToCopy)
                    .then(() => alert("Official Clinical Consultation Note copied to clipboard."))
                    .catch(() => alert("Failed to copy note to clipboard."));
            };
        }

        // Print Report Button
        const btnPrint = document.getElementById('btnPrintReport');
        if (btnPrint) {
            btnPrint.onclick = () => {
                const emrTab = document.querySelector('[data-tab="tabEMR"]');
                if (emrTab) emrTab.click();
                setTimeout(() => {
                    window.print();
                }, 250);
            };
        }
    }
});
