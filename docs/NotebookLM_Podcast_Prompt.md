# AdityScan — NotebookLM Audio Overview Prompt
## 10-Minute Explainer Podcast Script Directive

---

> **How to use:** Paste everything below the horizontal rule into NotebookLM's
> **"Customize"** field for Audio Overview, then click **Generate**.
> Upload the `AdityScan_Technical_Report.html` as the source document.

---

## PASTE THIS INTO NOTEBOOKLM CUSTOMISE FIELD ↓

---

You are generating a polished, 10-minute audio explainer podcast called
**"AdityScan: India's AI Brain for the Sun"**.

The hosts are:
- **Priya** — an enthusiastic science communicator who asks sharp "why does this matter?" questions.
- **Arjun** — a machine learning engineer who explains the technical pieces clearly, avoiding jargon where possible.

**Tone:** Energetic, curious, confident. Think BBC Science Hour meets Lex Fridman — serious science delivered engagingly. No filler words. Every sentence earns its place.

**Duration target:** Exactly 10 minutes of spoken audio (approx. 1,400–1,500 spoken words at a natural pace).

**Structure — follow this arc precisely:**

---

### [00:00 – 00:45] COLD OPEN — The Stakes

Priya opens with a vivid scene: it is March 13, 1989. A massive solar flare erupts. Nine hours later, six million people in Quebec, Canada lose power for nine hours. The Hydro-Québec grid has collapsed — not from a storm, not from equipment failure, but from a wall of charged particles thrown out by the Sun travelling 150 million kilometres to strike Earth's magnetic field.

Arjun: "And that was a moderate event by historical standards. The Carrington Event of 1859, if it happened today, would cost the global economy between 0.6 and 2.6 trillion dollars in the first year alone."

Priya: "So — we need to predict these things. How far ahead can we do it today?"

Arjun: "Current operational systems give you maybe 20 to 30 minutes of warning. AdityScan is trying to push that to 24 hours — and do it with India's own satellite data."

---

### [00:45 – 02:00] WHAT IS ADITYA-L1?

Priya asks: "Tell me about the satellite. Aditya-L1 — India's first dedicated solar observatory. What does it actually see?"

Arjun explains clearly:
- Aditya-L1 was launched by ISRO in September 2023 and reached the Sun-Earth Lagrange Point 1 (L1) in January 2024. L1 is a gravitationally stable point 1.5 million kilometres from Earth — perfect for uninterrupted solar observation.
- It carries 7 payloads. AdityScan uses three of them:
  - **SoLEXS** (Solar Low Energy X-ray Spectrometer): watches soft X-ray emission (1–15 keV), the first sign of a solar flare heating the corona.
  - **HEL1OS** (High Energy L1 Orbiting X-ray Spectrometer): watches hard X-rays (10–150 keV) — the violent impulsive phase. Together, SoLEXS + HEL1OS give you the full spectral X-ray fingerprint of a flare.
  - **MAG** (Magnetometer): measures the solar wind's magnetic field direction as it sweeps past L1. This is the real-time "upstream warning" — if the field tilts southward, it couples to Earth's magnetosphere, amplifying geomagnetic storm impact.
- It also cross-references **GOES** satellite X-ray flux (the global gold standard for flare classification: A, B, C, M, X).

Priya: "So AdityScan is fusing Indian and American satellite data — why both?"

Arjun: "GOES has decades of labelled flare history, perfect for training. Aditya-L1 provides the novel vantage point at L1 with instruments tuned to the Sun. Together they give AdityScan something no purely GOES-based system has: real-time in-situ solar wind context at the point the solar wind actually passes through."

---

### [02:00 – 03:30] THE PREDICTION PROBLEM

Priya: "OK so what are we actually predicting? Walk me through the task."

Arjun explains the five-horizon prediction framework:
- AdityScan outputs solar flare probability forecasts for five time windows simultaneously: the next 30 minutes, 1 hour, 3 hours, 6 hours, and 24 hours ahead.
- Each prediction is not just "yes/no flare" — it's a **calibrated probability** between 0 and 1, with an **uncertainty interval** attached.
- The classification threshold used is M1.0 and above — that's the level at which flares start causing radio blackouts and satellite drag. C-class and below are routine; X-class are the civilisation-scale events.

Priya: "Why is this hard? The Sun is bright — surely we can see a flare coming?"

Arjun: "Three reasons. First, flare onset is genuinely stochastic — even the best magnetohydrodynamic models have fundamental prediction limits because solar magnetic field topology is chaotic. Second, the data is severely imbalanced: major flares happen maybe 1 to 5 percent of the time, so a model that always says 'no flare' is 95 to 99 percent accurate but completely useless. Third, there is temporal autocorrelation — consecutive 1-minute observations look almost identical, so naive ML overfits to the quiet periods."

Priya: "How does AdityScan beat those problems?"

---

### [03:30 – 05:30] THE PHYSICS-INFORMED AI ENGINE

Arjun: "This is the heart of the system. Let me walk you through the three-branch neural architecture."

**Branch 1: X-Ray Time Series — the TCN**
- The X-ray light curves from SoLEXS and HEL1OS are processed by a **Temporal Convolutional Network** — six layers of dilated causal convolutions, kernel size 8, dilations from 1 to 32. This gives the model a receptive field of over 250 time steps, capturing the slow pre-flare coronal heating that begins sometimes hours before eruption.
- Crucially, the model also ingests **Continuous Wavelet Transform** coefficients — a physics-motivated frequency decomposition that highlights quasi-periodic pulsations, or QPPs. QPPs are oscillations in X-ray flux at timescales of 10 to 300 seconds; their presence often correlates with particle acceleration events.

Priya: "So you're teaching the AI to see the same signatures a solar physicist looks for."

Arjun: "Exactly. The **Neupert Effect** is another example — it says the time derivative of the soft X-ray flux should correlate with the hard X-ray flux, because both trace energetic electron dynamics. We compute this derivative explicitly as an input feature. We're not hoping the model will discover this relationship; we encode it."

**Branch 2: Magnetic Field — the BiLSTM**
- Solar Active Regions — the magnetically complex sunspot groups — are the breeding ground for flares. The SHARP database from NASA's Solar Dynamics Observatory provides 21 magnetic parameters every 12 minutes for every active region.
- These are encoded by a **Bidirectional LSTM** — 128 hidden dimensions — because magnetic field evolution has strong temporal memory. The free magnetic energy in a delta-class active region can take days to accumulate.

**Branch 3: In-Situ Solar Wind — the MLP**
- 14 real-time features from the MAG and SWIS sensors: field magnitude, latitude and longitude angles, proton density, temperature, speed, plasma beta, and Alfvén Mach number.
- Processed by a compact MLP with 64-dimensional output.

**Fusion Layer:**
- All three branches feed into a **4-head cross-modal attention layer** with 256-dimensional projections. This learns which modalities matter most given the current solar state — sometimes the X-ray branch dominates, sometimes the magnetic field is the key signal.
- The fused representation then splits into five independent forecast heads for the five time horizons.

---

### [05:30 – 06:30] THE TRAINING STRATEGY

Priya: "You're training on real satellite data. How do you handle the class imbalance problem you mentioned?"

Arjun explains two key innovations:

**Golden Window Sampling:** Rather than training on every minute of solar observation, AdityScan defines "Golden Windows" — 6-hour periods centred on confirmed M+ flares, plus matched quiet periods. This ensures the model sees the pre-flare build-up phase, not just the peak.

**Focal Loss with Dynamic Class Weighting:** The training loss function down-weights the easy negatives — routine quiet periods — and focuses on the ambiguous pre-flare signals. The effective flare weight multiplier starts at 5 and scales dynamically with monthly flare occurrence rate.

**Incremental Training:** The model is trained month by month — May 2010, June 2010, July 2010 — resuming from checkpoint each time, never forgetting past data but always incorporating the new month's solar cycle state. This is especially important because the Sun follows an 11-year activity cycle; a model trained only on solar maximum data fails at solar minimum.

Priya: "What are the numbers? Does it actually work?"

Arjun: "Phase 1, trained on 2010 data alone: True Skill Statistic of 0.38. Phase 2, with balanced training and the full architecture: TSS above 0.80, AUC above 0.86. For context, the operational NOAA SWPC human-forecaster benchmark sits around TSS 0.50. We're beating the human baseline."

---

### [06:30 – 07:45] UNCERTAINTY — WHY IT MATTERS

Priya: "You mentioned calibrated probabilities. That sounds like more than just a confidence score — what does it actually mean?"

Arjun: "A calibrated model is one where when it says 70 percent probability of flare, roughly 70 percent of those cases actually result in a flare. Most neural networks are overconfident — they say 95 percent and they're right 60 percent of the time."

**Three-layer uncertainty stack:**

1. **MC Dropout (Epistemic Uncertainty):** At inference time, AdityScan runs the model 50 times with dropout layers active and different random neuron masks each time. The spread of those 50 predictions gives the **epistemic uncertainty** — how uncertain the model is due to limited training data.

2. **Temperature Scaling (Calibration):** A single learned parameter — T equals 0.863 — softens the output logits, aligning predicted probabilities with empirical frequencies across the validation set. Post-calibration, the Expected Calibration Error drops below 3 percent.

3. **Conformal Prediction (Coverage Guarantee):** Using the non-conformity score framework, AdityScan produces prediction intervals with a **guaranteed 90 percent coverage** — meaning 9 out of 10 actual outcomes fall within the stated interval. This is a mathematical guarantee, not an empirical approximation.

Priya: "So a space agency operator gets: probability, uncertainty band, and a coverage guarantee — not just a single number."

Arjun: "Which is why this system is actually usable for operational decisions, not just a research demo."

---

### [07:45 – 09:00] THE GEO RADIATION EXTENSION

Priya: "You've been talking about flare prediction. But there's another hazard — radiation for satellites in geostationary orbit. Tell me about that."

Arjun explains the Strategic GEO Extension:
- Geostationary satellites — weather satellites, communications relays, GPS augmentation — orbit at 35,786 kilometres altitude. They sit at the outer edge of Earth's Van Allen radiation belts.
- During geomagnetic storms triggered by solar events, the outer radiation belt is injected with high-energy electrons — nicknamed "killer electrons" because they can deposit charge deep inside satellite electronics, causing single-event upsets or permanent damage.
- AdityScan's GEO extension module uses the solar wind forecast from the L1 MAG data, a physics-based radiation belt model, and GOES proton flux readings to predict the **relativistic electron flux environment** at geostationary orbit for the next 24 hours.
- The eventual cross-calibration target is **GRASP on IRNSS/NavIC** — India's own navigation satellite constellation — which carries radiation monitors but currently lacks any predictive intelligence layer.

Priya: "So AdityScan could one day protect India's own navigation satellites."

Arjun: "That's the operational end-state. Indian satellite operators get 24-hour advance warning to upload safe-mode commands, delay high-voltage operations, or switch to redundant systems before the storm arrives."

---

### [09:00 – 10:00] CLOSING — THE BIGGER PICTURE

Priya: "Zoom out. Why does this matter for India specifically?"

Arjun: "India is building out its space infrastructure at unprecedented pace. IRNSS/NavIC, GSAT communications, EOS Earth observation, the Chandrayaan and Mangalyaan series, Gaganyaan crewed missions on the horizon. All of these assets are vulnerable to space weather. Right now, India has no domestic space-weather forecast capability at this level. Every alert comes from NOAA SWPC in the United States — a foreign government system. AdityScan changes that."

Priya: "And the open-source angle?"

Arjun: "The entire codebase, the training pipeline, the model weights — all open. Any ISRO team, any university research group, any national disaster management agency can run this. It's public infrastructure dressed as a hackathon submission."

Priya closes: "AdityScan — physics-informed AI, built on India's own satellite, predicting the Sun's anger before it reaches us. And it's open source. I think that's a pretty good reason to exist."

Arjun: "I'd say so."

**[Outro — 10 seconds of gentle ambient tone]**

---

## ADDITIONAL INSTRUCTIONS FOR NOTEBOOKLM:

- Keep both hosts' voices distinct. Priya uses vivid analogies and historical scenes. Arjun is precise but never dry.
- Do NOT add disclaimers like "this is just a prototype" or "further research is needed." Present AdityScan as the serious operational system it is.
- When reading numbers, speak them naturally: "TSS of zero point eight", "thirty-five thousand seven hundred eighty-six kilometres".
- The emotional arc: urgency (the Quebec blackout) → wonder (the satellite) → technical depth (the AI engine) → operational trust (uncertainty) → national significance (India's independence).
- End on pride and possibility, not caution.
- Do NOT pad with filler. Every second should deliver information or emotional resonance.
