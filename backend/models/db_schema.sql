-- AdityScan v3 — TimescaleDB Schema
-- Run once on fresh PostgreSQL + TimescaleDB installation:
--   psql -U postgres -d adityscan -f db_schema.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Light curve data (high-frequency, from SoLEXS + HEL1OS) ──────────────────
CREATE TABLE IF NOT EXISTS lightcurves (
    time           TIMESTAMPTZ NOT NULL,
    instrument     TEXT NOT NULL,        -- 'SoLEXS_SDD2' | 'HEL1OS_CdTe1' | ...
    band           TEXT NOT NULL,        -- e.g. '1_8keV' | '30_40keV'
    count_rate     DOUBLE PRECISION,     -- counts/s
    goes_proxy     DOUBLE PRECISION,     -- W/m² (SoLEXS only)
    quality_flag   SMALLINT DEFAULT 1    -- 1=valid, 0=bad
);
SELECT create_hypertable('lightcurves', 'time', if_not_exists => TRUE);
CREATE INDEX ON lightcurves (instrument, time DESC);

-- ── Spectral fit results (from PyXSPEC fitting runs) ─────────────────────────
CREATE TABLE IF NOT EXISTS spectral_fits (
    time           TIMESTAMPTZ NOT NULL,
    instrument     TEXT NOT NULL,        -- 'SoLEXS' | 'HEL1OS_CdTe' | 'HEL1OS_CZT'
    model          TEXT NOT NULL,        -- 'chisoth' | 'vth+bknpower' | 'bpow'
    t_mk           DOUBLE PRECISION,     -- plasma temperature (MK)
    em_norm        DOUBLE PRECISION,     -- emission measure (cm⁻³ or norm)
    logt_err       DOUBLE PRECISION,     -- log10(T) uncertainty
    norm_err       DOUBLE PRECISION,     -- EM norm uncertainty
    chi2_red       DOUBLE PRECISION,     -- reduced χ²
    gamma_lo       DOUBLE PRECISION,     -- HEL1OS low-energy photon index (power-law)
    gamma_hi       DOUBLE PRECISION,     -- HEL1OS high-energy photon index
    break_e_kev    DOUBLE PRECISION,     -- break energy (keV)
    neupert_ratio  DOUBLE PRECISION,     -- Neupert ratio (dSXR/dt / HXR proxy)
    hope_flag      BOOLEAN DEFAULT FALSE -- HOPE precursor flag
);
SELECT create_hypertable('spectral_fits', 'time', if_not_exists => TRUE);

-- ── MAG in-situ data (from Aditya-L1 MAG L2, 10-s cadence) ─────────────────
CREATE TABLE IF NOT EXISTS mag_data (
    time           TIMESTAMPTZ NOT NULL,
    bx_gse         DOUBLE PRECISION,    -- nT
    by_gse         DOUBLE PRECISION,    -- nT
    bz_gse         DOUBLE PRECISION,    -- nT
    bx_gsm         DOUBLE PRECISION,    -- nT
    by_gsm         DOUBLE PRECISION,    -- nT
    bz_gsm         DOUBLE PRECISION,    -- nT
    b_total        DOUBLE PRECISION,    -- nT
    clock_angle    DOUBLE PRECISION,    -- degrees
    cone_angle     DOUBLE PRECISION,    -- degrees
    quality_flag   SMALLINT             -- 0=bad, 1=valid (from manual Table 3)
);
SELECT create_hypertable('mag_data', 'time', if_not_exists => TRUE);

-- ── ASPEX-SWIS solar wind bulk parameters ────────────────────────────────────
CREATE TABLE IF NOT EXISTS swis_bulk (
    time              TIMESTAMPTZ NOT NULL,
    proton_density    DOUBLE PRECISION,   -- cm⁻³
    proton_temperature DOUBLE PRECISION,  -- K
    proton_speed      DOUBLE PRECISION,   -- km/s
    dynamic_pressure  DOUBLE PRECISION,   -- nPa (derived)
    density_err       DOUBLE PRECISION,
    temperature_err   DOUBLE PRECISION,
    speed_err         DOUBLE PRECISION,
    x_gse_km          DOUBLE PRECISION,  -- spacecraft position
    y_gse_km          DOUBLE PRECISION,
    z_gse_km          DOUBLE PRECISION
);
SELECT create_hypertable('swis_bulk', 'time', if_not_exists => TRUE);

-- ── GOES XRS cross-reference data ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS goes_xrs (
    time          TIMESTAMPTZ NOT NULL,
    satellite     TEXT NOT NULL,          -- 'GOES-16' | 'GOES-18'
    flux_1_8      DOUBLE PRECISION,       -- W/m², 1–8 Å (GOES class channel)
    flux_0p5_4    DOUBLE PRECISION,       -- W/m², 0.5–4 Å
    goes_class    TEXT                    -- 'M3.7', 'X1.2', etc.
);
SELECT create_hypertable('goes_xrs', 'time', if_not_exists => TRUE);
CREATE INDEX ON goes_xrs (satellite, time DESC);

-- ── ML inference results ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_predictions (
    time                   TIMESTAMPTZ NOT NULL,
    activity_mode          TEXT NOT NULL,
    goes_class_predicted   TEXT,
    flare_prob_now         DOUBLE PRECISION,   -- P(flare now)
    flare_prob_now_std     DOUBLE PRECISION,   -- MC Dropout std
    p_flare_5min           DOUBLE PRECISION,
    p_flare_10min          DOUBLE PRECISION,
    p_flare_15min          DOUBLE PRECISION,
    p_flare_30min          DOUBLE PRECISION,
    p_flare_60min          DOUBLE PRECISION,
    -- Conformal intervals for 15-min horizon (primary operational horizon)
    p_15min_lower          DOUBLE PRECISION,
    p_15min_upper          DOUBLE PRECISION,
    cme_risk               DOUBLE PRECISION,
    active_modalities      TEXT[],             -- array of active branch names
    model_version          TEXT DEFAULT '3.0.0'
);
SELECT create_hypertable('ml_predictions', 'time', if_not_exists => TRUE);

-- ── Detected flare events (catalog) ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS flare_events (
    event_id       SERIAL PRIMARY KEY,
    start_time     TIMESTAMPTZ NOT NULL,
    peak_time      TIMESTAMPTZ,
    end_time       TIMESTAMPTZ,
    goes_class     TEXT,                   -- from Aditya-L1 detection
    goes_class_ref TEXT,                   -- from GOES cross-reference
    peak_t_mk      DOUBLE PRECISION,       -- peak plasma temperature (MK)
    peak_em        DOUBLE PRECISION,       -- peak emission measure
    peak_gamma_lo  DOUBLE PRECISION,       -- peak spectral index (HXR)
    peak_flux_wm2  DOUBLE PRECISION,       -- peak GOES-proxy flux (W/m²)
    hope_flagged   BOOLEAN DEFAULT FALSE,  -- HOPE precursor detected
    cme_associated BOOLEAN,               -- NULL = unknown
    active_region  TEXT,                   -- NOAA AR number if known
    neupert_ratio  DOUBLE PRECISION,       -- Neupert diagnostic
    detection_source TEXT DEFAULT 'AdityScan', -- 'AdityScan' | 'GOES' | 'HEK'
    notes          TEXT
);
CREATE INDEX ON flare_events (start_time DESC);
CREATE INDEX ON flare_events (goes_class);

-- ── Continuous aggregates (TimescaleDB — 1-minute rollups for dashboards) ─────
CREATE MATERIALIZED VIEW IF NOT EXISTS lightcurves_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    instrument,
    band,
    AVG(count_rate) AS mean_count_rate,
    MAX(count_rate) AS max_count_rate,
    AVG(goes_proxy) AS mean_goes_proxy
FROM lightcurves
WHERE quality_flag = 1
GROUP BY bucket, instrument, band
WITH NO DATA;

SELECT add_continuous_aggregate_policy('lightcurves_1min',
    start_offset => INTERVAL '1 hour',
    end_offset   => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- ── Data retention policies (keep 90 days of raw 1-s data) ───────────────────
SELECT add_retention_policy('lightcurves',  INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('spectral_fits', INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('mag_data',      INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('swis_bulk',     INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('goes_xrs',      INTERVAL '365 days', if_not_exists => TRUE);
SELECT add_retention_policy('ml_predictions', INTERVAL '365 days', if_not_exists => TRUE);
-- flare_events: no retention policy — keep forever (catalog)
