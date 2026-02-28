-- BLUR Observation Collector — Database Schema
-- Run once: mysql -u blurse_wxcoll -p blurse_weather < scripts/init_db.sql

CREATE TABLE IF NOT EXISTS stations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    station_code VARCHAR(20) NOT NULL,
    source ENUM('smhi', 'met_no', 'dmi', 'fmi') NOT NULL,
    name VARCHAR(100) NOT NULL,
    latitude DECIMAL(8,5) NOT NULL,
    longitude DECIMAL(8,5) NOT NULL,
    country CHAR(2) NOT NULL,
    race_area VARCHAR(50),
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_station_code (station_code)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS observations (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    station_code VARCHAR(20) NOT NULL,
    observed_at DATETIME NOT NULL,
    wind_speed_ms DECIMAL(5,2),
    wind_direction_deg DECIMAL(5,1),
    air_pressure_hpa DECIMAL(6,1),
    air_temperature_c DECIMAL(4,1),
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY idx_station_time (station_code, observed_at),
    INDEX idx_observed_at (observed_at),
    INDEX idx_station_observed (station_code, observed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS collection_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source ENUM('smhi', 'met_no', 'dmi', 'fmi') NOT NULL,
    stations_queried INT DEFAULT 0,
    observations_inserted INT DEFAULT 0,
    observations_skipped INT DEFAULT 0,
    errors INT DEFAULT 0,
    error_detail TEXT,
    duration_seconds DECIMAL(6,2)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
