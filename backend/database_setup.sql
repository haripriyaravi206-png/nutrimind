-- ============================================================
--  NutriMind Database Setup
--  Run this file in MySQL Workbench or terminal:
--  mysql -u root -p < database_setup.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS nutrimind_db;
USE nutrimind_db;

-- ============================================================
-- TABLE 1: users
--   Stores signup & login details
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(100)        NOT NULL,
    phone         VARCHAR(15)         NOT NULL,
    email         VARCHAR(150)        NOT NULL UNIQUE,
    password_hash VARCHAR(255)        NOT NULL,
    created_at    TIMESTAMP           DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- TABLE 2: health_forms
--   Stores the health form data submitted by the user
-- ============================================================
CREATE TABLE IF NOT EXISTS health_forms (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    user_id        INT                 NOT NULL,
    age            INT                 NOT NULL,
    gender         VARCHAR(10)         NOT NULL,
    height_cm      FLOAT               NOT NULL,
    weight_kg      FLOAT               NOT NULL,
    bmi            FLOAT               NOT NULL,
    water_intake   FLOAT               NOT NULL,
    activity_level VARCHAR(20)         NOT NULL,
    bp             INT                 NOT NULL,
    sugar          FLOAT               NOT NULL,
    alcohol        TINYINT(1)          NOT NULL DEFAULT 0,
    smoking        TINYINT(1)          NOT NULL DEFAULT 0,
    health_issues  VARCHAR(255)        DEFAULT NULL,
    submitted_at   TIMESTAMP           DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- ============================================================
-- TABLE 3: diet_results
--   Stores the ML prediction + 30-day diet plan result
-- ============================================================
CREATE TABLE IF NOT EXISTS diet_results (
    id                   INT AUTO_INCREMENT PRIMARY KEY,
    user_id              INT             NOT NULL,
    form_id              INT             NOT NULL,
    predicted_condition  VARCHAR(50)     NOT NULL,
    bmi                  FLOAT           NOT NULL,
    diet_plan_json       LONGTEXT        NOT NULL,
    generated_at         TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (form_id) REFERENCES health_forms(id) ON DELETE CASCADE
);

-- ============================================================
-- Indexes for faster lookups
-- ============================================================
CREATE INDEX idx_users_email   ON users(email);
CREATE INDEX idx_forms_user    ON health_forms(user_id);
CREATE INDEX idx_results_user  ON diet_results(user_id);

-- ============================================================
-- Verify
-- ============================================================
SHOW TABLES;
DESC users;
DESC health_forms;
DESC diet_results;
