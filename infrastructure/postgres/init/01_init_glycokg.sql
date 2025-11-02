-- Initialize GlycoKG PostgreSQL Database
-- This script sets up the core tables for caching and metadata

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS glycokg;
CREATE SCHEMA IF NOT EXISTS cache;
CREATE SCHEMA IF NOT EXISTS metadata;

-- Set search path
SET search_path = glycokg, cache, metadata, public;

-- Data source tracking
CREATE TABLE metadata.data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL UNIQUE,
    base_url VARCHAR(500),
    api_version VARCHAR(20),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB
);

-- Insert initial data sources
INSERT INTO metadata.data_sources (name, base_url, api_version, metadata) VALUES
('GlyTouCan', 'https://glytoucan.org/', 'v1', '{"description": "International glycan repository", "total_structures": 240000}'),
('GlyGen', 'https://glygen.org/', 'v2', '{"description": "Protein glycosylation database", "focus": "protein-glycan associations"}'),
('GlycoPOST', 'https://glycopost.glycosmos.org/', 'v1', '{"description": "MS/MS glycomics database", "data_types": ["spectra", "structures"]}'),
('UniCarbKB', 'http://unicarbkb.org/', 'v1', '{"description": "Curated glycan evidence database", "focus": "experimental_evidence"}');

-- Glycan structures cache table
CREATE TABLE cache.glycan_structures (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    glytoucan_id VARCHAR(20) UNIQUE NOT NULL,
    wurcs_sequence TEXT NOT NULL,
    glycoct TEXT,
    iupac_extended TEXT,
    iupac_condensed TEXT,
    mass_mono DECIMAL(12,6),
    mass_avg DECIMAL(12,6),
    composition JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_id UUID REFERENCES metadata.data_sources(id)
);

-- Create indexes for performance
CREATE INDEX idx_glycan_glytoucan_id ON cache.glycan_structures(glytoucan_id);
CREATE INDEX idx_glycan_wurcs ON cache.glycan_structures USING gin(wurcs_sequence gin_trgm_ops);
CREATE INDEX idx_glycan_mass_mono ON cache.glycan_structures(mass_mono);
CREATE INDEX idx_glycan_composition ON cache.glycan_structures USING gin(composition);

-- Protein-glycan associations cache
CREATE TABLE cache.protein_glycan_associations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    uniprot_id VARCHAR(20) NOT NULL,
    glytoucan_id VARCHAR(20) NOT NULL,
    glycosylation_site INTEGER,
    evidence_type VARCHAR(50),
    organism_taxid INTEGER,
    tissue VARCHAR(100),
    disease VARCHAR(100),
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_id UUID REFERENCES metadata.data_sources(id),
    UNIQUE(uniprot_id, glytoucan_id, glycosylation_site)
);

CREATE INDEX idx_pga_uniprot ON cache.protein_glycan_associations(uniprot_id);
CREATE INDEX idx_pga_glytoucan ON cache.protein_glycan_associations(glytoucan_id);
CREATE INDEX idx_pga_organism ON cache.protein_glycan_associations(organism_taxid);

-- MS/MS spectra cache
CREATE TABLE cache.ms_spectra (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    spectrum_id VARCHAR(50) NOT NULL,
    glytoucan_id VARCHAR(20),
    precursor_mz DECIMAL(12,6),
    charge_state INTEGER,
    collision_energy DECIMAL(6,2),
    peaks JSONB NOT NULL, -- Array of [mz, intensity] pairs
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source_id UUID REFERENCES metadata.data_sources(id)
);

CREATE INDEX idx_spectra_spectrum_id ON cache.ms_spectra(spectrum_id);
CREATE INDEX idx_spectra_glytoucan ON cache.ms_spectra(glytoucan_id);
CREATE INDEX idx_spectra_precursor_mz ON cache.ms_spectra(precursor_mz);

-- Data synchronization tracking
CREATE TABLE metadata.sync_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES metadata.data_sources(id),
    sync_type VARCHAR(50) NOT NULL, -- 'full', 'incremental'
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'running', -- 'running', 'completed', 'failed'
    records_processed INTEGER DEFAULT 0,
    records_added INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    errors JSONB,
    metadata JSONB
);

-- API usage statistics
CREATE TABLE metadata.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    user_agent TEXT,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- Create functions for maintenance
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at columns
CREATE TRIGGER update_glycan_structures_updated_at 
    BEFORE UPDATE ON cache.glycan_structures 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Performance monitoring views
CREATE VIEW metadata.cache_statistics AS
SELECT 
    'glycan_structures' AS table_name,
    COUNT(*) AS total_records,
    COUNT(DISTINCT source_id) AS unique_sources,
    MAX(updated_at) AS last_updated
FROM cache.glycan_structures
UNION ALL
SELECT 
    'protein_glycan_associations' AS table_name,
    COUNT(*) AS total_records,
    COUNT(DISTINCT source_id) AS unique_sources,
    MAX(created_at) AS last_updated
FROM cache.protein_glycan_associations
UNION ALL
SELECT 
    'ms_spectra' AS table_name,
    COUNT(*) AS total_records,
    COUNT(DISTINCT source_id) AS unique_sources,
    MAX(created_at) AS last_updated
FROM cache.ms_spectra;