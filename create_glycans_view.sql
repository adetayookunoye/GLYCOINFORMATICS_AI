-- Create view to map cache.glycan_structures to glycans table expected by training formatter
CREATE OR REPLACE VIEW glycans AS
SELECT
    glytoucan_id,
    wurcs_sequence as wurcs,
    glycoct,
    iupac_extended,
    mass_mono as mass,
    composition,
    created_at,
    updated_at
FROM cache.glycan_structures;