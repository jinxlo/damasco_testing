-- Enable accent removal extension (safe if already present)
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Canonicalise warehouse_name column
UPDATE products
SET warehouse_name = UPPER(unaccent(warehouse_name));

-- Rebuild composite IDs with canonicalised warehouse names
UPDATE products
SET id = item_code || '_' ||
         REGEXP_REPLACE(warehouse_name, '[^A-Z0-9_-]', '_', 'g');
