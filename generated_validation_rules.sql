-- Generated Validation Rules SQL Statements
-- Generated on: 2025-06-23 23:38:55
-- Generated using OPENAI gpt-3.5-turbo for intelligent rule creation
-- Target table: validation_rules

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  1, 'edipi_not_null', 'edipi', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  2, 'edipi_format_check', 'edipi', 'REGEX_MATCH', '^[0-9]{10}$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  3, 'ssn_not_null', 'ssn', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  4, 'ssn_format_check', 'ssn', 'REGEX_MATCH', '^[0-9]{3}-[0-9]{2}-[0-9]{4}$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  5, 'name_not_null', 'name', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  6, 'name_format_check', 'name', 'REGEX_MATCH', '^[a-zA-Z''-]{1,100}$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  7, 'date_of_birth_not_null', 'date_of_birth', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  8, 'date_of_birth_format_check', 'date_of_birth', 'REGEX_MATCH', '^(19|20)\d\d-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  9, 'pay_grade_not_null', 'pay_grade', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  10, 'pay_grade_format_check', 'pay_grade', 'REGEX_MATCH', '^(E|O|W)[1-9]$|^(O10|W[1-5])$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  11, 'unit_not_null', 'unit', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  12, 'unit_format_check', 'unit', 'REGEX_MATCH', '^[a-zA-Z0-9_ ]{1,50}$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  13, 'email_address_not_null', 'email_address', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  14, 'email_address_format_check', 'email_address', 'REGEX_MATCH', '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  15, 'phone_number_not_null', 'phone_number', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  16, 'phone_number_format_check', 'phone_number', 'REGEX_MATCH', '^[+]?[0-9]{10,15}$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  17, 'effective_date_not_null', 'effective_date', 'NOT_NULL', NULL,
  NULL, false, NULL, '{source_column} IS NULL',
  'REJECT', true, 1, 'validation', 
  current_timestamp(), current_timestamp()
);

INSERT INTO validation_rules (
  rule_id, rule_name, source_column, rule_type, rule_condition, 
  compare_column_or_table, explode_flag, threshold, expression_template, 
  action_type, is_enabled, priority, rule_category, created_date, updated_date
) VALUES (
  18, 'effective_date_format_check', 'effective_date', 'REGEX_MATCH', '^(19|20)\d\d-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$',
  NULL, false, NULL, 'NOT ({source_column} RLIKE ''{rule_condition}'')',
  'REJECT', true, 2, 'validation', 
  current_timestamp(), current_timestamp()
);

