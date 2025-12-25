# AI Agent Instructions for the mesa tool

## What this system does

The MESA system is made to process input datato support borth emergency response activities and land use use planning. 

## System complexity

The system consists of several python files which constitute different parts of the system.

Most of our potential users are not in a position to download and run python code. We have therefor chosen to compile the python project. In this context compiling it is a bit of a stretch. It is more like we wrap the interpreter (python.exe) and our code in several .exe-files. For the user this looks like a normal program. Be aware that some systems might have security settings (either local or governed by system managers) which could create challenges for the user.

### Area analysis helper

- `data_analysis.py` is a standalone helper that lets a power user digitise or import analysis polygons, intersect them with the asset catalogue and export a PDF summary (`output/MESA_area_analysis_report_YYYY_MM_DD.pdf`).
- Geometry is stored in `output/geoparquet/tbl_analysis_polygons.parquet` so edits are shared between the .py and packaged .exe variants.
- The helper mirrors the other Leaflet editors: drawing/editing happens inside a pywebview map, while the left-hand pane controls metadata and report generation.

It is important that functional consistency is available for the system. THe separate python files should run separately as python programs, as compiled programs and as "slaves" under the mesa.py or mesa.exe programs. This involves some setups for where all modes are supported. It is particularly storage places (paths) which is of concern.

## Code Modification Restrictions

### STRICT RULES - READ CAREFULLY

1. **NEVER modify the core architecture** - This is a single-file script by design
2. **NEVER add external dependencies** beyond the existing three: `requests`, `pandas`, `python-docx`
3. **NEVER change the configuration pattern** - Top-of-file variables must remain hardcoded
4. **NEVER alter the main data flow** - Fetch → Transform → Generate Word doc pipeline is fixed
5. **NEVER modify function signatures** without explicit permission
6. **Do not remove existing error handling** - Only add to it, ask before you do such changes
7. **CHECK your code for syntax errors** - So that you don not deliver code with obvious and basic errors.

### ALLOWED MODIFICATIONS

- Bug fixes within existing functions
- Adding new utility functions (following patterns below)
- Enhancing documentation and comments
- Adding optional parameters with defaults
- Improving error messages and logging

## Code Standards & Preferences

### Measurment standards
- **ALLWAYS use ISO standards like mm, cm, meters, litres etc** rather than gallons, inches, feet etc

### Function-Based Approach
- **ALWAYS prefer pure functions** over methods when possible
- **ALWAYS write functions that take inputs and return outputs** rather than modifying global state
- **ALWAYS separate data processing from I/O operations**
- Example pattern:
```python
def process_document_data(raw_docs: List[dict], lookup_maps: dict) -> pd.DataFrame:
    """Pure function that transforms data without side effects."""
    # Processing logic here
    return transformed_df
```

### Documentation Requirements
- **EVERY function must have a docstring** explaining purpose, parameters, and return value
- **EVERY complex operation must have inline comments** explaining the "why", not just the "what"
- **EVERY configuration variable must have a comment** explaining its purpose
- **Use type hints consistently** - this codebase already follows this pattern

### Code Organization Patterns
- **Group related functions together** with clear section comments
- **Keep configuration at the top** - never move hardcoded values deeper into the code
- **Maintain the existing import order** - standard library, third-party, local
- **Use descriptive variable names** - `df_group` not `grp`, `custom_field_id` not `cf_id`

### Error Handling Philosophy
- **Always provide context in error messages** - include what operation was being attempted
- **Use specific exception types** when possible (`PermissionError` vs generic `Exception`)
- **Exit codes must remain consistent** - Status 0: success, 1: API errors, 2: auth/connection
- **Never silence exceptions** - always log or re-raise with context

## Existing Architecture Patterns

### Configuration Management
```python
# Top-of-file pattern - NEVER change this approach
CUSTOM_FIELD_NAME = "value"    # Configuration variables
HARD_CODED_URL = "value"       # Must remain at top
# ... other config
```

### API Interaction Pattern
```python
# Pagination pattern - MAINTAIN this approach
def paged_get(session, base_url, path, params=None):
    # Handles pagination automatically
    # Includes error recovery for API versioning
```

### Data Transformation Pattern
```python
# Fetch → Process → Format → Output
# NEVER change this pipeline order
```

### Word Document Generation Pattern
```python
# A4 Landscape → TOC → Page Break → Content
# This sequence is required for proper formatting
```

## When Making Changes

### Before Modifying Code
1. **Understand the data flow** from API to Word document
2. **Check if your change affects the cronjob deployment model**
3. **Verify you're not breaking the single-file design**
4. **Consider impact on existing configuration variables**

### Required Testing Approach
- **Test with verbose flag** to verify logging changes
- **Test with different custom field types** (string, numeric, null)
- **Test pagination** with small page sizes
- **Verify Word document structure** remains intact

### Documentation Updates
- **Update docstrings** for any modified functions
- **Update comments** if logic changes
- **Maintain existing comment style** and formatting

## Common Modification Patterns

### Adding a New Column
```python
# Correct approach - extend DEFAULT_COLUMNS tuple
DEFAULT_COLUMNS: List[Tuple[str, str]] = [
    # ... existing columns
    ("new_field_key", "Display Name"),  # Add here
]
```

### Adding Data Processing
```python
def process_new_field(value: Any) -> str:
    """
    Process a new field type for display.
    
    Args:
        value: Raw field value from API
        
    Returns:
        Formatted string for display
    """
    # Processing logic here
    return formatted_value
```

### Enhancing Error Handling
```python
# Add context to existing patterns
try:
    result = api_operation()
except requests.HTTPError as e:
    print(f"ERROR: Failed during {operation_name}: {e}", file=sys.stderr)
    sys.exit(appropriate_exit_code)
```

## What NOT to Do

- ❌ Split the script into multiple files
- ❌ Add a requirements.txt or setup.py
- ❌ Change the hardcoded configuration pattern
- ❌ Modify the Word document structure (A4 landscape, TOC, etc.)
- ❌ Add complex command-line interfaces
- ❌ Remove or significantly alter existing functions
- ❌ Change the exit code meanings
- ❌ Add new external dependencies

## Questions to Ask Before Changes

1. Does this maintain the single-file cronjob design?
2. Will this still work with the existing configuration pattern?
3. Does this follow the function-based approach?
4. Is this properly documented?
5. Does this maintain backward compatibility?
6. Have I tested the Word document output?
