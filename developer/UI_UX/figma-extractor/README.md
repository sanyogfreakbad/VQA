# Figma Design Extractor

A Python tool to extract and normalize design values from Figma files via their REST API.

## Features

- Fetches design data from Figma's REST API
- Filters specific node types: `FRAME`, `TEXT`, `RECTANGLE`, `COMPONENT`
- Normalizes bounding box data (`absoluteBoundingBox` → `x`, `y`, `width`, `height`)
- Extracts type-specific properties (fills, strokes, text styles, etc.)
- Outputs clean, structured JSON

## Installation

1. Clone or download this project
2. Install dependencies:

```bash
cd figma-extractor
pip install -r requirements.txt
```

## Setup

### Get Your Figma API Token

1. Go to [Figma Account Settings](https://www.figma.com/settings)
2. Scroll to "Personal access tokens"
3. Click "Create new token"
4. Copy the token

### Configure the Token

**Option 1: Environment variable**
```bash
export FIGMA_TOKEN="your-token-here"
```

**Option 2: Create a `.env` file**
```bash
cp .env.example .env
# Edit .env and add your token
```

## Usage

### Basic Usage

```bash
python figma_extractor.py <file_key>
```

### Save to File

```bash
python figma_extractor.py <file_key> --output design.json
```

### With Inline Token

```bash
python figma_extractor.py <file_key> --token YOUR_TOKEN
```

### Compact JSON Output

```bash
python figma_extractor.py <file_key> --compact
```

## Getting Your Figma File Key

From your Figma file URL:
```
https://www.figma.com/file/ABC123xyz/My-Design-File
                          ^^^^^^^^^^
                          This is your file_key
```

## Output Format

The tool outputs JSON with the following structure:

```json
{
  "file_name": "My Design File",
  "file_key": "ABC123xyz",
  "last_modified": "2024-01-15T10:30:00Z",
  "extracted_at": "2024-01-15T12:00:00Z",
  "version": "123456789",
  "total_nodes_extracted": 42,
  "nodes": [
    {
      "id": "1:2",
      "name": "Button",
      "type": "FRAME",
      "x": 100,
      "y": 200,
      "width": 120,
      "height": 40,
      "fills": [...],
      "strokes": [...],
      "cornerRadius": 8
    },
    {
      "id": "1:3",
      "name": "Label",
      "type": "TEXT",
      "x": 110,
      "y": 210,
      "width": 100,
      "height": 20,
      "characters": "Click me",
      "fontSize": 14,
      "fontFamily": "Inter",
      "fills": [...]
    }
  ]
}
```

## Extracted Properties by Node Type

### TEXT
- `characters` - The text content
- `fontSize`, `fontFamily`, `fontWeight`
- `textAlignHorizontal`, `textAlignVertical`
- `letterSpacing`, `lineHeightPx`
- `fills` - Text color/gradient

### FRAME
- `fills`, `strokes`, `strokeWeight`
- `cornerRadius`
- `paddingLeft`, `paddingRight`, `paddingTop`, `paddingBottom`
- `itemSpacing`, `layoutMode` (for auto-layout frames)

### RECTANGLE
- `fills`, `strokes`, `strokeWeight`
- `cornerRadius`, `rectangleCornerRadii`

### COMPONENT
- `description`
- `fills`, `strokes`, `cornerRadius`

## Error Handling

The tool provides clear error messages for common issues:

- **401**: Invalid API token
- **404**: File not found (check your file key)
- **429**: Rate limited (wait and retry)
- **Network errors**: Connection issues

## License

MIT
