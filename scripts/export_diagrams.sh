#!/bin/bash

# Export HHPF Mermaid Diagrams to PNG/PDF
# Usage: ./scripts/export_diagrams.sh

set -e

echo "=========================================="
echo "HHPF Diagram Export Script"
echo "=========================================="

# Check if mermaid-cli is installed
if ! command -v mmdc &> /dev/null; then
    echo "❌ mermaid-cli (mmdc) not found"
    echo ""
    echo "To install mermaid-cli:"
    echo "  sudo npm install -g @mermaid-js/mermaid-cli"
    echo ""
    echo "Or use alternative methods documented in:"
    echo "  outputs/research_questions/DIAGRAM_EXPORT_GUIDE.md"
    exit 1
fi

# Create output directory
mkdir -p outputs/research_questions/figures

echo "✅ mermaid-cli found"
echo "📁 Output directory: outputs/research_questions/figures"
echo ""

# Note: Individual diagrams need to be extracted from PIPELINE_DIAGRAMS.md
# into separate .mmd files for mmdc to process

echo "⚠️  Note: This script requires individual .mmd files"
echo "    Please extract diagrams from docs/PIPELINE_DIAGRAMS.md"
echo "    into separate files in a temp directory."
echo ""
echo "Example workflow:"
echo "  1. Create temp/diagram1.mmd with first mermaid block"
echo "  2. Run: mmdc -i temp/diagram1.mmd -o outputs/research_questions/figures/pipeline_complete.png -w 2400 -H 1600"
echo "  3. Repeat for each diagram"
echo ""

# TODO: Extract mermaid blocks automatically
# For now, provide manual instructions

exit 0
