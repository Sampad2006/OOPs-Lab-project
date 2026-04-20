"""
Script to organize test data into structured folders:
- images/: copies of all image files
- digitized_pdfs/: PDFs with text and layout from JSON annotations
- txts/: plain text extracted from JSON annotations
"""

import os
import json
import shutil
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image


def create_directories():
    """Create the directory structure under test_data."""
    base_path = Path("test_data")
    dirs = {
        "images": base_path / "images",
        "digitized_pdfs": base_path / "digitized_pdfs",
        "txts": base_path / "txts",
    }
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_base_name(filename):
    """Extract base name without extension."""
    return Path(filename).stem


def copy_images(source_dir, target_dir):
    """Copy all image files to the images directory."""
    supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
    count = 0
    
    for file in Path(source_dir).iterdir():
        if file.suffix.lower() in supported_formats:
            dest = target_dir / file.name
            shutil.copy2(file, dest)
            count += 1
            print(f"✓ Copied image: {file.name}")
    
    return count


def extract_text_from_json(json_file):
    """Extract all text from JSON annotation file."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text from each annotation
        texts = [item.get("text", "") for item in data if isinstance(item, dict)]
        return " ".join(texts)
    except Exception as e:
        print(f"  ⚠ Error reading {json_file}: {e}")
        return ""


def save_text_file(base_name, text, target_dir):
    """Save extracted text to a .txt file."""
    output_file = target_dir / f"{base_name}.txt"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ Created text file: {base_name}.txt")
        return True
    except Exception as e:
        print(f"  ⚠ Error creating text file: {e}")
        return False


def create_pdf_with_layout(base_name, json_file, image_file, target_dir):
    """Create a PDF with text positioned according to JSON coordinates (text only, no background image)."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # Get image dimensions for coordinate scaling reference
        if image_file.exists():
            img = Image.open(image_file)
            img_width, img_height = img.size
            # Scale to fit on letter size (8.5 x 11 inches)
            max_width = letter[0] - 0.5 * inch
            max_height = letter[1] - 0.5 * inch
            scale = min(max_width / img_width, max_height / img_height)
        else:
            scale = 0.5
            img_width, img_height = 800, 600
        
        output_file = target_dir / f"{base_name}.pdf"
        c = canvas.Canvas(str(output_file), pagesize=letter)
        
        # Add text annotations with positioning (no background image)
        c.setFont("Helvetica", 10)
        c.setFillColor((0, 0, 0))
        
        for annotation in annotations:
            if isinstance(annotation, dict) and "text" in annotation and "polygon" in annotation:
                text = annotation["text"]
                poly = annotation["polygon"]
                
                # Use x0, y0 as reference point, scaled to fit on letter page
                x = (poly.get("x0", 0) * scale) / 72 * inch + 0.25 * inch
                y = letter[1] - (poly.get("y0", 0) * scale) / 72 * inch - 0.25 * inch
                
                try:
                    c.drawString(x, y, text[:50])  # Limit text length
                except Exception as e:
                    pass  # Skip problematic text
        
        c.save()
        print(f"✓ Created PDF: {base_name}.pdf")
        return True
    
    except Exception as e:
        print(f"  ⚠ Error creating PDF: {e}")
        return False


def main():
    """Main orchestration function."""
    print("\n" + "="*60)
    print("📂 Test Data Organizer")
    print("="*60 + "\n")
    
    # Create directory structure
    print("📁 Creating directory structure...")
    dirs = create_directories()
    print(f"✓ Directories created\n")
    
    # Source directory
    source_dir = Path("test_data/test")
    
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return
    
    print(f"📂 Processing files from: {source_dir}\n")
    
    # Step 1: Copy images
    print("📋 Step 1: Copying images...")
    image_count = copy_images(source_dir, dirs["images"])
    print(f"✓ Copied {image_count} image(s)\n")
    
    # Step 2: Process JSON files (create PDFs and text files)
    print("📋 Step 2: Processing annotations...")
    json_files = list(source_dir.glob("*.json"))
    
    for json_file in sorted(json_files):
        base_name = get_base_name(json_file.name)
        
        # Find corresponding image
        image_file = None
        for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            candidate = source_dir / f"{base_name}{ext}"
            if candidate.exists():
                image_file = candidate
                break
        
        if not image_file:
            print(f"⚠ No image found for {json_file.name}, skipping...")
            continue
        
        # Extract text and save to txt file
        text = extract_text_from_json(json_file)
        if text:
            save_text_file(base_name, text, dirs["txts"])
        
        # Create PDF with layout
        create_pdf_with_layout(base_name, json_file, image_file, dirs["digitized_pdfs"])
    
    print(f"\n✓ Processed {len(json_files)} file(s)")
    print("\n" + "="*60)
    print("✅ Organization complete!")
    print("="*60)
    print(f"\n📂 Output structure:")
    print(f"   test_data/images/          - {image_count} image(s)")
    print(f"   test_data/digitized_pdfs/  - {len(json_files)} PDF(s)")
    print(f"   test_data/txts/            - {len(json_files)} text file(s)")
    print()


if __name__ == "__main__":
    main()
