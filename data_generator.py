"""
Synthetic Data Generator for Hybrid RAG System
Generates product data and receipt PDFs for testing
"""

import json
import random
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from faker import Faker
import os

fake = Faker()


class SyntheticDataGenerator:
    """Generate synthetic product and receipt data"""
    
    def __init__(self):
        self.categories = [
            "Electronics", "Accessories", "Clothing", "Home & Garden",
            "Sports & Outdoors", "Books", "Toys & Games", "Office Supplies"
        ]
        
        self.manufacturers = {
            "Electronics": ["TechCorp", "ElectroMax", "DigitalWave", "SmartDevices"],
            "Accessories": ["PeripheralsCo", "GearUp", "AccessPlus", "ConnectTech"],
            "Clothing": ["FashionHub", "StyleWear", "TrendyThreads", "UrbanOutfit"],
            "Home & Garden": ["HomeEssentials", "GardenPro", "CozyLiving", "NatureSpace"],
            "Sports & Outdoors": ["ActiveLife", "SportsMaster", "OutdoorGear", "FitZone"],
            "Books": ["LitPublishing", "ReadMore", "BookWorld", "KnowledgePress"],
            "Toys & Games": ["PlayTime", "FunFactory", "GameMasters", "KidJoy"],
            "Office Supplies": ["OfficePro", "WorkSpace", "DeskMate", "BizSupply"]
        }
        
        self.product_templates = {
            "Electronics": [
                ("Laptop", {"ram": ["8GB", "16GB", "32GB"], "storage": ["256GB SSD", "512GB SSD", "1TB SSD"], 
                           "processor": ["Intel i5", "Intel i7", "AMD Ryzen 5", "AMD Ryzen 7"]}),
                ("Smartphone", {"ram": ["4GB", "6GB", "8GB"], "storage": ["64GB", "128GB", "256GB"], 
                               "screen": ["6.1 inch", "6.5 inch", "6.7 inch"]}),
                ("Tablet", {"ram": ["4GB", "8GB"], "storage": ["64GB", "128GB", "256GB"], 
                           "screen": ["10.2 inch", "11 inch", "12.9 inch"]}),
                ("Wireless Earbuds", {"battery": ["6 hours", "8 hours", "10 hours"], 
                                     "connectivity": ["Bluetooth 5.0", "Bluetooth 5.2"], 
                                     "noise_cancellation": ["Active", "Passive"]}),
                ("Smart Watch", {"battery": ["18 hours", "24 hours", "36 hours"], 
                               "display": ["AMOLED", "LCD"], "water_resistance": ["IP67", "IP68"]})
            ],
            "Accessories": [
                ("Wireless Mouse", {"dpi": ["1600", "3200", "6400"], "connectivity": ["Bluetooth", "2.4GHz"], 
                                   "battery": ["Rechargeable", "AA Battery"]}),
                ("Mechanical Keyboard", {"switches": ["Cherry MX Blue", "Cherry MX Red", "Cherry MX Brown"], 
                                        "connectivity": ["USB", "Bluetooth"], "backlight": ["RGB", "Single Color"]}),
                ("USB-C Hub", {"ports": ["5", "7", "9"], "power_delivery": ["60W", "100W"], 
                              "hdmi": ["4K@30Hz", "4K@60Hz"]}),
                ("Laptop Stand", {"material": ["Aluminum", "Plastic"], "adjustable": ["Yes", "No"], 
                                 "max_weight": ["5kg", "10kg"]})
            ],
            "Clothing": [
                ("T-Shirt", {"material": ["Cotton", "Polyester", "Cotton Blend"], 
                           "size": ["S", "M", "L", "XL"], "fit": ["Regular", "Slim", "Loose"]}),
                ("Jeans", {"material": ["Denim", "Stretch Denim"], "size": ["28", "30", "32", "34", "36"], 
                          "fit": ["Slim", "Regular", "Relaxed"]}),
                ("Jacket", {"material": ["Leather", "Polyester", "Cotton"], "size": ["S", "M", "L", "XL"], 
                           "style": ["Casual", "Formal", "Sports"]})
            ],
            "Home & Garden": [
                ("LED Bulb Set", {"wattage": ["9W", "12W", "15W"], "color_temp": ["Warm White", "Cool White"], 
                                 "quantity": ["4 pack", "6 pack", "10 pack"]}),
                ("Plant Pot", {"material": ["Ceramic", "Plastic", "Terracotta"], 
                              "size": ["Small", "Medium", "Large"], "drainage": ["Yes", "No"]})
            ],
            "Sports & Outdoors": [
                ("Yoga Mat", {"thickness": ["4mm", "6mm", "8mm"], "material": ["PVC", "TPE", "Natural Rubber"], 
                             "length": ["68 inch", "72 inch"]}),
                ("Water Bottle", {"capacity": ["500ml", "750ml", "1L"], "material": ["Stainless Steel", "Plastic"], 
                                 "insulation": ["Yes", "No"]})
            ]
        }
        
        self.payment_methods = [
            "Visa ending in 4532", "Mastercard ending in 8901", "Amex ending in 2345",
            "PayPal", "Debit Card ending in 6789", "Credit Card ending in 1234"
        ]
        
        self.warranties = [
            "1 year manufacturer warranty",
            "2 years manufacturer warranty",
            "3 years manufacturer warranty",
            "1 year limited warranty",
            "18 months manufacturer warranty",
            "6 months warranty"
        ]
        
        self.return_policies = [
            "30-day money-back guarantee",
            "60-day return window",
            "90-day return policy",
            "14-day return policy",
            "No returns on opened items",
            "30-day exchange only"
        ]
    
    def generate_product(self, product_id=None, category=None):
        """Generate a single product"""
        if category is None:
            category = random.choice(self.categories)
        
        if category not in self.product_templates:
            category = "Electronics"
        
        product_type, spec_options = random.choice(self.product_templates[category])
        manufacturer = random.choice(self.manufacturers[category])
        
        # Generate specifications
        specs = {}
        for spec_key, spec_values in spec_options.items():
            specs[spec_key] = random.choice(spec_values)
        
        # Generate product name
        product_name = f"{product_type} {random.choice(['Pro', 'Plus', 'Ultra', 'Max', 'Elite', ''])}"
        product_name = product_name.strip()
        
        # Generate price based on category
        price_ranges = {
            "Electronics": (299, 2499),
            "Accessories": (19, 299),
            "Clothing": (19, 199),
            "Home & Garden": (9, 149),
            "Sports & Outdoors": (15, 299),
            "Books": (9, 49),
            "Toys & Games": (9, 99),
            "Office Supplies": (5, 149)
        }
        
        min_price, max_price = price_ranges.get(category, (10, 500))
        price = round(random.uniform(min_price, max_price), 2)
        
        if product_id is None:
            product_id = f"P{random.randint(1000, 9999)}"
        
        receipt_path = f"./receipts/{product_id.lower()}_receipt.pdf"
        
        return {
            "id": product_id,
            "name": product_name,
            "category": category,
            "price": price,
            "manufacturer": manufacturer,
            "specs": specs,
            "receipt_path": receipt_path
        }
    
    def generate_receipt_data(self, product):
        """Generate receipt data for a product"""
        
        # Generate purchase date (random date within last 6 months)
        days_ago = random.randint(1, 180)
        purchase_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        
        # Generate invoice number
        invoice_types = ["INV", "RCP", "SR"]
        invoice_prefix = random.choice(invoice_types)
        invoice_number = f"{invoice_prefix}-{datetime.now().year}-{random.randint(100, 999)}"
        
        # Generate serial number (for electronics)
        serial_number = None
        if product['category'] in ["Electronics", "Accessories"]:
            serial_number = f"{product['id']}-{datetime.now().year}{random.randint(1000, 9999)}-{random.randint(1000, 9999)}"
        
        # Calculate tax and total
        tax_rate = 0.08  # 8% tax
        subtotal = product['price']
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)
        
        # Apply random discount
        discount = 0
        discount_desc = None
        if random.random() < 0.3:  # 30% chance of discount
            discount = round(random.uniform(5, 25), 0)
            discount_desc = f"{int(discount)}% discount applied"
            subtotal = round(subtotal * (1 - discount/100), 2)
            tax = round(subtotal * tax_rate, 2)
            total = round(subtotal + tax, 2)
        
        return {
            "invoice_number": invoice_number,
            "purchase_date": purchase_date,
            "product": product,
            "serial_number": serial_number,
            "quantity": 1,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "discount": discount_desc,
            "payment_method": random.choice(self.payment_methods),
            "warranty": random.choice(self.warranties),
            "return_policy": random.choice(self.return_policies),
            "store_name": f"{product['manufacturer']} {random.choice(['Store', 'Shop', 'Outlet', 'Direct'])}",
            "store_email": f"support@{product['manufacturer'].lower().replace(' ', '')}.com",
            "store_phone": fake.phone_number()
        }
    
    def create_receipt_pdf(self, receipt_data, output_path):
        """Create a PDF receipt"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#666666'),
            alignment=1
        )
        
        # Store header
        story.append(Paragraph(receipt_data['store_name'], title_style))
        story.append(Paragraph(receipt_data['store_email'], header_style))
        story.append(Paragraph(receipt_data['store_phone'], header_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Invoice info
        story.append(Paragraph(f"<b>INVOICE #{receipt_data['invoice_number']}</b>", styles['Heading2']))
        story.append(Paragraph(f"Date: {receipt_data['purchase_date']}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Product details
        product_data = [
            ['Product', 'Quantity', 'Unit Price', 'Total'],
            [receipt_data['product']['name'], 
             str(receipt_data['quantity']), 
             f"${receipt_data['product']['price']:.2f}",
             f"${receipt_data['product']['price']:.2f}"]
        ]
        
        product_table = Table(product_data, colWidths=[3*inch, 1*inch, 1.5*inch, 1.5*inch])
        product_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(product_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Specifications
        if receipt_data['product']['specs']:
            story.append(Paragraph("<b>Product Specifications:</b>", styles['Heading3']))
            specs_text = ", ".join([f"{k}: {v}" for k, v in receipt_data['product']['specs'].items()])
            story.append(Paragraph(specs_text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Serial number
        if receipt_data['serial_number']:
            story.append(Paragraph(f"<b>Serial Number:</b> {receipt_data['serial_number']}", styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
        
        # Pricing breakdown
        pricing_data = [
            ['', ''],
            ['Subtotal:', f"${receipt_data['subtotal']:.2f}"]
        ]
        
        if receipt_data['discount']:
            pricing_data.append(['Discount:', receipt_data['discount']])
        
        pricing_data.extend([
            ['Tax (8%):', f"${receipt_data['tax']:.2f}"],
            ['Total:', f"${receipt_data['total']:.2f}"]
        ])
        
        pricing_table = Table(pricing_data, colWidths=[5*inch, 2*inch])
        pricing_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, -1), (-1, -1), 14),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.black),
            ('TOPPADDING', (0, -1), (-1, -1), 12),
        ]))
        
        story.append(pricing_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Payment and warranty info
        story.append(Paragraph(f"<b>Payment Method:</b> {receipt_data['payment_method']}", styles['Normal']))
        story.append(Paragraph(f"<b>Warranty:</b> {receipt_data['warranty']}", styles['Normal']))
        story.append(Paragraph(f"<b>Return Policy:</b> {receipt_data['return_policy']}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#999999'),
            alignment=1
        )
        story.append(Paragraph("Thank you for your purchase!", footer_style))
        story.append(Paragraph(f"For support, contact: {receipt_data['store_email']}", footer_style))
        
        # Build PDF
        doc.build(story)
        
        return output_path
    
    def generate_dataset(self, num_products=10, output_dir="./synthetic_data"):
        """Generate complete dataset with products and receipts"""
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/receipts", exist_ok=True)
        
        products = []
        receipts_info = []
        
        for i in range(num_products):
            # Generate product
            product_id = f"P{1000 + i}"
            product = self.generate_product(product_id=product_id)
            products.append(product)
            
            # Generate receipt
            receipt_data = self.generate_receipt_data(product)
            
            # Create PDF
            pdf_path = f"{output_dir}/receipts/{product_id.lower()}_receipt.pdf"
            self.create_receipt_pdf(receipt_data, pdf_path)
            
            # Update product with correct path
            product['receipt_path'] = pdf_path
            
            receipts_info.append({
                "product_id": product_id,
                "invoice_number": receipt_data['invoice_number'],
                "purchase_date": receipt_data['purchase_date'],
                "total": receipt_data['total'],
                "pdf_path": pdf_path
            })
            
            print(f"✓ Generated product {product_id}: {product['name']}")
        
        # Save products JSON
        with open(f"{output_dir}/products.json", 'w') as f:
            json.dump(products, f, indent=2)
        
        # Save receipts info JSON
        with open(f"{output_dir}/receipts_info.json", 'w') as f:
            json.dump(receipts_info, f, indent=2)
        
        # Create import script
        self.create_import_script(output_dir)
        
        print(f"\n✓ Generated {num_products} products with receipts")
        print(f"✓ Products saved to: {output_dir}/products.json")
        print(f"✓ Receipts saved to: {output_dir}/receipts/")
        print(f"✓ Import script: {output_dir}/import_data.py")
        
        return products, receipts_info
    
    def create_import_script(self, output_dir):
        """Create a script to import generated data into the RAG system"""
        
        script_content = """'''
Import synthetic data into Hybrid RAG System
'''

import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env in the project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import HybridRAGSystem

def import_data(neo4j_uri, neo4j_user, neo4j_password, openai_key):
    print("Initializing RAG system...")
    rag_system = HybridRAGSystem(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        openai_key=openai_key
    )

    rag_system.setup_graph_schema()

    with open('products.json', 'r') as f:
        products = json.load(f)

    print(f"\\nImporting {len(products)} products...")

    for product in products:
        success, msg = rag_system.add_product_to_graph(product)
        if success:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")

        if os.path.exists(product['receipt_path']):
            success, msg = rag_system.load_pdf_to_vector_store(
                product['receipt_path'],
                product['id']
            )
            if success:
                print(f"    ✓ Loaded receipt PDF")
            else:
                print(f"    ✗ {msg}")

    print("\\n✓ Import complete!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Import synthetic data')
    parser.add_argument('--neo4j-uri',      default=os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687"))
    parser.add_argument('--neo4j-user',     default=os.getenv("NEO4J_USER",     "neo4j"))
    parser.add_argument('--neo4j-password', default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument('--openai-key',     default=os.getenv("OPENAI_API_KEY", ""))

    args = parser.parse_args()

    if not args.openai_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or pass --openai-key.")
        import sys; sys.exit(1)

    import_data(
        args.neo4j_uri,
        args.neo4j_user,
        args.neo4j_password,
        args.openai_key
    )
"""
        
        with open(f"{output_dir}/import_data.py", 'w') as f:
            f.write(script_content)


def main():
    """Main execution"""
    print("=" * 60)
    print("Synthetic Data Generator for Hybrid RAG")
    print("=" * 60)
    print()
    
    generator = SyntheticDataGenerator()
    
    # Generate dataset
    num_products = 20  # Change this number as needed
    products, receipts = generator.generate_dataset(
        num_products=num_products,
        output_dir="./synthetic_data"
    )
    
    print("\n" + "=" * 60)
    print("Sample Products Generated:")
    print("=" * 60)
    for product in products[:5]:
        print(f"\n{product['name']} ({product['id']})")
        print(f"  Category: {product['category']}")
        print(f"  Price: ${product['price']:.2f}")
        print(f"  Manufacturer: {product['manufacturer']}")
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review generated data in ./synthetic_data/")
    print("2. Update import_data.py with your RAG module import")
    print("3. Run: cd synthetic_data && python import_data.py --neo4j-password YOUR_PASSWORD --openai-key YOUR_KEY")


if __name__ == "__main__":
    main()
