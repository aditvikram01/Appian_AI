# -- coding: utf-8 --
"""
PDF Document Management System with Advanced Classification
"""

import os
import sys
import pytesseract
from pdf2image import convert_from_path
import spacy
import re
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from enum import Enum
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from typing import Dict, List, Optional, Union
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

class DocumentType(Enum):
    PAN = "pan_card"
    AADHAR = "aadhar_card"
    INCOME_STATEMENT = "income_statement"
    SALARY_SLIP = "salary_slip"
    ITR = "income_tax_return"
    BANK_STATEMENT = "bank_statement"
    CREDIT_CARD = "credit_card_application"
    LOAN = "loan_application"

class DocumentCategory(Enum):
    IDENTITY = "identity_document"
    FINANCIAL = "financial_document"
    BANK_APPLICATION = "bank_application"

class DocumentClassifier:
    """Handles document classification using pattern matching and scoring"""

    def __init__(self):
        self.patterns = {
            DocumentType.PAN: {
                'required': [
                    'permanent account number',
                    'income tax department',
                    'पैन कार्ड'
                ],
                'optional': [
                    'pan',
                    'father',
                    'date of birth',
                    'signature'
                ],
                'id_pattern': r'[A-Z]{5}[0-9]{4}[A-Z]'
            },
            DocumentType.AADHAR: {
                'required': [
                    'aadhaar',
                    'unique identification',
                    'uidai',
                    'आधार'
                ],
                'optional': [
                    'dob',
                    'enrollment',
                    'biometric',
                    'demographic'
                ],
                'id_pattern': r'\d{4}\s*\d{4}\s*\d{4}'
            },
            DocumentType.INCOME_STATEMENT: {
                'required': [
                    'income statement',
                    'total income',
                    'revenue'
                ],
                'optional': [
                    'profit',
                    'loss',
                    'earnings',
                    'fiscal year'
                ]
            },
            DocumentType.SALARY_SLIP: {
                'required': [
                    'salary slip',
                    'pay slip',
                    'employee'
                ],
                'optional': [
                    'basic pay',
                    'deductions',
                    'net pay',
                    'gross salary'
                ]
            }
            # Add patterns for other document types...
        }

    def classify(self, text: str) -> tuple[DocumentType, float]:
        """
        Classify document based on text content and return type with confidence score
        """
        text = text.lower()
        scores = {}

        for doc_type, patterns in self.patterns.items():
            score = 0

            # Check required patterns
            required_matches = sum(1 for pattern in patterns['required']
                                if pattern in text)
            if required_matches == 0:
                continue

            score += required_matches * 2  # Weight required patterns more heavily

            # Check optional patterns
            score += sum(1 for pattern in patterns['optional']
                        if pattern in text)

            # Check ID patterns if they exist
            if 'id_pattern' in patterns:
                id_matches = len(re.findall(patterns['id_pattern'], text))
                score += id_matches * 2

            scores[doc_type] = score

        if not scores:
            return None, 0.0

        max_score_type = max(scores.items(), key=lambda x: x[1])
        confidence = min(max_score_type[1] / 10, 1.0)  # Normalize to 0-1

        return max_score_type[0], confidence

class PDFProcessor:
    """Handles PDF processing and text extraction"""

    def __init__(self):
        self.temp_dir = Path("temp_images")
        self.temp_dir.mkdir(exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF using OCR
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text = ""

            # Process each page
            for i, image in enumerate(images):
                # Save temporary image
                temp_image_path = self.temp_dir / f"page_{i}.png"
                image.save(temp_image_path)

                # Extract text using OCR
                text += pytesseract.image_to_string(
                    Image.open(temp_image_path),
                    lang='eng+hin'  # Support both English and Hindi
                )

                # Clean up
                temp_image_path.unlink()

            return text

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

class DocumentProcessor:
    """Main document processing class"""

    def __init__(self, storage_path: str = "document_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.db_path = self.storage_path / "document_db.json"

        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.classifier = DocumentClassifier()
        self.nlp = self._initialize_nlp()

        self.load_database()

    def _initialize_nlp(self):
        """Initialize NLP model"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            console.print("[yellow]Downloading SpaCy model...[/yellow]")
            os.system("python -m spacy download en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def load_database(self):
        """Load document database"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                self.documents = json.load(f)
        else:
            self.documents = []
            self.save_database()

    def save_database(self):
        """Save document database"""
        with open(self.db_path, 'w') as f:
            json.dump(self.documents, f, indent=4, default=str)

    def extract_entity_info(self, text: str) -> Dict:
        """Extract person and ID information from text"""
        doc = self.nlp(text)

        # Extract person names using NER
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        # Extract document numbers
        pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
        aadhar_pattern = r'\d{4}\s?\d{4}\s?\d{4}'

        pan_matches = re.findall(pan_pattern, text)
        aadhar_matches = re.findall(aadhar_pattern, text)

        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)

        return {
            "name": person_names[0] if person_names else None,
            "pan": pan_matches[0] if pan_matches else None,
            "aadhar": aadhar_matches[0] if aadhar_matches else None,
            "email": email_matches[0] if email_matches else None,
            "all_matches": {
                "names": person_names,
                "pan_numbers": pan_matches,
                "aadhar_numbers": aadhar_matches,
                "emails": email_matches
            }
        }

    def process_document(self, file_path: str) -> Dict:
        """Process and classify document"""
        file_path = Path(file_path)

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() != '.pdf':
            raise ValueError("Only PDF files are supported")

        try:
            # Extract text from PDF
            text = self.pdf_processor.extract_text_from_pdf(file_path)

            # Classify document
            doc_type, confidence = self.classifier.classify(text)
            if not doc_type:
                raise ValueError("Unable to classify document")

            # Extract information
            entity_info = self.extract_entity_info(text)

            # Generate document ID
            doc_id = f"DOC_{len(self.documents) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Create document record
            doc_info = {
                "id": doc_id,
                "document_type": doc_type.value,
                "category": self._get_category(doc_type).value,
                "person_name": entity_info["name"] or "Unknown",
                "person_id": entity_info.get("pan") or entity_info.get("aadhar") or "Unknown",
                "confidence_score": confidence,
                "metadata": {
                    **entity_info,
                    "file_type": "pdf",
                    "extraction_date": datetime.now().isoformat()
                }
            }

            # Store document
            stored_path = self.storage_path / f"{doc_id}.pdf"
            shutil.copy2(file_path, stored_path)
            doc_info['file_path'] = str(stored_path)

            # Save to database
            self.documents.append(doc_info)
            self.save_database()

            return doc_info

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def _get_category(self, doc_type: DocumentType) -> DocumentCategory:
        """Get document category based on type"""
        category_mapping = {
            DocumentType.PAN: DocumentCategory.IDENTITY,
            DocumentType.AADHAR: DocumentCategory.IDENTITY,
            DocumentType.INCOME_STATEMENT: DocumentCategory.FINANCIAL,
            DocumentType.SALARY_SLIP: DocumentCategory.FINANCIAL,
            DocumentType.ITR: DocumentCategory.FINANCIAL,
            DocumentType.BANK_STATEMENT: DocumentCategory.FINANCIAL,
            DocumentType.CREDIT_CARD: DocumentCategory.BANK_APPLICATION,
            DocumentType.LOAN: DocumentCategory.BANK_APPLICATION
        }
        return category_mapping[doc_type]

    def get_documents_by_person(self, person_identifier: str) -> List[Dict]:
        """Get all documents for a person"""
        return [doc for doc in self.documents
                if doc['person_name'] == person_identifier
                or doc['person_id'] == person_identifier]

    def get_documents_by_type(self, doc_type: str) -> List[Dict]:
        """Get all documents of a specific type"""
        return [doc for doc in self.documents
                if doc['document_type'] == doc_type]

class DocumentManagementSystem:
    """User interface for document management"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.console = Console()

    def run(self):
        """Run the document management system"""
        try:
            while True:
                choice = self.display_menu()

                if choice == "1":
                    self.upload_document()
                elif choice == "2":
                    self.view_documents_by_person()
                elif choice == "3":
                    self.view_documents_by_type()
                elif choice == "4":
                    self.display_summary()
                elif choice == "5":
                    break

                if choice != "5":
                    if not Confirm.ask("\nDo you want to perform another operation?"):
                        break

            self.console.print("[green]Thank you for using Document Management System![/green]")

        except Exception as e:
            self.console.print(f"[red]Error: {str(e)}[/red]")
        finally:
            self.processor.pdf_processor.cleanup()

    def display_menu(self):
        menu = """
        1. Upload New Document
        2. View Documents by Person
        3. View Documents by Type
        4. View Document Summary
        5. Exit
        """
        self.console.print(Panel(menu, title="Document Management System",
                               border_style="blue"))
        return Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])

    def upload_document(self):
        self.console.print("\n[bold blue]Upload New Document[/bold blue]")
        file_path = Prompt.ask("Enter the path to PDF document")

        try:
            doc_info = self.processor.process_document(file_path)
            self.console.print("\n[green]Document uploaded successfully![/green]")
            self.display_document_info(doc_info)
        except Exception as e:
            self.console.print(f"\n[red]Error: {str(e)}[/red]")

    def display_document_info(self, doc_info: Dict):
        table = Table(title="Document Information")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in doc_info.items():
            if key != "metadata":
                table.add_row(key, str(value))

        self.console.print(table)

    def view_documents_by_person(self):
        self.console.print("\n[bold blue]View Documents by Person[/bold blue]")
        identifier = Prompt.ask("Enter person name or ID")

        docs = self.processor.get_documents_by_person(identifier)
        self.display_documents_list(docs)

    def view_documents_by_type(self):
        self.console.print("\n[bold blue]View Documents by Type[/bold blue]")

        # Display available document types
        table = Table(title="Available Document Types")
        table.add_column("Type", style="cyan")
        for doc_type in DocumentType:
            table.add_row(doc_type.value)
        self.console.print(table)

        doc_type = Prompt.ask("Enter document type")
        docs = self.processor.get_documents_by_type(doc_type)
        self.display_documents_list(docs)

    def display_documents_list(self, documents: List[Dict]):
        if not documents:
            self.console.print("[yellow]No documents found matching the criteria.[/yellow]")
            return

        table = Table(title="Documents List")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Person Name", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Upload Date", style="blue")

        for doc in documents:
            table.add_row(
                doc['id'],
                doc['document_type'],
                doc['person_name'],
                f"{doc.get('confidence_score', 0):.2f}",
                doc.get('metadata', {}).get('extraction_date', 'N/A')
            )

        self.console.print(table)

    def display_summary(self):
        self.console.print("\n[bold blue]Document Summary[/bold blue]")

        if not self.processor.documents:
            self.console.print("[yellow]No documents found in the system.[/yellow]")
            return

        # Create summary by document type
        summary = {}
        for doc in self.processor.documents:
            doc_type = doc['document_type']
            summary[doc_type] = summary.get(doc_type, 0) + 1

        table = Table(title="Document Summary")
        table.add_column("Document Type", style="cyan")
        table.add_column("Count", style="magenta")

        for doc_type, count in summary.items():
            table.add_row(str(doc_type), str(count))

        self.console.print(table)

def main():
    """Main entry point of the application"""
    try:
        # Print welcome message
        console = Console()
        console.print("\n[bold blue]Welcome to Document Management System[/bold blue]")
        console.print("This system helps you organize and classify your PDF documents.\n")

        # Check for required dependencies
        requirements = ["tesseract", "poppler"]
        missing_deps = []

        # Check Tesseract
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            missing_deps.append("tesseract")

        if missing_deps:
            console.print("[red]Missing required system dependencies:[/red]")
            for dep in missing_deps:
                console.print(f"- {dep}")
            console.print("\nPlease install the missing dependencies and try again.")
            console.print("Installation instructions:")
            console.print("Ubuntu/Debian: sudo apt-get install tesseract-ocr poppler-utils")
            console.print("MacOS: brew install tesseract poppler")
            console.print("Windows: Download and install Tesseract and add to PATH")
            return

        # Initialize and run the system
        dms = DocumentManagementSystem()
        dms.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Program interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]An unexpected error occurred: {str(e)}[/red]")
    finally:
        console.print("\n[blue]Closing Document Management System...[/blue]")

if __name__ == "_main_":
    main()

main()
