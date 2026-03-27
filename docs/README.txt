SAMPLE CLINIC DOCUMENT — place real PDFs here for RAG

This folder (docs/) is for sample or reference PDF documents that can be
uploaded via the chat interface for RAG-based Q&A.

Example documents you might place here:
  - clinic_services.pdf      (list of services, pricing, doctors)
  - faq.pdf                  (frequently asked questions)
  - opening_hours.pdf        (clinic hours, location, contact)
  - patient_guide.pdf        (pre-appointment instructions)

How to use:
  1. Run the Streamlit app
  2. In the sidebar under "Upload Clinic Documents", upload any PDF
  3. The system will extract, chunk, embed, and index the content
  4. Users can then ask questions like "What are your opening hours?"
     and get answers drawn directly from the document
