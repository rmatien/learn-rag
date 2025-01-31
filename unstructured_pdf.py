PDF_FILE = 'pdf/draf panduan SNBP web 2025.pdf'

# from unstructured.partition.auto import partition
# elements = partition(PDF_FILE)

# from unstructured.partition.pdf import partition_pdf
# elements = partition_pdf(PDF_FILE)

from collections import Counter
# print(Counter(type(element) for element in elements))


from unstructured.partition.html import partition_html

url = "https://docs.google.com/document/d/e/2PACX-1vR4g4xwgQzdcpQ07ZV5awOD9OhQ4lLtgCTyQ3CX0ExkuOSTJI0ItppvZdKSNefPSNlXgDslsERq5oTc/pub"
elements = partition_html(url=url)
# print(Counter(type(element) for element in elements))

# # Print each element's type and content
# for element in elements:
#     print(f"Type: {type(element)}")
#     print(f"Content: {element}")
#     print("-" * 50)  # Separator line

# # Print summary of element types
# print("Summary of element types:")
# print(Counter(type(element) for element in elements))

# Print only elements of type 'table'
from unstructured.documents.elements import Table, Text, ListItem, NarrativeText, Title 
for element in elements:
    if isinstance(element, NarrativeText):
        print(f"Type: {type(element)}")
        print(f"Content: {element}")
        print("-" * 50)  # Separator line