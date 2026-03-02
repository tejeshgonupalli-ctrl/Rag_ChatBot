from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# embeddings = client.models.embeddings.create(model="gemini-embedding-001")

# embeddings.embed_content("hello, world!")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
print(embeddings.embed_query("hello, world!"))

# print(embeddings)  # Verify that the embeddings object is created successfully