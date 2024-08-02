# from langchain_community.vectorstores import PGVector
# from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import os
# from dotenv import load_dotenv
# load_dotenv()

# collection = "Name of your collection"
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# connection_string = f"postgresql+psycopg2://{os.getenv('PGVECTOR_USER')}:{os.getenv('PGVECTOR_PASSWORD')}@{os.getenv('PGVECTOR_HOST')}:{os.getenv('PGVECTOR_PORT')}/{os.getenv('PGVECTOR_DATABASE')}"

# docs = [
#     Document(
#         page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
#         metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
#     ),
#     Document(
#         page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
#         metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
#     ),
#     Document(
#         page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
#         metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
#     ),
#     Document(
#         page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
#         metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
#     ),
#     Document(
#         page_content="Toys come alive and have a blast doing so",
#         metadata={"year": 1995, "genre": "animated"},
#     ),
#     Document(
#         page_content="Three men walk into the Zone, three men walk out of the Zone",
#         metadata={
#             "year": 1979,
#             "director": "Andrei Tarkovsky",
#             "genre": "science fiction",
#             "rating": 9.9,
#         },
#     ),
# ]
# vectorstore = PGVector.from_documents(
#     docs,
#     embeddings,
#     collection_name=collection,
#     connection_string=connection_string,
# )


# from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI

# metadata_field_info = [
#     AttributeInfo(
#         name="genre",
#         description="The genre of the movie",
#         type="string or list[string]",
#     ),
#     AttributeInfo(
#         name="year",
#         description="The year the movie was released",
#         type="integer",
#     ),
#     AttributeInfo(
#         name="director",
#         description="The name of the movie director",
#         type="string",
#     ),
#     AttributeInfo(
#         name="rating", description="A 1-10 rating for the movie", type="float"
#     ),
# ]
# document_content_description = "Brief summary of a movie"
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
# retriever = SelfQueryRetriever.from_llm(
#     llm, vectorstore, document_content_description, metadata_field_info, verbose=True
# )

# print(retriever.invoke("What are some movies about dinosaurs"))


# --------------------------------------------------------------
# use text file and metadata
# --------------------------------------------------------------

import os
import json
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Function to read and parse the metadata field info from a text file
def read_metadata_field_info(file_path):
    with open(file_path, 'r') as file:
        metadata_field_info_list = json.load(file)
    return [
        AttributeInfo(
            name=field["name"],
            description=field["description"],
            type=field["type"]
        ) for field in metadata_field_info_list
    ]

metadata_field_info = read_metadata_field_info("metadata_field_info.txt")

# Reading and parsing the documents text file
def read_documents(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    docs = []
    entries = content.strip().split("\n\n")
    for entry in entries:
        lines = entry.strip().split("\n")
        page_content = lines[1]
        metadata = json.loads(lines[3])
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

docs = read_documents("documents.txt")

# Setting up embeddings and vectorstore
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

connection_string = f"postgresql+psycopg2://{os.getenv('PGVECTOR_USER')}:{os.getenv('PGVECTOR_PASSWORD')}@{os.getenv('PGVECTOR_HOST')}:{os.getenv('PGVECTOR_PORT')}/{os.getenv('PGVECTOR_DATABASE')}"

vectorstore = PGVector.from_documents(
    docs,
    embeddings,
    collection_name="my_custom_table",
    connection_string=connection_string,
)

document_content_description = "Brief summary of a movie"
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)

print(retriever.invoke("What are some movies about dinosaurs"))


# --------------------------------------------------------------
# use text file and metadata custom table
# --------------------------------------------------------------


# import os
# import json
# from sqlalchemy import create_engine, Table, Column, Integer, Text, JSON, MetaData
# from sqlalchemy.dialects.postgresql import insert
# from sqlalchemy.orm import sessionmaker
# from langchain_community.vectorstores import PGVector
# from langchain_core.documents import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv
# from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Load environment variables
# load_dotenv()

# # Database connection string
# connection_string = f"postgresql+psycopg2://{os.getenv('PGVECTOR_USER')}:{os.getenv('PGVECTOR_PASSWORD')}@{os.getenv('PGVECTOR_HOST')}:{os.getenv('PGVECTOR_PORT')}/{os.getenv('PGVECTOR_DATABASE')}"

# # Set up database engine and metadata
# engine = create_engine(connection_string)
# metadata = MetaData()

# # Define custom table schema
# movie_vectors = Table(
#     'movie_vectors', metadata,
#     Column('id', Integer, primary_key=True),
#     Column('content', Text),
#     Column('metadata', JSON),
#     Column('embedding', Text)  # Adjust type if necessary for VECTOR
# )

# # Create table
# metadata.create_all(engine)

# # Function to read documents from a text file
# def read_documents(file_path):
#     with open(file_path, 'r') as file:
#         content = file.read()

#     docs = []
#     entries = content.strip().split("\n\n")
#     for entry in entries:
#         lines = entry.strip().split("\n")
#         page_content = lines[1]
#         metadata = json.loads(lines[3])
#         docs.append(Document(page_content=page_content, metadata=metadata))
#     return docs

# # Function to read and parse the metadata field info from a text file
# def read_metadata_field_info(file_path):
#     with open(file_path, 'r') as file:
#         metadata_field_info_list = json.load(file)
#     return [
#         AttributeInfo(
#             name=field["name"],
#             description=field["description"],
#             type=field["type"]
#         ) for field in metadata_field_info_list
#     ]

# # Load documents and metadata field info from text files
# docs = read_documents("documents.txt")
# metadata_field_info = read_metadata_field_info("metadata_field_info.txt")

# # Set up embeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Insert documents into custom table
# Session = sessionmaker(bind=engine)
# session = Session()

# for doc in docs:
#     embedding = embeddings.embed_documents(doc.page_content)
#     insert_stmt = insert(movie_vectors).values(
#         content=doc.page_content,
#         metadata=doc.metadata,
#         embedding=embedding
#     ).on_conflict_do_nothing()
#     session.execute(insert_stmt)

# session.commit()

# # Document content description
# document_content_description = "Brief summary of a movie"
# llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# # Create PGVector instance with custom table
# vectorstore = PGVector(
#     embedding_function=embeddings,
#     collection_name="movie_vectors",
#     connection_string=connection_string,
#     table=movie_vectors
# )

# # Set up SelfQueryRetriever
# retriever = SelfQueryRetriever.from_llm(
#     llm, vectorstore, document_content_description, metadata_field_info, verbose=True
# )

# # Example query
# print(retriever.invoke("What are some movies about dinosaurs"))
