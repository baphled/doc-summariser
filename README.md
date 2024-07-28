# Document Summariser

## Introduction

This project is a document summariser that uses Chroma to embed a single
document into a vector space. The document is then summarised by selecting the
most representative sentences from the document. The summarisation is done by
selecting the sentences that are closest to the centroid of the document in the
vector space.

We then use Ollama to generate a summary of the document and to ask questions
about the document. The questions are generated by selecting the most
representative sentences from the document and then using these sentences to
generate questions.

We've intentionally baked in the question generation into the script, but it is
easy enough to customise to be tailored to your needs.

The purpose of this script is to get a quick and dirty summary of a document and
to generate questions about the document. This can be useful for quickly
understanding the contents of a document.

## Installation

To install the required packages, run the following command:

```bash
python -m venv .doc-summariser
```

```bash
. .doc-summariser/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
ollama pull mistral
```

```bash
ollama pull nomic-embed-text
```

## Usage

Then run the following command:

```bash
python summariser.py <path_to_document>
```