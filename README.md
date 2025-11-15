# PartSelect Assistant

An intelligent multi-agent system for appliance parts discovery, troubleshooting, and repair guidance. Built with a 3-agent architecture powered by DeepSeek AI.

<img width="1918" height="1006" alt="image" src="https://github.com/user-attachments/assets/bde06106-a475-4f42-a5ea-d043e31a78c2" />


## Overview

A conversational AI system that helps users find appliance parts, troubleshoot issues, and access repair guides. Uses specialized agents that collaborate to provide comprehensive responses with full product data, repair instructions, and video tutorials.

## Data Pipeline

### Web Scraping & JSON Database Creation

The foundation of this project is a comprehensive JSON-based database created through systematic web scraping of PartSelect's product catalog, repair guides, and blog articles.

**Data Sources:**
1. **Products Database** (`products/*.json`)
   - Part numbers, names, and descriptions
   - Pricing and availability
   - Brand and model compatibility
   - Specifications and images
   - Symptoms fixed by each part

2. **Repair Guides** (`repairs/*.json`)
   - Step-by-step repair instructions
   - Required tools and parts
   - Video tutorials
   - Difficulty ratings and time estimates

3. **Blog Articles** (`blogs/*.json`)
   - How-to guides
   - Troubleshooting tips
   - Appliance maintenance advice

**Scraping Process:**
- Custom Python scrapers for each data type
- JSON schema validation
- Deduplication and data cleaning
- Regular updates to maintain currency

## System Architecture

The system uses a **3-agent orchestration pattern** with specialized roles:

<img width="753" height="775" alt="image" src="https://github.com/user-attachments/assets/f0cf23bd-cbb4-4c4f-adf6-8f12673b75b9" />


### Components

#### 1. **Router Agent** (`agent_router.py`)
- **Role:** Query analysis and tool selection
- **Tools:**
  - `search_products` - Find parts and pricing
  - `search_repairs` - Locate repair guides
  - `search_blogs` - Search articles and tips
- **Features:**
  - Multi-tool calling capability
  - Context-aware routing
  - Fallback handling

#### 2. **Data Retriever** (`data_retriever.py`)
- **Role:** Fetch data from JSON databases
- **Operations:**
  - Text-based search with fuzzy matching
  - Brand and appliance type filtering
  - Top-K retrieval (default: 5 per tool)
- **Databases:**
  - Products: `./database/products/`
  - Repairs: `./database/repairs/`
  - Blogs: `./database/blogs/`

#### 3. **Response Agent** (`response_agent.py`)
- **Role:** Natural language generation
- **Capabilities:**
  - Full context awareness (all retrieved data)
  - Product comparisons and recommendations
  - Detailed repair instructions
  - URL citation and linking
  - Media extraction (images, videos)
- **Output:**
  - Response text
  - Formatted products for UI
  - Media dictionary (images, videos)

#### 4. **Summary Agent** (`summary_agent.py`)
- **Role:** Conversation context management
- **Features:**
  - Rolling conversation history
  - Context compression for long conversations
  - Session reset capability

#### 5. **Orchestrator** (`orchestrator.py`)
- **Role:** Pipeline coordination
- **Workflow:**
  ```
  Query → Router → Data Retrieval → Response Generation → Context Update → Output
  ```

## Features

**Core Capabilities:**
- Intelligent multi-database search with relevance ranking
- Context-aware multi-turn conversations
- Step-by-step repair instructions with video tutorials
- Product comparison and recommendations
- Shopping cart with running totals
- Streamlit-based UI

## Installation

### Prerequisites
- Python 3.8+
- DeepSeek API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd partselect-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file:
   ```env
   DEEPSEEK_API_KEY=your_api_key_here
   DEEPSEEK_MODEL=deepseek-chat
   DEEPSEEK_MODEL2=deepseek-chat  # For evaluation
   ```

4. **Verify database structure**
   ```
   database/
   ├── products/
   │   └── *.json
   ├── repairs/
   │   └── *.json
   └── blogs/
       └── *.json
   ```

## Usage

### Running the Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`

### Example Queries

**Product Search:**
```
"Find Whirlpool dishwasher door latches under $50"
"Compare Samsung refrigerator water filters"
```

**Troubleshooting:**
```
"My GE washer won't spin"
"Refrigerator is leaking water"
```

**Repair Guidance:**
```
"How do I replace a dryer heating element?"
"Steps to fix a dishwasher that won't drain"
```

**General Information:**
```
"How often should I clean my dishwasher filter?"
"What causes a refrigerator to freeze food?"
```

## Project Structure

```
partselect-assistant/
├── app.py                  # Streamlit UI application
├── orchestrator.py         # Main coordination logic
├── agent_router.py         # Agent 1: Query routing
├── response_agent.py       # Agent 2: Response generation
├── data_retriever.py       # Data access layer
├── summary_agent.py        # Context management
├── utils/
│   └── llm.py             # DeepSeek API client
├── evaluation.py          # Evaluation framework
├── database/
│   ├── products/          # Product JSON files
│   ├── repairs/           # Repair guide JSON files
│   ├── blogs/             # Article JSON files
│   └── images/            # UI assets
├── eval/
│   └── eval_set.json      # Test cases
├── requirements.txt
└── .env
```

## Evaluation

Automated evaluation framework using DeepSeek-reasoner as a judge.

### Running Evaluation

```bash
python evaluation.py
```


