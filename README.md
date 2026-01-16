# AgenticAI-Bootcamp

## Setup Instructions

### Windows (Anaconda Prompt)
Navigate to the current working directory and run the following commands:

```bash
conda create --prefix ./env python=3.12 -y
conda activate ./env
pip install -r requirements.txt
pip install python-dotenv
python main.py
```
GOOGLE_API_KEY : https://aistudio.google.com/api-keys

## Git Helper Commands

To remove a file from git's index (stop tracking):

```bash
git rm --cached path/to/your/file.txt
git commit -m "Stop tracking and ignore path/to/your/file.txt"
```

### Langchain
- Open-source framework for LLM apps
- Compose prompts, tools, memory and workflows
- Use with many providers (OpenAI, NVIDIA, local models)

## why use Langchain
- Fast from prototype --> production
- Batteries included: retrievers, agents, evaluators

## Langchain Core building blocks
- Models(chat/LLM, embeddings)
- Prompts(PromptTemplate, ChatPromptTemplate)
- Chains( Sequiential, router, map-reduce)
- Memory (buffer/window, vector)
- Tools( Custom functions, APIs)
- Retrievers & VectorStores

# Chain Types at a Glance
- LLMChain: prompt --> model --> parsed output
- SequentialChain: multi-step workflows
- RouterChain: route by topic/task

# Prompts & Output Parsing
- ChatPromptTemplate with variables and system/human messages
- Structured output with Pydantic/JSON parser
- Guard against empty / unknown fields

# PromptTemplate Basics
- Template strings with {variables}
- Provide values at runtime
- Partial variables for fixed context

# ChatPromptTemplate
- Helps us to assemble multimessage prompt
- System message: behavior and constraints
- Human message: tasks and inputs
- AI message: context carryover when needed

# Style & Tone Controls
- Role instruction: persona,audience, constraints
- Output format: bullet list, table, JSON keys
- Length, temperature hints, guard words

# Few-Shot Prompting
- Add 1-3 examples to teach pattern
- Keep short and consistent
- Swap examples per domain ( routing )

# Structured output
- Request JSON with explicit keys
- Use Output Parsers ( JSON / Pydantic)
- Validate fields; fallback on parse errors

# Guards and Safety Cues
- Define boundaries in system message
- Ask for 'cannot comply' when out-of-scope
- Mask secrets and avoid speculative claims