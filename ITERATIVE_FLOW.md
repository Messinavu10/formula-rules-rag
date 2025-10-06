# Agent Flow with Multi-Tool Orchestration

## Advanced Architecture with Parallel and Sequential Tool Execution

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                    REASON NODE                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Classify Intent (6 categories)                           ││
│  │ 2. Determine if multi-tool orchestration needed             ││
│  │ 3. Select single tool OR multiple tools                     ││
│  │ 4. Generate reasoning for tool selection                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │  SINGLE TOOL?    │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   YES: Execute    │
        │   single tool     │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   NO: Multi-tool  │
        │   orchestration   │
        └─────────┬─────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  ACT NODE                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ SINGLE TOOL EXECUTION:                                      ││
│  │ • Execute selected tool                                     ││
│  │ • Store result in state                                     ││
│  │                                                             ││
│  │ MULTI-TOOL ORCHESTRATION:                                   ││
│  │ • Execute tools in parallel where possible                  ││
│  │ • Combine results intelligently                             ││
│  │ • Handle tool dependencies                                  ││
│  │ • Store all results in multi_tool_results                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  REFLECT NODE                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ SINGLE TOOL:                                                ││
│  │ • Generate final answer from tool result                    ││
│  │                                                             ││
│  │ MULTI-TOOL:                                                 ││
│  │ • Combine results from multiple tools                       ││
│  │ • Generate comprehensive answer                             ││
│  │ • Cross-reference information                               ││
│  │ • Store combined answer in state                            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│              SHOULD_CONTINUE NODE                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Check if tool result(s) exist                            ││
│  │ 2. Use LLM to assess answer quality (1-10 scale)            ││
│  │ 3. Check iteration count (max 3 attempts)                   ││
│  │                                                             ││
│  │ Decision Logic:                                             ││
│  │ • No result → CONTINUE                                      ││
│  │ • Quality < 7/10 → CONTINUE                                 ││
│  │ • Quality >= 7/10 → END                                     ││
│  │ • Max iterations reached → END                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   CONTINUE?      │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   YES: Go back    │
        │   to REASON       │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   NO: Return      │
        │   final answer    │
        └───────────────────┘

```

## 🔄 Multi-Tool Orchestration Strategies

### 1. **Single Tool Execution**
```
Simple queries that need one specific tool:
- "What are the safety requirements?" → regulation_search
- "Compare 2024 vs 2025 regulations" → regulation_comparison
- "What are the penalties?" → penalty_lookup
```

### 2. **Multi-Tool Orchestration**
```
Complex queries requiring multiple tools:
- "What are the safety requirements and penalties?" 
  → [regulation_search, penalty_lookup]
- "Summarize technical regulations and compare years"
  → [regulation_summary, regulation_comparison]
- "Find safety rules and penalty information for violations"
  → [regulation_search, penalty_lookup, regulation_summary]
```

### 3. **Tool Combination Strategies**
```
PARALLEL EXECUTION:
- Independent tools run simultaneously
- Results combined intelligently
- Faster response times

SEQUENTIAL EXECUTION:
- Tools with dependencies run in order
- Results from one tool inform the next
- More comprehensive analysis
```

### 4. **Query Refinement**
```
Original: "What are the safety requirements?"
Iteration 1: "Formula 1 car safety requirements technical regulations"
Iteration 2: "FIA technical regulations safety systems crash protection"
```

### 5. **Tool Switching**
```
Original Tool: regulation_search
Switch To: regulation_summary (for comprehensive analysis)
```

### 6. **Quality Assessment**
```
LLM evaluates on 4 criteria:
- Completeness (1-10)
- Accuracy (1-10) 
- Clarity (1-10)
- Specificity (1-10)

Threshold: 7/10 to continue
```

## 🛡️ Safety Mechanisms

### 1. **Maximum Iterations**
- Hard limit: 3 iterations
- Prevents infinite loops
- Graceful degradation

### 2. **Error Handling**
- Try/catch around all operations
- Fallback to previous result
- Detailed error logging

### 3. **State Management**
- Track all reasoning steps
- Store tool results
- Maintain conversation context

## 📊 Example Execution Flows

### Single Tool Execution
```
Question: "What are the safety requirements for Formula 1 cars?"

Iteration 1:
├── Intent: SEARCH
├── Tool: regulation_search
├── Query: "What are the safety requirements for Formula 1 cars?"
├── Result: "Safety requirements include..."
├── Quality: 8/10 (complete)
└── Decision: END

Final Answer: "Safety requirements with citations..."
```

### Multi-Tool Orchestration
```
Question: "What are the safety requirements and penalties for violations?"

Iteration 1:
├── Intent: MULTI_TOOL
├── Tools: [regulation_search, penalty_lookup]
├── Parallel Execution:
│   ├── regulation_search: "Safety requirements include..."
│   └── penalty_lookup: "Penalties for violations include..."
├── Combined Result: "Comprehensive safety and penalty information..."
├── Quality: 9/10 (complete)
└── Decision: END

Final Answer: "Combined safety requirements and penalties with citations..."
```

### Iterative Refinement with Multi-Tool
```
Question: "Summarize all safety requirements for Formula 1 cars"

Iteration 1:
├── Intent: SUMMARY
├── Tool: regulation_summary
├── Query: "Summarize all safety requirements for Formula 1 cars"
├── Result: "Basic safety info..."
├── Quality: 6/10 (incomplete)
└── Decision: CONTINUE

Iteration 2:
├── Strategy: MULTI_TOOL_ORCHESTRATION
├── Tools: [regulation_search, regulation_summary]
├── Parallel Execution:
│   ├── regulation_search: "Detailed safety requirements..."
│   └── regulation_summary: "Comprehensive safety overview..."
├── Combined Result: "Comprehensive safety requirements..."
├── Quality: 8/10 (complete)
└── Decision: END

Final Answer: "Comprehensive safety requirements with citations..."
```

## 🎯 Benefits

### Multi-Tool Orchestration Benefits
1. **Comprehensive Answers**: Multiple tools provide complete coverage
2. **Parallel Processing**: Faster execution for complex queries
3. **Intelligent Combination**: Results are intelligently merged
4. **Cross-Validation**: Multiple sources verify information accuracy

### Iterative Refinement Benefits
1. **Better Results**: Iterative refinement leads to higher quality answers
2. **Adaptive Strategy**: Can switch tools or refine queries as needed
3. **Quality Control**: LLM-based quality assessment ensures good results
4. **Safety**: Maximum iteration limits prevent infinite loops

### System Benefits
5. **Transparency**: Full reasoning steps are logged and visible
6. **Flexibility**: Can handle both simple and complex queries effectively
7. **Scalability**: Can orchestrate multiple tools efficiently
8. **Robustness**: Handles tool failures gracefully with fallbacks

## 🔧 Configuration

```python
# Maximum iterations (configurable)
MAX_ITERATIONS = 3

# Quality threshold (configurable)  
QUALITY_THRESHOLD = 7.0

# Multi-tool orchestration settings
MULTI_TOOL_ENABLED = True
MAX_PARALLEL_TOOLS = 3
TOOL_TIMEOUT_SECONDS = 30

# Available refinement strategies
STRATEGIES = [
    "REFINE_QUERY", 
    "SWITCH_TOOL", 
    "MULTI_TOOL_ORCHESTRATION"
]

# Available tools for single and multi-tool execution
TOOLS = [
    "regulation_search",
    "regulation_summary", 
    "general_rag",
    "regulation_comparison",
    "penalty_lookup",
    "out_of_scope_handler"
]

# Tool combination strategies
TOOL_COMBINATIONS = {
    "safety_and_penalties": ["regulation_search", "penalty_lookup"],
    "comprehensive_analysis": ["regulation_search", "regulation_summary"],
    "comparison_analysis": ["regulation_comparison", "regulation_summary"]
}

# Parallel execution settings
PARALLEL_EXECUTION = {
    "enabled": True,
    "max_workers": 3,
    "timeout": 30
}
```
