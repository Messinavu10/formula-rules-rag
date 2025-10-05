# 🔄 Iterative Agent Flow

## Enhanced Architecture with Multi-Tool Execution

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUESTION                            │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                    REASON NODE                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Classify Intent (6 categories)                           ││
│  │ 2. Select Tool based on intent                              ││
│  │ 3. Generate reasoning for tool selection                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────-┐
│                     ACT NODE                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Check if this is a refinement attempt:                      │ │
│  │                                                             │ │
│  │ IF First Attempt:                                           │ │
│  │   • Use original question                                   │ │
│  │   • Execute selected tool                                   │ │
│  │                                                             │ │
│  │ IF Refinement Attempt:                                      │ │
│  │   • Analyze previous result quality                         │ │
│  │   • Choose strategy:                                        │ │
│  │     - REFINE_QUERY: Better search terms                     │ │
│  │     - SWITCH_TOOL: Try different tool                       │ │
│  │   • Execute with refined approach                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────────────────────────-┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│                  REFLECT NODE                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Generate final answer from tool result                   ││
│  │ 2. Store answer in state                                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────────┐
│              SHOULD_CONTINUE NODE                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Check if tool result exists                              ││
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

## 🔄 Iteration Strategies

### 1. **Query Refinement**
```
Original: "What are the safety requirements?"
Iteration 1: "Formula 1 car safety requirements technical regulations"
Iteration 2: "FIA technical regulations safety systems crash protection"
```

### 2. **Tool Switching**
```
Original Tool: regulation_search
Switch To: regulation_summary (for comprehensive analysis)
```

### 3. **Quality Assessment**
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

## 📊 Example Execution Flow

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
├── Strategy: REFINE_QUERY
├── Refined Query: "FIA technical regulations safety systems crash protection roll cage"
├── Result: "Comprehensive safety requirements..."
├── Quality: 8/10 (complete)
└── Decision: END

Final Answer: "Comprehensive safety requirements with citations..."
```

## 🎯 Benefits

1. **Better Results**: Iterative refinement leads to higher quality answers
2. **Adaptive Strategy**: Can switch tools or refine queries as needed
3. **Quality Control**: LLM-based quality assessment ensures good results
4. **Safety**: Maximum iteration limits prevent infinite loops
5. **Transparency**: Full reasoning steps are logged and visible
6. **Flexibility**: Can handle both simple and complex queries effectively

## 🔧 Configuration

```python
# Maximum iterations (configurable)
MAX_ITERATIONS = 3

# Quality threshold (configurable)  
QUALITY_THRESHOLD = 7.0

# Available refinement strategies
STRATEGIES = ["REFINE_QUERY", "SWITCH_TOOL"]

# Available tools for switching
TOOLS = [
    "regulation_search",
    "regulation_summary", 
    "general_rag",
    "regulation_comparison",
    "penalty_lookup"
]
```
