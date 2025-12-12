# LiaPlusAI
This project implements a web-based chatbot with both conversation-level and statement-level sentiment analysis. It maintains full conversation history and summarises the trend or shift in mood across the conversation.

# How to run

1. Install dependencies

```python
pip install streamlit nltk
```
2. Download VADER Lexicon

```python
import nltk
nltk.download('vader_lexicon')
```
3. Paste the source code

4. Run the Streamlit app in the project directory:
```python
streamlit run app.py
```
5. Streamlit will open the chat interface in your browser

# Chosen Technologies

1. Language: 

   Python 3.8+

2. Libraries: 

   Streamlit — Web UI

   NLTK (VADER) — Sentiment analysis

   Statistics module — Aggregations

3. VADER:

   Lightweight and optimized for short, conversational sentences.

   Provides deterministic scoring suitable for consistent evaluation.

   No API keys or large model dependencies.

# Explanation of sentiment logic

1. Conversation-level sentiment (Tier 1)

   -Take all user compound scores\
   -Compute the average\
   -Convert the average to a label using the same thresholds\
   -Display:
   
       Final label
       Average compound score
       Explanation ("Overall positive sentiment", "Mixed/neutral", etc.)

2. Per-message sentiment (Tier 2)

   Each user message is passed to VADER:
   
       scores = {
         "neg": ...,
         "neu": ...,
         "pos": ...,
         "compound": ...
       }

   The compond score is used to determine sentiment:

          | Compound Value | Sentiment Label |
          | -------------- | --------------- |
          | ≥ 0.05         | Positive        |
          | ≤ −0.05        | Negative        |
          | otherwise      | Neutral         |

    Each message is stored in the conversation history with its sentiment label and raw scores.

3. Mood Trend Analysis (Tier 2 Enhancement)

     To detect sentiment shift:
   
     -Split compound scores into first vs last third\
     -Compare averages\
     -Output examples:
   
           “Conversation shifted towards more positive sentiment.”
           “Sentiment remained stable."
           “Conversation shifted towards more negative sentiment.”


# Status of Tier 2 implementation

  Fully implemented that includes:
  
  -Sentiment evaluation for every user message individually\
  -Display of each message with sentiment\
  -Trend shift and summary
