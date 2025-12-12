import streamlit as st
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from statistics import mean

nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict:
        """
        Returns the VADER scores dictionary, e.g. {neg, neu, pos, compound}
        """
        return self.analyzer.polarity_scores(text)

    def label_from_compound(self, compound: float, thresholds=(0.05, -0.05)) -> str:
        """
        Map compound score to label.
        thresholds: (positive_threshold, negative_threshold)
        """
        pos_th, neg_th = thresholds
        if compound >= pos_th:
            return "Positive"
        elif compound <= neg_th:
            return "Negative"
        else:
            return "Neutral"

    def conversation_sentiment(self, compound_scores: list) -> dict:
        """
        Aggregate conversation-level sentiment.
        We'll compute average compound and map to label.
        Also return min/max and simple interpretation.
        """
        if not compound_scores:
            return {"avg_compound": 0.0, "label": "Neutral", "explanation": "No messages."}
        avg = mean(compound_scores)
        label = self.label_from_compound(avg)
        explanation = ""
        if label == "Positive":
            explanation = "Overall positive sentiment."
        elif label == "Negative":
            explanation = "Overall negative sentiment."
        else:
            explanation = "Mixed/neutral sentiment."
        return {"avg_compound": avg, "label": label, "explanation": explanation}

    def trend_summary(self, compound_scores: list) -> str:
        """
        Very simple trend: compare first third vs last third average to detect shift.
        Returns a short summary.
        """
        n = len(compound_scores)
        if n < 2:
            return "Not enough messages to detect trend."
        first_third = compound_scores[: max(1, n // 3)]
        last_third = compound_scores[- max(1, n // 3):]
        avg_first = mean(first_third)
        avg_last = mean(last_third)
        diff = avg_last - avg_first
        if abs(diff) < 0.05:
            return "Sentiment remained roughly stable over the conversation."
        elif diff > 0:
            return "Conversation shifted towards more positive sentiment over time."
        else:
            return "Conversation shifted towards more negative sentiment over time."

st.set_page_config(page_title="Chatbot with Sentiment Analysis", layout="wide")

st.title("Chatbot with Conversation & Statement-level Sentiment")
st.write("Implements Tier 1 (conversation-level) and Tier 2 (statement-level).")

if "history" not in st.session_state:
    st.session_state.history = []
if "compound_scores" not in st.session_state:
    st.session_state.compound_scores = []

sent = SentimentAnalyzer()

with st.sidebar:
    st.header("Options")
    show_trend = st.checkbox("Show sentiment trend summary", value=True)
    show_per_message = st.checkbox("Show per-message sentiment (Tier 2)", value=True)
    model_info = st.expander("Sentiment logic (VADER)")
    model_info.write("""
    We're using NLTK VADER:
    - Produces `neg`, `neu`, `pos`, and `compound` scores.
    - `compound` is used to label sentiments:
      compound >= 0.05 → Positive
      compound <= -0.05 → Negative
      otherwise → Neutral
    """)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    s = sent.score(user_input)
    label = sent.label_from_compound(s["compound"])
    st.session_state.history.append({"sender": "User", "text": user_input, "sentiment": {"scores": s, "label": label}})
    st.session_state.compound_scores.append(s["compound"])

    if label == "Negative":
        bot_reply = "I'm sorry to hear that. Could you share more details so I can help?"
    elif label == "Positive":
        bot_reply = "That's great to hear! Tell me more."
    else:
        bot_reply = "Thanks for sharing. Anything else you'd like to add?"

    st.session_state.history.append({"sender": "Bot", "text": bot_reply, "sentiment": None})

st.subheader("Conversation")
for entry in st.session_state.history:
    if entry["sender"] == "User":
        if show_per_message:
            scores = entry["sentiment"]["scores"]
            label = entry["sentiment"]["label"]
            st.markdown(f"**User:** {entry['text']}  \n> Sentiment: **{label}** (compound={scores['compound']:.3f})")
        else:
            st.markdown(f"**User:** {entry['text']}")
    else:
        st.markdown(f"**Chatbot:** {entry['text']}")

st.markdown("---")
if st.button("End Conversation and Analyze"):
    conv = sent.conversation_sentiment(st.session_state.compound_scores)
    st.subheader("Conversation-level Sentiment (Tier 1)")
    st.write(f"**Overall sentiment:** {conv['label']}")
    st.write(f"**Average compound score:** {conv['avg_compound']:.3f}")
    st.write(f"**Interpretation:** {conv['explanation']}")

    if show_trend:
        st.subheader("Trend Summary")
        st.write(sent.trend_summary(st.session_state.compound_scores))

    st.markdown("**Per-message details**")
    for i, entry in enumerate(st.session_state.history):
        if entry["sender"] == "User":
            scores = entry["sentiment"]["scores"]
            label = entry["sentiment"]["label"]
            st.write(f"{i+1}. \"{entry['text']}\" → {label} (compound={scores['compound']:.3f})")

    st.button("Start New Conversation", key="reset_button", on_click=lambda: reset_state())

def reset_state():
    st.session_state.history = []
    st.session_state.compound_scores = []


with st.expander("Raw Conversation History (Debug)"):
    st.json(st.session_state.history)
