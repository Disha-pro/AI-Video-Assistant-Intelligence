from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "english") -> dict:
    print("\n🚀 Starting AI Video Assistant...\n")

    # Step 1: Download/Process Audio
    chunks = process_input(source)

    # Step 2: Transcription
    transcript = transcribe_all(chunks, language=language)

    print("\n" + "=" * 60)
    print("📝 Transcript Preview")
    print("=" * 60)
    print(transcript[:300] + "...\n")

    # Step 3: LLM Tasks
    title = generate_title(transcript)
    summary = summarize(transcript)

    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    open_questions = extract_questions(transcript)

    # Step 4: Build RAG
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": open_questions,
        "rag_chain": rag_chain,
    }


def main():
    source = input("🎥 Enter YouTube URL or local file path:\n> ").strip()

    language = (
        input("🌐 Language (english/hinglish) [english]: ").strip().lower()
        or "english"
    )

    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print("📌 TITLE")
    print("=" * 60)
    print(result["title"])

    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(result["summary"])

    print("\n" + "=" * 60)
    print("✅ ACTION ITEMS")
    print("=" * 60)
    print(result["action_items"])

    print("\n" + "=" * 60)
    print("🔑 KEY DECISIONS")
    print("=" * 60)
    print(result["key_decisions"])

    print("\n" + "=" * 60)
    print("❓ OPEN QUESTIONS")
    print("=" * 60)
    print(result["open_questions"])

    print("\n" + "=" * 60)
    print("💬 Chat with your Meeting")
    print("Type 'exit' to quit")
    print("=" * 60)

    rag_chain = result["rag_chain"]

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant:\n{answer}")


if __name__ == "__main__":
    main()